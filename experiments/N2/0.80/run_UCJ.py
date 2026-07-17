"""Run the polynomial-time UCJ backpropagation energy estimate, then
variationally optimize the circuit parameters, for the N2 molecule at a
fixed bond distance.

Template for use with launch.sh: it copies this file into each bond-distance
subdirectory and replaces the placeholder bond distance below with that
directory's value before running.
"""

import glob
import os
import sys

import numpy as np
import pyscf
import pyscf.tools.fcidump
import ffsim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from fermiprop import UCJBackPropagator

# Molecule / active space parameters.
bond_distance = 0.80  # Angstrom; substituted by launch.sh from the subdirectory name.

# Parameters of the (L)UCJ ansatz.
half_layer = False                       # If True, appends a final rotation to the circuit, but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit.

# Variational optimization settings. See fermiprop/propagator.py's
# optimize_jax() docstring: chunk_size must evenly divide num_orb**4.
# None (fully unchunked) needs ~48GB for this active space -- fine on an
# H200 (143GB) but OOMs on a 32GB V100, and cluster jobs here can land on
# either, so default to something that fits comfortably on the smallest
# GPU in rotation. num_orb**3 (~26 chunks, ~1-2GB/chunk by linear
# extrapolation from the 48GB/1-chunk figure) leaves plenty of headroom on
# a 32GB card; drop to num_orb**2 (~676 chunks) if that's still too much,
# or raise towards num_orb**4 (i.e. None) if you know you're on an H200.
optimizer_method = "L-BFGS-B"
optimizer_options = {"maxiter": 500, "gtol": 1e-9, "ftol": 1e-9}
optimizer_chunk_size = None  # set below to num_orb**3 once num_orb is known

# Each bond-distance subdirectory has its own pre-generated FCIDUMP (active
# space integrals for this R), matching the 4Fe-4S workflow.
[fcidump_filename] = glob.glob("*_fcidump.txt")

# Run Hartree-Fock. N2 dissociation curves are a known hard case for RHF
# convergence (near-degenerate/symmetry-breaking orbitals as the bond
# stretches), so start with damping + level-shifting for stability, and
# only escalate to the second-order (Newton) SCF solver if that alone
# isn't enough.
mf = pyscf.tools.fcidump.to_scf(fcidump_filename)
mf.max_cycle = 300
mf.conv_tol = 1e-9
mf.level_shift = 0.3
mf.damp = 0.3
mf.kernel()
if not mf.converged:
    mf = mf.newton()
    mf.max_cycle = 200
    mf.kernel()
assert mf.converged, "SCF did not converge"

# Run CCSD. Default max_cycle=50 isn't always enough at stretched bond
# lengths (confirmed: R=3.00 needs ~more than 50 iterations here, converges
# fine with more room).
ccsd = pyscf.cc.CCSD(mf)
ccsd.max_cycle = 200
eccsd, *_ = ccsd.kernel()
assert ccsd.converged, "CCSD did not converge"

# Run CISD.
cisd = pyscf.ci.CISD(mf)
cisd.max_cycle = 200
ecisd, *_ = cisd.kernel()
assert cisd.converged, "CISD did not converge"

# Extract second-quantized Hamiltonian and Hamiltonian parameters.
constant = pyscf.tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
h1e = mf.get_hcore()
num_orb = h1e.shape[0]
n_qubits = 2 * num_orb
h2e = pyscf.ao2mo.restore(1, mf._eri, num_orb)
nelec = pyscf.tools.fcidump.read(fcidump_filename)["NELEC"]

if optimizer_chunk_size is None:
    optimizer_chunk_size = num_orb ** 3

print(f"Bond distance: {bond_distance} Angstrom")
print(f"FCIDUMP: {fcidump_filename}")
print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
print(f"optimize_jax chunk_size: {optimizer_chunk_size}")
# print("Hartree-Fock energy:", mf.e_tot)
# print("CCSD correlation energy:", eccsd)
# print("CCSD total energy:", ccsd.e_tot)
# print("CISD correlation energy:", ecisd)
# print("CISD total energy:", cisd.e_tot)

nelec = (nelec // 2, nelec // 2)  # Convert to (n_alpha, n_beta) tuple.

# Build the UCJ Operation. half_layer needs a second repetition's worth of
