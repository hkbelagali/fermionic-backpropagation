"""Recompute HF, CCSD, CISD, and the CCSD-parameterized UCJ energy for the
N2 molecule at a fixed bond distance, WITHOUT running the variational
optimization loop. Updates this directory's existing UCJ_results.npz with
just those fields, leaving ucj_optimized_energy (and anything else already
saved there, e.g. hci_energy) untouched.

Use this to redo HF/CCSD/CISD (e.g. after they failed to converge) without
paying for the optimize_jax() run again.

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
bond_distance = 1.60  # Angstrom; substituted by launch.sh from the subdirectory name.

# Parameters of the (L)UCJ ansatz.
half_layer = False                       # If True, appends a final rotation to the circuit, but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit.

# Existing results to preserve -- must already exist (from a prior run_UCJ.py
# run); we're only refreshing HF/CCSD/CISD/ucj_ccsd_energy below.
existing_results = dict(np.load("UCJ_results.npz"))

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
# lengths (confirmed: R=3.00 needs more than 50 iterations here, converges
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

print(f"Bond distance: {bond_distance} Angstrom")
print(f"FCIDUMP: {fcidump_filename}")
print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")

nelec = (nelec // 2, nelec // 2)  # Convert to (n_alpha, n_beta) tuple.

# Build the UCJ Operation.
base_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=ccsd.t2, n_reps=1,  # The polynomial time algorithm applies to one repetition/layer of the UCJ ansatz.
    interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
)
if half_layer:
    ucj_op = ffsim.UCJOpSpinBalanced(
        diag_coulomb_mats=base_op.diag_coulomb_mats[:1],
        orbital_rotations=base_op.orbital_rotations[:1],
        final_orbital_rotation=base_op.orbital_rotations[1].conj().T,
    )
else:
    ucj_op = base_op

backprop = UCJBackPropagator(ucj_op, nelec=nelec, num_orb=num_orb, h1e=h1e, h2e=h2e, ecore=constant)

# CCSD-parameterized UCJ energy. NOTE: no optimize_jax() call here --
# ucj_optimized_energy is left as whatever was already saved.
ucj_ccsd_energy = backprop.propagate()

print(f"Hartree-Fock energy: {mf.e_tot:.10f} Ha")
print(f"CCSD energy: {ccsd.e_tot:.10f} Ha")
print(f"CISD energy: {cisd.e_tot:.10f} Ha")
print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.10f} Ha")
if "ucj_optimized_energy" in existing_results:
    print(f"Variationally optimized UCJ energy (unchanged): {float(existing_results['ucj_optimized_energy']):.10f} Ha")

existing_results.update(
    bond_distance=bond_distance,
    hf_energy=mf.e_tot,
    ccsd_energy=ccsd.e_tot,
    cisd_energy=cisd.e_tot,
    ucj_ccsd_energy=ucj_ccsd_energy,
)
np.savez("UCJ_results.npz", **existing_results)
print("Updated UCJ_results.npz (hf/ccsd/cisd/ucj_ccsd_energy only; ucj_optimized_energy and any other fields left untouched)")
