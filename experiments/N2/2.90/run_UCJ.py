"""Run the polynomial-time UCJ backpropagation energy estimate, then
variationally optimize the circuit parameters, for the N2 molecule at a
fixed bond distance.

Template for use with launch.sh: it copies this file into each bond-distance
subdirectory and replaces the placeholder bond distance below with that
directory's value before running.
"""

import os
import sys

import numpy as np
import pyscf
import ffsim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from fermiprop import UCJBackPropagator

# Molecule / active space parameters.
bond_distance = 2.90  # Angstrom; substituted by launch.sh from the subdirectory name.
basis = "sto-3g"       # Full active space (no CAS/frozen-core reduction) -- adjust for a larger/smaller problem.

# Parameters of the (L)UCJ ansatz.
half_layer = False                       # If True, appends a final rotation to the circuit, but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit.

# Variational optimization settings. See fermiprop/propagator.py's
# optimize_jax() docstring: chunk_size must evenly divide num_orb**4;
# num_orb**2 is always a safe, valid choice.
optimizer_method = "L-BFGS-B"
optimizer_options = {"maxiter": 500, "gtol": 1e-9, "ftol": 1e-9}
optimizer_chunk_size = None  # set below to num_orb**2 once num_orb is known

# Build the N2 molecule and run Hartree-Fock.
mol = pyscf.gto.M(
    atom=f"N 0 0 0; N 0 0 {bond_distance}",
    basis=basis,
    verbose=0,
)   
mf = pyscf.scf.RHF(mol)
mf.kernel()
assert mf.converged, "SCF did not converge"

# Run CCSD.
ccsd = pyscf.cc.CCSD(mf)
eccsd, *_ = ccsd.kernel()

# Run CISD.
cisd = pyscf.ci.CISD(mf)
ecisd, *_ = cisd.kernel()

# Extract second-quantized Hamiltonian and Hamiltonian parameters, in the
# HF molecular-orbital basis (no active-space reduction here).
constant = mol.energy_nuc()
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
num_orb = h1e.shape[0]
n_qubits = 2 * num_orb
h2e = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, mf.mo_coeff), num_orb)
nelec = mol.nelectron

print(f"Bond distance: {bond_distance} Angstrom")
print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
print("Hartree-Fock energy:", mf.e_tot)
print("CCSD correlation energy:", eccsd)
print("CCSD total energy:", ccsd.e_tot)
print("CISD correlation energy:", ecisd)
print("CISD total energy:", cisd.e_tot)

if optimizer_chunk_size is None:
    optimizer_chunk_size = num_orb ** 2

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

# CCSD-parameterized UCJ energy, before variational optimization.
ucj_ccsd_energy = backprop.propagate()

# Variationally optimize the circuit parameters starting from the
# CCSD-derived parameters, using analytic (JAX autodiff) gradients.
result = backprop.optimize_jax(
    interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
    chunk_size=optimizer_chunk_size,
    method=optimizer_method,
    options=optimizer_options,
)
ucj_optimized_energy = backprop.propagate(show_progress=False)

print(f"Hartree-Fock energy: {mf.e_tot:.10f} Ha")
print(f"CCSD energy: {ccsd.e_tot:.10f} Ha")
print(f"CISD energy: {cisd.e_tot:.10f} Ha")
print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.10f} Ha")
print(f"Variationally optimized UCJ energy: {ucj_optimized_energy:.10f} Ha")

np.savez(
    "UCJ_results.npz",
    bond_distance=bond_distance,
    hf_energy=mf.e_tot,
    ccsd_energy=ccsd.e_tot,
    cisd_energy=cisd.e_tot,
    ucj_ccsd_energy=ucj_ccsd_energy,
    ucj_optimized_energy=ucj_optimized_energy,
)
print("Saved results to UCJ_results.npz")
