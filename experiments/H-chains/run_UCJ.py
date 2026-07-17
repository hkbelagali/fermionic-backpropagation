"""Run the polynomial-time UCJ backpropagation energy estimate, then
variationally optimize the circuit parameters, for the H2 molecule.
"""

import os
import sys
import time

import numpy as np
import pyscf
import pyscf.cc
import ffsim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from fermiprop import UCJBackPropagator

# Molecule parameters.
atom = "H"
natoms = 2
atomic_distance = 0.74  # Angstrom.

# Parameters of the (L)UCJ ansatz.
half_layer = False                       # If True, appends a final rotation to the circuit, but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit.

# Variational optimization settings. See fermiprop/propagator.py's
# optimize_jax() docstring: chunk_size must evenly divide num_orb**4.
optimizer_method = "L-BFGS-B"
optimizer_options = {"maxiter": 500, "gtol": 1e-9, "ftol": 1e-9}
optimizer_chunk_size = None  # set below to num_orb**4 (i.e. unchunked) once num_orb is known -- H2 is tiny.


def generate_linear_geometry(atom: str, natoms: int, atomic_distance: float = 1.0) -> str:
    return "; ".join([f"{atom} 0 0 {i * atomic_distance}" for i in range(natoms)])

mol = pyscf.gto.Mole()
mol.build(
    atom = generate_linear_geometry(atom, natoms, atomic_distance),
    basis = "sto-6g",
)

n_frozen = 0
active_space = range(n_frozen, mol.nao_nr())

scf = pyscf.scf.RHF(mol).run()

norb = len(active_space)
n_electrons = int(sum(scf.mo_occ[active_space]))
n_alpha = (n_electrons + mol.spin) // 2
n_beta = (n_electrons - mol.spin) // 2
nelec = (n_alpha, n_beta)

# Extract second-quantized Hamiltonian and Hamiltonian parameters for the
# active space.
mol_data = ffsim.MolecularData.from_scf(scf, active_space=active_space)
num_orb = mol_data.norb
n_qubits = 2 * num_orb
h1e = mol_data.one_body_integrals
h2e = pyscf.ao2mo.restore(1, mol_data.two_body_integrals, num_orb)
constant = mol_data.core_energy

if optimizer_chunk_size is None:
    optimizer_chunk_size = num_orb ** 4

print(f"Atomic distance: {atomic_distance} Angstrom")
print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
print(f"optimize_jax chunk_size: {optimizer_chunk_size}")

# Run CCSD.
ccsd = pyscf.cc.CCSD(scf)
ccsd.max_cycle = 200
eccsd, *_ = ccsd.kernel()
assert ccsd.converged, "CCSD did not converge"

# Build the UCJ Operation. half_layer needs a second repetition's worth of
# orbital rotations to promote into a final_orbital_rotation (n_reps=1 --
# required by the polynomial-time energy algorithm's single-layer assumption
# -- only ever returns one), so ask for n_reps=2 in that case and keep only
# the first layer's diag_coulomb_mats/orbital_rotation as the actual ansatz.
base_op = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
    t2=ccsd.t2, n_reps=(2 if half_layer else 1),
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
t_start = time.perf_counter()
ucj_ccsd_energy = backprop.propagate()
propagate_runtime = time.perf_counter() - t_start

# Variationally optimize the circuit parameters starting from the
# CCSD-derived parameters, using analytic (JAX autodiff) gradients.
t_start = time.perf_counter()
result = backprop.optimize_jax(
    interaction_pairs=(alpha_alpha_indices(num_orb), alpha_beta_indices(num_orb)),
    chunk_size=optimizer_chunk_size,
    method=optimizer_method,
    options=optimizer_options,
)
optimize_runtime = time.perf_counter() - t_start
ucj_optimized_energy = backprop.propagate(show_progress=False)

print(f"Hartree-Fock energy: {scf.e_tot:.10f} Ha")
print(f"CCSD energy: {ccsd.e_tot:.10f} Ha")
print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.10f} Ha")
print(f"Variationally optimized UCJ energy: {ucj_optimized_energy:.10f} Ha")
print(f"Backpropagation (CCSD-parameterized) runtime: {propagate_runtime:.4f} s")
print(f"Variational optimization runtime: {optimize_runtime:.4f} s")

np.savez(
    "UCJ_results.npz",
    atomic_distance=atomic_distance,
    hf_energy=scf.e_tot,
    ccsd_energy=ccsd.e_tot,
    ucj_ccsd_energy=ucj_ccsd_energy,
    ucj_optimized_energy=ucj_optimized_energy,
    propagate_runtime=propagate_runtime,
    optimize_runtime=optimize_runtime,
)
print("Saved results to UCJ_results.npz")
