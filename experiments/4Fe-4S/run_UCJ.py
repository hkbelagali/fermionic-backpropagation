"""Run the polynomial-time UCJ backpropagation energy estimate, then
variationally optimize the circuit parameters, for the n = 72 qubit iron
sulfur cluster UCJ circuit from [1] https://www.science.org/doi/10.1126/sciadv.adu9991.
"""

import os
import sys

import numpy as np
import pyscf
import ffsim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from fermiprop import UCJBackPropagator

# Parameters of the (L)UCJ ansatz.
half_layer = False                       # If True, appends a final rotation to the circuit as in [1], but makes the energy worse.
alpha_alpha_indices = lambda norb: None  # Use lambda norb: [(p, p + 1) for p in range(norb - 1)] for an LUCJ circuit as in [1]. Use None to run a UCJ circuit with more gates that improves the energy.
alpha_beta_indices  = lambda norb: None  # Use lambda norb: [(p, p) for p in range(0, norb, 4) if p <= 16] for a (truncated) LUCJ circuit as in [1]. Use None to run a UCJ circuit with more gates that improves the energy.

# Variational optimization settings. For the full 36-orbital active space
# this ansatz has ~2600-3900 free parameters, which made the previous
# gradient-free (Powell/Nelder-Mead) approach impractical: those methods
# need on the order of n_params evaluations just to take a single step.
# optimize_jax() uses analytic gradients (via fermiprop.jax_propagator/JAX
# autodiff), so each step costs about one forward+backward pass instead of
# ~n_params forward passes, making gradient-based methods like L-BFGS-B
# tractable here. chunk_size bounds memory for the norb**4 two-body sum
# (must evenly divide norb**4; norb**2 is always a safe, valid choice).
optimizer_method = "L-BFGS-B"
optimizer_options = {"maxiter": 50}
optimizer_chunk_size = None  # set below to num_orb**2 once num_orb is known

fcidump_filename = os.path.join(os.path.dirname(__file__), "fcidump_Fe4S4_MO.txt")

# Run Hartree-Fock.
mf_as = pyscf.tools.fcidump.to_scf(fcidump_filename)
mf_as.max_cycle = 100
mf_as.conv_tol = 1e-9
mf_as = mf_as.newton()
mf_as.kernel()
assert mf_as.converged, "SCF did not converge"

# Run CCSD.
ccsd = pyscf.cc.CCSD(mf_as)
eccsd, *_ = ccsd.kernel()

# Extract second-quantized Hamiltonian and Hamiltonian parameters.
constant = pyscf.tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
h1e = mf_as.get_hcore()
num_orb = h1e.shape[0]
n_qubits = 2 * num_orb
h2e = pyscf.ao2mo.restore(1, mf_as._eri, num_orb)
nelec = pyscf.tools.fcidump.read(fcidump_filename)["NELEC"]

print(f"Number of spatial orbitals: {num_orb}, Number of qubits: {n_qubits}")
print("CCSD correlation energy:", eccsd)
print("CCSD total energy:", ccsd.e_tot)

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

print(f"Hartree-Fock energy: {mf_as.e_tot:.10f} Ha")
print(f"CCSD energy: {ccsd.e_tot:.10f} Ha")
print(f"CCSD-parameterized UCJ energy: {ucj_ccsd_energy:.10f} Ha")
print(f"Variationally optimized UCJ energy: {ucj_optimized_energy:.10f} Ha")

out_path = os.path.join(os.path.dirname(__file__), "UCJ_results.npz")
np.savez(
    out_path,
    hf_energy=mf_as.e_tot,
    ccsd_energy=ccsd.e_tot,
    ucj_ccsd_energy=ucj_ccsd_energy,
    ucj_optimized_energy=ucj_optimized_energy,
)
print(f"Saved results to {out_path}")
