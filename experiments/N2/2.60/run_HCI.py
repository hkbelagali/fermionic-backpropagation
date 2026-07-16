"""Run Heat-bath/selected CI (pyscf.fci.selected_ci) for the N2 molecule at
a fixed bond distance, and add the result to this directory's
UCJ_results.npz (produced by run_UCJ.py), preserving all fields already
saved there.

Template for use with launch.sh: it copies this file into each bond-distance
subdirectory and replaces the placeholder bond distance below with that
directory's value before running. Molecule/basis settings must match
run_UCJ.py's for the energies to be comparable.
"""

import numpy as np
import pyscf
from pyscf.fci import selected_ci

# Molecule parameters -- must match run_UCJ.py's settings for this sweep.
bond_distance = 2.60  # Angstrom; substituted by launch.sh from the subdirectory name.
basis = "sto-3g"

mol = pyscf.gto.M(
    atom=f"N 0 0 0; N 0 0 {bond_distance}",
    basis=basis,
    verbose=0,
)
mf = pyscf.scf.RHF(mol)
mf.kernel()
assert mf.converged, "SCF did not converge"

# Extract second-quantized Hamiltonian parameters, in the HF molecular-orbital
# basis (no active-space reduction here) -- same convention as run_UCJ.py.
constant = mol.energy_nuc()
h1e = mf.mo_coeff.T @ mf.get_hcore() @ mf.mo_coeff
num_orb = h1e.shape[0]
h2e = pyscf.ao2mo.restore(1, pyscf.ao2mo.kernel(mol, mf.mo_coeff), num_orb)
nelec = mol.nelectron
nelec = (nelec // 2, nelec // 2)

myci = selected_ci.SelectedCI()
hci_energy, _ = myci.kernel(h1e, h2e, num_orb, nelec, ecore=constant)

print(f"Bond distance: {bond_distance} Angstrom")
print(f"HCI (selected CI) total energy: {hci_energy:.10f} Ha")

results = dict(np.load("UCJ_results.npz"))
results["hci_energy"] = hci_energy
np.savez("UCJ_results.npz", **results)
print("Updated UCJ_results.npz with hci_energy")
