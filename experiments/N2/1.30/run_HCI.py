"""Run Heat-bath/selected CI (pyscf.fci.selected_ci) for the N2 molecule at
a fixed bond distance, and add the result to this directory's
UCJ_results.npz (produced by run_UCJ.py), preserving all fields already
saved there.

Template for use with launch.sh: it copies this file into each bond-distance
subdirectory and replaces the placeholder bond distance below with that
directory's value before running. Molecule/basis settings must match
run_UCJ.py's for the energies to be comparable.
"""

import glob

import numpy as np
import pyscf
import pyscf.tools.fcidump
from pyscf.fci import selected_ci

# Molecule parameters -- must match run_UCJ.py's settings for this sweep.
bond_distance = 1.30  # Angstrom; substituted by launch.sh from the subdirectory name.

# Same pre-generated FCIDUMP that run_UCJ.py uses, so the Hamiltonians match.
[fcidump_filename] = glob.glob("*_fcidump.txt")

# N2 dissociation curves are a known hard case for RHF convergence
# (near-degenerate/symmetry-breaking orbitals as the bond stretches), so
# start with damping + level-shifting for stability, and only escalate to
# the second-order (Newton) SCF solver if that alone isn't enough.
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

# Extract second-quantized Hamiltonian parameters -- same convention as run_UCJ.py.
constant = pyscf.tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
h1e = mf.get_hcore()
num_orb = h1e.shape[0]
h2e = pyscf.ao2mo.restore(1, mf._eri, num_orb)
nelec = pyscf.tools.fcidump.read(fcidump_filename)["NELEC"]
nelec = (nelec // 2, nelec // 2)

myci = selected_ci.SelectedCI()
hci_energy, _ = myci.kernel(h1e, h2e, num_orb, nelec, ecore=constant)

print(f"Bond distance: {bond_distance} Angstrom")
print(f"HCI (selected CI) total energy: {hci_energy:.10f} Ha")

results = dict(np.load("UCJ_results.npz"))
results["hci_energy"] = hci_energy
np.savez("UCJ_results.npz", **results)
print("Updated UCJ_results.npz with hci_energy")
