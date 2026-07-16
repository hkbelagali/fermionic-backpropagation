"""Plot HF, CCSD, CISD, UCJ (CCSD-parameterized), UCJ (optimized), and HCI
energies against the N2 bond length R, collected from each bond-distance
subdirectory's UCJ_results.npz (written by run_UCJ.py and run_HCI.py).
"""

import glob
import os

import numpy as np
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(script_dir, "..", "presentation.mplstyle"))

# Collect one row of energies per bond-distance subdirectory that has a
# UCJ_results.npz, keyed on the bond_distance saved inside the file itself
# (robust to directory naming) rather than the directory name.
rows = []
for path in sorted(glob.glob(os.path.join(script_dir, "*", "UCJ_results.npz"))):
    data = np.load(path)
    rows.append({
        "R": float(data["bond_distance"]),
        "hf": float(data["hf_energy"]),
        "ccsd": float(data["ccsd_energy"]),
        "cisd": float(data["cisd_energy"]),
        "ucj": float(data["ucj_ccsd_energy"]),
        "ucj_opt": float(data["ucj_optimized_energy"]),
        "hci": float(data["hci_energy"]) if "hci_energy" in data.files else None,
    })

if not rows:
    raise SystemExit(f"No UCJ_results.npz files found under {script_dir}/*/")

rows.sort(key=lambda row: row["R"])
missing_hci = [row["R"] for row in rows if row["hci"] is None]
if missing_hci:
    print(f"Warning: no hci_energy for R = {missing_hci} (run_HCI.py not run there yet); "
          f"skipping those points on the HCI curve.")

R = np.array([row["R"] for row in rows])
series = [
    ("hf", "HF", "x"),
    ("ccsd", "CCSD", "o"),
    ("cisd", "CISD", "s"),
    ("ucj", "UCJ (CCSD params)", "^"),
    ("ucj_opt", "UCJ (optimized)", "D"),
    ("hci", "HCI", "v"),
]

fig, ax = plt.subplots()
for key, label, marker in series:
    xs = [row["R"] for row in rows if row[key] is not None]
    ys = [row[key] for row in rows if row[key] is not None]
    ax.plot(xs, ys, marker=marker, label=label)

ax.set_xlabel(r"$R$ (\AA)" if plt.rcParams["text.usetex"] else "R (Angstrom)")
ax.set_ylabel("Energy (Ha)")
ax.set_title(r"N$_2$ dissociation curve")
ax.legend()

out_path = os.path.join(script_dir, "energy_vs_R.pdf")
fig.savefig(out_path)
print(f"Saved plot to {out_path}")
