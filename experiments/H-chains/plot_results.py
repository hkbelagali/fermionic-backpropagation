import glob
import os
import re

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

script_dir = os.path.dirname(os.path.abspath(__file__))
plt.style.use(os.path.join(script_dir, "..", "presentation.mplstyle"))
plt.rcParams.update({"text.usetex": True})


def bold_ticks(ax):
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight("bold")
        tick_label.set_color("black")
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")
    legend = ax.get_legend()
    if legend is not None:
        for text in legend.get_texts():
            text.set_color("black")

rows = []
for path in sorted(glob.glob(os.path.join(script_dir, "n*", "UCJ_results.npz"))):
    subdir = os.path.basename(os.path.dirname(path))
    match = re.fullmatch(r"n(\d+)", subdir)
    if not match:
        continue
    data = np.load(path)
    rows.append({
        "n": int(match.group(1)),
        "hf": float(data["hf_energy"]),
        "ccsd": float(data["ccsd_energy"]),
        "ucj": float(data["ucj_ccsd_energy"]),
        "ucj_opt": float(data["ucj_optimized_energy"]),
        "hci": float(data["hci_energy"]) if "hci_energy" in data.files else None,
        "propagate_runtime": float(data["propagate_runtime"]),
        "optimize_runtime": float(data["optimize_runtime"]),
    })

if not rows:
    raise SystemExit(f"No UCJ_results.npz files found under {script_dir}/n*/")

rows.sort(key=lambda row: row["n"])

energy_series = [
    ("hf", "HF", "x"),
    ("ccsd", "CCSD", "o"),
    ("ucj", "UCJ (CCSD params)", "^"),
    ("ucj_opt", "UCJ (optimized)", "D"),
]

colors = {key: f"C{i}" for i, (key, _, _) in enumerate(energy_series)}

error_series = [(key, label, marker) for key, label, marker in energy_series if key != "ccsd"]

runtime_series = [
    ("propagate_runtime", "UCJ (CCSD params)", "^"),
    ("optimize_runtime", "Variational optimization", "D"),
]

fig, (ax_energy, ax_err) = plt.subplots(2, 1, sharex=True)

for key, label, marker in energy_series:
    xs = [row["n"] for row in rows]
    ys = [row[key] for row in rows]
    ax_energy.plot(xs, ys, marker=marker, label=label, color=colors[key])

ax_energy.set_ylabel(r"Energy (Ha)")
ax_energy.legend(loc="best")

for key, label, marker in error_series:
    xs = [row["n"] for row in rows]
    ys = [1000.0 * abs(row[key] - row["ccsd"]) for row in rows]
    ax_err.plot(xs, ys, marker=marker, label=label, color=colors[key])

chemical_accuracy = 1.5936  # mHa, i.e. 1 kcal/mol.
chem_line = ax_err.axhline(chemical_accuracy, color="gray", linestyle="--", label="Chemical accuracy")

ax_err.set_yscale("log")
ax_err.set_yticks([10**e for e in range(-9, 5, 2)])
ax_err.minorticks_off()
ax_err.set_xlabel(r"Chain length ($N$)")
ax_err.set_ylabel(r"$|E - E_{\mathrm{CCSD}}|$ (mHa)")
ax_err.set_xlim(1, 21)

ax_err.legend(handles=[chem_line], loc="upper left", bbox_to_anchor=(0.0, 1.06),
              frameon=False, fontsize="small")

ax_inset = inset_axes(ax_err, width="38%", height="35%", loc="lower right", borderpad=1.2)
for key, label, marker in runtime_series:
    xs = [row["n"] for row in rows]
    ys = [row[key] for row in rows]
    ax_inset.plot(xs, ys, marker=marker, label=label, color="black", markersize=4, linewidth=1)

ax_inset.set_yscale("log")
ax_inset.set_yticks([1e-3, 1e-1, 1e1, 1e3])
ax_inset.set_ylim(1e-4, 1e4)
ax_inset.minorticks_off()
ax_inset.set_ylabel(r"Runtime (s)", fontsize="small")
ax_inset.set_xlim(1, 21)
ax_inset.tick_params(labelsize="small")
ax_inset.set_xticklabels([])

inset_labels = [
    ("D", 1.10, "Variational optimization"),
    ("^", 0.91, "UCJ (CCSD params)"),
]
for marker, y, label in inset_labels:
    ax_inset.plot([0.05], [y], marker=marker, color="black", markersize=4,
                  transform=ax_inset.transAxes, clip_on=False)
    ax_inset.text(0.12, y, label, transform=ax_inset.transAxes,
                  fontsize="xx-small", ha="left", va="center")

bold_ticks(ax_energy)
bold_ticks(ax_err)

plt.tight_layout()

out_path = os.path.join(script_dir, "energy_and_runtime_vs_N.svg")
fig.savefig(out_path)
print(f"Saved plot to {out_path}")
