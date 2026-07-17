"""Recompute HF, CCSD, CISD, and the CCSD-parameterized UCJ energy for every
N2 bond-distance directory, walking R sequentially with HF orbital
continuation (each geometry's SCF is warm-started from the previous, smaller-R
geometry's converged density matrix), instead of solving each R independently.

This fixes a real physics bug: solving each R independently from a generic
initial guess lets RHF land on different, discontinuous solutions at
different geometries once N2's orbitals go near-degenerate away from
equilibrium (confirmed directly: R=1.00 -> R=1.10 jumped by 0.65 Ha with
independent solves). Continuation removes that jump and produces a smooth,
monotonic HF curve across the whole sweep.

Does NOT touch optimize_jax() or ucj_optimized_energy -- that's already been
run and is unaffected by how HF was initialized, so this only refreshes
hf_energy/ccsd_energy/cisd_energy/ucj_ccsd_energy in each directory's existing
UCJ_results.npz, leaving ucj_optimized_energy (and hci_energy, if present)
untouched. Run directly (not templated/copied by launch.sh -- this is a
single sequential process over all directories, not one process per R).
"""

import glob
import os
import sys

import numpy as np
import pyscf
import pyscf.tools.fcidump
import ffsim

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from fermiprop import UCJBackPropagator

script_dir = os.path.dirname(os.path.abspath(__file__))

# Parameters of the (L)UCJ ansatz -- must match whatever run_UCJ.py used
# originally, since we're rebuilding the same UCJ operator from CCSD t2.
half_layer = False  # See conversation/session notes: half_layer=True's energy looked
                    # catastrophically wrong on the real system (investigated but not
                    # yet resolved) -- reverted to restore known-good data.
alpha_alpha_indices = lambda norb: None
alpha_beta_indices = lambda norb: None

distances = ["0.70", "0.80", "0.90", "1.00", "1.10", "1.20", "1.30", "1.40", "1.50",
             "1.60", "1.70", "1.80", "1.90", "2.00", "2.10", "2.20", "2.30", "2.40",
             "2.50", "2.60", "2.70", "2.80", "2.90", "3.00"]

prev_dm = None
failures = []

for r in distances:
    dirpath = os.path.join(script_dir, r)
    fcidump_matches = glob.glob(os.path.join(dirpath, "*_fcidump.txt"))
    if not fcidump_matches:
        print(f"R={r}: no FCIDUMP found in {dirpath}, skipping")
        continue
    [fcidump_filename] = fcidump_matches

    results_path = os.path.join(dirpath, "UCJ_results.npz")
    if not os.path.exists(results_path):
        print(f"R={r}: no existing UCJ_results.npz in {dirpath}, skipping "
              f"(need a prior run_UCJ.py run there to preserve ucj_optimized_energy)")
        continue
    existing_results = dict(np.load(results_path))

    # Hartree-Fock. First point (no prev_dm): damping + level-shifting for
    # stability, since there's nothing to warm-start from yet. Every later
    # point: warm-start from the previous geometry's converged density
    # matrix, which is what actually fixes the discontinuity.
    mf = pyscf.tools.fcidump.to_scf(fcidump_filename)
    mf.verbose = 0
    mf.max_cycle = 300
    mf.conv_tol = 1e-9
    if prev_dm is None:
        mf.level_shift = 0.3
        mf.damp = 0.3
        mf.kernel()
    else:
        mf.kernel(dm0=prev_dm)
    if not mf.converged:
        mf = mf.newton()
        mf.max_cycle = 200
        mf.kernel(dm0=prev_dm) if prev_dm is not None else mf.kernel()
    if not mf.converged:
        print(f"R={r}: WARNING -- HF did not converge, skipping this point entirely "
              f"(not updating its UCJ_results.npz, not advancing continuation)")
        failures.append((r, "HF"))
        continue
    prev_dm = mf.make_rdm1()

    # CCSD. Known to become unreliable near dissociation (increasingly
    # multi-reference character) -- try an escalating sequence of
    # damping/level-shift settings (found by direct experimentation: no
    # single fixed setting works everywhere, but trying several in sequence
    # and keeping the first that converges rescues most points), warn
    # rather than crash on total failure, and skip the downstream UCJ
    # rebuild for this point if it's unusable (NaN).
    ccsd_configs = [
        dict(max_cycle=200),
        dict(max_cycle=300, iterative_damping=0.3),
        dict(max_cycle=400, iterative_damping=0.5, level_shift=0.2),
        dict(max_cycle=500, iterative_damping=0.7, level_shift=0.5),
        dict(max_cycle=500, iterative_damping=0.5, level_shift=0.5, diis_space=12),
        # Found by grid search directly against R=2.80: a larger level_shift
        # (1.0-2.0) reliably finds a genuine, consistent CCSD solution
        # (multiple different configs in this range converged to the same
        # energy, ~0.4 Ha below anything the smaller-level_shift attempts
        # above found) that the smaller level_shift values above miss.
        dict(max_cycle=500, iterative_damping=0.3, level_shift=1.0),
        dict(max_cycle=500, iterative_damping=0.3, level_shift=2.0),
        dict(max_cycle=500, iterative_damping=0.1, level_shift=1.0),
    ]
    ccsd = None
    for cfg in ccsd_configs:
        ccsd = pyscf.cc.CCSD(mf)
        ccsd.verbose = 0
        for k, v in cfg.items():
            setattr(ccsd, k, v)
        ccsd.kernel()
        if ccsd.converged:
            break
    if not ccsd.converged:
        print(f"R={r}: WARNING -- CCSD did not converge after all fallback settings (E={ccsd.e_tot})")
        failures.append((r, "CCSD"))
    if not np.isfinite(ccsd.e_tot) or not np.all(np.isfinite(ccsd.t2)):
        print(f"R={r}: CCSD energy/amplitudes are NaN/inf, skipping this point's "
              f"CCSD/CISD/UCJ update entirely")
        failures.append((r, "CCSD-NaN"))
        continue

    # CISD.
    cisd = pyscf.ci.CISD(mf)
    cisd.verbose = 0
    cisd.max_cycle = 200
    cisd.kernel()
    if not cisd.converged:
        print(f"R={r}: WARNING -- CISD did not converge (E={cisd.e_tot})")
        failures.append((r, "CISD"))

    # Extract second-quantized Hamiltonian and Hamiltonian parameters.
    constant = pyscf.tools.fcidump.read(fcidump_filename).get("ECORE", 0.0)
    h1e = mf.get_hcore()
    num_orb = h1e.shape[0]
    h2e = pyscf.ao2mo.restore(1, mf._eri, num_orb)
    nelec = pyscf.tools.fcidump.read(fcidump_filename)["NELEC"]
    nelec = (nelec // 2, nelec // 2)

    # Build the UCJ Operation from the (now-continuous) CCSD amplitudes.
    # half_layer needs a second repetition's worth of orbital rotations to
    # promote into a final_orbital_rotation (from_t_amplitudes with n_reps=1
    # -- required by the polynomial-time energy algorithm's single-layer
    # assumption -- only ever returns one), so ask for n_reps=2 here and
    # keep only the first layer's diag_coulomb_mats/orbital_rotation as the
    # actual ansatz.
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

    # CCSD-parameterized UCJ energy. NOTE: no optimize_jax() call -- it's
    # already been run, and ucj_optimized_energy is preserved as-is below.
    ucj_ccsd_energy = backprop.propagate(show_progress=False)

    print(f"R={r}: HF={mf.e_tot:.8f}  CCSD={ccsd.e_tot:.8f}  CISD={cisd.e_tot:.8f}  "
          f"UCJ(ccsd)={ucj_ccsd_energy:.8f}")
    if "ucj_optimized_energy" in existing_results:
        print(f"       UCJ(optimized, unchanged)={float(existing_results['ucj_optimized_energy']):.8f}")

    existing_results.update(
        bond_distance=float(r),
        hf_energy=mf.e_tot,
        ccsd_energy=ccsd.e_tot,
        cisd_energy=cisd.e_tot,
        ucj_ccsd_energy=ucj_ccsd_energy,
    )
    np.savez(results_path, **existing_results)

print()
if failures:
    print("Points with convergence issues (see warnings above):")
    for r, stage in failures:
        print(f"  R={r}: {stage}")
else:
    print("All points converged cleanly.")
