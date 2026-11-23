"""plot_same_shape_overlay.py -- combine every fitter's saved curve from
same_shape_overlay.py into one plot against the TRUE curve, at both
surprisal levels. Rebuilds the shared dataset (same seed/generator) only
to recover gt and basis -- no refitting.

Usage (after all 7 same_shape_overlay.py runs have completed):

    python plot_same_shape_overlay.py --generator A-B2 --seed 777
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from basis.basis_functions import make_raised_cosine_basis
from recovery_grid import build_dataset_from_generator

SCALE_PRESETS = {
    "pilot": dict(n_patients=2, n_electrodes=2, n_times=400, n_basis=5, tau_max=100.0),
    "full":  dict(n_patients=4, n_electrodes=3, n_times=1000, n_basis=6, tau_max=200.0),
}

FITTERS = ["Standard", "A-B1", "A-B2", "B-Matern-B1", "B-Matern-B2", "B-SE-B1", "B-SE-B2"]
COLORS = {
    "Standard": "gray", "A-B1": "tab:orange", "A-B2": "tab:red",
    "B-Matern-B1": "tab:blue", "B-Matern-B2": "tab:cyan",
    "B-SE-B1": "tab:green", "B-SE-B2": "tab:olive",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generator", default="A-B2")
    ap.add_argument("--scale", default="full", choices=list(SCALE_PRESETS))
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--snr-target", type=float, default=5.0)
    ap.add_argument("--overlay-dir", default="figures/overlay")
    ap.add_argument("--out-path", default="figures/overlay/same_shape_overlay.png")
    args = ap.parse_args()

    p = SCALE_PRESETS[args.scale]
    dt = 10.0
    basis = make_raised_cosine_basis(n_basis=p["n_basis"], tau_max=p["tau_max"], c=5.0)
    n_lags = int(np.floor(p["tau_max"] / dt)) + 1
    taus = np.arange(n_lags) * dt

    print(f"Rebuilding the SAME shared dataset (generator={args.generator}, seed={args.seed}) "
          f"to recover the ground truth for the true curve -- not refitting anything.")
    data = build_dataset_from_generator(
        args.generator, basis, taus, p["n_patients"], p["n_electrodes"], p["n_times"], dt,
        snr_target=args.snr_target, seed=args.seed, return_ground_truth=True,
    )
    gt = data["ground_truth"]

    results = {}
    missing = []
    surp_lo = surp_hi = None
    for fitter in FITTERS:
        path = os.path.join(args.overlay_dir, f"{fitter}.json")
        if not os.path.exists(path):
            missing.append(fitter)
            continue
        with open(path) as f:
            r = json.load(f)
        results[fitter] = r
        surp_lo, surp_hi = r["surp_lo"], r["surp_hi"]

    if missing:
        print(f"WARNING: missing results for {missing} -- plotting the {len(results)} available fitters only. "
              f"Run same_shape_overlay.py --fitter <name> for each missing one.")
    if not results:
        raise SystemExit("No fitter results found at all -- nothing to plot.")

    # True curve, computed the same way _true_h0_any_family does for the
    # basis family (A-B2's generator), but here we need the FULL
    # surprisal-dependent curve (mu_eff + beta_eff * surp), not just h0.
    if not hasattr(gt, "weight_for"):
        raise SystemExit(f"--generator {args.generator} isn't a basis-family generator "
                          "(no weight_for()) -- this script currently only builds the true "
                          "curve for Standard/A-B1/A-B2-style generators.")
    mu_eff_true, beta_eff_true = gt.weight_for(patient=0, electrode=0)
    phi = basis.eval(taus)
    true_lo = phi @ (mu_eff_true + beta_eff_true * surp_lo)
    true_hi = phi @ (mu_eff_true + beta_eff_true * surp_hi)

    fig, (ax_lo, ax_hi) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, true_curve, surp_val, title in [
        (ax_lo, true_lo, surp_lo, f"surp = {surp_lo:.1f} bits (low)"),
        (ax_hi, true_hi, surp_hi, f"surp = {surp_hi:.1f} bits (high)"),
    ]:
        ax.plot(taus, true_curve, "k--", linewidth=2.5, label="TRUE", zorder=10)
        for fitter, r in results.items():
            fitted_taus = np.asarray(r["taus"])
            h = np.asarray(r["h_lo"] if ax is ax_lo else r["h_hi"])
            label = fitter
            if r["max_rhat"] is not None:
                tag = "OK" if r["max_rhat"] <= 1.05 else "NOT CONVERGED"
                label += f" (max Rhat={r['max_rhat']:.2f}, {tag})"
            else:
                label += " (closed-form, no Rhat)"
            ax.plot(fitted_taus, h, color=COLORS.get(fitter, None), linewidth=1.5, alpha=0.85, label=label)
        ax.set_title(title)
        ax.set_xlabel("lag (ms)")
        ax.axhline(0, color="lightgray", linewidth=0.8, zorder=0)
    ax_lo.set_ylabel("h(tau | surp)")
    ax_hi.legend(fontsize=7, loc="best")
    fig.suptitle(f"Same-shape overlay: all fitters on ONE {args.generator}-generated dataset "
                 f"(seed={args.seed})", fontsize=11)
    fig.tight_layout()
    plt.savefig(args.out_path, dpi=120, bbox_inches="tight")
    print(f"Saved {args.out_path}")


if __name__ == "__main__":
    main()