"""
plot_recovery_grid.py
======================

Reads recovery_grid.py's checkpointed JSON results and renders the
6x7 (generator x fitter) grid as two heatmaps side by side:

  left  : held-out Pearson r for every (generator, fitter) cell.
          Diagonal cells (fitter recovering its own generator's data)
          are boxed -- that's the direct self-recovery answer.
  right : max R-hat per cell (Bayesian fitters only; Standard has no
          posterior, shown as hatched grey). A cell can have a high r
          in the left panel and still be untrustworthy if this panel
          shows it didn't converge -- exactly the pattern that made
          the original shared-dataset comparison misleading.

Run after recovery_grid.py has completed at least some cells --
partial grids render fine (unfinished cells shown as hatched grey with
"pending").
"""

from __future__ import annotations

import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from recovery_grid import GENERATOR_NAMES, FITTER_NAMES


def plot_recovery_grid(results_path: str = "figures/recovery_grid_results.json",
                        output_path: str = "figures/recovery_grid.png"):
    with open(results_path) as f:
        results = json.load(f)

    n_gen, n_fit = len(GENERATOR_NAMES), len(FITTER_NAMES)
    r_matrix = np.full((n_gen, n_fit), np.nan)
    rhat_matrix = np.full((n_gen, n_fit), np.nan)
    status = np.empty((n_gen, n_fit), dtype=object)  # "ok" | "not_converged" | "no_posterior" | "pending" | "error"

    for gi, gname in enumerate(GENERATOR_NAMES):
        for fi, fname in enumerate(FITTER_NAMES):
            key = f"{gname} -> {fname}"
            cell = results.get(key)
            if cell is None:
                status[gi, fi] = "pending"
                continue
            if "error" in cell:
                status[gi, fi] = "error"
                continue
            r_matrix[gi, fi] = cell["r"]
            diag = cell.get("diagnostics")
            if diag is None:
                status[gi, fi] = "no_posterior"
            else:
                rhat_matrix[gi, fi] = diag["max_rhat"]
                status[gi, fi] = "ok" if (diag["max_rhat"] <= 1.01 and diag["n_divergences"] == 0) else "not_converged"

    fig, (ax_r, ax_rhat) = plt.subplots(1, 2, figsize=(17, 7))

    # --- left panel: r heatmap -------------------------------------------
    im_r = ax_r.imshow(r_matrix, cmap="RdYlGn", vmin=0.0, vmax=1.0, aspect="auto")
    ax_r.set_xticks(range(n_fit)); ax_r.set_xticklabels(FITTER_NAMES, rotation=45, ha="right", fontsize=8)
    ax_r.set_yticks(range(n_gen)); ax_r.set_yticklabels(GENERATOR_NAMES, fontsize=8)
    ax_r.set_xlabel("Fitter"); ax_r.set_ylabel("Generator (true data-generating model)")
    ax_r.set_title("Held-out Pearson r, every (generator, fitter) pair")
    plt.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)

    for gi, gname in enumerate(GENERATOR_NAMES):
        for fi, fname in enumerate(FITTER_NAMES):
            s = status[gi, fi]
            if s == "pending":
                ax_r.add_patch(plt.Rectangle((fi - 0.5, gi - 0.5), 1, 1, facecolor="lightgray", hatch="..", edgecolor="gray"))
                ax_r.text(fi, gi, "pending", ha="center", va="center", fontsize=6, color="gray")
            elif s == "error":
                ax_r.add_patch(plt.Rectangle((fi - 0.5, gi - 0.5), 1, 1, facecolor="black", alpha=0.7))
                ax_r.text(fi, gi, "FAILED", ha="center", va="center", fontsize=6, color="white")
            else:
                ax_r.text(fi, gi, f"{r_matrix[gi, fi]:.2f}", ha="center", va="center", fontsize=7,
                          color="black" if r_matrix[gi, fi] > 0.4 else "white")
            if s == "not_converged":
                ax_r.add_patch(plt.Rectangle((fi - 0.5, gi - 0.5), 1, 1, fill=False, hatch="////", edgecolor="red", lw=0))
            if gname == fname:  # diagonal: box it, this is the self-recovery cell
                ax_r.add_patch(plt.Rectangle((fi - 0.5, gi - 0.5), 1, 1, fill=False, edgecolor="blue", lw=2.5))

    # --- right panel: max R-hat heatmap ------------------------------------
    im_rhat = ax_rhat.imshow(rhat_matrix, cmap="RdYlGn_r", vmin=1.0, vmax=1.1, aspect="auto")
    ax_rhat.set_xticks(range(n_fit)); ax_rhat.set_xticklabels(FITTER_NAMES, rotation=45, ha="right", fontsize=8)
    ax_rhat.set_yticks(range(n_gen)); ax_rhat.set_yticklabels(GENERATOR_NAMES, fontsize=8)
    ax_rhat.set_xlabel("Fitter"); ax_rhat.set_ylabel("Generator")
    ax_rhat.set_title("Max R-hat (green <= 1.01 = converged; grey = no posterior / pending)")
    plt.colorbar(im_rhat, ax=ax_rhat, fraction=0.046, pad=0.04)

    for gi in range(n_gen):
        for fi in range(n_fit):
            s = status[gi, fi]
            if s in ("no_posterior", "pending", "error"):
                ax_rhat.add_patch(plt.Rectangle((fi - 0.5, gi - 0.5), 1, 1, facecolor="lightgray", hatch="..", edgecolor="gray"))
            else:
                ax_rhat.text(fi, gi, f"{rhat_matrix[gi, fi]:.2f}", ha="center", va="center", fontsize=7)
            if GENERATOR_NAMES[gi] == FITTER_NAMES[fi]:
                ax_rhat.add_patch(plt.Rectangle((fi - 0.5, gi - 0.5), 1, 1, fill=False, edgecolor="blue", lw=2.5))

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved recovery grid figure to {output_path}")

    # --- text summary of the diagonal specifically -------------------------
    print("\n=== Diagonal (self-recovery) cells ===")
    for name in GENERATOR_NAMES:
        key = f"{name} -> {name}"
        cell = results.get(key)
        if cell is None:
            print(f"  {name:15s}: pending")
        elif "error" in cell:
            print(f"  {name:15s}: FAILED ({cell['error']})")
        else:
            diag = cell.get("diagnostics")
            if diag is None:
                print(f"  {name:15s}: r={cell['r']:.4f}  (no posterior)")
            else:
                ok = "OK" if (diag["max_rhat"] <= 1.01 and diag["n_divergences"] == 0) else "NOT CONVERGED"
                print(f"  {name:15s}: r={cell['r']:.4f}  max_rhat={diag['max_rhat']:.3f}  "
                      f"divergences={diag['n_divergences']}  [{ok}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", default="figures/recovery_grid_results.json")
    parser.add_argument("--output-path", default="figures/recovery_grid.png")
    args = parser.parse_args()
    plot_recovery_grid(results_path=args.results_path, output_path=args.output_path)