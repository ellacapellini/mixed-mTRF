"""
plot_identifiability.py
========================

Reads identifiability_study.py's JSON-lines checkpoint file and renders
the actual deliverable: a P(correct kernel identified) heatmap over
(SNR x n_trials), with a Wilson score confidence interval per cell
(not just a point estimate -- with a finite number of repeats, the
proportion itself has real sampling uncertainty that a bare heatmap
number would hide), annotated with the mean "compensation score" of the
LOSING kernel where relevant -- a cell where the wrong kernel wins by a
large margin is a different finding from a cell where it wins by
successfully mimicking the true kernel's covariance curve (see
identifiability_study.py's module docstring on the compensation
diagnostic).
"""

from __future__ import annotations

import json
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Wilson score interval -- better-behaved than the naive normal
    approximation at small n or p near 0/1, both of which are realistic
    here (early sweep points may have few repeats, and P(correct) is
    expected to sit near 0 or 1 away from the identifiability boundary)."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p_hat = successes / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return p_hat, max(0.0, center - half_width), min(1.0, center + half_width)


def load_results(results_path: str) -> list[dict]:
    records = []
    with open(results_path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def aggregate(records: list[dict]) -> dict:
    """Group by (snr, n_times), returns {(snr, n_times): {"n": int, "correct": int,
    "mean_loser_compensation": float}}."""
    cells = defaultdict(lambda: {"n": 0, "correct": 0, "loser_compensations": []})
    for rec in records:
        t = rec["task"]
        cell_key = (t["snr_target"], t["n_times"])
        cells[cell_key]["n"] += 1
        cells[cell_key]["correct"] += int(rec["correct"])
        loser = [k for k in rec["per_kernel"] if k != rec["winner"]]
        if loser:
            cells[cell_key]["loser_compensations"].append(
                rec["per_kernel"][loser[0]]["compensation_score"]
            )
    return cells


def plot_identifiability(results_path: str, output_path: str = "figures/identifiability_kernel_family.png"):
    records = load_results(results_path)
    if not records:
        print(f"No completed tasks in {results_path} yet.")
        return

    fit_mode = records[0]["fit_mode"]
    cells = aggregate(records)
    snrs = sorted(set(k[0] for k in cells))
    n_times_vals = sorted(set(k[1] for k in cells))

    p_matrix = np.full((len(snrs), len(n_times_vals)), np.nan)
    n_matrix = np.zeros_like(p_matrix, dtype=int)
    ci_lo = np.full_like(p_matrix, np.nan)
    ci_hi = np.full_like(p_matrix, np.nan)
    comp_matrix = np.full_like(p_matrix, np.nan)

    for i, snr in enumerate(snrs):
        for j, nt in enumerate(n_times_vals):
            cell = cells.get((snr, nt))
            if cell is None:
                continue
            p_hat, lo, hi = _wilson_interval(cell["correct"], cell["n"])
            p_matrix[i, j] = p_hat
            n_matrix[i, j] = cell["n"]
            ci_lo[i, j], ci_hi[i, j] = lo, hi
            if cell["loser_compensations"]:
                comp_matrix[i, j] = np.mean(cell["loser_compensations"])

    fig, ax = plt.subplots(figsize=(1.8 * len(n_times_vals) + 3, 1.2 * len(snrs) + 2.5))
    im = ax.imshow(p_matrix, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", origin="lower")
    ax.set_xticks(range(len(n_times_vals))); ax.set_xticklabels(n_times_vals)
    ax.set_yticks(range(len(snrs))); ax.set_yticklabels(snrs)
    ax.set_xlabel("n_times (trial count proxy)")
    ax.set_ylabel("SNR target")
    warn = "" if fit_mode == "nuts" else "\n(fit_mode=point_estimate -- HARNESS VALIDATION ONLY, not a publishable result)"
    ax.set_title(f"P(Matern vs SE correctly identified by LOO), Wilson 95% CI shown{warn}", fontsize=10)
    plt.colorbar(im, ax=ax, label="P(correct)")

    for i in range(len(snrs)):
        for j in range(len(n_times_vals)):
            if np.isnan(p_matrix[i, j]):
                continue
            label = f"{p_matrix[i,j]:.2f}\n[{ci_lo[i,j]:.2f},{ci_hi[i,j]:.2f}]\nn={n_matrix[i,j]}"
            if not np.isnan(comp_matrix[i, j]) and comp_matrix[i, j] < 0.15:
                label += "\n(disguise)"
            ax.text(j, i, label, ha="center", va="center", fontsize=7,
                     color="black" if p_matrix[i, j] > 0.4 else "white")

    plt.tight_layout()
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved identifiability map to {output_path}")

    print("\n=== Summary ===")
    for i, snr in enumerate(snrs):
        for j, nt in enumerate(n_times_vals):
            if np.isnan(p_matrix[i, j]):
                continue
            print(f"  SNR={snr}, n_times={nt}: P(correct)={p_matrix[i,j]:.2f} "
                  f"[{ci_lo[i,j]:.2f}, {ci_hi[i,j]:.2f}] (n={n_matrix[i,j]} repeats)"
                  + (f", mean loser compensation={comp_matrix[i,j]:.3f}"
                     + (" (LOW -- 'disguise' signature)" if comp_matrix[i, j] < 0.15 else "")
                     if not np.isnan(comp_matrix[i, j]) else ""))

    if fit_mode != "nuts":
        print("\n*** fit_mode was 'point_estimate' -- this validates the harness runs "
              "correctly, but is NOT the publishable identifiability result. Re-run with "
              "--fit-mode nuts at full settings (ideally on ETH hardware) for the real map. ***")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-path", default="figures/identifiability_kernel_family.jsonl")
    parser.add_argument("--output-path", default="figures/identifiability_kernel_family.png")
    args = parser.parse_args()
    plot_identifiability(args.results_path, args.output_path)