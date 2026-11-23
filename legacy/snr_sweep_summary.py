"""snr_sweep_summary.py -- turn a set of diagnose_cell.py runs at different
--snr-target values into the identifiability map Section 9 goal #3 asks for:
does a convergence pathology (like the Standard-generator non-identifiability)
get better or worse as data quality changes, or is it a fixed property of the
model regardless of SNR?

Reads each SNR run's summary.txt (for max Rhat) and bfmi.txt (for per-chain
BFMI), assuming diagnose_cell.py's own naming convention:
    figures/diagnose/<generator>_<fitter>_snr<N>/   (non-default SNR)
    figures/diagnose/<generator>_<fitter>/          (SNR == 5.0, the default)

Usage (on Euler, after running diagnose_cell.py at each SNR value):

    python snr_sweep_summary.py --generator Standard --fitter B-Matern-B1 \
        --snr-values 1 3 5 10 20

Outputs, under figures/diagnose/<generator>_<fitter>_snr_sweep/:
    snr_sweep.png   -- max Rhat and mean BFMI vs SNR, side by side
    snr_sweep.txt   -- the same numbers as a plain table
"""
from __future__ import annotations

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _folder_for(generator: str, fitter: str, snr: float) -> str:
    suffix = "" if snr == 5.0 else f"_snr{snr:g}"
    return f"figures/diagnose/{generator}_{fitter}{suffix}"


def _read_max_rhat(folder: str) -> float | None:
    path = os.path.join(folder, "summary.txt")
    if not os.path.exists(path):
        return None
    # summary.txt is az.summary()'s to_string() output, sorted worst-Rhat-first
    # by diagnose_cell.py. az.summary()'s column set isn't fixed -- it can
    # include eti89_lb/eti89_ub/mcse_mean/mcse_sd depending on arviz version
    # and scale, so r_hat is NOT reliably the last column (an earlier version
    # of this function assumed it was, and silently read mcse_sd instead on
    # full-scale runs). Locate "r_hat" by its header name instead.
    with open(path) as f:
        lines = [l for l in f.read().splitlines() if l.strip()]
    if len(lines) < 2:
        return None
    header_cols = lines[0].split()
    try:
        rhat_col_idx = header_cols.index("r_hat")
    except ValueError:
        return None
    row_tokens = lines[1].split()
    # row_tokens[0] is the parameter name (the index), so value columns are
    # offset by one relative to header_cols.
    value_idx = rhat_col_idx + 1
    try:
        return float(row_tokens[value_idx])
    except (ValueError, IndexError):
        return None


def _read_bfmi(folder: str) -> np.ndarray | None:
    path = os.path.join(folder, "bfmi.txt")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        text = f.read()
    match = re.search(r"\[([^\]]*)\]", text)
    if not match:
        return None
    try:
        return np.array([float(x) for x in match.group(1).split(",")])
    except ValueError:
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", required=True)
    ap.add_argument("--fitter", required=True)
    ap.add_argument("--snr-values", type=float, nargs="+", required=True)
    args = ap.parse_args()

    rows = []
    for snr in args.snr_values:
        folder = _folder_for(args.generator, args.fitter, snr)
        max_rhat = _read_max_rhat(folder)
        bfmi = _read_bfmi(folder)
        if max_rhat is None or bfmi is None:
            print(f"WARNING: missing results in {folder} (max_rhat={max_rhat}, bfmi={bfmi}) -- skipping this SNR.")
            continue
        rows.append((snr, max_rhat, bfmi.mean(), bfmi.min()))

    if not rows:
        raise SystemExit("No complete SNR runs found -- nothing to summarize. "
                          "Did you run diagnose_cell.py with --snr-target for each value first?")

    rows.sort(key=lambda r: r[0])
    out_dir = f"figures/diagnose/{args.generator}_{args.fitter}_snr_sweep"
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "snr_sweep.txt"), "w") as f:
        header = f"{'SNR':>8} {'max_rhat':>10} {'mean_bfmi':>10} {'min_bfmi':>10}"
        print(header)
        f.write(header + "\n")
        for snr, max_rhat, mean_bfmi, min_bfmi in rows:
            line = f"{snr:8.2f} {max_rhat:10.3f} {mean_bfmi:10.3f} {min_bfmi:10.3f}"
            print(line)
            f.write(line + "\n")

    snrs, max_rhats, mean_bfmis, min_bfmis = zip(*rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(snrs, max_rhats, "o-", color="C3")
    ax1.axhline(1.01, color="gray", linestyle="--", linewidth=1, label="Rhat=1.01 (converged)")
    ax1.set_xlabel("SNR target")
    ax1.set_ylabel("max Rhat")
    ax1.set_title(f"{args.generator} -> {args.fitter}: convergence vs SNR")
    ax1.legend(fontsize=8)

    ax2.plot(snrs, mean_bfmis, "o-", color="C0", label="mean BFMI")
    ax2.plot(snrs, min_bfmis, "s--", color="C1", label="min BFMI (worst chain)")
    ax2.axhline(0.3, color="gray", linestyle="--", linewidth=1, label="BFMI=0.3 (concern threshold)")
    ax2.set_xlabel("SNR target")
    ax2.set_ylabel("BFMI")
    ax2.set_title("Sampler health vs SNR")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    plt.savefig(os.path.join(out_dir, "snr_sweep.png"), dpi=110, bbox_inches="tight")
    print(f"\nSaved {out_dir}/snr_sweep.png and {out_dir}/snr_sweep.txt")


if __name__ == "__main__":
    main()