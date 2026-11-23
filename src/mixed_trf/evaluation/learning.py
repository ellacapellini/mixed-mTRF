"""Learning-evolution figures from stage-2 jsonl rows (python -m mixed_trf plot-learning results/*.jsonl --out figs/).

  learning_convergence.png   R-hat and ESS of the population TRF vs fraction of the MCMC draws  (has the sampler converged?)
  learning_data.png          |estimate - truth| and 90 % interval width vs amount of data (minutes/patient)  (data efficiency)
  learning_sequential.png    posterior of the effect as patients are added one by one            (sequential updating)
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        for f in glob.glob(p):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows


def _effect(r):
    m = r["mode"]
    return {"gain": "gain", "latency": "shift_ms", "width": "width", "null": "gain"}[m]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--mode", default="gain", help="simulated effect to plot (effects have different units, so one at a time)")
    ap.add_argument("--hz", type=float, default=100.0)
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = load(a.paths)
    if a.mode:
        rows = [r for r in rows if r["mode"] == a.mode]
    if not rows:
        raise SystemExit("no rows")
    os.makedirs(a.out, exist_ok=True)
    models = sorted({r["model"] for r in rows})
    col = dict(zip(models, plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    # 1 convergence curve
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in models:
        cur = collections.defaultdict(list)
        for r in rows:
            if r["model"] == m:
                for c in r["convergence_curve_h_pop"]:
                    cur[c["draw_frac"]].append((c["max_rhat"], c["min_ess"]))
        fr = sorted(cur)
        ax[0].plot(fr, [np.nanmedian([v[0] for v in cur[f]]) for f in fr], "o-", color=col[m], label=m)
        ax[1].plot(fr, [np.nanmedian([v[1] for v in cur[f]]) for f in fr], "o-", color=col[m], label=m)
    ax[0].axhline(1.01, ls="--", c="grey", lw=1); ax[0].set_ylabel("max R-hat of h_pop (median over runs)")
    ax[1].axhline(400, ls="--", c="grey", lw=1); ax[1].set_ylabel("min bulk ESS of h_pop")
    for x in ax:
        x.set_xlabel("fraction of draws used"); x.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{a.out}/learning_convergence.png", dpi=200); plt.close(fig)

    # 2 data efficiency
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for m in models:
        g = collections.defaultdict(list)
        for r in rows:
            if r["model"] == m and r["mode"] != "null":
                k = _effect(r)
                lo, hi = r["est_90ci"][k]
                g[r["n_times"] / a.hz / 60].append((abs(r["est"][k] - r["truth"][k]), hi - lo))
        xs = sorted(g)
        if xs:
            ax[0].plot(xs, [np.mean([v[0] for v in g[x]]) for x in xs], "o-", color=col[m], label=m)
            ax[1].plot(xs, [np.mean([v[1] for v in g[x]]) for x in xs], "o-", color=col[m], label=m)
    ax[0].set_ylabel("|estimate - truth|"); ax[1].set_ylabel("90 % interval width")
    for x in ax:
        x.set_xlabel("minutes per electrode"); x.set_xscale("log"); x.legend(frameon=False)
    fig.tight_layout(); fig.savefig(f"{a.out}/learning_data.png", dpi=200); plt.close(fig)

    # 3 sequential updating
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for m in models:
        g = collections.defaultdict(list)
        for r in rows:
            if r["model"] == m and r["mode"] != "null":
                k = _effect(r); g[r["fit_patients"]].append((r["est"][k], *r["est_90ci"][k], r["truth"][k]))
        xs = sorted(g)
        if len(xs) > 1:
            ax.plot(xs, [np.mean([v[0] for v in g[x]]) for x in xs], "o-", color=col[m], label=m)
            ax.fill_between(xs, [np.mean([v[1] for v in g[x]]) for x in xs], [np.mean([v[2] for v in g[x]]) for x in xs],
                            color=col[m], alpha=0.2)
            tr = np.mean([v[3] for x in xs for v in g[x]])
    if "tr" in dir():
        ax.axhline(tr, c="k", ls="--", lw=1, label="truth")
    ax.set_xlabel("patients included in the fit"); ax.set_ylabel("effect estimate (90 % interval)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{a.out}/learning_sequential.png", dpi=200); plt.close(fig)
    print("wrote", a.out)
