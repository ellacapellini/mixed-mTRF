"""All presentation figures.

    python -m mixed_trf plot-stage1 --glob 'results/stage1/ieeg1_fin_*_seed*.jsonl' --out results/figures/stage1_main
    python -m mixed_trf plot-stage1 --glob 'results/stage1/ieeg1_es*_seed*.jsonl' 'results/stage1/ieeg1_fin_*_seed*.jsonl' --out results/figures/power_map
    python -m mixed_trf plot-stage2 --glob 'results/stage2/*.jsonl' --out results/figures/stage2

Stage 1 (one PNG each; every one is skipped with a message if the needed rows are absent):
  s1_recovery.png        estimate vs truth for each parameter, all seeds, per SNR and band (null vs active mechanism)
  s1_estimator_bias.png  first-order vs binned estimator: bias in units of the smallest effect of interest (SESOI)
  s1_detection.png       P(90% interval excludes 0), P(interval beyond SESOI) (= claim rule), and false-positive rate under the null
  s1_coverage.png        90% interval coverage with Wilson intervals
  s1_selection.png       electrode-selection precision/recall and leave-one-patient-out gain of the surprisal model
  s1_power_map.png       detection probability vs effect size (needs runs with different --effect_scale)
Stage 2:
  s2_trf_recovery.png    recovered h0(tau) and h1(tau) (posterior mean, 90% band) vs the simulated truth, per model
  s2_ladder.png          estimate +/- 90% interval vs truth for gain / shift / width, per model and simulated effect
  s2_diagnostics.png     divergences and R-hat of the population TRF per model
  s2_heldout.png         explained share of an unseen patient predicted from the population TRF alone
  s2_smoothness.png      does B-pop-hyp recover known patient differences in smoothness? (corr of log length scale with log true width)
Learning-evolution plots are separate: `plot-learning`.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np

PARAMS = ["gain (per SD)", "shift (ms per SD)", "width (log, per SD)"]
SESOI = np.array([0.10, 10.0, 0.10])
ACTIVE = {"gain": 0, "latency": 1, "width": 2}                       # mechanism -> parameter it changes
MODE_COL = {"null": "#7f7f7f", "gain": "#1f77b4", "latency": "#d62728", "width": "#2ca02c"}
MODEL_COL = {"A-pop": "#1f77b4", "B-pop-amp": "#ff7f0e", "B-pop-hyp": "#2ca02c",
             "B-pop-amp-se": "#9467bd", "B-pop-hyp-se": "#8c564b"}


def wilson(k, n, z=1.645):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def load(globs):
    seen, rows = set(), []
    for g in globs:
        for f in sorted(glob.glob(g)):
            for l in open(f):
                if not l.strip():
                    continue
                r = json.loads(l)
                key = (r.get("band"), r.get("snr"), r.get("mode"), r.get("seed"), r.get("effect_scale", 1.0),
                       r.get("rhythm", "natural"), r.get("electrodes"), r.get("estimator"))
                if key not in seen:
                    seen.add(key); rows.append(r)
    return rows


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})
    return plt


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, name); fig.savefig(p, dpi=200, bbox_inches="tight"); print("wrote", p)


def _cells(rows):
    c = collections.defaultdict(list)
    for r in rows:
        c[(r["band"], r["snr"], r["mode"], r.get("effect_scale", 1.0))].append(r)
    return c


# ------------------------------------------------------------------------------------------------ stage 1
def s1_recovery(rows, out, plt):
    cells = _cells(rows); bands = sorted({k[0] for k in cells}); snrs = sorted({k[1] for k in cells})
    fig, ax = plt.subplots(len(bands), 3, figsize=(11, 3.3 * len(bands)), squeeze=False)
    for bi, band in enumerate(bands):
        for j in range(3):
            a = ax[bi, j]
            for si, snr in enumerate(snrs):
                for mi, mode in enumerate(("null", [m for m, p in ACTIVE.items() if p == j][0])):
                    rs = cells.get((band, snr, mode, 1.0), [])
                    if not rs:
                        continue
                    est = np.array([r["selected"]["est"][j] for r in rs]); tru = np.mean([r["selected"]["truth"][j] for r in rs])
                    x = si + (-0.18 if mi == 0 else 0.18)
                    a.scatter(x + np.random.default_rng(0).uniform(-0.07, 0.07, len(est)), est, s=5, alpha=0.25, color=MODE_COL[mode], lw=0)
                    lo, md, hi = np.percentile(est, [5, 50, 95])
                    a.vlines(x, lo, hi, color="k", lw=1.5); a.plot([x], [md], "o", color="k", ms=4)
                    a.hlines(tru, x - 0.13, x + 0.13, color="crimson", lw=2, zorder=5)
            a.axhline(0, c="k", lw=0.6); a.set_xticks(range(len(snrs))); a.set_xticklabels([f"{s:g}" for s in snrs])
            a.set_xlabel("SNR"); a.set_title(f"{band}: {PARAMS[j]}")
    fig.suptitle("Recovery across seeds. Left of each pair: no effect (null). Right: the mechanism that changes this parameter.\n"
                 "dots = seeds, black = median and 5-95 %, red = truth on the selected electrodes", y=1.02)
    fig.tight_layout(); _save(fig, out, "s1_recovery.png")


def s1_estimator_bias(rows, out, plt):
    rows = [r for r in rows if "first_order" in r and r.get("effect_scale", 1.0) == 1.0]
    if not rows:
        return print("s1_estimator_bias: no rows with paired first-order estimates (binned runs only)")
    snrs = sorted({r["snr"] for r in rows}); bands = sorted({r["band"] for r in rows})
    fig, ax = plt.subplots(len(bands), 3, figsize=(11, 3.2 * len(bands)), squeeze=False)
    for bi, band in enumerate(bands):
        for mode, j in ACTIVE.items():
            a = ax[bi, j]
            for k, (lab, getter, col) in enumerate((("first-order", lambda r: r["first_order"]["selected"]["est"][j], "#bbbbbb"),
                                                     ("binned", lambda r: r["selected"]["est"][j], "#1f77b4"))):
                m, s = [], []
                for snr in snrs:
                    b = np.array([(getter(r) - r["selected"]["truth"][j]) / SESOI[j] for r in rows
                                  if r["band"] == band and r["snr"] == snr and r["mode"] == mode])
                    m.append(b.mean() if len(b) else np.nan); s.append(b.std(ddof=1) / np.sqrt(len(b)) if len(b) > 1 else 0)
                a.errorbar(np.arange(len(snrs)) + (k - 0.5) * 0.12, m, 1.645 * np.array(s), fmt="o-", color=col, label=lab, capsize=2)
            a.axhspan(-0.3, 0.3, color="g", alpha=0.12); a.axhline(0, c="k", lw=0.6)
            a.set_xticks(range(len(snrs))); a.set_xticklabels([f"{s:g}" for s in snrs]); a.set_xlabel("SNR")
            a.set_title(f"{band}, {mode} -> {PARAMS[j]}"); a.set_ylabel("bias / SESOI")
            if bi == 0 and j == 0:
                a.legend(frameon=False)
    fig.suptitle("Estimator bias in units of the smallest effect of interest (green = tolerance 0.3 x SESOI)", y=1.02)
    fig.tight_layout(); _save(fig, out, "s1_estimator_bias.png")


def _detect(rs, j):
    lo = np.array([r["selected"]["lo"][j] for r in rs]); hi = np.array([r["selected"]["hi"][j] for r in rs])
    return (np.sum((lo > 0) | (hi < 0)), np.sum((lo > SESOI[j]) | (hi < -SESOI[j])), len(rs))


def s1_detection(rows, out, plt):
    cells = _cells(rows); bands = sorted({k[0] for k in cells}); snrs = sorted({k[1] for k in cells})
    fig, ax = plt.subplots(len(bands), 3, figsize=(11, 3.3 * len(bands)), squeeze=False)
    for bi, band in enumerate(bands):
        for j in range(3):
            a = ax[bi, j]; mech = [m for m, p in ACTIVE.items() if p == j][0]
            for lab, mode, which, ls, col in (("power: interval excludes 0", mech, 0, "-", "#1f77b4"),
                                              ("claim rule: whole interval beyond SESOI", mech, 1, "--", "#ff7f0e"),
                                              ("false positive (null); nominal 0.10 for a 90 % interval", "null", 0, ":", "#d62728")):
                ps, los, his = [], [], []
                for snr in snrs:
                    rs = cells.get((band, snr, mode, 1.0), [])
                    if not rs:
                        ps.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                    d = _detect(rs, j); p, l, h = wilson(d[which], d[2]); ps.append(p); los.append(l); his.append(h)
                x = np.arange(len(snrs)); ps, los, his = map(np.array, (ps, los, his))
                a.plot(x, ps, ls, color=col, marker="o", label=lab); a.fill_between(x, los, his, color=col, alpha=0.12)
            a.axhline(0.9, c="grey", lw=0.6); a.axhline(0.10, c="grey", lw=0.6, ls="--")   # 90 % interval: nominal false-positive rate = 0.10
            a.set_ylim(-0.02, 1.02); a.set_xticks(range(len(snrs))); a.set_xticklabels([f"{s:g}" for s in snrs])
            a.set_xlabel("SNR"); a.set_title(f"{band}: {PARAMS[j]}"); a.set_ylabel("probability")
            if bi == 0 and j == 0:
                a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); _save(fig, out, "s1_detection.png")


def s1_coverage(rows, out, plt):
    cells = _cells(rows); bands = sorted({k[0] for k in cells}); snrs = sorted({k[1] for k in cells})
    fig, ax = plt.subplots(len(bands), 3, figsize=(11, 3.2 * len(bands)), squeeze=False)
    for bi, band in enumerate(bands):
        for j in range(3):
            a = ax[bi, j]
            for mi, mode in enumerate(("null", "gain", "latency", "width")):
                xs, ps, ls_, hs = [], [], [], []
                for si, snr in enumerate(snrs):
                    rs = cells.get((band, snr, mode, 1.0), [])
                    if rs:
                        k = sum(bool(r["selected"]["covered"][j]) for r in rs); p, l, h = wilson(k, len(rs))
                        xs.append(si + (mi - 1.5) * 0.1); ps.append(p); ls_.append(p - l); hs.append(h - p)
                a.errorbar(xs, ps, [ls_, hs], fmt="o", color=MODE_COL[mode], label=mode, capsize=2)
            a.axhline(0.9, c="k", lw=0.8); a.set_ylim(0.4, 1.02); a.set_xticks(range(len(snrs))); a.set_xticklabels([f"{s:g}" for s in snrs])
            a.set_xlabel("SNR"); a.set_title(f"{band}: {PARAMS[j]}"); a.set_ylabel("90 % interval coverage")
            if bi == 0 and j == 0:
                a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); _save(fig, out, "s1_coverage.png")


def s1_selection(rows, out, plt):
    rows = [r for r in rows if r.get("effect_scale", 1.0) == 1.0]
    bands = sorted({r["band"] for r in rows}); snrs = sorted({r["snr"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3))
    for band, ls in zip(bands, ("-", "--")):
        pr = [np.mean([r["sel_precision"] for r in rows if r["band"] == band and r["snr"] == s and r["mode"] == "null"]) for s in snrs]
        rc = [np.mean([r["sel_recall"] for r in rows if r["band"] == band and r["snr"] == s and r["mode"] == "null"]) for s in snrs]
        ax[0].plot(range(len(snrs)), pr, ls, marker="o", label=f"{band} precision"); ax[0].plot(range(len(snrs)), rc, ls, marker="s", label=f"{band} recall")
        for mode in ("null", "gain"):
            f = [np.mean([r["selected"]["lopo_dR2"] > 0 for r in rows if r["band"] == band and r["snr"] == s and r["mode"] == mode]) for s in snrs]
            ax[1].plot(range(len(snrs)), f, ls, marker="o", color=MODE_COL[mode], label=f"{band} {mode}")
            d = [np.mean([r["selected"]["lopo_dR2"] for r in rows if r["band"] == band and r["snr"] == s and r["mode"] == mode]) for s in snrs]
            ax[2].plot(range(len(snrs)), d, ls, marker="o", color=MODE_COL[mode], label=f"{band} {mode}")
    ax[0].set_title("electrode selection (dev data)"); ax[0].set_ylabel("share"); ax[0].legend(frameon=False, fontsize=7)
    ax[1].set_title("held-out patient: P(surprisal model helps)"); ax[1].axhline(0.5, c="grey", lw=0.6); ax[1].legend(frameon=False, fontsize=7)
    ax[2].set_title("mean R$^2$ gain of surprisal model"); ax[2].axhline(0, c="k", lw=0.6)
    for a in ax:
        a.set_xticks(range(len(snrs))); a.set_xticklabels([f"{s:g}" for s in snrs]); a.set_xlabel("SNR")
    fig.tight_layout(); _save(fig, out, "s1_selection.png")


def s1_power_map(rows, out, plt):
    scales = sorted({r.get("effect_scale", 1.0) for r in rows})
    if len(scales) < 2:
        return print("s1_power_map: need runs with at least two different --effect_scale values")
    snrs = sorted({r["snr"] for r in rows}); bands = sorted({r["band"] for r in rows})
    fig, ax = plt.subplots(len(bands), 3, figsize=(11, 3.2 * len(bands)), squeeze=False)
    for bi, band in enumerate(bands):
        for mode, j in ACTIVE.items():
            a = ax[bi, j]
            for snr, ls in zip(snrs, ("-", "--", ":")):
                if len({r.get("effect_scale", 1.0) for r in rows if r["band"] == band and r["snr"] == snr}) < 2:
                    continue                                   # a single effect size is not a curve
                ps, lo, hi, xs = [], [], [], []
                for sc in scales:
                    rs = [r for r in rows if r["band"] == band and r["snr"] == snr and r["mode"] == mode and r.get("effect_scale", 1.0) == sc]
                    if rs:
                        p, l, h = wilson(_detect(rs, j)[1], len(rs)); xs.append(sc); ps.append(p); lo.append(l); hi.append(h)
                if xs:
                    a.plot(xs, ps, ls, color=MODE_COL[mode], marker="o", label=f"SNR {snr:g}"); a.fill_between(xs, lo, hi, color=MODE_COL[mode], alpha=0.12)
            a.axhline(0.8, c="grey", lw=0.6); a.set_ylim(-0.02, 1.02); a.set_xscale("log", base=2)
            a.set_xticks(scales); a.set_xticklabels([f"{s:g}" for s in scales])
            a.set_xlabel("true effect size (x default)"); a.set_ylabel("P(claim rule met)"); a.set_title(f"{band}: {mode} -> {PARAMS[j]}")
            a.legend(frameon=False, fontsize=7)
    fig.tight_layout(); _save(fig, out, "s1_power_map.png")


# ------------------------------------------------------------------------------------------------ stage 2
def _eff_of(r):
    return {"gain": "gain", "latency": "shift_ms", "width": "width"}.get(r["mode"])


def s2_trf_recovery(rows, out, plt):
    rows = [r for r in rows if "trf" in r]
    if not rows:
        return print("s2_trf_recovery: no rows with saved TRFs (re-run stage 2 with this version)")
    models = sorted({r["model"] for r in rows}); modes = sorted({r["mode"] for r in rows if r["mode"] != "null"}) or ["null"]
    fig, ax = plt.subplots(len(modes) * 2, len(models), figsize=(3.6 * len(models), 2.6 * len(modes) * 2), squeeze=False)
    for mi, mode in enumerate(modes):
        for ci, model in enumerate(models):
            rs = [r for r in rows if r["model"] == model and r["mode"] == mode]
            for k, name in enumerate(("h0 (baseline response)", "h1 (change per SD of surprisal)")):
                a = ax[mi * 2 + k, ci]
                if not rs:
                    a.axis("off"); continue
                t = np.array(rs[0]["trf"]["taus"])
                for r in rs:                                            # all seeds, normalised by the truth's h0 peak
                    sc = np.max(np.abs(r["trf"]["truth"][0])); tr = np.array(r["trf"]["truth"][k]) / sc
                    # posterior is on standardised-y scale: rescale by the least-squares scale between h0 mean and truth h0
                    m0, t0 = np.array(r["trf"]["mean"][0]), np.array(r["trf"]["truth"][0])
                    s = float(m0 @ t0 / (m0 @ m0 + 1e-12))
                    m, lo, hi = (np.array(r["trf"][q][k]) * s / sc for q in ("mean", "lo", "hi"))
                    a.fill_between(t, lo, hi, color=MODEL_COL.get(model, "C0"), alpha=0.15); a.plot(t, m, color=MODEL_COL.get(model, "C0"), lw=1)
                a.plot(t, tr, "k--", lw=1.5, label="truth"); a.axhline(0, c="grey", lw=0.5)
                a.set_title(f"{model} | {mode}\n{name}", fontsize=8); a.set_xlabel("lag (ms)")
    fig.suptitle("Recovered response curves (shape; scale-matched to truth). Band = 90 % posterior, one per seed. h1 truth = effect per +1 SD", y=1.01)
    fig.tight_layout(); _save(fig, out, "s2_trf_recovery.png")


def s2_ladder(rows, out, plt):
    rows = [r for r in rows if r["mode"] != "null" and _eff_of(r)]
    if not rows:
        return print("s2_ladder: no non-null stage-2 rows")
    models = sorted({r["model"] for r in rows}); modes = sorted({r["mode"] for r in rows})
    fig, ax = plt.subplots(1, len(modes), figsize=(4.2 * len(modes), 3.4), squeeze=False)
    for mi, mode in enumerate(modes):
        a = ax[0, mi]
        for ci, model in enumerate(models):
            rs = [r for r in rows if r["model"] == model and r["mode"] == mode]
            for k, r in enumerate(rs):
                key = _eff_of(r); lo, hi = r["est_90ci"][key]
                a.vlines(ci + (k - len(rs) / 2) * 0.06, lo, hi, color=MODEL_COL.get(model, "C0"), lw=1.5)
                a.plot([ci + (k - len(rs) / 2) * 0.06], [r["est"][key]], "o", color=MODEL_COL.get(model, "C0"), ms=4)
        tr = rows and np.mean([r["truth"][_eff_of(r)] for r in rows if r["mode"] == mode]); a.axhline(tr, c="crimson", lw=2, label="truth")
        a.axhline(0, c="k", lw=0.5); a.set_xticks(range(len(models))); a.set_xticklabels(models, rotation=20)
        a.set_title(f"{mode}: {PARAMS[ACTIVE[mode]]}"); a.legend(frameon=False)
    fig.tight_layout(); _save(fig, out, "s2_ladder.png")


def s2_diagnostics(rows, out, plt):
    models = sorted({r["model"] for r in rows})
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    for ci, model in enumerate(models):
        rs = [r for r in rows if r["model"] == model]
        ax[0].bar(ci, np.mean([r["n_divergences"] for r in rs]), color=MODEL_COL.get(model, "C0"))
        ax[1].scatter([ci] * len(rs), [r["convergence_curve_h_pop"][-1]["max_rhat"] for r in rs], color=MODEL_COL.get(model, "C0"))
        ax[2].scatter([ci] * len(rs), [r["convergence_curve_h_pop"][-1]["min_ess"] for r in rs], color=MODEL_COL.get(model, "C0"))
    ax[0].set_title("divergences per fit (mean)"); ax[1].set_title("max R-hat of h_pop"); ax[1].axhline(1.01, c="k", ls="--", lw=0.8)
    ax[2].set_title("min bulk ESS of h_pop"); ax[2].axhline(400, c="k", ls="--", lw=0.8)
    for a in ax:
        a.set_xticks(range(len(models))); a.set_xticklabels(models, rotation=20)
    fig.tight_layout(); _save(fig, out, "s2_diagnostics.png")


def s2_heldout(rows, out, plt):
    rows = [r for r in rows if r.get("heldout_explained_share") is not None]
    if not rows:
        return print("s2_heldout: no rows with --holdout")
    models = sorted({r["model"] for r in rows}); modes = sorted({r["mode"] for r in rows})
    fig, a = plt.subplots(figsize=(6, 3.3))
    for mi, mode in enumerate(modes):
        for ci, model in enumerate(models):
            v = [r["heldout_explained_share"] for r in rows if r["model"] == model and r["mode"] == mode]
            a.scatter([ci + (mi - len(modes) / 2) * 0.12] * len(v), v, color=MODE_COL.get(mode, "C0"), label=mode if ci == 0 else None, s=18)
    a.axhline(0, c="k", lw=0.6); a.set_xticks(range(len(models))); a.set_xticklabels(models, rotation=20)
    a.set_ylabel("explained share, unseen patient"); a.legend(frameon=False, title="simulated effect"); fig.tight_layout(); _save(fig, out, "s2_heldout.png")


def s2_smoothness(rows, out, plt):
    rows = [r for r in rows if "corr_log_ell_vs_log_true_width" in r]
    if not rows:
        return print("s2_smoothness: no B-pop-hyp rows with patient_width_sd > 0")
    fig, a = plt.subplots(figsize=(4.5, 3.3))
    v = [r["corr_log_ell_vs_log_true_width"] for r in rows]
    a.scatter(range(len(v)), v, color=MODEL_COL["B-pop-hyp"]); a.axhline(0, c="k", lw=0.6); a.set_ylim(-1, 1)
    a.set_xlabel("fit"); a.set_ylabel("corr(log length scale, log true width)"); a.set_title("smoothness recovery (B-pop-hyp)")
    fig.tight_layout(); _save(fig, out, "s2_smoothness.png")


def main_stage1(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", default=["results/stage1/ieeg1_fin_*_seed*.jsonl"])
    ap.add_argument("--out", default="results/figures/stage1")
    a = ap.parse_args(argv); plt = _plt(); rows = load(a.glob)
    if not rows:
        raise SystemExit(f"no rows for {a.glob}")
    print(len(rows), "rows")
    for f in (s1_recovery, s1_estimator_bias, s1_detection, s1_coverage, s1_selection, s1_power_map):
        f(rows, a.out, plt)


def main_stage2(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--glob", nargs="+", default=["results/stage2/*.jsonl"])
    ap.add_argument("--out", default="results/figures/stage2")
    a = ap.parse_args(argv); plt = _plt(); rows = load_stage2(a.glob)
    if not rows:
        raise SystemExit(f"no rows for {a.glob}")
    print(len(rows), "rows")
    for f in (s2_trf_recovery, s2_ladder, s2_diagnostics, s2_heldout, s2_smoothness):
        f(rows, a.out, plt)


def load_stage2(globs):
    rows = []
    for g in globs:
        for f in sorted(glob.glob(g)):
            rows += [json.loads(l) for l in open(f) if l.strip()]
    return rows