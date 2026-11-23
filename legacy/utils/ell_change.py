"""
ell_change.py
=============
Report how much Model B-C2's surprisal-sensitive kernel changes across a
surprisal range, WITH uncertainty -- instead of reporting delta alone.

Why not just delta?
-------------------
delta is "change in log(length-scale) per bit". The data only constrain
delta x (the range of surprisal actually spanned), so a per-bit number hides
how much (or how little) the data say, and its units invite extrapolation.
What is scientifically readable is the RATIO

        ell_h1(s_hi) / ell_h1(s_lo)  =  exp( delta * (s_hi - s_lo) )

with a credible interval, P(ratio < 1) ("does higher surprisal shorten the
length scale?"), and -- crucially -- how much the data narrowed the interval
compared with the PRIOR alone ("contraction"). If contraction ~ 1 the data
told you nothing about delta, however confident the point estimate looks.

Which range?  Pick it deliberately and SAY which:
  * the spread of per-unit mean surprisal (what the current C2 actually
    observes -- one number per unit), or
  * the word-level range (e.g. 5th-95th percentile of word surprisal) --
    scientifically natural, but for the current unit-mean C2 this is an
    EXTRAPOLATION beyond what the design saw.

Usage
-----
    from utils.ell_change import ell_change_summary, format_ell_change
    s = ell_change_summary(trace, s_lo=2.1, s_hi=3.0)
    print(format_ell_change(s, label="unit-mean range"))

Population-level only: the absolute ell values returned exclude patient and
electrode deviations (they describe the 'typical unit').
"""
from __future__ import annotations

import numpy as np


def _flat(trace, name):
    return np.asarray(trace.posterior[name].values).reshape(-1)


def _eti(x, prob):
    lo = (1.0 - prob) / 2.0
    q = np.quantile(x, [lo, 1.0 - lo])
    return float(q[0]), float(q[1])


def _one(coef_post, ds, prior_sd, prob, rng):
    """Ratio summary for exp(coef * ds), coef ~ posterior vs coef ~ N(0, prior_sd)."""
    log_ratio = coef_post * ds
    ratio = np.exp(log_ratio)
    lo, hi = _eti(ratio, prob)
    prior_ratio = np.exp(rng.normal(0.0, prior_sd, size=20000) * ds)
    plo, phi = _eti(prior_ratio, prob)
    prior_sd_lr = prior_sd * abs(ds)
    return {
        "median": float(np.median(ratio)), "lo": lo, "hi": hi,
        "p_below_1": float((ratio < 1.0).mean()),
        "prior_lo": plo, "prior_hi": phi,
        # 1.0 = data added nothing beyond the prior; 0 = fully determined
        "sd_post_over_sd_prior": float(np.std(log_ratio) / prior_sd_lr) if prior_sd_lr > 0 else float("nan"),
    }


def ell_change_summary(trace, s_lo, s_hi, *, prior_sd_delta=0.3, prior_sd_gamma=0.3,
                       prob=0.9, seed=0):
    """Summarise delta (and gamma, if present) as ratios over [s_lo, s_hi]."""
    if not s_hi > s_lo:
        raise ValueError("need s_hi > s_lo (a zero range -- e.g. a shared stimulus where every unit has the same "
                         "mean surprisal -- means the ratio over the OBSERVED range is undefined)")
    rng = np.random.default_rng(seed)
    ds = float(s_hi - s_lo)
    out = {"s_lo": float(s_lo), "s_hi": float(s_hi), "prob": prob}
    delta = _flat(trace, "delta")
    out["ell_ratio"] = _one(delta, ds, prior_sd_delta, prob, rng)
    out["delta"] = {"mean": float(delta.mean()), "sd": float(delta.std()),
                    "lo": _eti(delta, prob)[0], "hi": _eti(delta, prob)[1]}

    # per-chain means: a split here (chains disagree) is a different failure
    # from "wide but unimodal" -- report it, don't average it away.
    d = np.asarray(trace.posterior["delta"].values)             # (chain, draw)
    out["delta_chain_means"] = d.mean(axis=1).tolist()

    if "log_ell0_h1" in trace.posterior:
        l0 = _flat(trace, "log_ell0_h1")
        out["ell_h1_at_lo"] = dict(zip(("lo", "hi"), _eti(np.exp(l0 + delta * s_lo), prob)),
                                   median=float(np.median(np.exp(l0 + delta * s_lo))))
        out["ell_h1_at_hi"] = dict(zip(("lo", "hi"), _eti(np.exp(l0 + delta * s_hi), prob)),
                                   median=float(np.median(np.exp(l0 + delta * s_hi))))
        # Aliasing diagnostic: delta * (mean surprisal) is ~constant across
        # units, so delta trades off against log_ell0_h1 along a ridge.
        out["corr_delta_log_ell0_h1"] = float(np.corrcoef(delta, l0)[0, 1])

    if "gamma" in trace.posterior:
        out["sigma_ratio"] = _one(_flat(trace, "gamma"), ds, prior_sd_gamma, prob, rng)
    return out


def format_ell_change(s, label=""):
    r = s["ell_ratio"]
    pc = int(round(100 * s["prob"]))
    lines = [
        f"Length-scale change of h1 over surprisal {s['s_lo']:.2f} -> {s['s_hi']:.2f} bits"
        + (f" ({label})" if label else ""),
        f"  ell(high)/ell(low): median {r['median']:.2f}, {pc}% CI [{r['lo']:.2f}, {r['hi']:.2f}]"
        f"   (prior alone: [{r['prior_lo']:.2f}, {r['prior_hi']:.2f}])",
        f"  P(higher surprisal -> shorter ell) = {r['p_below_1']:.2f}",
        f"  posterior sd / prior sd (log ratio) = {r['sd_post_over_sd_prior']:.2f}"
        "   (1.0 = data added nothing)",
    ]
    if "ell_h1_at_lo" in s:
        a, b = s["ell_h1_at_lo"], s["ell_h1_at_hi"]
        lines.append(f"  typical-unit ell_h1: {a['median']:.1f} [{a['lo']:.1f}, {a['hi']:.1f}] -> "
                     f"{b['median']:.1f} [{b['lo']:.1f}, {b['hi']:.1f}]  (same units as lag axis)")
    lines.append("  delta chain means: " + ", ".join(f"{x:+.2f}" for x in s["delta_chain_means"]))
    if "corr_delta_log_ell0_h1" in s:
        lines.append(f"  corr(delta, log_ell0_h1) = {s['corr_delta_log_ell0_h1']:+.2f}"
                     "   (|corr| near 1 = aliased with the intercept)")
    if "sigma_ratio" in s:
        g = s["sigma_ratio"]
        lines.append(f"  output-scale ratio s(high)/s(low): median {g['median']:.2f}, "
                     f"{pc}% CI [{g['lo']:.2f}, {g['hi']:.2f}]")
    return "\n".join(lines)
