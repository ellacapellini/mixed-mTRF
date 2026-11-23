"""Stage 2: fit ONE Bayesian model of the ladder on simulated testbed data and report what matters.

    python -m mixed_trf stage2 --model B-pop-hyp --mode gain --seed 3

Reported (one JSON row, appended to --out):
  * convergence on FUNCTIONALS (population TRF h_pop, and unit length scales), not raw parameters:
    R-hat, bulk ESS, divergences, BFMI;
  * a convergence curve (R-hat / ESS of h_pop using the first 25/50/75/100 % of the draws);
  * gain / shift / width decomposition of the posterior population TRF vs the simulation truth;
  * held-out-patient explained share, from the population mean alone (--holdout P drops patient P from the fit);
  * with --fit_patients k only the first k patients are used (sequential updating / data efficiency).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from mixed_trf.core.design import unit_design
from mixed_trf.estimators.decomposition import decompose
from mixed_trf.models.registry import MODELS, fit
from mixed_trf.simulation.templates import DT, STRENGTH
from mixed_trf.simulation.testbed import make_dataset


def _bfmi(trace):
    e = np.asarray(trace.sample_stats["energy"].values)           # (chain, draw)
    return [float(np.sum(np.diff(c) ** 2) / np.sum((c - c.mean()) ** 2)) for c in e]


def _functional_diag(trace, var, fracs=(0.25, 0.5, 0.75, 1.0)):
    import arviz as az
    post = trace.posterior
    n = post.sizes["draw"]
    out = []
    for f in fracs:
        k = max(4, int(n * f))
        sub = trace.isel(draw=slice(0, k)) if hasattr(trace, "isel") else trace
        try:
            rh = float(np.nanmax(az.rhat(sub, var_names=[var])[var].values))
            es = float(np.nanmin(az.ess(sub, var_names=[var])[var].values))
        except Exception:
            rh, es = float("nan"), float("nan")
        out.append({"draw_frac": f, "max_rhat": rh, "min_ess": es})
    return out


def run(a):
    taus = np.arange(int(a.tau_max / DT) + 1) * DT
    L = len(taus)
    rng = np.random.default_rng(a.seed)
    units = make_dataset(rng, n_patients=a.n_patients, n_electrodes=a.n_electrodes, n_times=a.n_times, taus=taus,
                         mode=a.mode, effect_frac=1.0, effect_on="surprisal", shared=a.shared_stimulus, snr=a.snr,
                         noise="white", common_frac=0.0, rho=0.0, surp_mean=7.0, surp_sd=3.0,
                         patient_width_sd=a.patient_width_sd)
    from mixed_trf.models.pop_trf import prepare_suffstats
    designs = [unit_design(u, a.n_times, L, ["surp"]) for u in units]
    ss_all = prepare_suffstats(designs, [u["r"] for u in units])
    pat_all = np.array([u["patient"] for u in units])
    keep = pat_all < (a.fit_patients or a.n_patients)
    if a.holdout is not None:
        keep &= pat_all != a.holdout
    sub = {k: v[keep] for k, v in ss_all.items()}
    pat = pat_all[keep]
    _, pat = np.unique(pat, return_inverse=True)

    t0 = time.time()
    res = fit(a.model, sub, pat, taus, draws=a.draws, tune=a.tune, chains=a.chains, cores=a.cores, seed=a.seed,
              target_accept=a.target_accept)
    tr = res.trace
    hp = res.h_pop.mean(axis=(0, 1))                                    # (2, L)
    g, sh, om, _ = decompose(hp[0], hp[1], taus)
    row = {"model": a.model, "mode": a.mode, "seed": a.seed, "snr": a.snr, "n_patients": a.n_patients,
           "fit_patients": int(a.fit_patients or a.n_patients), "n_electrodes": a.n_electrodes,
           "n_times": a.n_times, "holdout": a.holdout, "elapsed_s": time.time() - t0,
           "n_divergences": int(tr.sample_stats["diverging"].sum()), "bfmi": _bfmi(tr),
           "convergence_curve_h_pop": _functional_diag(tr, "h_pop"),
           "truth": {"gain": STRENGTH["gain"] if a.mode == "gain" else 0.0,
                     "shift_ms": STRENGTH["shift"] if a.mode == "latency" else 0.0,
                     "width": STRENGTH["width"] if a.mode == "width" else 0.0},
           "est": {"gain": float(g), "shift_ms": float(sh), "width": float(om)}}
    # uncertainty on the decomposition: apply it to a thinned set of posterior draws of h_pop
    hs = res.h_pop.reshape(-1, 2, L)
    idx = np.linspace(0, len(hs) - 1, min(200, len(hs))).astype(int)
    dec = np.array([decompose(hs[i, 0], hs[i, 1], taus)[:3] for i in idx])
    row["est_90ci"] = {k: [float(np.percentile(dec[:, j], 5)), float(np.percentile(dec[:, j], 95))]
                       for j, k in enumerate(("gain", "shift_ms", "width"))}
    if a.model.endswith("hyp") or "hyp" in a.model:
        ell = np.asarray(tr.posterior["ell_unit_0"].values).mean(axis=(0, 1))
        wt = np.array([u["wscale"] for u in units])[keep]
        row["corr_log_ell_vs_log_true_width"] = float(np.corrcoef(np.log(ell), np.log(wt))[0, 1])
    if a.holdout is not None:                                           # population-only prediction of unseen units
        h = np.concatenate([hp[0], hp[1]])
        m = pat_all == a.holdout
        XtX, Xty, yty = ss_all["XtX"][m], ss_all["Xty"][m], ss_all["yty"][m]
        expl = (2 * Xty @ h - np.einsum("i,uij,j->u", h, XtX, h)) / yty
        row["heldout_explained_share"] = float(np.mean(expl))
    line = json.dumps(row)
    print(line)
    if a.out:
        with open(a.out, "a") as f:
            f.write(line + "\n")
    return row


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--mode", default="gain", choices=["null", "gain", "latency", "width"])
    ap.add_argument("--n_patients", type=int, default=6)
    ap.add_argument("--n_electrodes", type=int, default=2)
    ap.add_argument("--n_times", type=int, default=24000)
    ap.add_argument("--tau_max", type=float, default=800.0)
    ap.add_argument("--snr", type=float, default=0.2)
    ap.add_argument("--patient_width_sd", type=float, default=0.0)
    ap.add_argument("--shared_stimulus", action="store_true")
    ap.add_argument("--fit_patients", type=int, default=None, help="use only the first k patients")
    ap.add_argument("--holdout", type=int, default=None, help="leave this patient out of the fit")
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--target_accept", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap


def main(argv=None):
    return run(build_parser().parse_args(argv))
