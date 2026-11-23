"""
delta_identifiability.py
========================
Can the data identify delta (surprisal -> log length-scale of h1) at all?

Two modes
---------
bound : NO MCMC, seconds. The BEST CASE: pretend every unit's true
        log ell_h1 is observed exactly, then ask how well a regression on
        the per-unit mean surprisal can pin delta, given the generator's
        patient/electrode random-effect sizes and the prior sd(delta).
        If this best case already fails, no sampler fix can rescue C2 as
        coded. Also shows the SHARED-STIMULUS case (every unit hears the
        same words => identical mean surprisal => zero information),
        which is what real iEEG with one audiobook looks like.

fit   : the real thing. One (design, true delta, seed) per call: simulate
        from B-<kernel>-B2 with the given true delta, fit Model B-C2 with
        the NON-centered GP, report delta's posterior, contraction vs
        prior, the ell ratio over the observed range (utils/ell_change.py),
        sampler diagnostics, and coverage of the truth. Append one JSON
        line per call to --out (safe for Slurm arrays).

    python delta_identifiability.py --mode bound
    python delta_identifiability.py --mode fit --n_patients 4 --n_electrodes 3 \
        --delta_true -0.3 --seed 0 --out figures/delta_ident.jsonl

Caveats (also printed): bound mode treats random effects as independent
unit-level noise and the intercept as free -- approximate and optimistic by
construction. fit mode simulates a DIFFERENT word stream per unit (which is
what makes unit-mean surprisal vary at all); real shared-stimulus data is
strictly worse for delta.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

# Mirror ground_truth_gp.add_hierarchical_random_effects defaults for log ell_h1.
TAU_PATIENT_ELL = 0.3
TAU_ELECTRODE_ELL = 0.15
PRIOR_SD_DELTA = 0.3          # = pm.Normal("delta", 0, 0.3) in bayesian_gp_trf.py


# ---------------------------------------------------------------- bound mode
def _unit_means(n_units, n_times, dt, rng, shared):
    from data.simulate import make_word_stream
    if shared:
        m = float(make_word_stream(n_times, dt, 3.0, rng)[1].mean())
        return np.full(n_units, m)
    return np.array([make_word_stream(n_times, dt, 3.0, rng)[1].mean() for _ in range(n_units)])


def bound_one(n_patients, n_electrodes, delta_true, reps, n_times, dt, shared, seed,
              word_range=(0.13, 7.5)):
    rng = np.random.default_rng(seed)
    n = n_patients * n_electrodes
    pat = np.repeat(np.arange(n_patients), n_electrodes)
    sig2 = TAU_PATIENT_ELL ** 2 + TAU_ELECTRODE_ELL ** 2
    post_sd, post_mean, m_range = [], [], []
    for _ in range(reps):
        m = _unit_means(n, n_times, dt, rng, shared)
        a = rng.normal(0, TAU_PATIENT_ELL, n_patients)
        b = rng.normal(0, TAU_ELECTRODE_ELL, n)
        y = delta_true * m + a[pat] + b                       # oracle log ell_i (up to intercept)
        mc = m - m.mean()
        sxx = float((mc ** 2).sum())
        prec = 1.0 / PRIOR_SD_DELTA ** 2 + sxx / sig2
        sd = prec ** -0.5
        post_sd.append(sd)
        post_mean.append(sd ** 2 * float(mc @ y) / sig2)
        m_range.append(m.max() - m.min())
    post_sd, post_mean = np.array(post_sd), np.array(post_mean)
    ds_unit = float(np.mean(m_range))
    ds_word = word_range[1] - word_range[0]
    z = 1.645                                                  # 90% interval excludes 0
    power = float((np.abs(post_mean) / post_sd > z).mean()) if delta_true != 0 else float("nan")
    cover = float((np.abs(post_mean - delta_true) < z * post_sd).mean())
    return {
        "n_units": n, "shared_stimulus": shared, "delta_true": delta_true,
        "post_sd_delta": float(post_sd.mean()),
        "sd_post_over_sd_prior": float(post_sd.mean() / PRIOR_SD_DELTA),
        "ci90_halfwidth_log_ratio_unit_range": float(z * post_sd.mean() * ds_unit),
        "ci90_halfwidth_log_ratio_word_range": float(z * post_sd.mean() * ds_word),
        "power_90": power, "coverage_90": cover, "mean_unit_mean_range": ds_unit,
    }


def run_bound(args):
    rows = []
    for (P, E) in [(2, 2), (4, 3), (6, 4), (10, 10)]:
        for shared in (False, True):
            for dt_true in (-0.3, -0.8):
                rows.append(bound_one(P, E, dt_true, args.reps, 1000, 10.0, shared, args.seed))
    print("BEST-CASE bound on delta (oracle per-unit log ell; approximate, optimistic)")
    print("sd/prior=1.0 means the data add nothing. 'ratio CI' = 90% half-width of log(ell_hi/ell_lo);")
    print("e.g. 0.5 means ell could differ by a factor exp(0.5)=1.6 either way and still fit.\n")
    hdr = f"{'units':>5} {'shared':>6} {'delta':>6} {'post sd':>8} {'sd/prior':>9} {'ratioCI unit':>13} {'ratioCI word':>13} {'power90':>8} {'cover90':>8}"
    print(hdr)
    for r in rows:
        print(f"{r['n_units']:>5} {str(r['shared_stimulus']):>6} {r['delta_true']:>6.1f} "
              f"{r['post_sd_delta']:>8.3f} {r['sd_post_over_sd_prior']:>9.2f} "
              f"{r['ci90_halfwidth_log_ratio_unit_range']:>13.2f} {r['ci90_halfwidth_log_ratio_word_range']:>13.2f} "
              f"{r['power_90']:>8.2f} {r['coverage_90']:>8.2f}")
    if args.out:
        with open(args.out, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------- fit mode
def run_fit(args):
    import arviz as az
    import recovery_grid as rg
    from basis.basis_functions import make_raised_cosine_basis
    from models.bayesian_gp_trf import fit_bayesian_gp_trf
    from utils.ell_change import ell_change_summary, format_ell_change

    dt = 10.0
    n_lags = int(np.floor(args.tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt
    basis = make_raised_cosine_basis(n_basis=6, tau_max=args.tau_max, c=5.0)
    gen = "B-Matern-B2" if args.kernel == "matern52" else "B-SE-B2"
    data = rg.build_dataset_from_generator(gen, basis, taus, args.n_patients, args.n_electrodes,
                                           args.n_times, dt, args.snr, args.seed,
                                           delta_true=args.delta_true,
                                           shared_stimulus=args.shared_stimulus)
    ums = data["unit_mean_surp"]
    t0 = time.time()
    res = fit_bayesian_gp_trf(
        data["S0_raw"], data["S1_raw"], data["r"], data["patient_idx"], taus, ums,
        kernel_type=args.kernel, surprisal_in_kernel=True,
        draws=args.draws, tune=args.tune, chains=args.chains,
        cores=(args.cores if args.cores is not None else args.chains),
        target_accept=args.target_accept, seed=args.seed, noncentered=True,
    )
    trace = res.trace
    unit_range_defined = float(ums.max() - ums.min()) > 1e-9
    # Shared stimulus => zero observed range => the "what the design saw" ratio is
    # UNDEFINED (not small). Only the word-range (extrapolation) summary exists, and it
    # will be ~ the prior. Use the word-range summary as the carrier of delta stats then.
    s_unit = (ell_change_summary(trace, float(ums.min()), float(ums.max()), prob=0.9)
              if unit_range_defined else None)
    s_word = ell_change_summary(trace, args.word_lo, args.word_hi, prob=0.9)
    s_ref = s_unit if s_unit is not None else s_word
    summ = az.summary(trace, var_names=["delta", "log_ell0_h1"], round_to=4)
    en = np.asarray(trace.sample_stats["energy"].values)
    bfmi = [float(np.sum(np.diff(e) ** 2) / len(e) / np.var(e)) for e in en]
    d = s_ref["delta"]
    out = {
        "kernel": args.kernel, "n_patients": args.n_patients, "n_electrodes": args.n_electrodes,
        "n_units": args.n_patients * args.n_electrodes, "seed": args.seed, "snr": args.snr,
        "delta_true": args.delta_true, "shared_stimulus": args.shared_stimulus,
        "delta_mean": d["mean"], "delta_sd": d["sd"], "delta_ci90": [d["lo"], d["hi"]],
        "delta_truth_in_ci90": bool(d["lo"] <= args.delta_true <= d["hi"]),
        "delta_sd_over_prior": d["sd"] / PRIOR_SD_DELTA,
        "unit_mean_surp_range": [float(ums.min()), float(ums.max())],
        "ell_ratio_unit_range": (s_unit["ell_ratio"] if s_unit else None),
        "ell_ratio_word_range": s_word["ell_ratio"],
        "word_range": [args.word_lo, args.word_hi],
        "delta_chain_means": s_ref["delta_chain_means"],
        "corr_delta_log_ell0_h1": s_ref.get("corr_delta_log_ell0_h1"),
        "unit_range_defined": unit_range_defined,
        "rhat_delta": float(summ.loc["delta", "r_hat"]), "ess_delta": float(summ.loc["delta", "ess_bulk"]),
        "bfmi": bfmi, "n_divergences": int(trace.sample_stats["diverging"].sum()),
        "elapsed_s": time.time() - t0,
    }
    if s_unit is not None:
        print(format_ell_change(s_unit, label="unit-mean range = what the design observed"))
    else:
        print("Unit-mean surprisal range = 0 (shared stimulus): the ratio over the range the "
              "design observed is UNDEFINED. Only the word-range extrapolation below exists.")
    print(format_ell_change(s_word, label="word range = EXTRAPOLATION for unit-mean C2"))
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("ell_ratio")}, indent=1))
    if args.out:
        with open(args.out, "a") as f:
            f.write(json.dumps(out) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bound", "fit"], required=True)
    ap.add_argument("--reps", type=int, default=2000)               # bound
    ap.add_argument("--kernel", default="matern52", choices=["matern52", "squared_exponential"])
    ap.add_argument("--n_patients", type=int, default=4)
    ap.add_argument("--n_electrodes", type=int, default=3)
    ap.add_argument("--delta_true", type=float, default=-0.3)
    ap.add_argument("--n_times", type=int, default=1000)
    ap.add_argument("--tau_max", type=float, default=200.0)
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--word_lo", type=float, default=0.13)        # ~5th pct of Exp(mean 2.5)
    ap.add_argument("--word_hi", type=float, default=7.5)         # ~95th pct of Exp(mean 2.5)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1500)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--shared_stimulus", action="store_true",
                    help="fit mode: all units hear the same word stream (realistic). Expect "
                         "delta's posterior ~ prior and corr(delta, log_ell0_h1) ~ -1.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    (run_bound if args.mode == "bound" else run_fit)(args)


if __name__ == "__main__":
    main()
