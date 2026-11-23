"""same_shape_overlay.py -- the "same shape" test: fit ONE given fitter to
a SHARED dataset (same seed every time this script is called, regardless of
which --fitter is passed) and save its recovered h(tau|surp) curve at two
surprisal levels. Run once per fitter (7 separate Euler jobs, one per
fitter), then run plot_same_shape_overlay.py to combine all 7 into one
overlay against the true curve.

Default generator is A-B2: the most flexible basis-family generator, and
(Section 4) a strict special case of every Model B kernel -- so every one
of the 7 fitters should in principle be ABLE to represent it. If a fitter
disagrees sharply from the others here, that's a fitter problem, not a
"maybe the true process isn't expressible in this model" excuse.

Usage (on Euler, from the project root, venv activated):

    python same_shape_overlay.py --fitter Standard
    python same_shape_overlay.py --fitter A-B1
    python same_shape_overlay.py --fitter A-B2
    python same_shape_overlay.py --fitter B-Matern-B1
    python same_shape_overlay.py --fitter B-Matern-B2
    python same_shape_overlay.py --fitter B-SE-B1
    python same_shape_overlay.py --fitter B-SE-B2

All seven MUST use the same --seed (default below) and --scale so they
share the identical dataset -- don't override --seed per-call.

Output: figures/overlay/<fitter>.json
    {fitter, surp_lo, surp_hi, taus, h_lo, h_hi, max_rhat (null for Standard),
     n_divergences (null for Standard)}
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from basis.basis_functions import make_raised_cosine_basis
from recovery_grid import build_dataset_from_generator, split_train_test
from models.mtrf import RidgemTRF
from models.bayesian_mixed_trf import fit_bayesian_mixed_trf
from models.bayesian_gp_trf import fit_bayesian_gp_trf
from metrics import pearson_r

SCALE_PRESETS = {
    "pilot": dict(n_patients=2, n_electrodes=2, n_times=400, n_basis=5,
                  tau_max=100.0, draws=150, tune=300, chains=2),
    "full":  dict(n_patients=4, n_electrodes=3, n_times=1000, n_basis=6,
                  tau_max=200.0, draws=500, tune=1000, chains=4),
}

FITTERS = ["Standard", "A-B1", "A-B2", "B-Matern-B1", "B-Matern-B2", "B-SE-B1", "B-SE-B2"]


def _posterior_mean(trace, var_name):
    return trace.posterior[var_name].mean(dim=("chain", "draw")).values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fitter", required=True, choices=FITTERS)
    ap.add_argument("--generator", default="A-B2",
                     help="dataset ALL 7 fitters share -- default A-B2 (real signal, "
                          "strict special case of every Model B kernel per Section 4)")
    ap.add_argument("--scale", default="full", choices=list(SCALE_PRESETS))
    ap.add_argument("--seed", type=int, default=777,
                     help="MUST be identical across all 7 fitter calls so they share one dataset")
    ap.add_argument("--snr-target", type=float, default=5.0)
    ap.add_argument("--surp-lo", type=float, default=1.0)
    ap.add_argument("--surp-hi", type=float, default=5.0)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--max-treedepth", type=int, default=14)
    ap.add_argument("--out-dir", default="figures/overlay")
    args = ap.parse_args()

    p = SCALE_PRESETS[args.scale]
    n_patients, n_electrodes, n_times = p["n_patients"], p["n_electrodes"], p["n_times"]
    n_basis, tau_max = p["n_basis"], p["tau_max"]
    draws = args.draws or p["draws"]
    tune = args.tune or p["tune"]
    chains = args.chains or p["chains"]
    cores = args.cores or chains
    dt = 10.0

    os.makedirs(args.out_dir, exist_ok=True)

    basis = make_raised_cosine_basis(n_basis=n_basis, tau_max=tau_max, c=5.0)
    n_lags = int(np.floor(tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt

    print(f"Building SHARED dataset: generator={args.generator}, seed={args.seed} "
          f"(must match every other fitter's call for this to be a valid overlay)")
    data = build_dataset_from_generator(
        args.generator, basis, taus, n_patients, n_electrodes, n_times, dt,
        snr_target=args.snr_target, seed=args.seed, return_ground_truth=True,
    )
    train, test = split_train_test(data)
    patient_idx = data["patient_idx"]
    unit_mean_surp = data["unit_mean_surp"]
    n_units = len(patient_idx)

    max_rhat = None
    n_divergences = None

    print(f"Fitting: fitter={args.fitter}")
    if args.fitter == "Standard":
        X_train = np.concatenate([train["S0_raw"][i] for i in range(n_units)], axis=0)
        y_train = np.concatenate([train["r"][i] for i in range(n_units)], axis=0)
        model = RidgemTRF(lags=taus, alpha=1.0).fit(X_train, y_train)
        # No surprisal modulation at all -- the same flat curve at both
        # surprisal levels. That flatness IS the point: this is the thing
        # the comparison is meant to show Standard cannot do.
        h_lo = model.coef_.copy()
        h_hi = model.coef_.copy()

    elif args.fitter in ("A-B1", "A-B2"):
        result = fit_bayesian_mixed_trf(
            train["S0_basis"], train["S1_basis"], train["r"], patient_idx,
            variant=args.fitter, draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=args.target_accept, max_treedepth=args.max_treedepth, seed=args.seed,
        )
        trace = result.trace
        mu = _posterior_mean(trace, "mu")
        # Recompute unit-0's effective (mu_eff, beta_eff) the same way
        # fit_predict_model_A does, but we only need unit 0 (patient 0,
        # electrode 0) here, not every unit.
        u = np.stack([_posterior_mean(trace, f"tau_p_u_{j}") * _posterior_mean(trace, f"raw_p_u_{j}")
                       for j in range(n_basis)], axis=1)  # (n_patients, n_basis)
        v = np.stack([_posterior_mean(trace, f"tau_e_u_{j}") * _posterior_mean(trace, f"raw_e_u_{j}")
                       for j in range(n_basis)], axis=1)  # (n_units, n_basis)
        u_beta = np.stack([_posterior_mean(trace, f"tau_p_beta_{j}") * _posterior_mean(trace, f"raw_p_beta_{j}")
                            for j in range(n_basis)], axis=1)
        v_beta = np.stack([_posterior_mean(trace, f"tau_e_beta_{j}") * _posterior_mean(trace, f"raw_e_beta_{j}")
                            for j in range(n_basis)], axis=1)

        if args.fitter == "A-B2":
            beta_fixed = _posterior_mean(trace, "beta")
        else:
            kappa = float(_posterior_mean(trace, "kappa"))
            beta_fixed = kappa * mu

        mu_eff_0 = mu + u[patient_idx[0]] + v[0]
        beta_eff_0 = beta_fixed + u_beta[patient_idx[0]] + v_beta[0]

        phi = basis.eval(taus)  # (n_lags, n_basis)
        h_lo = phi @ (mu_eff_0 + beta_eff_0 * args.surp_lo)
        h_hi = phi @ (mu_eff_0 + beta_eff_0 * args.surp_hi)

        import arviz as az
        max_rhat = float(az.summary(trace)["r_hat"].max())
        n_divergences = int(trace.sample_stats["diverging"].sum().values) if "diverging" in trace.sample_stats else None

    else:  # Model B
        kernel_type = "matern52" if "Matern" in args.fitter else "squared_exponential"
        surprisal_in_kernel = args.fitter.endswith("B2")
        result = fit_bayesian_gp_trf(
            train["S0_raw"], train["S1_raw"], train["r"], patient_idx, taus, unit_mean_surp,
            kernel_type=kernel_type, surprisal_in_kernel=surprisal_in_kernel,
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=args.target_accept, max_treedepth=args.max_treedepth, seed=args.seed,
        )
        trace = result.trace
        h0_0 = _posterior_mean(trace, "h0_0")
        h1_0 = _posterior_mean(trace, "h1_0")
        h_lo = h0_0 + h1_0 * args.surp_lo
        h_hi = h0_0 + h1_0 * args.surp_hi

        import arviz as az
        max_rhat = float(az.summary(trace)["r_hat"].max())
        n_divergences = int(trace.sample_stats["diverging"].sum().values) if "diverging" in trace.sample_stats else None

    out = dict(
        fitter=args.fitter, generator=args.generator, seed=args.seed,
        surp_lo=args.surp_lo, surp_hi=args.surp_hi,
        taus=taus.tolist(), h_lo=np.asarray(h_lo).tolist(), h_hi=np.asarray(h_hi).tolist(),
        max_rhat=max_rhat, n_divergences=n_divergences,
    )
    out_path = os.path.join(args.out_dir, f"{args.fitter}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {out_path}  (max_rhat={max_rhat})")


if __name__ == "__main__":
    main()