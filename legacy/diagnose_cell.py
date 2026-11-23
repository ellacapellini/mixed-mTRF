"""diagnose_cell.py -- deep-dive diagnostics for ONE (generator, fitter) cell.

Reproduces exactly what recovery_grid.py would have fit for a given cell
(same seed-per-generator scheme, same scale presets), but keeps the full
PyMC trace in memory (recovery_grid.py's checkpointed JSON only stores
summary numbers) so we can run diagnostics recovery_grid.py never computes:
BFMI, per-parameter Rhat (not just the max), and a prior-vs-posterior
overlap for the kernel hyperparameters -- the check that tells you whether
a bad Rhat means "not enough data to identify this" (posterior ~= prior)
versus a real sampler/geometry problem (posterior is some OTHER wrong
place, not just spread out over the prior).

Also doubles as the "realistic scale" driver: pass --n-patients/--n-
electrodes to run the SAME cell at a bigger size than the full-scale grid,
to see whether more clusters actually fixes convergence (Bates et al. 2015 /
Section 9's own prediction) rather than just resembling it.

Usage (on Euler, from the project root, venv activated):

    python diagnose_cell.py --generator Standard --fitter B-Matern-B1

    # same cell, but with more patients/electrodes -- the "realistic scale"
    # identifiability check:
    python diagnose_cell.py --generator Standard --fitter B-Matern-B1 \
        --n-patients 8 --n-electrodes 12

Outputs, all under --out-dir (default figures/diagnose/<generator>_<fitter>/):
    trace.nc                  -- full arviz InferenceData, for later reuse
    summary.txt               -- per-parameter Rhat/ESS, sorted worst-first
    bfmi.txt                  -- one BFMI value per chain
    prior_posterior.png       -- overlay for the kernel hyperparameters
                                  (Model B) or mu/beta (Model A)
    trace_worst.png           -- trace plot for the 6 worst-Rhat parameters
"""
from __future__ import annotations

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import arviz as az

from basis.basis_functions import make_raised_cosine_basis
from recovery_grid import (
    GENERATOR_NAMES, FITTER_NAMES, _GENERATOR_SEED_OFFSET,
    build_dataset_from_generator, split_train_test,
)
from models.bayesian_mixed_trf import fit_bayesian_mixed_trf
from models.bayesian_gp_trf import fit_bayesian_gp_trf

SCALE_PRESETS = {
    "pilot": dict(n_patients=2, n_electrodes=2, n_times=400, n_basis=5,
                  tau_max=100.0, draws=150, tune=300, chains=2),
    "full":  dict(n_patients=4, n_electrodes=3, n_times=1000, n_basis=6,
                  tau_max=200.0, draws=500, tune=1000, chains=4),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--generator", required=True, choices=GENERATOR_NAMES)
    ap.add_argument("--fitter", required=True, choices=FITTER_NAMES)
    ap.add_argument("--scale", default="full", choices=list(SCALE_PRESETS))
    ap.add_argument("--n-patients", type=int, default=None, help="override preset (the 'realistic scale' check)")
    ap.add_argument("--n-electrodes", type=int, default=None, help="override preset")
    ap.add_argument("--n-times", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="base seed, same convention as recovery_grid.py")
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--tune", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--cores", type=int, default=None, help="default: cores=chains")
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--max-treedepth", type=int, default=14)
    ap.add_argument("--snr-target", type=float, default=5.0,
                     help="signal-to-noise ratio used when simulating the dataset "
                          "(recovery_grid.py's full grid always used 5.0 -- this lets "
                          "you sweep it to see whether a convergence pathology is "
                          "specific to that one idealized noise level)")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    p = SCALE_PRESETS[args.scale]
    n_patients = args.n_patients or p["n_patients"]
    n_electrodes = args.n_electrodes or p["n_electrodes"]
    n_times = args.n_times or p["n_times"]
    n_basis = p["n_basis"]
    tau_max = p["tau_max"]
    draws = args.draws or p["draws"]
    tune = args.tune or p["tune"]
    chains = args.chains or p["chains"]
    cores = args.cores or chains
    dt = 10.0  # matches recovery_grid.py's implicit dt (tau_max / (n_lags-1)); see note below

    # snr_target folded into the dir name (only when non-default) so a sweep's
    # runs land in separate folders instead of overwriting each other; a
    # default-SNR run keeps the old plain name for backward compatibility
    # with folders you've already downloaded.
    snr_suffix = "" if args.snr_target == 5.0 else f"_snr{args.snr_target:g}"
    out_dir = args.out_dir or f"figures/diagnose/{args.generator}_{args.fitter}{snr_suffix}"
    os.makedirs(out_dir, exist_ok=True)

    basis = make_raised_cosine_basis(n_basis=n_basis, tau_max=tau_max, c=5.0)
    n_lags = int(np.floor(tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt

    gen_seed = args.seed + _GENERATOR_SEED_OFFSET[args.generator]
    print(f"Building dataset: generator={args.generator}, seed={gen_seed}, "
          f"n_patients={n_patients}, n_electrodes={n_electrodes}, n_times={n_times}")
    data = build_dataset_from_generator(
        args.generator, basis, taus, n_patients, n_electrodes, n_times, dt,
        snr_target=args.snr_target, seed=gen_seed, return_ground_truth=True,
    )
    train, _test = split_train_test(data)
    patient_idx = data["patient_idx"]
    unit_mean_surp = data["unit_mean_surp"]
    gt = data["ground_truth"]

    print(f"Fitting: fitter={args.fitter}, draws={draws}, tune={tune}, chains={chains}, cores={cores}")
    if args.fitter in ("A-B1", "A-B2"):
        result = fit_bayesian_mixed_trf(
            train["S0_basis"], train["S1_basis"], train["r"], patient_idx,
            variant=args.fitter, draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=args.target_accept, max_treedepth=args.max_treedepth, seed=args.seed,
        )
        prior_posterior_vars = ["mu", "beta"] if args.fitter == "A-B2" else ["mu", "kappa"]
    elif args.fitter == "Standard":
        raise SystemExit("Standard is a closed-form ridge fit -- no trace/NUTS diagnostics to run. "
                          "Use --fitter one of the Bayesian models instead.")
    else:
        kernel_type = "matern52" if "Matern" in args.fitter else "squared_exponential"
        surprisal_in_kernel = args.fitter.endswith("B2")
        result = fit_bayesian_gp_trf(
            train["S0_raw"], train["S1_raw"], train["r"], patient_idx, taus, unit_mean_surp,
            kernel_type=kernel_type, surprisal_in_kernel=surprisal_in_kernel,
            draws=draws, tune=tune, chains=chains, cores=cores,
            target_accept=args.target_accept, max_treedepth=args.max_treedepth, seed=args.seed,
        )
        prior_posterior_vars = ["log_ell0_h0", "log_sigma0_h0", "log_ell0_h1", "log_sigma0_h1"]
        if surprisal_in_kernel:
            prior_posterior_vars += ["delta", "gamma"]

    trace = result.trace
    # Saving is best-effort: a fit can take hours, and a missing NetCDF
    # backend (netCDF4/h5netcdf) once threw away a completed 7-hour fit
    # because this line ran first and raised. Never let a save failure
    # block the diagnostics below.
    try:
        trace.to_netcdf(os.path.join(out_dir, "trace.nc"))
        print(f"Saved full trace to {out_dir}/trace.nc")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: could not save trace.nc ({exc!r}); falling back to pickle.")
        try:
            import pickle
            with open(os.path.join(out_dir, "trace.pkl"), "wb") as f:
                pickle.dump(trace, f)
            print(f"Saved full trace to {out_dir}/trace.pkl")
        except Exception as exc2:  # noqa: BLE001
            print(f"WARNING: pickle fallback also failed ({exc2!r}); continuing without a saved trace.")

    # --- 1. Per-parameter Rhat/ESS, sorted worst-first (not just the max) --
    summary = az.summary(trace)
    summary_sorted = summary.sort_values("r_hat", ascending=False)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(summary_sorted.to_string())
    print("\nWorst 10 parameters by Rhat:")
    print(summary_sorted.head(10)[["mean", "sd", "ess_bulk", "ess_tail", "r_hat"]])

    # --- 2. BFMI: catches sampler pathologies (esp. funnels) that Rhat/
    # divergence-count alone can miss ------------------------------------
    bfmi_raw = az.bfmi(trace)
    # az.bfmi's return type has changed across arviz versions (plain array
    # vs. a Dataset/DataTree wrapping an "energy" variable). Pull out a
    # flat numpy array either way so the printed/saved value is just the
    # per-chain numbers, not an object repr.
    try:
        bfmi = np.asarray(bfmi_raw["energy"].values if hasattr(bfmi_raw, "__getitem__") else bfmi_raw)
    except Exception:
        bfmi = np.asarray(bfmi_raw)
    with open(os.path.join(out_dir, "bfmi.txt"), "w") as f:
        f.write(f"BFMI per chain: {bfmi.tolist()}\n")
        f.write("Rule of thumb: values well below ~0.3 indicate the sampler is\n"
                "struggling to explore the posterior's energy distribution,\n"
                "independent of what Rhat/divergences say.\n")
    print(f"\nBFMI per chain: {bfmi.tolist()}")

    # --- 3. Trace plot for the worst offenders ---------------------------
    # Each plotting block is its own try/except: a failure in one (e.g. an
    # arviz/matplotlib version mismatch) must never cost the numeric
    # results already computed and saved above, or the other plot below.
    # az.summary() reports per-element names like "h0_3[2]" for vector
    # parameters, but plot_trace's var_names filter only accepts base
    # variable names ("h0_3") -- passing the indexed form raises a KeyError
    # even though the variable genuinely exists. Strip the "[...]" suffix
    # and dedupe, preserving the worst-first order.
    seen = set()
    worst_vars = []
    for name in summary_sorted.index:
        base = name.split("[")[0]
        if base not in seen:
            seen.add(base)
            worst_vars.append(base)
        if len(worst_vars) == 6:
            break
    try:
        with az.rc_context(rc={"plot.max_subplots": 120}):
            az.plot_trace(trace, var_names=worst_vars, compact=False)
        plt.savefig(os.path.join(out_dir, "trace_worst.png"), dpi=110, bbox_inches="tight")
        plt.close("all")
        print(f"Saved trace plot to {out_dir}/trace_worst.png")
    except Exception as exc:  # noqa: BLE001
        print(f"WARNING: trace_worst.png failed ({exc!r}); skipping.")
        plt.close("all")

    # --- 4. Prior-posterior overlap --------------------------------------
    # If the posterior for a kernel hyperparameter looks ~identical to its
    # prior, that's evidence the likelihood genuinely isn't informative for
    # THIS dataset -- non-identifiability, not a sampler bug. If instead the
    # posterior is confidently in some OTHER wrong place (not matching the
    # prior, not matching truth), that points at a real bug or a genuine
    # geometry problem worth reparameterizing.
    #
    # Built manually rather than via az.plot_dist_comparison: that function
    # is absent in this environment's arviz install (a version where
    # top-level plotting is being split into arviz_base/arviz_plots/
    # arviz_stats, and this one hasn't landed under any name we could find).
    # Priors below are hardcoded from models/bayesian_gp_trf.py's actual
    # pm.Normal(...) calls (ell0_prior=50.0, sigma0_prior=1.0 defaults) --
    # if you ever change those defaults, update PRIOR_SPECS to match.
    PRIOR_SPECS = {
        "log_ell0_h0":   (np.log(50.0), 0.3),
        "log_sigma0_h0": (np.log(1.0), 0.3),
        "log_ell0_h1":   (np.log(50.0), 0.3),
        "log_sigma0_h1": (np.log(1.0), 0.3),
        "delta":         (0.0, 0.3),
        "gamma":         (0.0, 0.3),
    }
    present_vars = [v for v in prior_posterior_vars if v in trace.posterior and v in PRIOR_SPECS]
    if present_vars:
        try:
            fig, axes = plt.subplots(1, len(present_vars), figsize=(4 * len(present_vars), 3.5))
            if len(present_vars) == 1:
                axes = [axes]
            for ax, var in zip(axes, present_vars):
                mu, sd = PRIOR_SPECS[var]
                post = trace.posterior[var].values.flatten()
                xmin = min(post.min(), mu - 4 * sd)
                xmax = max(post.max(), mu + 4 * sd)
                xs = np.linspace(xmin, xmax, 300)
                prior_density = np.exp(-0.5 * ((xs - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))
                ax.plot(xs, prior_density, "k--", label="prior", linewidth=1.5)
                ax.hist(post, bins=40, density=True, alpha=0.6, color="C0", label="posterior")
                ax.set_title(var, fontsize=10)
                ax.legend(fontsize=8)
            fig.suptitle("Prior (dashed) vs posterior (filled) -- overlap = non-identifiable given this data,\n"
                          "not necessarily a sampler bug", fontsize=9)
            fig.tight_layout()
            plt.savefig(os.path.join(out_dir, "prior_posterior.png"), dpi=110, bbox_inches="tight")
            plt.close("all")
            print(f"Saved prior-posterior overlay for {present_vars} to {out_dir}/prior_posterior.png")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: prior_posterior.png failed ({exc!r}); skipping.")
            plt.close("all")
    else:
        print(f"None of {prior_posterior_vars} have a known prior spec -- skipping overlay.")

    print(f"\nAll diagnostics written to {out_dir}/")


if __name__ == "__main__":
    main()