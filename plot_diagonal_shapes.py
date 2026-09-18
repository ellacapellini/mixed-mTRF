"""
plot_diagonal_shapes.py
========================

Two deliverables for Tim, from ONE fit per model (not two, to avoid
doubling runtime):

1. Recovered TRF SHAPE h(tau) -- true vs. recovered, per model, plus
   one combined overlay. This is "what does the model think the
   impulse response looks like."

2. Raw TIME SERIES r(t) -- the actual observed (noisy) recording for
   one unit, vs. that model's PREDICTED r(t) (recovered h0/h1
   convolved back through the real design matrix, no noise). This is
   "does the model's fit actually explain the data over time," a
   different, complementary question from (1) -- a model can get the
   general shape roughly right while still fitting the moment-to-
   moment time series poorly, or vice versa.

Both come from the SAME representative unit (patient=0, electrode=0)
for every model, so comparisons across models are apples-to-apples.

Does NOT modify or depend on recovery_grid.py's checkpointed results
(those only store scalar r + diagnostics) -- re-fits the 7 diagonal
cells directly via the same underlying functions used everywhere else
in this project (fit_bayesian_mixed_trf / fit_bayesian_gp_trf / RidgemTRF).

BUG FIXED HERE, worth knowing about: an earlier version of this script
reconstructed "true" ground truth from a freshly-seeded RNG, separate
from the one build_dataset_from_generator used internally. For Model B,
ground_truth_gp.py's shape_for() is a LAZY draw -- it only consumes RNG
state the first time a given unit is requested, and build_dataset_from_
generator's own per-unit loop already advances the RNG (stimulus, word
stream) before it first calls shape_for for that unit. Reconstructing a
fresh RNG and calling shape_for immediately, without replaying those
same draws first, silently drew a DIFFERENT, unrelated h0/h1 and
displayed it as "true" -- this produced a completely misleading true-
vs-recovered mismatch for every Model B panel that looked like a real
recovery failure but was actually just this bug. Model A was
unaffected (weight_for() is a pure array lookup; all of Model A's
random structure is drawn upfront, not lazily per unit). THE FIX: get
ground truth directly from build_dataset_from_generator itself (which
already has the correctly-synced object internally), not from a
separate reconstruction -- robust by construction, not by bookkeeping.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from basis.basis_functions import make_raised_cosine_basis
from models.mtrf import RidgemTRF
from models.bayesian_mixed_trf import fit_bayesian_mixed_trf
from models.bayesian_gp_trf import fit_bayesian_gp_trf
from recovery_grid import build_dataset_from_generator, split_train_test, _GENERATOR_SEED_OFFSET

MODELS = ["Standard", "A-B1", "A-B2", "B-Matern-B1", "B-Matern-B2", "B-SE-B1", "B-SE-B2"]


def _posterior_mean(trace, name):
    return trace.posterior[name].mean(dim=("chain", "draw")).values


def get_true_and_recovered(
    model_name: str, n_patients: int, n_electrodes: int, n_times: int,
    n_basis: int, tau_max: float, dt: float, fit_kwargs: dict, seed: int,
):
    """
    Returns a dict with, all for unit (patient=0, electrode=0):
      taus, true_h0                                -- TRF shape, ground truth vs recovered
      recovered_h0
      t_train, true_r_train, predicted_r_train      -- IN-SAMPLE time series (fit on this data)
      t_test, true_r_test, predicted_r_test         -- HELD-OUT time series (never seen in fitting)

    The train/test split matters a lot here, not just cosmetically: an
    under-converged or overly flexible model can chase noise in its own
    training data and look like an excellent in-sample fit while its
    recovered parameters are still wrong (exactly what the shape-recovery
    comparison in this same function can independently reveal). Held-out
    prediction is the honest check -- this is the whole reason this
    project scores everything on held-out r / LOO rather than in-sample
    fit quality, made visible here as a picture instead of a number.
    """
    basis = make_raised_cosine_basis(n_basis=n_basis, tau_max=tau_max, c=5.0)
    n_lags = int(np.floor(tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt

    data = build_dataset_from_generator(
        model_name, basis, taus, n_patients, n_electrodes, n_times, dt,
        snr_target=5.0, seed=seed + _GENERATOR_SEED_OFFSET[model_name],
        return_ground_truth=True,
    )
    train, test = split_train_test(data)
    patient_idx = data["patient_idx"]
    unit_mean_surp = data["unit_mean_surp"]
    gt = data["ground_truth"]

    t_train = np.arange(train["r"].shape[1]) * dt
    t_test = np.arange(train["r"].shape[1], train["r"].shape[1] + test["r"].shape[1]) * dt
    true_r_train = train["r"][0]
    true_r_test = test["r"][0]

    # --- true h0 (see module docstring for why this must come from `gt`
    # returned by build_dataset_from_generator, not a fresh reconstruction) ---
    if model_name == "Standard" or model_name.startswith("A-"):
        mu_eff_true, _ = gt.weight_for(patient=0, electrode=0)
        true_h0 = basis.eval(taus) @ mu_eff_true
    else:
        true_h0, _ = gt.shape_for(patient=0, electrode=0, rng=None, surp=unit_mean_surp[0])

    # --- fit once, extract BOTH the recovered shape and the recovered
    # time-series prediction from the SAME fit ---
    if model_name == "Standard":
        n_units = train["S0_raw"].shape[0]
        X_train = np.concatenate([train["S0_raw"][i] for i in range(n_units)], axis=0)
        y_train = np.concatenate([train["r"][i] for i in range(n_units)], axis=0)
        model = RidgemTRF(lags=taus, alpha=1.0).fit(X_train, y_train)
        recovered_h0 = model.coef_
        predicted_r_train = model.predict(train["S0_raw"][0])
        predicted_r_test = model.predict(test["S0_raw"][0])

    elif model_name in ("A-B1", "A-B2"):
        result = fit_bayesian_mixed_trf(
            train["S0_basis"], train["S1_basis"], train["r"], patient_idx,
            variant=model_name, **fit_kwargs,
        )
        trace = result.trace
        mu = _posterior_mean(trace, "mu")
        u = np.array([_posterior_mean(trace, f"tau_p_u_{j}") * _posterior_mean(trace, f"raw_p_u_{j}")
                      for j in range(n_basis)]).T
        v = np.array([_posterior_mean(trace, f"tau_e_u_{j}") * _posterior_mean(trace, f"raw_e_u_{j}")
                      for j in range(n_basis)]).T
        mu_eff_unit0 = mu + u[0] + v[0]
        recovered_h0 = basis.eval(taus) @ mu_eff_unit0

        # beta_eff: same structure as mu_eff, but beta_fixed = kappa*mu (A-B1)
        # or free beta (A-B2), per bayesian_mixed_trf.py's actual model --
        # NOT kappa*mu_eff, a distinction that file's own comments flag as
        # a bug they already found and fixed once.
        if model_name == "A-B2":
            beta_fixed = _posterior_mean(trace, "beta")
        else:
            kappa = float(_posterior_mean(trace, "kappa"))
            beta_fixed = kappa * mu
        u_beta = np.array([_posterior_mean(trace, f"tau_p_beta_{j}") * _posterior_mean(trace, f"raw_p_beta_{j}")
                            for j in range(n_basis)]).T
        v_beta = np.array([_posterior_mean(trace, f"tau_e_beta_{j}") * _posterior_mean(trace, f"raw_e_beta_{j}")
                            for j in range(n_basis)]).T
        beta_eff_unit0 = beta_fixed + u_beta[0] + v_beta[0]

        predicted_r_train = train["S0_basis"][0] @ mu_eff_unit0 + train["S1_basis"][0] @ beta_eff_unit0
        predicted_r_test = test["S0_basis"][0] @ mu_eff_unit0 + test["S1_basis"][0] @ beta_eff_unit0

    else:
        kernel_type = "matern52" if "Matern" in model_name else "squared_exponential"
        surprisal_in_kernel = model_name.endswith("B2")
        result = fit_bayesian_gp_trf(
            train["S0_raw"], train["S1_raw"], train["r"], patient_idx, taus, unit_mean_surp,
            kernel_type=kernel_type, surprisal_in_kernel=surprisal_in_kernel, **fit_kwargs,
        )
        recovered_h0 = _posterior_mean(result.trace, "h0_0")
        recovered_h1 = _posterior_mean(result.trace, "h1_0")
        predicted_r_train = train["S0_raw"][0] @ recovered_h0 + train["S1_raw"][0] @ recovered_h1
        predicted_r_test = test["S0_raw"][0] @ recovered_h0 + test["S1_raw"][0] @ recovered_h1

    return {
        "taus": taus, "true_h0": true_h0, "recovered_h0": recovered_h0,
        "t_train": t_train, "true_r_train": true_r_train, "predicted_r_train": predicted_r_train,
        "t_test": t_test, "true_r_test": true_r_test, "predicted_r_test": predicted_r_test,
    }


def main(
    n_patients=2, n_electrodes=2, n_times=300, n_basis=4, tau_max=60.0, dt=10.0,
    chains=4, draws=50, tune=50, cores=4, seed=0,
    output_path="figures/diagonal_shapes_for_tim.png",
):
    fit_kwargs = dict(draws=draws, tune=tune, chains=chains, cores=cores,
                       target_accept=0.99, max_treedepth=14, seed=seed)

    results = {}
    for model_name in MODELS:
        print(f"Fitting {model_name}...", flush=True)
        results[model_name] = get_true_and_recovered(
            model_name, n_patients, n_electrodes, n_times, n_basis, tau_max, dt,
            fit_kwargs, seed,
        )
        print(f"  done.")

    # === Deliverable 1: TRF shape h(tau), true vs recovered ===
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    axes = axes.flatten()
    for i, model_name in enumerate(MODELS):
        r = results[model_name]
        ax = axes[i]
        ax.plot(r["taus"], r["true_h0"], "k--", label="true", linewidth=2)
        ax.plot(r["taus"], r["recovered_h0"], color="C0", label="recovered", linewidth=2)
        ax.set_title(model_name, fontsize=11)
        ax.set_xlabel("lag (ms)")
        ax.axhline(0, color="gray", lw=0.5)
        ax.legend(fontsize=8)
    axes[-1].axis("off")
    plt.suptitle("Diagonal self-recovery: true vs. recovered h(tau) [TRF SHAPE, not a time series]", fontsize=13)
    plt.tight_layout()
    shape_path = output_path.replace(".png", "_shape_small_multiples.png")
    plt.savefig(shape_path, dpi=150)
    print(f"Saved shape small multiples to {shape_path}")

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(MODELS)))
    for model_name, color in zip(MODELS, colors):
        r = results[model_name]
        ax2.plot(r["taus"], r["recovered_h0"], label=model_name, color=color, linewidth=2)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("lag (ms)")
    ax2.set_ylabel("recovered h(tau)")
    ax2.set_title("All 7 models' recovered TRF shape, same unit, overlaid")
    ax2.legend(fontsize=9)
    plt.tight_layout()
    shape_overlay_path = output_path.replace(".png", "_shape_overlay.png")
    plt.savefig(shape_overlay_path, dpi=150)
    print(f"Saved shape overlay to {shape_overlay_path}")

    # === Deliverable 2: raw TIME SERIES r(t) -- IN-SAMPLE (top) vs HELD-OUT (bottom) ===
    # Showing both stacked is deliberate: a model that looks great in-sample
    # but falls apart held-out is exactly the overfitting/non-convergence
    # signature this project's whole held-out-r/LOO framework exists to catch.
    fig3, axes3 = plt.subplots(2, 7, figsize=(28, 8), sharex="col")
    for i, model_name in enumerate(MODELS):
        r = results[model_name]
        ax_top, ax_bot = axes3[0, i], axes3[1, i]
        ax_top.plot(r["t_train"], r["true_r_train"], color="gray", alpha=0.6, linewidth=1)
        ax_top.plot(r["t_train"], r["predicted_r_train"], color="C1", linewidth=1.6)
        ax_top.set_title(model_name, fontsize=10)
        if i == 0:
            ax_top.set_ylabel("in-sample")
        ax_bot.plot(r["t_test"], r["true_r_test"], color="gray", alpha=0.6, linewidth=1, label="observed")
        ax_bot.plot(r["t_test"], r["predicted_r_test"], color="C3", linewidth=1.6, label="predicted")
        ax_bot.set_xlabel("time (ms)")
        if i == 0:
            ax_bot.set_ylabel("held-out")
            ax_bot.legend(fontsize=7)
    plt.suptitle("Diagonal self-recovery: raw TIME SERIES r(t), in-sample (top, orange) vs. held-out (bottom, red) -- "
                 "a model that only looks good in-sample is overfit/non-converged, not recovered", fontsize=12)
    plt.tight_layout()
    ts_path = output_path.replace(".png", "_timeseries_small_multiples.png")
    plt.savefig(ts_path, dpi=150)
    print(f"Saved time series small multiples to {ts_path}")

    fig4, (ax4a, ax4b) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)
    for model_name, color in zip(MODELS, colors):
        r = results[model_name]
        ax4a.plot(r["t_train"], r["predicted_r_train"], label=model_name, color=color, linewidth=1.5)
        ax4b.plot(r["t_test"], r["predicted_r_test"], label=model_name, color=color, linewidth=1.5)
    ax4a.set_title("In-sample predicted time series, all 7 models")
    ax4a.set_ylabel("predicted r(t)")
    ax4a.legend(fontsize=8)
    ax4b.set_title("Held-out predicted time series, all 7 models")
    ax4b.set_xlabel("time (ms)")
    ax4b.set_ylabel("predicted r(t)")
    ax4b.legend(fontsize=8)
    plt.tight_layout()
    ts_overlay_path = output_path.replace(".png", "_timeseries_overlay.png")
    plt.savefig(ts_overlay_path, dpi=150)
    print(f"Saved time series overlay to {ts_overlay_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-patients", type=int, default=2)
    parser.add_argument("--n-electrodes", type=int, default=2)
    parser.add_argument("--n-times", type=int, default=300)
    parser.add_argument("--n-basis", type=int, default=4)
    parser.add_argument("--tau-max", type=float, default=60.0)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=50)
    parser.add_argument("--tune", type=int, default=50)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--output-path", default="figures/diagonal_shapes_for_tim.png")
    args = parser.parse_args()
    main(
        n_patients=args.n_patients, n_electrodes=args.n_electrodes, n_times=args.n_times,
        n_basis=args.n_basis, tau_max=args.tau_max, chains=args.chains,
        draws=args.draws, tune=args.tune, cores=args.cores, output_path=args.output_path,
    )