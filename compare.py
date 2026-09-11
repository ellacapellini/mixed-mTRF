"""
compare.py
==========

THE presentation deliverable: fits all 7 models on one shared,
white-noise synthetic dataset, computes held-out predictive accuracy
for each, and produces a comparison figure.

Models 1-7 (see equations doc Section 8):
  1. Standard mTRF (mtrf.py)            -- no surprisal, no hierarchy
  2. Model A, A-C1  (bayesian_mixed_trf.py, variant="A-C1")
  3. Model A, A-C2  (bayesian_mixed_trf.py, variant="A-C2")
  4. Model B, Matern 5/2, C1  (bayesian_gp_trf.py)
  5. Model B, Matern 5/2, C2
  6. Model B, SE, C1
  7. Model B, SE, C2

HOW THE SHARED DATASET IS BUILT: one ground truth (Model A's stage-3
"full modulation" + hierarchical random effects, ground_truth.py --
chosen because it's the most thoroughly tested ground-truth generator
in the whole project, lowest risk to build the final comparison on)
generates the TRUE per-unit (mu_eff, beta_eff). For every patient/
electrode unit, the SAME underlying stimulus and surprisal sequence is
used to build BOTH a basis-projected design matrix (for Model A) and a
raw-lag design matrix (for Model B) -- Section 6 of the equations doc
proves these are numerically consistent (raw-lag projected through the
basis reproduces the basis-projected version exactly), so this is a
fair, apples-to-apples comparison, not different data per model.

WHY MODELS 2-7's PREDICTION LOGIC IS RECONSTRUCTED HERE, not added as
methods on the fitted result objects: fit_bayesian_mixed_trf and
fit_bayesian_gp_trf were built and tested under real time pressure
without a predict() method -- adding prediction support here, in the
one place it's actually needed, was lower-risk under deadline than
going back to re-test the already-working fitting code.

RUN THIS ON A MACHINE WITH REAL COMPUTE, not the sandbox this was
developed in (no BLAS, 1 CPU -- painfully slow). Default settings
below are deliberately modest (moderate draws/tune, 2 chains) to be
tractable on a normal laptop in a reasonable time; increase draws/tune
for a more polished final version if time allows.
"""

from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import arviz as az

from basis.basis_functions import make_raised_cosine_basis
from ground_truth import make_stage3_full_modulation, add_hierarchical_random_effects
from utils.design_matrix import build_design_matrix, build_raw_lag_design_matrix, expand_word_level_to_samples
from data.simulate import make_word_stream
from metrics import pearson_r, train_test_split_contiguous
from models.mtrf import RidgemTRF
from models.bayesian_mixed_trf import fit_bayesian_mixed_trf
from models.bayesian_gp_trf import fit_bayesian_gp_trf


# ============================================================================
# 1. Build the shared dataset
# ============================================================================

def build_shared_dataset(
    n_patients: int = 4,
    n_electrodes: int = 3,
    n_times: int = 1000,
    dt: float = 10.0,
    n_basis: int = 6,
    tau_max: float = 200.0,
    snr_target: float = 5.0,
    seed: int = 0,
):
    """
    One dataset, shared by all seven models. Returns a dict with
    everything each model family needs -- basis-projected design
    matrices for Model A, raw-lag design matrices for Model B, the
    response, patient index, per-unit mean surprisal (for B2), and
    the lag grid.
    """
    rng = np.random.default_rng(seed)
    basis = make_raised_cosine_basis(n_basis=n_basis, tau_max=tau_max, c=5.0)
    n_lags = int(np.floor(tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt

    gt = make_stage3_full_modulation(basis, rng)
    gt = add_hierarchical_random_effects(gt, n_patients=n_patients, n_electrodes=n_electrodes, rng=rng)

    S0_basis_all, S1_basis_all = [], []
    S0_raw_all, S1_raw_all = [], []
    r_all = []
    patient_idx = []
    unit_mean_surp = []

    for p in range(n_patients):
        for e in range(n_electrodes):
            stimulus = rng.normal(0, 1, size=n_times)
            word_onsets, word_surp = make_word_stream(n_times, dt, words_per_second=3.0, rng=rng)
            surp_at_t = expand_word_level_to_samples(n_times, dt, word_onsets, word_surp)

            design_basis = build_design_matrix(stimulus, surp_at_t, basis, dt)
            design_raw = build_raw_lag_design_matrix(stimulus, surp_at_t, taus, dt)

            mu_eff, beta_eff = gt.weight_for(patient=p, electrode=e)
            signal = design_basis.S0 @ mu_eff + design_basis.S1 @ beta_eff
            sigma_noise = signal.std() / snr_target if signal.std() > 1e-10 else 0.1
            noise = rng.normal(0, sigma_noise, size=n_times)
            r = signal + noise

            S0_basis_all.append(design_basis.S0)
            S1_basis_all.append(design_basis.S1)
            S0_raw_all.append(design_raw.S0)
            S1_raw_all.append(design_raw.S1)
            r_all.append(r)
            patient_idx.append(p)
            unit_mean_surp.append(float(word_surp.mean()))

    return {
        "basis": basis, "taus": taus, "dt": dt,
        "S0_basis": np.array(S0_basis_all), "S1_basis": np.array(S1_basis_all),
        "S0_raw": np.array(S0_raw_all), "S1_raw": np.array(S1_raw_all),
        "r": np.array(r_all),
        "patient_idx": np.array(patient_idx),
        "unit_mean_surp": np.array(unit_mean_surp),
        "n_units": n_patients * n_electrodes,
        "n_times": n_times,
    }


def split_train_test(data: dict, test_fraction: float = 0.2):
    """Same contiguous train/test split applied identically to every
    unit's design matrices and response -- one split, reused for all
    seven models, so held-out comparison is on identical held-out data."""
    train_idx, test_idx = train_test_split_contiguous(data["n_times"], test_fraction=test_fraction)

    def _split(arr):
        return arr[:, train_idx, :] if arr.ndim == 3 else arr[:, train_idx]

    train = {
        "S0_basis": data["S0_basis"][:, train_idx, :], "S1_basis": data["S1_basis"][:, train_idx, :],
        "S0_raw": data["S0_raw"][:, train_idx, :], "S1_raw": data["S1_raw"][:, train_idx, :],
        "r": data["r"][:, train_idx],
    }
    test = {
        "S0_basis": data["S0_basis"][:, test_idx, :], "S1_basis": data["S1_basis"][:, test_idx, :],
        "S0_raw": data["S0_raw"][:, test_idx, :], "S1_raw": data["S1_raw"][:, test_idx, :],
        "r": data["r"][:, test_idx],
    }
    return train, test


# ============================================================================
# 2. Fit + predict, one function per model
# ============================================================================

def fit_predict_model1_standard(train, test, taus):
    """Model 1: standard mTRF, raw lags, no surprisal, fully pooled
    (no patient hierarchy at all -- the plain baseline)."""
    n_units = train["S0_raw"].shape[0]
    X_train = np.concatenate([train["S0_raw"][i] for i in range(n_units)], axis=0)
    y_train = np.concatenate([train["r"][i] for i in range(n_units)], axis=0)
    X_test = np.concatenate([test["S0_raw"][i] for i in range(n_units)], axis=0)
    y_test = np.concatenate([test["r"][i] for i in range(n_units)], axis=0)

    model = RidgemTRF(lags=taus, alpha=1.0).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return pearson_r(y_test, y_pred)


def _posterior_mean(trace, name):
    return trace.posterior[name].mean(dim=("chain", "draw")).values


import model_selection


def _convergence_diagnostics(trace) -> dict:
    """Max Rhat, min bulk-ESS, and total divergence count across every
    parameter in the trace. This is the honest complement to held-out
    r: a model can post a good r purely because posterior_mean happened
    to land somewhere reasonable, even when the chains never actually
    converged to it (exactly what happened for SE-B1/B2 in the first
    seven-model run) -- Rhat/ESS/divergences are what tell you whether
    that r is something you're allowed to trust.

    ALSO includes by-unit LOO (model_selection.loo_by_unit) computed
    from this SAME trace -- LOO's importance-sampling approach estimates
    held-out performance from a single fit on the training data alone,
    so this costs no extra NUTS fit on top of the held-out-r evaluation
    already done elsewhere; it's a second, complementary reliability
    signal (Pareto-k specifically) from the fit we're already paying
    for, not a duplicate of the temporal held-out r check."""
    summary = az.summary(trace)
    n_divergent = int(trace.sample_stats["diverging"].sum()) if "diverging" in trace.sample_stats else 0
    diag = {
        "max_rhat": float(summary["r_hat"].max()),
        "min_ess_bulk": float(summary["ess_bulk"].min()),
        "n_divergences": n_divergent,
    }
    try:
        loo_result = model_selection.loo_by_unit(trace)
        diag.update({
            "elpd_loo": loo_result.elpd_loo,
            "elpd_loo_se": loo_result.elpd_loo_se,
            "max_pareto_k": loo_result.max_pareto_k,
            "n_bad_pareto_k": loo_result.n_bad_pareto_k,
            "n_units_loo": loo_result.n_units,
            "elpd_waic": loo_result.elpd_waic,
        })
    except Exception as exc:
        # LOO is a bonus diagnostic, not required for the cell's core
        # r/rhat result -- a LOO-specific failure (e.g. a trace missing
        # log_likelihood because pm.compute_log_likelihood wasn't called
        # by an older code path) shouldn't take down the whole cell.
        diag["loo_error"] = str(exc)
    return diag


def fit_predict_model_A(train, test, patient_idx, n_basis, variant, **fit_kwargs):
    """Models 2 (A-C1) and 3 (A-B2)."""
    result = fit_bayesian_mixed_trf(
        train["S0_basis"], train["S1_basis"], train["r"], patient_idx,
        variant=variant, **fit_kwargs,
    )
    trace = result.trace
    mu = _posterior_mean(trace, "mu")
    n_patients = int(patient_idx.max()) + 1
    n_units = len(patient_idx)

    u = np.zeros((n_patients, n_basis))
    u_beta = np.zeros((n_patients, n_basis))
    v = np.zeros((n_units, n_basis))
    v_beta = np.zeros((n_units, n_basis))
    for j in range(n_basis):
        u[:, j] = _posterior_mean(trace, f"tau_p_u_{j}") * _posterior_mean(trace, f"raw_p_u_{j}")
        u_beta[:, j] = _posterior_mean(trace, f"tau_p_beta_{j}") * _posterior_mean(trace, f"raw_p_beta_{j}")
        v[:, j] = _posterior_mean(trace, f"tau_e_u_{j}") * _posterior_mean(trace, f"raw_e_u_{j}")
        v_beta[:, j] = _posterior_mean(trace, f"tau_e_beta_{j}") * _posterior_mean(trace, f"raw_e_beta_{j}")

    if variant == "A-B2":
        beta_fixed = _posterior_mean(trace, "beta")
    else:
        kappa = float(_posterior_mean(trace, "kappa"))
        beta_fixed = kappa * mu

    mu_eff = mu[None, :] + u[patient_idx, :] + v
    beta_eff = beta_fixed[None, :] + u_beta[patient_idx, :] + v_beta

    y_pred_list, y_true_list = [], []
    for i in range(n_units):
        y_pred_list.append(test["S0_basis"][i] @ mu_eff[i] + test["S1_basis"][i] @ beta_eff[i])
        y_true_list.append(test["r"][i])
    r = pearson_r(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    return r, _convergence_diagnostics(trace)


def fit_predict_model_B(train, test, patient_idx, taus, unit_mean_surp, kernel_type, surprisal_in_kernel, **fit_kwargs):
    """Models 4-7: Model B, one of the four kernel/B1-B2 corners."""
    result = fit_bayesian_gp_trf(
        train["S0_raw"], train["S1_raw"], train["r"], patient_idx, taus, unit_mean_surp,
        kernel_type=kernel_type, surprisal_in_kernel=surprisal_in_kernel, **fit_kwargs,
    )
    trace = result.trace
    n_units = len(patient_idx)

    y_pred_list, y_true_list = [], []
    for i in range(n_units):
        h0_i = _posterior_mean(trace, f"h0_{i}")
        h1_i = _posterior_mean(trace, f"h1_{i}")
        y_pred_list.append(test["S0_raw"][i] @ h0_i + test["S1_raw"][i] @ h1_i)
        y_true_list.append(test["r"][i])
    r = pearson_r(np.concatenate(y_true_list), np.concatenate(y_pred_list))
    return r, _convergence_diagnostics(trace)


# ============================================================================
# 3. Run all seven, plot, save
# ============================================================================

def run_comparison(
    n_patients: int = 4,
    n_electrodes: int = 3,
    n_times: int = 1000,
    n_basis: int = 6,
    draws: int = 500,
    tune: int = 1000,
    chains: int = 4,
    seed: int = 0,
    output_path: str = "figures/07_seven_model_comparison.png",
):
    print("Building shared dataset...")
    data = build_shared_dataset(
        n_patients=n_patients, n_electrodes=n_electrodes, n_times=n_times,
        n_basis=n_basis, seed=seed,
    )
    train, test = split_train_test(data)
    patient_idx = data["patient_idx"]
    taus = data["taus"]
    unit_mean_surp_train = data["unit_mean_surp"]  # computed on full recording; fine as a representative scalar

    fit_kwargs_A = dict(draws=draws, tune=tune, chains=chains, target_accept=0.99, max_treedepth=14, seed=seed)
    fit_kwargs_B = dict(draws=draws, tune=tune, chains=chains, target_accept=0.99, max_treedepth=14, seed=seed)

    results = {}
    diagnostics = {}  # None for models with no posterior (Model 1, plain ridge)

    print("[1/7] Standard mTRF...")
    results["Standard\n(no surprisal)"] = fit_predict_model1_standard(train, test, taus)
    diagnostics["Standard\n(no surprisal)"] = None

    print("[2/7] Model A, ABC1...")
    results["Model A\nA-B1"], diagnostics["Model A\nA-B1"] = fit_predict_model_A(
        train, test, patient_idx, n_basis, "A-B1", **fit_kwargs_A
    )

    print("[3/7] Model A, ABC2...")
    results["Model A\nA-B2"], diagnostics["Model A\nA-B2"] = fit_predict_model_A(
        train, test, patient_idx, n_basis, "A-B2", **fit_kwargs_A
    )

    print("[4/7] Model B, Matern, B1...")
    results["Model B\nMatern-B1"], diagnostics["Model B\nMatern-B1"] = fit_predict_model_B(
        train, test, patient_idx, taus, unit_mean_surp_train, "matern52", False, **fit_kwargs_B
    )

    print("[5/7] Model B, Matern, B2...")
    results["Model B\nMatern-B2"], diagnostics["Model B\nMatern-B2"] = fit_predict_model_B(
        train, test, patient_idx, taus, unit_mean_surp_train, "matern52", True, **fit_kwargs_B
    )

    print("[6/7] Model B, SE, B1...")
    results["Model B\nSE-B1"], diagnostics["Model B\nSE-B1"] = fit_predict_model_B(
        train, test, patient_idx, taus, unit_mean_surp_train, "squared_exponential", False, **fit_kwargs_B
    )

    print("[7/7] Model B, SE, B2...")
    results["Model B\nSE-B2"], diagnostics["Model B\nSE-B2"] = fit_predict_model_B(
        train, test, patient_idx, taus, unit_mean_surp_train, "squared_exponential", True, **fit_kwargs_B
    )

    print("\n=== Held-out Pearson r, all seven models ===")
    for name, r in results.items():
        print(f"  {name.replace(chr(10), ' '):30s}: {r:.4f}")

    print("\n=== Convergence diagnostics (six Bayesian models; Standard has no posterior) ===")
    for name, d in diagnostics.items():
        if d is None:
            print(f"  {name.replace(chr(10), ' '):30s}: n/a (point estimate, no posterior)")
        else:
            flag = "  <-- NOT CONVERGED" if (d["max_rhat"] > 1.01 or d["n_divergences"] > 0) else ""
            print(f"  {name.replace(chr(10), ' '):30s}: max_rhat={d['max_rhat']:.3f}  "
                  f"min_ess_bulk={d['min_ess_bulk']:.0f}  n_divergences={d['n_divergences']}{flag}")

    names = list(results.keys())
    values = list(results.values())
    colors = ["gray"] + ["#4C72B0"] * 2 + ["#DD8452"] * 4

    fig, (ax_r, ax_diag) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # --- top panel: held-out r, same as before ---
    bars = ax_r.bar(names, values, color=colors)
    # hatch out any model that didn't converge, so a reader can't read the
    # top panel's height as trustworthy without also looking at the bottom
    for bar, name in zip(bars, names):
        d = diagnostics[name]
        if d is not None and (d["max_rhat"] > 1.01 or d["n_divergences"] > 0):
            bar.set_hatch("////")
            bar.set_edgecolor("red")
    ax_r.set_ylabel("Held-out Pearson r")
    ax_r.set_title("Seven-model comparison, shared white-noise synthetic dataset\n"
                    "(hatched red = did not converge -- see diagnostics below)")
    ax_r.axhline(0, color="black", lw=0.5)

    # --- bottom panel: max Rhat per model, with the standard 1.01 threshold ---
    rhat_values = [d["max_rhat"] if d is not None else np.nan for d in diagnostics.values()]
    diag_colors = [
        "lightgray" if d is None else ("crimson" if d["max_rhat"] > 1.01 else "seagreen")
        for d in diagnostics.values()
    ]
    ax_diag.bar(names, rhat_values, color=diag_colors)
    ax_diag.axhline(1.01, color="black", lw=1, ls="--", label="Rhat = 1.01 threshold")
    ax_diag.set_ylabel("Max R-hat")
    ax_diag.set_ylim(0.99, max([v for v in rhat_values if not np.isnan(v)] + [1.05]) * 1.05)
    ax_diag.legend(loc="upper left", fontsize=8)
    for i, (name, d) in enumerate(diagnostics.items()):
        if d is not None and d["n_divergences"] > 0:
            ax_diag.annotate(f"{d['n_divergences']} div.", (i, d["max_rhat"]),
                              textcoords="offset points", xytext=(0, 6),
                              ha="center", fontsize=7, color="crimson")

    plt.xticks(rotation=0, fontsize=9)
    plt.tight_layout()

    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved comparison + diagnostics figure to {output_path}")

    return results, diagnostics


if __name__ == "__main__":
    run_comparison()