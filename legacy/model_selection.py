"""
model_selection.py
===================

The decision-statistic layer for model recovery / identifiability
studies (see recovery_grid.py for the underlying generator/fitter
machinery this sits on top of).

WHY THIS FILE EXISTS: held-out Pearson r answers "how well does this
model predict", not "would a fair comparison against its rivals pick
this model" -- those are different questions, and only the second one
licenses a claim like "the data can distinguish C1 from C2". The
correct tool for the second question is WAIC/LOO (Vehtari et al. 2017,
already cited in the equations doc's Section 6), but getting it right
here required fixing two things that were NOT yet correct anywhere in
the codebase:

1. Neither bayesian_gp_trf.py nor bayesian_mixed_trf.py captured
   pointwise log-likelihood at all (no idata_kwargs / compute_log_
   likelihood call), so az.loo()/az.waic() could not be computed from
   any existing trace, full stop. Fixed: both now call
   pm.compute_log_likelihood(trace, model=model) after pm.sample().

2. THE MORE IMPORTANT FIX: `pm.Normal("obs", ..., observed=r_all)`
   with r_all shape (n_units, n_times) makes PyMC store one log-
   likelihood value per (unit, TIMEPOINT) pair. Calling az.loo()
   directly on that treats every timepoint as an independent,
   exchangeable observation for leave-one-out purposes -- which is
   wrong for this design specifically: r_all comes from a CONVOLUTION
   (design_matrix.py), so adjacent timepoints share overlapping
   stimulus history and are highly autocorrelated. Leave-one-timepoint
   -out silently violates the exchangeability LOO/WAIC formally
   require, and would be systematically overconfident (the "left out"
   point is barely informative to predict because its neighbours,
   still in the training set, already encode almost the same
   information). The unit that IS genuinely exchangeable here is the
   unit itself (patient x electrode) -- that's what the hierarchy
   (Section 2.5/5.3 of the equations doc) is actually built over, and
   units are conditionally independent given the population/patient/
   electrode-level parameters. So: sum log-likelihood over the time
   axis WITHIN each unit first, THEN run LOO/WAIC treating units as
   the leave-one-out axis. This is the standard fix for hierarchical
   models with within-group dependence (sometimes called "leave-one-
   group-out"); see Vehtari et al. 2017's own discussion of grouped/
   structured data, and Bürkner et al. 2020 on LOO for time series and
   other non-exchangeable structures specifically.

3. `az.waic()` does not exist in the installed arviz version (1.3.0 --
   API changed; LOO is the modern default and Vehtari et al. themselves
   recommend it over WAIC anyway). A minimal WAIC is hand-rolled below
   as a cross-check, not because it's preferred, but because having
   two independent estimates of the same quantity is a legitimate case
   check for the sweep's key numbers.

4. IMPORTANT LIMITATION TO CARRY FORWARD, NOT PAPER OVER: PSIS-LOO's
   importance-sampling approximation needs enough independent units to
   be reliable (Pareto-k <= 0.7 as the usual cutoff; <=0.5 for "good").
   With n_patients=4 x n_electrodes=3 = 12 units (your compare.py
   production default), or fewer in early pilots, Pareto-k warnings are
   a realistic possibility, not a hypothetical edge case -- loo_by_unit
   below surfaces this automatically on every call rather than letting
   a bad k slip through silently into a reported number.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import xarray as xr
import arviz as az


@dataclass
class ByUnitModelFit:
    elpd_loo: float
    elpd_loo_se: float
    p_loo: float
    max_pareto_k: float
    n_bad_pareto_k: int      # count with k > 0.7 -- LOO unreliable for these units
    n_units: int
    elpd_waic: float | None  # hand-rolled, see module docstring point 3
    p_waic: float | None


def loo_by_unit(trace, obs_var_name: str = "obs") -> ByUnitModelFit:
    """
    Correct LOO for this project's hierarchical-over-units, correlated-
    within-unit-in-time design: sums pointwise log-likelihood over the
    time axis WITHIN each unit before handing it to PSIS-LOO, so units
    (not individual timepoints) are the leave-one-out axis. See module
    docstring for why this matters -- the naive az.loo(trace) call on
    the raw per-timepoint log-likelihood would silently be invalid.

    Requires the trace to have been produced by a model that called
    pm.compute_log_likelihood(trace, model=model) after pm.sample()
    (both bayesian_gp_trf.py and bayesian_mixed_trf.py do this).
    """
    ll = trace.log_likelihood[obs_var_name]
    # time is whichever obs_dim is NOT the unit dim -- both models here
    # store observed as shape (n_units, n_times), so PyMC's default
    # dim naming gives obs_dim_0=units, obs_dim_1=time; sum over the
    # SECOND obs dimension explicitly rather than assuming the name, in
    # case a future model changes variable naming.
    obs_dims = [d for d in ll.dims if d not in ("chain", "draw")]
    if len(obs_dims) != 2:
        raise ValueError(
            f"loo_by_unit expects a 2D observed variable (n_units, n_times), "
            f"got obs dims {obs_dims} -- update this function if the model's "
            f"observation structure has changed"
        )
    unit_dim, time_dim = obs_dims
    ll_by_unit = ll.sum(dim=time_dim)
    n_units = ll_by_unit.sizes[unit_dim]

    trace_reduced = copy.deepcopy(trace)
    trace_reduced.log_likelihood = xr.Dataset({obs_var_name: ll_by_unit})

    loo_result = az.loo(trace_reduced, var_name=obs_var_name, pointwise=True)

    pareto_k = np.asarray(loo_result.pareto_k)
    n_bad = int(np.sum(pareto_k > 0.7))
    if n_bad > 0:
        import warnings
        warnings.warn(
            f"loo_by_unit: {n_bad}/{n_units} units have Pareto-k > 0.7 -- "
            f"PSIS-LOO's importance-sampling approximation is unreliable for "
            f"these units (max k = {pareto_k.max():.2f}). With only {n_units} "
            f"units this is a realistic outcome, not necessarily a bug -- but "
            f"don't trust elpd_loo at face value here without either (a) more "
            f"units, (b) exact refitting for the flagged units, or (c) treating "
            f"this as evidence the comparison itself needs more data to resolve.",
            stacklevel=2,
        )

    # Hand-rolled WAIC as an independent cross-check (az.waic unavailable in
    # arviz 1.3.0). Standard formula (Vehtari et al. 2017, Eq 11-12; Gelman
    # et al. 2014): elpd_waic_i = log(mean_s exp(ll_i,s)) - var_s(ll_i,s),
    # computed per unit i (already summed over time above) then totalled.
    ll_vals = ll_by_unit.stack(sample=("chain", "draw")).values  # (n_units, n_samples)
    log_mean_exp = np.array([_log_mean_exp(row) for row in ll_vals])
    p_waic_i = np.var(ll_vals, axis=1, ddof=1)
    elpd_waic_i = log_mean_exp - p_waic_i

    return ByUnitModelFit(
        elpd_loo=float(loo_result.elpd),
        elpd_loo_se=float(loo_result.se),
        p_loo=float(loo_result.p),
        max_pareto_k=float(pareto_k.max()) if n_units > 0 else float("nan"),
        n_bad_pareto_k=n_bad,
        n_units=n_units,
        elpd_waic=float(elpd_waic_i.sum()),
        p_waic=float(p_waic_i.sum()),
    )


def _log_mean_exp(log_vals: np.ndarray) -> float:
    """Numerically stable log(mean(exp(log_vals)))."""
    m = np.max(log_vals)
    return m + np.log(np.mean(np.exp(log_vals - m)))


# ============================================================================
# Point-estimate proxy -- for FAST LOCAL DEBUGGING of the sweep harness only.
# Never the basis for a reported scientific claim; see module docstring.
# NUTS + loo_by_unit above is the real thing, meant to run at ETH scale.
# ============================================================================

def held_out_gaussian_loglik_by_unit(y_true_by_unit, y_pred_by_unit, noise_var: float) -> float:
    """
    Cheap proxy for elpd_loo, usable with a point estimate (no posterior
    at all): plug in a single fitted noise_var and score held-out data
    under a homoskedastic Gaussian likelihood, summed over time within
    each unit then across units -- same "unit is the resampling axis"
    logic as loo_by_unit, just without a posterior to integrate over.
    Use this ONLY to validate that the sweep harness itself (data
    generation, fitting dispatch, aggregation, plotting) is bug-free
    with cheap/fast fits and small settings, before running the real
    NUTS-based sweep. A model "winning" here is not a publishable
    result by itself.
    """
    total = 0.0
    for y_true_i, y_pred_i in zip(y_true_by_unit, y_pred_by_unit):
        resid = y_true_i - y_pred_i
        n = len(resid)
        total += -0.5 * n * np.log(2 * np.pi * noise_var) - 0.5 * np.sum(resid**2) / noise_var
    return total


# ============================================================================
# Covariance-compensation diagnostic
# ============================================================================

def covariance_curve(kernel_type: str, ell: float, sigma: float, taus: np.ndarray) -> np.ndarray:
    """
    K(tau, 0) as a function of tau -- the 1-D "how correlated is lag
    tau with lag 0" curve implied by a fitted or true kernel. Two
    kernels with very different (ell, sigma) can still imply nearly
    the same curve at the SPECIFIC lags present in the data (the
    ell/sigma/noise trade-off the person flagged) -- comparing the
    curves directly, not just the raw hyperparameters, is how you catch
    that a fitted model "succeeded" by disguising itself as another
    kernel rather than by genuinely being distinguishable from it.
    """
    from utils.kernels import Matern52, SquaredExponential
    kernel_cls = {"matern52": Matern52, "squared_exponential": SquaredExponential}[kernel_type]
    kernel = kernel_cls(output_scale=sigma, length_scale=ell)
    return kernel(taus, np.array([0.0])).flatten()


def compensation_score(true_curve: np.ndarray, fit_curve: np.ndarray) -> float:
    """
    Normalised RMSE between two covariance curves -- low score despite
    very different (ell, sigma) between generator and fitter is exactly
    the "successfully disguised" signature: the fitted kernel reproduces
    the observed covariance structure well even though its own
    hyperparameters don't match the true generative ones. Report this
    ALONGSIDE hyperparameter recovery, never as a replacement for it.
    """
    scale = np.std(true_curve) if np.std(true_curve) > 1e-10 else 1.0
    return float(np.sqrt(np.mean((true_curve - fit_curve) ** 2)) / scale)


# ============================================================================
# Self-tests
# ============================================================================

def _self_test() -> None:
    import pymc as pm

    rng = np.random.default_rng(0)
    n_units, n_times = 8, 30
    # Real hierarchical pooling (mu_i ~ Normal(population_mu, tau)), NOT
    # independent per-unit means -- matters for this self-test specifically:
    # with independent means and no pooling, leave-one-unit-out removes ALL
    # information about that unit (the LOO posterior reverts to the raw
    # prior), which is a much more extreme perturbation than this project's
    # actual hierarchical models ever produce and manufactures pathological
    # Pareto-k warnings that have nothing to do with loo_by_unit's own
    # correctness. Pooling here keeps the self-test representative.
    population_mu = 0.5
    true_tau = 0.4
    true_unit_mu = population_mu + rng.normal(0, true_tau, size=n_units)

    with pm.Model() as model:
        pop_mu = pm.Normal("pop_mu", 0, 1)
        tau = pm.HalfCauchy("tau", beta=1.0)
        raw = pm.Normal("raw", 0, 1, shape=(n_units,))
        unit_mu = pop_mu + tau * raw
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)
        y_obs = true_unit_mu[:, None] + rng.normal(0, 0.3, size=(n_units, n_times))
        pm.Normal("obs", mu=unit_mu[:, None] * np.ones((n_units, n_times)), sigma=sigma_obs, observed=y_obs)
        trace = pm.sample(draws=500, tune=500, chains=4, cores=1, progressbar=False, random_seed=0)
        pm.compute_log_likelihood(trace, model=model)

    result = loo_by_unit(trace)
    print(f"model_selection.py: loo_by_unit -> elpd_loo={result.elpd_loo:.2f} "
          f"(se={result.elpd_loo_se:.2f}), p_loo={result.p_loo:.2f}, "
          f"max_pareto_k={result.max_pareto_k:.2f}, n_bad={result.n_bad_pareto_k}, "
          f"n_units={result.n_units}")
    assert result.n_units == n_units, f"expected {n_units} units in LOO output, got {result.n_units}"
    assert result.elpd_waic is not None and np.isfinite(result.elpd_waic)
    # WAIC and LOO should roughly agree for a simple, well-behaved model like this one
    assert abs(result.elpd_loo - result.elpd_waic) < 0.1 * abs(result.elpd_loo), (
        f"elpd_loo ({result.elpd_loo:.2f}) and hand-rolled elpd_waic "
        f"({result.elpd_waic:.2f}) disagree by more than 10%% -- check the "
        f"hand-rolled WAIC formula"
    )
    print("model_selection.py: LOO/WAIC cross-check agreement passed.")

    # --- covariance-compensation diagnostic sanity check ---
    taus = np.linspace(0, 90, 10)
    true_curve = covariance_curve("matern52", ell=30.0, sigma=1.5, taus=taus)
    same_curve = covariance_curve("matern52", ell=30.0, sigma=1.5, taus=taus)
    different_curve = covariance_curve("matern52", ell=5.0, sigma=1.5, taus=taus)
    assert compensation_score(true_curve, same_curve) < 1e-8
    assert compensation_score(true_curve, different_curve) > 0.1
    print("model_selection.py: compensation_score sanity check passed "
          f"(identical params -> {compensation_score(true_curve, same_curve):.4f}, "
          f"different ell -> {compensation_score(true_curve, different_curve):.4f}).")

    print("model_selection.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()