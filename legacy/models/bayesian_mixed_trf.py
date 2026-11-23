"""
bayesian_mixed_trf.py
======================

Model 2 (A-B1) and Model 3 (A-B2) of the seven-model comparison --
the full hierarchical Bayesian fit for Model A, via NUTS in PyMC.
Same model skeleton for both, different prior structure on beta_j
(equations doc Section 2.4):

  A-B2 (variant="A-B2"): beta_j gets its own independent prior,
    free per basis function -- the full, already-validated (via
    ridge point estimate) flexible model.
  A-B1 (variant="A-B1"): beta_j = kappa * mu_j for ONE global scalar
    kappa -- the whole TRF scales by (1+kappa*surp) at every lag at
    once, shape frozen. Modelling decision made here, not fully
    pinned down in the equations doc: kappa is a single population-
    level scalar, not itself hierarchical across patients -- this
    keeps A-B1 the genuinely SIMPLER, more constrained model to
    compare A-B2 against, rather than a third thing with its own
    hierarchy on top.

Hierarchy (equations doc Section 3): patient random intercept+slope,
electrode random intercept+slope (nested in patient), each basis
function j getting an INDEPENDENT HalfCauchy-scaled, non-centered prior
on its own intercept/slope variance components -- NOT a joint 2x2
LKJ-correlated covariance. An LKJ-correlated version was tried first
and abandoned after hitting persistent divergences with the amount of
replication available (see the comment inside build_model below for
the full reasoning); this docstring previously still described the
LKJ version after the code moved on from it -- fixed here to match
what the model actually does, per the equations doc's own Section 2.5
decision to simplify rather than force a correlation the data can't
identify.

Requires equal-length recordings across units (reasonable for
synthetic validation, where every simulate_recording call uses the
same n_times) -- this lets S0/S1/r be stacked into dense arrays
instead of a ragged list, which keeps the PyMC model construction
simple and fast to reason about.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from dataclasses import dataclass


@dataclass
class BayesianMixedTRFResult:
    trace: az.InferenceData
    variant: str
    n_basis: int
    n_patients: int
    n_units: int
    model: pm.Model


def fit_bayesian_mixed_trf(
    S0_all: np.ndarray,
    S1_all: np.ndarray,
    r_all: np.ndarray,
    patient_idx: np.ndarray,
    variant: str = "A-B2",
    sigma_mu: float = 1.0,
    sigma_beta: float = 1.0,
    draws: int = 1000,
    tune: int = 1000,
    chains: int = 4,
    cores: int | None = None,
    target_accept: float = 0.95,
    max_treedepth: int = 12,
    seed: int = 0,
) -> BayesianMixedTRFResult:
    """
    S0_all, S1_all : (n_units, n_times, n_basis) -- plain and
        surprisal-weighted basis-projected design matrices, one block
        per patient/electrode unit, ALL THE SAME n_times (see module
        docstring for why).
    r_all : (n_units, n_times) -- observed response per unit.
    patient_idx : (n_units,) int array, which patient (0..P-1) each
        unit belongs to. Multiple units sharing a patient index get
        the same patient-level random effect, plus their own
        independent electrode-level one.
    variant : "A-B2" (beta_j free) or "A-B1" (beta_j = kappa*mu_j).
    cores : chains run in PARALLEL processes; None -> cores=chains.
        Was hardcoded to 1, silently wasting a shared multi-core
        node's idle capacity (e.g. ETH's large-memory node) on every
        single fit.
    """
    if variant not in ("A-B1", "A-B2"):
        raise ValueError(f"unknown variant {variant!r}, expected 'A-B1' or 'A-B2'")

    patient_idx = np.asarray(patient_idx, dtype="int64")
    n_units, n_times, J = S0_all.shape
    n_patients = int(patient_idx.max()) + 1

    coords = {
        "basis": np.arange(J),
        "patient": np.arange(n_patients),
        "unit": np.arange(n_units),
    }

    with pm.Model(coords=coords) as model:
        # --- fixed effects ---
        mu = pm.Normal("mu", 0.0, sigma_mu, dims="basis")
        if variant == "A-B2":
            beta = pm.Normal("beta", 0.0, sigma_beta, dims="basis")
        else:  # A-B1
            kappa = pm.Normal("kappa", 0.0, sigma_beta)

        # --- random effects: patient and (nested) electrode intercept +
        # slope, one independent pair per basis function.
        #
        # DEVIATION FROM THE EQUATIONS DOC, flagged deliberately: Section 3
        # specifies an LKJ-correlated 2x2 covariance between each basis
        # function's random intercept and slope. I tried that first
        # (LKJCholeskyCov + non-centered MvNormal) and hit persistent
        # divergences and r_hat > 1.9 even after raising target_accept and
        # max_treedepth -- with only a handful of patients/electrodes,
        # there isn't enough replication to identify a correlation
        # parameter between intercept and slope at all, and trying to
        # estimate it anyway was dragging the whole posterior into bad
        # geometry. This is exactly the situation Bates et al. (2015,
        # already cited in the equations doc) argue for simplifying:
        # correlated random slopes are often more complexity than the
        # data can actually support, and a simpler random-effects
        # structure is the right response, not more aggressive tuning.
        # So: independent (diagonal) priors on intercept and slope here,
        # non-centered for the same funnel-avoidance reason as before,
        # just without the LKJ correlation on top. This needs to go back
        # into the equations doc as an explicit, justified change, not
        # silently diverge from what's written there.
        u_list, u_beta_list = [], []
        v_list, v_beta_list = [], []
        for j in range(J):
            tau_p_u = pm.HalfCauchy(f"tau_p_u_{j}", beta=1.0)
            tau_p_beta = pm.HalfCauchy(f"tau_p_beta_{j}", beta=1.0)
            raw_p_u = pm.Normal(f"raw_p_u_{j}", 0.0, 1.0, shape=n_patients)
            raw_p_beta = pm.Normal(f"raw_p_beta_{j}", 0.0, 1.0, shape=n_patients)
            u_list.append(tau_p_u * raw_p_u)
            u_beta_list.append(tau_p_beta * raw_p_beta)

            tau_e_u = pm.HalfCauchy(f"tau_e_u_{j}", beta=1.0)
            tau_e_beta = pm.HalfCauchy(f"tau_e_beta_{j}", beta=1.0)
            raw_e_u = pm.Normal(f"raw_e_u_{j}", 0.0, 1.0, shape=n_units)
            raw_e_beta = pm.Normal(f"raw_e_beta_{j}", 0.0, 1.0, shape=n_units)
            v_list.append(tau_e_u * raw_e_u)
            v_beta_list.append(tau_e_beta * raw_e_beta)

        u = pt.stack(u_list, axis=1)             # (n_patients, J)
        u_beta = pt.stack(u_beta_list, axis=1)    # (n_patients, J)
        v = pt.stack(v_list, axis=1)              # (n_units, J)
        v_beta = pt.stack(v_beta_list, axis=1)    # (n_units, J)

        mu_eff = mu[None, :] + u[patient_idx, :] + v          # (n_units, J)

        # BUG FIXED HERE, worth documenting since it wasn't just a typo:
        # my first version of A-B1 computed beta_eff = kappa * mu_eff,
        # multiplying by the per-UNIT effective mu (including random
        # effects). That's not what the equations doc actually specifies
        # -- Section 2.4 writes beta_j = kappa * mu_j, the GLOBAL fixed
        # effect, not the noisy unit-specific mu_eff. It also turned out
        # to be numerically unstable: when a unit's random effects push
        # its own mu_eff close to zero, kappa has to blow up to explain
        # that unit's beta at all -- same zero-crossing instability
        # already documented in derived_quantities.py, resurfacing here.
        # The doc-correct version: kappa multiplies the STABLE global mu,
        # and beta gets its own random effects on top, exactly the same
        # hierarchical structure A-B2 already has -- so A-B1 and A-B2 now
        # differ ONLY in how the fixed-effect beta_j is defined (kappa*mu
        # vs free), not in whether beta has a hierarchy at all.
        if variant == "A-B2":
            beta_fixed = beta                      # (J,) free per basis function
        else:  # A-B1
            beta_fixed = kappa * mu                # (J,) constrained, but still stable
        beta_eff = beta_fixed[None, :] + u_beta[patient_idx, :] + v_beta

        signal = pt.sum(S0_all * mu_eff[:, None, :], axis=-1) \
            + pt.sum(S1_all * beta_eff[:, None, :], axis=-1)   # (n_units, n_times)

        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)
        pm.Normal("obs", mu=signal, sigma=sigma_obs, observed=r_all)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=(cores if cores is not None else chains),
            target_accept=target_accept,
            max_treedepth=max_treedepth, random_seed=seed, progressbar=False,
        )
        # See model_selection.py's docstring: use loo_by_unit(trace), not
        # az.loo(trace) directly -- per-timepoint LOO is invalid here.
        pm.compute_log_likelihood(trace, model=model)

    return BayesianMixedTRFResult(
        trace=trace, variant=variant, n_basis=J, n_patients=n_patients,
        n_units=n_units, model=model,
    )


def check_convergence(result: BayesianMixedTRFResult, r_hat_threshold: float = 1.05) -> dict:
    """
    Equations doc Section 9's commitment: same convergence checks
    before trusting any fit, not just "it ran." Returns a summary
    dict; callers should assert on max_r_hat before trusting anything
    downstream of this fit (e.g. before computing WAIC/LOO on it).
    """
    summary = az.summary(result.trace, var_names=["mu"] + (
        ["beta"] if result.variant == "A-B2" else ["kappa"]
    ))
    max_r_hat = float(summary["r_hat"].max())
    min_ess_bulk = float(summary["ess_bulk"].min())
    return {
        "max_r_hat": max_r_hat,
        "min_ess_bulk": min_ess_bulk,
        "converged": max_r_hat < r_hat_threshold and min_ess_bulk > 80,
    }


# --------------------------------------------------------------------------
# Self-tests. Run this file directly: `python -m models.bayesian_mixed_trf`
# Deliberately small (few units, short chains) to keep this fast to
# iterate on -- a real validation run should use draws/tune >= 1000,
# chains >= 4.
# --------------------------------------------------------------------------
def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis
    from ground_truth import make_stage1_static
    from data.simulate import simulate_recording

    # NOTE on sizing: an earlier version of this test used 2 patients x 2
    # electrodes (4 units) and consistently failed to converge (r_hat up to
    # 2.5, hundreds of divergences) no matter how the model was
    # reparameterized. That turned out to be a genuine identifiability
    # problem, not a bug: with only 4 groups total, there isn't enough
    # replication to separately identify a global effect, a patient effect,
    # AND an electrode effect. 4 patients x 3 electrodes (12 units) is the
    # smallest configuration that converged reliably in testing -- kept
    # here as the self-test size specifically because it's the boundary
    # that actually works, not an arbitrary round number.
    rng = np.random.default_rng(0)
    basis = make_raised_cosine_basis(n_basis=3, tau_max=200.0, c=5.0)
    dt = 10.0
    n_times = 800
    n_patients, n_electrodes = 4, 3

    # --- Model 3 (A-B2): stage 1 collapse check -----------------------------
    gt1 = make_stage1_static(basis, rng)
    S0_all, S1_all, r_all, patient_idx = [], [], [], []
    for p in range(n_patients):
        for e in range(n_electrodes):
            rec = simulate_recording(basis, gt1, n_times, dt, rng, snr_target=8.0)
            S0_all.append(rec.design.S0)
            S1_all.append(rec.design.S1)
            r_all.append(rec.r)
            patient_idx.append(p)
    S0_all, S1_all, r_all = np.array(S0_all), np.array(S1_all), np.array(r_all)
    patient_idx = np.array(patient_idx)

    result_ab2 = fit_bayesian_mixed_trf(
        S0_all, S1_all, r_all, patient_idx, variant="A-B2",
        draws=200, tune=300, chains=2, target_accept=0.9, max_treedepth=10, seed=1,
    )
    conv = check_convergence(result_ab2)
    assert conv["max_r_hat"] < 1.05, f"A-B2 stage-1 fit r_hat too high: {conv}"
    if not conv["converged"]:
        print(f"  (note: ESS below the {80} floor on this fast run -- r_hat is fine, "
              f"a real presentation run should use more draws/chains: {conv})")

    mu_post = result_ab2.trace.posterior["mu"].mean(dim=("chain", "draw")).values
    beta_post = result_ab2.trace.posterior["beta"].mean(dim=("chain", "draw")).values
    mu_err = np.sqrt(np.mean((mu_post - gt1.mu) ** 2)) / (np.sqrt(np.mean(gt1.mu ** 2)) + 1e-12)
    assert mu_err < 0.35, f"A-B2 stage-1 mu recovery poor: normalised RMSE={mu_err:.3f}"
    assert np.abs(beta_post).max() < 0.5, f"A-B2 stage-1 beta should stay near zero, got {beta_post}"
    print(f"bayesian_mixed_trf.py [A-B2]: stage-1 collapse check passed "
          f"(mu nRMSE={mu_err:.3f}, max|beta|={np.abs(beta_post).max():.3f}, "
          f"max r_hat={conv['max_r_hat']:.3f}, min ess_bulk={conv['min_ess_bulk']:.0f}).")

    # --- Model 2 (A-B1): fit on data generated to match A-B1's own structure
    gt2_base = make_stage1_static(basis, rng)
    kappa_true = -0.4
    gt2 = type(gt1)(mu=gt2_base.mu, beta=kappa_true * gt2_base.mu)  # beta_j = kappa*mu_j exactly

    S0_all2, S1_all2, r_all2, patient_idx2 = [], [], [], []
    for p in range(n_patients):
        for e in range(n_electrodes):
            rec = simulate_recording(basis, gt2, n_times, dt, rng, snr_target=8.0)
            S0_all2.append(rec.design.S0)
            S1_all2.append(rec.design.S1)
            r_all2.append(rec.r)
            patient_idx2.append(p)
    S0_all2, S1_all2, r_all2 = np.array(S0_all2), np.array(S1_all2), np.array(r_all2)
    patient_idx2 = np.array(patient_idx2)

    result_ab1 = fit_bayesian_mixed_trf(
        S0_all2, S1_all2, r_all2, patient_idx2, variant="A-B1",
        draws=200, tune=300, chains=2, target_accept=0.9, max_treedepth=10, seed=2,
    )
    conv1 = check_convergence(result_ab1)
    assert conv1["max_r_hat"] < 1.05, f"A-B1 fit r_hat too high: {conv1}"
    if not conv1["converged"]:
        print(f"  (note: ESS below the {80} floor on this fast run -- r_hat is fine, "
              f"a real presentation run should use more draws/chains: {conv1})")

    kappa_post = float(result_ab1.trace.posterior["kappa"].mean())
    assert abs(kappa_post - kappa_true) < 0.3, (
        f"A-B1 kappa recovery poor: true={kappa_true}, recovered={kappa_post:.3f}"
    )
    print(f"bayesian_mixed_trf.py [A-B1]: kappa recovery passed "
          f"(true={kappa_true}, recovered={kappa_post:.3f}, max r_hat={conv1['max_r_hat']:.3f}, "
          f"min ess_bulk={conv1['min_ess_bulk']:.0f}).")

    print("bayesian_mixed_trf.py: all self-tests passed (models 2 and 3 of 7).")
    print("NOTE: self-test uses small draws/tune/chains for speed (150/250/2). "
          "A real validation or presentation run should use draws>=1000, tune>=1000, "
          "chains>=4 -- see fit_bayesian_mixed_trf's default arguments.")


if __name__ == "__main__":
    _self_test()