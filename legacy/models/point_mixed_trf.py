"""
point_mixed_trf.py
===================

The A-B1 (restricted) point estimate: beta_j = kappa * mu_j for one
shared scalar kappa. Flagged in the equations doc (Section 2.4, "The
part I need to be careful about when I actually built this") as NOT a
flag you can add to ordinary ridge:

    r(t) = sum_j mu_j * [s0_j(t) + kappa * s1_j(t)] + eps(t)

is bilinear in (mu_j, kappa) -- both unknown at once, multiplying each
other -- and ordinary ridge only works when the thing being solved for
enters linearly. This file was a placeholder scaffold until now; there
was no A-B1 point estimate anywhere in the codebase, so any script
needing to fit A-B1 without paying for a full NUTS run (e.g. a model-
recovery / identifiability sweep needing hundreds of fits) had nothing
to call.

APPROACH: alternating least squares (Gorski et al. 2007), exactly as
specified in the doc: fix kappa, solve mu_j by ordinary ridge on the
single combined regressor [s0_j(t) + kappa*s1_j(t)]; fix mu_j, solve
kappa by a simple 1-D search; repeat until it settles. Each half-step
is an easy, convex problem; the whole thing is not jointly convex, so
this can in principle land in a local optimum -- mitigated here by
initialising kappa at 0 (the "no modulation" starting point, itself a
meaningful default rather than an arbitrary one) and iterating to a
fixed point in the training residual.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
from sklearn.linear_model import Ridge

from utils.design_matrix import DesignMatrix


@dataclass
class ALSFitResult:
    mu_hat: np.ndarray       # (n_basis,) estimated population-average weights
    kappa_hat: float         # estimated single scalar surprisal-gain
    beta_hat: np.ndarray     # (n_basis,) = kappa_hat * mu_hat, for interface
                             # parity with fit_ridge_mixed_trf's RidgeFitResult
    alpha: float
    n_iterations: int
    r_train: float
    r_test: float
    noise_var: float         # residual variance on TRAIN data, for held-out
                              # log-likelihood-based model comparison


def fit_als_mixed_trf(
    design_train: DesignMatrix,
    r_train: np.ndarray,
    design_test: DesignMatrix,
    r_test: np.ndarray,
    n_basis: int,
    alpha: float = 1.0,
    max_iterations: int = 25,
    tol: float = 1e-6,
) -> ALSFitResult:
    """
    Fit A-B1's restricted point estimate: beta_j = kappa * mu_j.

    design_train.S0, design_train.S1 : (n_times, n_basis) each, the
    plain and surprisal-weighted basis-projected regressors (same
    convention as fit_ridge_mixed_trf / design_matrix.py throughout).
    """
    S0, S1 = design_train.S0, design_train.S1

    kappa = 0.0  # start at "no modulation" -- a meaningful, not arbitrary, init
    mu = np.zeros(n_basis)
    prev_residual = np.inf
    it = 0

    for it in range(1, max_iterations + 1):
        # --- fix kappa, solve mu by ordinary ridge on the combined regressor ---
        combined = S0 + kappa * S1  # (n_times, n_basis)
        ridge = Ridge(alpha=alpha, fit_intercept=False)
        ridge.fit(combined, r_train)
        mu = ridge.coef_

        # --- fix mu, solve kappa by a simple 1-D search ---
        # r(t) - sum_j mu_j*s0_j(t) = kappa * sum_j mu_j*s1_j(t) + eps(t)
        # a 1-D regression of the mu-weighted S0 residual against the
        # mu-weighted S1 regressor -- written as a bounded 1-D search
        # (rather than the closed form) to stay easy to extend if the
        # surprisal link is later made nonlinear (equations doc Section
        # 2.3's open question) without restructuring this alternation.
        resid = r_train - S0 @ mu
        weighted_s1 = S1 @ mu

        def _sq_err(k, resid=resid, weighted_s1=weighted_s1):
            return np.sum((resid - weighted_s1 * k) ** 2)

        result = minimize_scalar(_sq_err, bounds=(-50.0, 50.0), method="bounded")
        kappa = float(result.x)

        pred_train = S0 @ mu + kappa * (S1 @ mu)
        current_residual = float(np.sum((r_train - pred_train) ** 2))
        if abs(prev_residual - current_residual) < tol * max(1.0, prev_residual):
            break
        prev_residual = current_residual

    beta_hat = kappa * mu
    pred_train = S0 @ mu + kappa * (S1 @ mu)
    pred_test = design_test.S0 @ mu + kappa * (design_test.S1 @ mu)

    from metrics import pearson_r
    noise_var = float(np.var(r_train - pred_train))

    return ALSFitResult(
        mu_hat=mu, kappa_hat=kappa, beta_hat=beta_hat, alpha=alpha,
        n_iterations=it, r_train=pearson_r(r_train, pred_train),
        r_test=pearson_r(r_test, pred_test), noise_var=noise_var,
    )


# Self-tests
def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis
    from ground_truth import make_stage2_amplitude_only, make_stage3_full_modulation
    from data.simulate import simulate_recording
    from metrics import train_test_split_contiguous, tf_parameter_recovery_error

    rng = np.random.default_rng(42)
    basis = make_raised_cosine_basis(n_basis=8, tau_max=400.0, c=5.0)
    dt = 5.0
    n_times = 20000  # long recording, high SNR -> should recover cleanly

    # --- Positive control: data IS A-B1-shaped (beta = gain*mu). ALS should
    # recover kappa close to the true gain, and mu close to the true mu. ---
    true_gain = -0.3
    gt = make_stage2_amplitude_only(basis, rng, gain=true_gain)
    rec = simulate_recording(basis, gt, n_times, dt, rng, snr_target=8.0)

    train_idx, test_idx = train_test_split_contiguous(n_times, test_fraction=0.2)
    d_train = DesignMatrix(S0=rec.design.S0[train_idx], S1=rec.design.S1[train_idx], X=rec.design.X[train_idx])
    d_test = DesignMatrix(S0=rec.design.S0[test_idx], S1=rec.design.S1[test_idx], X=rec.design.X[test_idx])
    r_train, r_test = rec.r[train_idx], rec.r[test_idx]

    fit = fit_als_mixed_trf(d_train, r_train, d_test, r_test, n_basis=8, alpha=1.0)
    mu_err = tf_parameter_recovery_error(gt.mu, fit.mu_hat)

    print(f"ALS positive control (true kappa={true_gain}): recovered kappa={fit.kappa_hat:.4f}, "
          f"held-out r={fit.r_test:.3f}, iterations={fit.n_iterations}")
    print(f"  mu recovery: normalised RMSE = {mu_err['normalised_rmse']:.3f}, corr = {mu_err['correlation']:.3f}")

    assert fit.r_test > 0.7, f"A-B1 positive control held-out r too low: {fit.r_test}"
    assert abs(fit.kappa_hat - true_gain) < 0.1, f"kappa recovery poor: got {fit.kappa_hat}, true {true_gain}"
    assert mu_err["correlation"] > 0.8, f"mu recovery poor: {mu_err}"

    # --- Negative control: data is A-B2-shaped (beta free per basis function,
    # NOT proportional to mu). A-B1 is mis-specified here by construction, so
    # it should NOT achieve the near-perfect fit A-B2 itself would -- if it
    # did, that would mean the ALS fit is silently ignoring the beta=kappa*mu
    # constraint rather than actually enforcing it. ---
    gt2 = make_stage3_full_modulation(basis, rng)
    rec2 = simulate_recording(basis, gt2, n_times, dt, rng, snr_target=8.0)
    d2_train = DesignMatrix(S0=rec2.design.S0[train_idx], S1=rec2.design.S1[train_idx], X=rec2.design.X[train_idx])
    d2_test = DesignMatrix(S0=rec2.design.S0[test_idx], S1=rec2.design.S1[test_idx], X=rec2.design.X[test_idx])
    r2_train, r2_test = rec2.r[train_idx], rec2.r[test_idx]

    fit2 = fit_als_mixed_trf(d2_train, r2_train, d2_test, r2_test, n_basis=8, alpha=1.0)
    print(f"ALS negative control (A-B2-shaped data, A-B1 is mis-specified): held-out r={fit2.r_test:.3f}")
    cos_sim = np.dot(gt2.mu, gt2.beta) / (np.linalg.norm(gt2.mu) * np.linalg.norm(gt2.beta) + 1e-12)
    assert abs(cos_sim) < 0.9, (
        f"negative control isn't discriminating: true beta is already nearly proportional "
        f"to true mu (cosine similarity={cos_sim:.3f}) by chance for this seed -- rerun "
        f"with a different seed rather than trusting this control"
    )
    assert fit2.r_test < fit.r_test, (
        f"A-B1 (restricted) should fit its OWN positive-control data at least as well as "
        f"mis-specified A-B2-shaped data, since the constraint removes real flexibility "
        f"here -- got r_test={fit2.r_test:.3f} on mis-specified data vs {fit.r_test:.3f} "
        f"on well-specified data"
    )

    print("point_mixed_trf.py: all self-tests passed (positive control recovers "
          "kappa/mu; negative control confirms the beta=kappa*mu constraint is "
          "actually being enforced, not silently ignored).")


if __name__ == "__main__":
    _self_test()