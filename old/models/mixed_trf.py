"""
Ridge-regression point estimate of the FIXED-EFFECT part of the model
(mu_j, beta_j), no hierarchy, no priors. 
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Ridge

from utils.design_matrix import DesignMatrix


@dataclass
class RidgeFitResult:
    mu_hat: np.ndarray      #(n_basis,) estimated fixed-effect intercept weights
    beta_hat: np.ndarray    # n_basis,) estimated fixed-effect surprisal slopes
    alpha: float            # ridge regularisation strength used
    r_train: float
    r_test: float


def fit_ridge_mixed_trf(
    design_train: DesignMatrix,
    r_train: np.ndarray,
    design_test: DesignMatrix,
    r_test: np.ndarray,
    n_basis: int,
    alpha: float = 1.0,
) -> RidgeFitResult:
    """
    Fit ridge regression on [S0 | S1] 
    """
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(design_train.X, r_train)

    coefs = model.coef_
    mu_hat = coefs[:n_basis]
    beta_hat = coefs[n_basis:]

    pred_train = model.predict(design_train.X)
    pred_test = model.predict(design_test.X)

    from metrics import pearson_r

    return RidgeFitResult(
        mu_hat=mu_hat,
        beta_hat=beta_hat,
        alpha=alpha,
        r_train=pearson_r(r_train, pred_train),
        r_test=pearson_r(r_test, pred_test),
    )


# Self-tests
def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis
    from ground_truth import make_stage1_static, make_stage2_amplitude_only
    from data.simulate import simulate_recording
    from metrics import train_test_split_contiguous, tf_parameter_recovery_error

    rng = np.random.default_rng(42)
    basis = make_raised_cosine_basis(n_basis=8, tau_max=400.0, c=5.0)
    dt = 5.0
    n_times = 20000  #long recording, high SNR -> should recover cleanly

    # 1 check: static TRF, beta=0, > SNR
    gt1 = make_stage1_static(basis, rng)
    rec1 = simulate_recording(basis, gt1, n_times, dt, rng, snr_target=8.0)

    train_idx, test_idx = train_test_split_contiguous(n_times, test_fraction=0.2)

    d_train = DesignMatrix(
        S0=rec1.design.S0[train_idx], S1=rec1.design.S1[train_idx],
        X=rec1.design.X[train_idx],
    )
    d_test = DesignMatrix(
        S0=rec1.design.S0[test_idx], S1=rec1.design.S1[test_idx],
        X=rec1.design.X[test_idx],
    )
    r_train, r_test = rec1.r[train_idx], rec1.r[test_idx]

    fit1 = fit_ridge_mixed_trf(d_train, r_train, d_test, r_test, n_basis=8, alpha=1.0)
    mu_err = tf_parameter_recovery_error(gt1.mu, fit1.mu_hat)
    beta_err = tf_parameter_recovery_error(gt1.beta, fit1.beta_hat)

    print(f"Stage 1 (static): held-out r = {fit1.r_test:.3f}")
    print(f"  mu recovery: normalised RMSE = {mu_err['normalised_rmse']:.3f}, "
          f"corr = {mu_err['correlation']:.3f}")
    print(f"  beta (true=0) recovery: raw RMSE = {beta_err['rmse']:.4f}")

    assert fit1.r_test > 0.7, f"stage 1 held-out r too low: {fit1.r_test}"
    assert mu_err["normalised_rmse"] < 0.3, f"mu recovery poor: {mu_err}"
    assert beta_err["rmse"] < 0.3, f"beta should be near zero: {beta_err}"

    #2 check: amplitude-only modulation 
    gt2 = make_stage2_amplitude_only(basis, rng, gain=-0.3)
    rec2 = simulate_recording(basis, gt2, n_times, dt, rng, snr_target=8.0)
    d2_train = DesignMatrix(
        S0=rec2.design.S0[train_idx], S1=rec2.design.S1[train_idx],
        X=rec2.design.X[train_idx],
    )
    d2_test = DesignMatrix(
        S0=rec2.design.S0[test_idx], S1=rec2.design.S1[test_idx],
        X=rec2.design.X[test_idx],
    )
    r2_train, r2_test = rec2.r[train_idx], rec2.r[test_idx]
    fit2 = fit_ridge_mixed_trf(d2_train, r2_train, d2_test, r2_test, n_basis=8, alpha=1.0)
    mu_err2 = tf_parameter_recovery_error(gt2.mu, fit2.mu_hat)
    beta_err2 = tf_parameter_recovery_error(gt2.beta, fit2.beta_hat)

    print(f"Stage 2 (amplitude-only): held-out r = {fit2.r_test:.3f}")
    print(f"  mu recovery: normalised RMSE = {mu_err2['normalised_rmse']:.3f}")
    print(f"  beta recovery: normalised RMSE = {beta_err2['normalised_rmse']:.3f}, "
          f"corr = {beta_err2['correlation']:.3f}")

    assert fit2.r_test > 0.7, f"stage 2 held-out r too low: {fit2.r_test}"
    assert mu_err2["normalised_rmse"] < 0.3
    assert beta_err2["normalised_rmse"] < 0.4, f"beta recovery poor: {beta_err2}"

    print("mixed_trf.py: all self-tests passed (stages 1-2 recovered by ridge baseline).")


if __name__ == "__main__":
    _self_test()