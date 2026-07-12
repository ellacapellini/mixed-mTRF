"""
Post-hoc amplitude / latency / temporal-scale decomposition from
Section 6, computed from an already-fitted basis-weight TRF
h(tau | surp). 
"""

from __future__ import annotations

import numpy as np
from basis.basis_functions import RaisedCosineBasis


def trf_at_surprisal(
    mu: np.ndarray, beta: np.ndarray, basis: RaisedCosineBasis, surp: float, tau: np.ndarray
) -> np.ndarray:
    """
    Evaluate h(tau | surp) = sum_j [mu_j + beta_j * surp] * phi_j(tau)
    at given surp level, over grid of lags.
    """
    Phi = basis.eval(tau)  # (n_lags, n_basis)
    w = mu + beta * surp   # (n_basis,)
    return Phi @ w          # (n_lags,)


def amplitude_latency_scale(
    mu: np.ndarray,
    beta: np.ndarray,
    basis: RaisedCosineBasis,
    surp: float,
    tau: np.ndarray,
) -> dict:
    """
    Compute A(surp), mu_lat(surp), sigma_scale(surp) as defined in
    Section 6, restricted to the causal window tau in (0, tau_max].
    """
    h = trf_at_surprisal(mu, beta, basis, surp, tau)
    causal = tau > 0
    tau_c, h_c = tau[causal], h[causal]

    abs_h = np.abs(h_c)
    peak_idx = int(np.argmax(abs_h))
    A = float(abs_h[peak_idx])
    mu_lat = float(tau_c[peak_idx])

    denom = np.sum(h_c ** 2)
    if denom < 1e-12:
        sigma_scale = float("nan")
    else:
        sigma_scale = float(np.sqrt(np.sum((tau_c - mu_lat) ** 2 * h_c ** 2) / denom))

    return {"amplitude": A, "latency": mu_lat, "scale": sigma_scale}


def surprisal_sweep(
    mu: np.ndarray,
    beta: np.ndarray,
    basis: RaisedCosineBasis,
    surp_levels: np.ndarray,
    tau: np.ndarray,
    min_relative_amplitude: float = 0.05,
) -> dict:
    """
    Compute A/latency/scale across a range of surprisal levels, for
    plotting against Lalor's Figure 3-style summary and for the
    stage-2-vs-stage-3 diagnostic: does amplitude alone change (gain
    control) or do latency/scale change too (predictive coding)?
    """
    A_vals, lat_vals, scale_vals = [], [], []
    for s in surp_levels:
        out = amplitude_latency_scale(mu, beta, basis, s, tau)
        A_vals.append(out["amplitude"])
        lat_vals.append(out["latency"])
        scale_vals.append(out["scale"])

    A_vals = np.asarray(A_vals)
    lat_vals = np.asarray(lat_vals)
    scale_vals = np.asarray(scale_vals)

    peak_A = A_vals.max()
    if peak_A > 1e-12:
        unreliable = A_vals < (min_relative_amplitude * peak_A)
        lat_vals = np.where(unreliable, np.nan, lat_vals)
        scale_vals = np.where(unreliable, np.nan, scale_vals)

    return {
        "surp_levels": np.asarray(surp_levels),
        "amplitude": A_vals,
        "latency": lat_vals,
        "scale": scale_vals,
    }


# Self-tests
def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis
    from ground_truth import make_stage2_amplitude_only, make_stage3_full_modulation

    rng = np.random.default_rng(7)
    basis = make_raised_cosine_basis(n_basis=8, tau_max=400.0, c=5.0)
    tau = np.linspace(0, 400, 801)
    surp_levels = np.array([0.0, 2.0, 4.0, 6.0])

    # 2 (amplitude-only): latency & scale should stay ~constant 
    gt2 = make_stage2_amplitude_only(basis, rng, gain=-0.3)
    sweep2 = surprisal_sweep(gt2.mu, gt2.beta, basis, surp_levels, tau)

    lat_range2 = sweep2["latency"].max() - sweep2["latency"].min()
    scale_range2 = np.nanmax(sweep2["scale"]) - np.nanmin(sweep2["scale"])
    amp_range2 = sweep2["amplitude"].max() - sweep2["amplitude"].min()

    assert amp_range2 > 0.1, "stage 2 should show a real amplitude change"
    assert lat_range2 < 1e-6, (
        f"stage 2 (amplitude-only) should NOT shift latency, got range {lat_range2}"
    )
    print(f"derived_quantities.py: stage 2 -- amplitude range {amp_range2:.3f}, "
          f"latency range {lat_range2:.6f} (expect ~0), "
          f"scale range {scale_range2:.6f} (expect ~0)")

    # Regression test for the zero-crossing numerical artifact: sweep a
    # DENSE grid across the region where the stage-2 gain factor
    # (1 + gain*surp) crosses zero (gain=-0.3 -> crossing at surp=1/0.3),
    # using slightly-perturbed (not exact) weights to simulate the kind
    # of residual estimation error a real fit has. 
    from data.simulate import simulate_recording
    from utils.design_matrix import DesignMatrix
    from models.mixed_trf import fit_ridge_mixed_trf
    from metrics import train_test_split_contiguous

    rng3 = np.random.default_rng(123)
    basis10 = make_raised_cosine_basis(n_basis=10, tau_max=600.0, c=5.0)
    tau10 = np.linspace(0, 600, 1201)
    gt2_real = make_stage2_amplitude_only(basis10, rng3, gain=-0.3)
    rec = simulate_recording(basis10, gt2_real, n_times=24000, dt=5.0, rng=rng3, snr_target=6.0)
    train_idx, test_idx = train_test_split_contiguous(24000, test_fraction=0.2)
    d_train = DesignMatrix(S0=rec.design.S0[train_idx], S1=rec.design.S1[train_idx], X=rec.design.X[train_idx])
    d_test = DesignMatrix(S0=rec.design.S0[test_idx], S1=rec.design.S1[test_idx], X=rec.design.X[test_idx])
    fit = fit_ridge_mixed_trf(d_train, rec.r[train_idx], d_test, rec.r[test_idx], n_basis=10, alpha=1.0)

    dense_surp = np.sort(np.concatenate([np.linspace(0.0, 6.0, 13), np.linspace(3.0, 3.6, 61)]))
    dense_sweep = surprisal_sweep(fit.mu_hat, fit.beta_hat, basis10, dense_surp, tau10)

    n_nan = np.isnan(dense_sweep["scale"]).sum()
    assert n_nan > 0, (
        "expected the amplitude zero-crossing to trigger the unreliable-region "
        "guard, but no points were masked -- guard may not be working"
    )
    valid_scale = dense_sweep["scale"][~np.isnan(dense_sweep["scale"])]
    assert np.nanmax(valid_scale) < 100, (
        f"masking guard let a blown-up scale value through: max={np.nanmax(valid_scale):.1f} "
        "(expected values near true scale ~22, not >100)"
    )
    print(f"derived_quantities.py: zero-crossing guard masked {n_nan}/{len(dense_surp)} "
          f"points near the amplitude sign-flip; remaining max scale = {np.nanmax(valid_scale):.2f} "
          "(no spike leaking through)")

    #3 (full modulation): latency should ACTUALLY shift 
    gt3 = make_stage3_full_modulation(basis, rng)
    sweep3 = surprisal_sweep(gt3.mu, gt3.beta, basis, surp_levels, tau)
    lat_range3 = sweep3["latency"].max() - sweep3["latency"].min()

    assert lat_range3 > 1.0, (
        f"stage 3 (full modulation) should shift latency measurably, got {lat_range3}"
    )
    print(f"derived_quantities.py: stage 3 -- latency range {lat_range3:.2f} ms "
          f"(correctly detected as nonzero, unlike stage 2)")

    print("derived_quantities.py: all self-tests passed "
          "(correctly distinguishes gain-control from predictive-coding ground truth).")


if __name__ == "__main__":
    _self_test()