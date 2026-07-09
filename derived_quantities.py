"""
Post-hoc amplitude / latency / temporal-scale decomposition from
Section 6, computed from an already-fitted basis-weight TRF
h(tau | surp). 
"""

from __future__ import annotations

import numpy as np
from basis_functions import RaisedCosineBasis


def trf_at_surprisal(
    mu: np.ndarray, beta: np.ndarray, basis: RaisedCosineBasis, surp: float, tau: np.ndarray
) -> np.ndarray:
    """
    Evaluate h(tau | surp) = sum_j [mu_j + beta_j * surp] * phi_j(tau)
    at a given surprisal level, over a grid of lags.
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
    return {
        "surp_levels": np.asarray(surp_levels),
        "amplitude": np.asarray(A_vals),
        "latency": np.asarray(lat_vals),
        "scale": np.asarray(scale_vals),
    }


# Self-tests
def _self_test() -> None:
    from basis_functions import make_raised_cosine_basis
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