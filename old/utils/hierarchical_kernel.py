"""
hierarchical_kernel.py
=======================

The layer that makes kernel choice (Matern 5/2 vs squared-exponential)
and surprisal placement (B1 vs B2) orthogonal configuration axes
instead of four separate model implementations -- see equations doc
Section 7.3-7.4 and the meeting-prep notes on the extension plan.

WHAT this module does: composes a patient/electrode/(optionally
surprisal-)specific pair of kernel hyperparameters (length_scale,
output_scale) from a global value plus hierarchical deviations, on
the log scale (since both must stay strictly positive):

    log ell_{p,e}(surp) = log ell_0 + a_p^ell + b_{e(p)}^ell [+ delta * surp]
    log sigma_{p,e}(surp) = log sigma_0 + a_p^sigma + b_{e(p)}^sigma [+ gamma * surp]

The bracketed surprisal term is the ENTIRE difference between B1 and
B2 (equations doc Section 7.4): omit it (delta=None) for B1, include
it for B2. Everything else -- the hierarchy, the kernel formula, the
covariance construction -- is identical between the two, which is the
whole point of writing this as one function rather than duplicating
model code per B-variant.

WHY log scale, not raw scale: length_scale and output_scale must be
strictly positive. Composing on the raw scale (ell_0 + a_p + b_e) can
produce a negative lengthscale for a large enough negative deviation,
which is nonsensical and would break the kernel functions in
kernels.py silently (they don't validate their inputs). Composing on
the log scale and exponentiating at the end makes a non-positive
result structurally impossible, not just unlikely -- Eq. 15-16 in the
equations doc are written on the log scale for exactly this reason.

HOW this fits the pipeline: ground_truth_gp.py calls this (via
effective_covariance) to build the covariance matrix used to draw
synthetic h(0)/h(1) ground truth. bayesian_gp_trf.py will need its
own PyTensor-flavoured version of the SAME formula for the actual
PyMC model (see the module-level note below on why that can't just
import this file directly).

IMPORTANT LIMITATION, flagged deliberately rather than discovered
later: this module is plain NumPy, used for concrete numeric
ground-truth generation. Inside an actual PyMC model
(bayesian_gp_trf.py), the patient/electrode deviations are PyTensor
random variables, and np.exp does not work on those -- pytensor.tensor.exp
(or pm.math.exp) is needed instead. So bayesian_gp_trf.py will contain
a SEPARATE implementation of this exact same log-scale composition
formula using PyTensor ops, not an import of this file. The two
implementations computing the same formula in two different tensor
backends is an intentional, documented parallel, but it is a real risk
of silent drift if one gets edited without the other. A parity test
comparing the two on matched concrete inputs (numpy vs. pytensor's
eval()) should be added once bayesian_gp_trf.py exists, to catch that
drift automatically rather than relying on remembering to update both.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from utils.kernels import Matern52, SquaredExponential, WhiteNoise, AdditiveKernel

# Numerical floor for length_scale/output_scale -- see effective_hyperparams
# docstring. 1e-6 ms is far below any physiologically meaningful lengthscale
# (the whole model operates on a 0-800ms lag grid), so this floor only ever
# engages in genuinely pathological corners of parameter space, not in the
# normal operating range.
MIN_SCALE = 1e-6


@dataclass
class HierarchicalKernelParams:
    """
    All hyperparameters needed to compose an effective (length_scale,
    output_scale) pair for one component (h^(0) or h^(1)), one
    patient, one electrode, and -- for B2 only -- one surprisal value.

    Global values and deviations are all stored on the LOG scale
    already (i.e. log_ell_0 is log(ell_0), not ell_0 itself) so that
    composition is a plain sum -- see module docstring for why.

    delta, gamma: surprisal coefficients on the log-lengthscale and
    log-output-scale respectively. None means B1 (surprisal does not
    touch the kernel hyperparameters at all). A float means B2.
    """

    log_ell_0: float
    log_sigma_0: float
    a_p_ell: dict          # {patient_id: deviation}
    a_p_sigma: dict
    b_e_ell: dict           # {(patient_id, electrode_id): deviation}
    b_e_sigma: dict
    delta: float | None = None   # B1 if None, B2 if set
    gamma: float | None = None


def effective_hyperparams(
    params: HierarchicalKernelParams,
    patient: int,
    electrode: int,
    surp: float = 0.0,
) -> tuple[float, float]:
    """
    Compute the effective (length_scale, output_scale) for one
    patient/electrode/surprisal value, per Eq. 15-16 of the equations
    doc. Returns values on the ORIGINAL (positive) scale -- exponentiates
    internally, callers never see the log-scale intermediates.

    Numerical floor: composing on the log scale guarantees positivity
    in exact arithmetic (exp of any finite real is positive), but not
    in float64 -- a sufficiently extreme combination of deviations can
    underflow exp() to exactly 0.0, which would then hit a division by
    length_scale inside kernels.py's Matern52/SquaredExponential and
    produce NaN/Inf. Clipping to a small positive floor (MIN_SCALE)
    turns a silent NaN-propagation failure, likely surfacing confusingly
    deep inside a NUTS run, into "an extremely small but well-behaved
    lengthscale," which is both numerically safe and the mathematically
    honest limit of what an extreme deviation should mean anyway.
    """
    log_ell = params.log_ell_0 + params.a_p_ell.get(patient, 0.0) + params.b_e_ell.get((patient, electrode), 0.0)
    log_sigma = params.log_sigma_0 + params.a_p_sigma.get(patient, 0.0) + params.b_e_sigma.get((patient, electrode), 0.0)

    if params.delta is not None:
        log_ell = log_ell + params.delta * surp
    if params.gamma is not None:
        log_sigma = log_sigma + params.gamma * surp

    ell = max(float(np.exp(log_ell)), MIN_SCALE)
    sigma = max(float(np.exp(log_sigma)), MIN_SCALE)
    return ell, sigma


def build_kernel(kernel_type: str, length_scale: float, output_scale: float):
    """
    Factory dispatching a kernel_type string to the right class in
    kernels.py. Centralising this in one place means kernel choice is
    genuinely a config string threaded through the pipeline, not a
    scattering of if/else branches in every module that needs a kernel.

    Raises ValueError on an unrecognised kernel_type rather than
    silently defaulting to something -- a typo in a config dict should
    fail loudly at the point of the typo, not quietly fit the wrong
    model and produce a confusing result three steps later.
    """
    if kernel_type == "matern52":
        return Matern52(output_scale=output_scale, length_scale=length_scale)
    elif kernel_type == "squared_exponential":
        return SquaredExponential(output_scale=output_scale, length_scale=length_scale)
    else:
        raise ValueError(
            f"unknown kernel_type {kernel_type!r}, expected 'matern52' or 'squared_exponential'"
        )


def effective_covariance(
    params: HierarchicalKernelParams,
    patient: int,
    electrode: int,
    surp: float,
    taus: np.ndarray,
    kernel_type: str,
    nugget: float = 1e-6,
) -> np.ndarray:
    """
    End-to-end: hierarchical hyperparameters -> kernel -> covariance
    matrix on the given lag grid, with a small WhiteNoise nugget added
    for numerical stability ahead of Cholesky decomposition (same
    pattern as kernels.py's AdditiveKernel self-test).

    This is the single function ground_truth_gp.py calls to draw
    synthetic h(0)/h(1); bayesian_gp_trf.py's PyMC model will build
    the equivalent covariance matrix per patient/electrode using its
    own PyTensor version of effective_hyperparams, then call the same
    kernel functions from kernels.py (which are plain NumPy-style
    functions that work fine given already-concrete ell/sigma values
    at the point they're called -- only the hyperparameter COMPOSITION
    step needs a PyTensor-aware version, not the kernel formulas
    themselves).
    """
    ell, sigma = effective_hyperparams(params, patient, electrode, surp)
    base_kernel = build_kernel(kernel_type, length_scale=ell, output_scale=sigma)
    combined = AdditiveKernel(base_kernel, WhiteNoise(output_scale=nugget))
    return combined(taus, taus)


# --------------------------------------------------------------------------
# Self-tests. Run this file directly: `python -m utils.hierarchical_kernel`
# --------------------------------------------------------------------------
def _self_test() -> None:
    from utils.kernels import assert_valid_covariance

    # --- Test 1: no deviations, no surprisal -> exactly recovers globals ---
    params_flat = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={}, a_p_sigma={}, b_e_ell={}, b_e_sigma={},
    )
    ell, sigma = effective_hyperparams(params_flat, patient=0, electrode=0, surp=0.0)
    assert np.isclose(ell, 30.0), f"expected ell=30.0 with no deviations, got {ell}"
    assert np.isclose(sigma, 2.0), f"expected sigma=2.0 with no deviations, got {sigma}"
    print("hierarchical_kernel.py: no-deviation baseline recovers global hyperparameters exactly.")

    # --- Test 2: patient deviation composes correctly on the log scale -----
    params_dev = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={5: np.log(2.0)},  # patient 5's lengthscale should DOUBLE
        a_p_sigma={}, b_e_ell={}, b_e_sigma={},
    )
    ell_p5, _ = effective_hyperparams(params_dev, patient=5, electrode=0, surp=0.0)
    ell_other, _ = effective_hyperparams(params_dev, patient=99, electrode=0, surp=0.0)
    assert np.isclose(ell_p5, 60.0), f"expected patient 5's ell to double to 60.0, got {ell_p5}"
    assert np.isclose(ell_other, 30.0), f"expected an untouched patient to get the global 30.0, got {ell_other}"
    print("hierarchical_kernel.py: patient-level log-scale deviation composes correctly "
          f"(patient 5: ell={ell_p5}, unaffected patient: ell={ell_other}).")

    # --- Test 3: electrode deviation nests correctly under patient ---------
    params_nested = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={5: np.log(2.0)},
        a_p_sigma={}, b_e_ell={(5, 3): np.log(1.5)},  # electrode 3 of patient 5 only
        b_e_sigma={},
    )
    ell_p5_e3, _ = effective_hyperparams(params_nested, patient=5, electrode=3, surp=0.0)
    ell_p5_e0, _ = effective_hyperparams(params_nested, patient=5, electrode=0, surp=0.0)
    assert np.isclose(ell_p5_e3, 30.0 * 2.0 * 1.5), (
        f"expected patient 5 electrode 3 to compose both deviations, got {ell_p5_e3}"
    )
    assert np.isclose(ell_p5_e0, 60.0), (
        f"expected patient 5's OTHER electrodes to only see the patient-level deviation, got {ell_p5_e0}"
    )
    print(f"hierarchical_kernel.py: electrode deviation correctly nests under patient "
          f"(p5/e3={ell_p5_e3}, p5/other-electrode={ell_p5_e0}).")

    # --- Test 4: B1 vs B2 -- the actual thing this module exists to test ---
    params_b1 = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={}, a_p_sigma={}, b_e_ell={}, b_e_sigma={},
        delta=None,  # B1: surprisal does not touch the kernel
    )
    ell_low_surp, _ = effective_hyperparams(params_b1, 0, 0, surp=0.0)
    ell_high_surp, _ = effective_hyperparams(params_b1, 0, 0, surp=6.0)
    assert np.isclose(ell_low_surp, ell_high_surp), (
        "B1 (delta=None) should be completely insensitive to surprisal, "
        f"got ell={ell_low_surp} at surp=0 vs ell={ell_high_surp} at surp=6"
    )
    print("hierarchical_kernel.py: B1 (delta=None) correctly ignores surprisal entirely.")

    params_b2 = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={}, a_p_sigma={}, b_e_ell={}, b_e_sigma={},
        delta=-0.15,  # B2: negative delta -> lengthscale SHRINKS as surprisal rises
                      # (the Yu & Dayan / Mathys et al. direction: high surprisal ->
                      # faster adaptation -> shorter effective memory)
    )
    ell_b2_low, _ = effective_hyperparams(params_b2, 0, 0, surp=0.0)
    ell_b2_high, _ = effective_hyperparams(params_b2, 0, 0, surp=6.0)
    assert ell_b2_high < ell_b2_low, (
        f"B2 with negative delta should shrink ell as surprisal rises, "
        f"got ell={ell_b2_low} at surp=0 vs ell={ell_b2_high} at surp=6 (should be smaller)"
    )
    print(f"hierarchical_kernel.py: B2 (delta=-0.15) correctly shrinks lengthscale as "
          f"surprisal rises (surp=0: ell={ell_b2_low:.2f}, surp=6: ell={ell_b2_high:.2f}) "
          "-- matches the neuromodulation-literature direction cited in the equations doc.")

    # --- Test 5a: positivity holds under REALISTIC large deviations --------
    # "Realistic" here means large relative to the weakly-informative
    # Half-Cauchy(0,1) priors on the variance components in the equations
    # doc (Section 3) -- a log-scale deviation of a few units is already a
    # long way into the tail of that prior, not something NUTS would
    # sample in ordinary operation.
    #
    # NOTE on a sizing mistake caught while writing this test: my first
    # attempt used a_p_ell=-5.0, delta=-2.0, surp=6.0, which composes to
    # log(30) - 5 - 12 = -13.6, i.e. ell = 1.24e-6 -- only 1.24x above the
    # MIN_SCALE floor (1e-6), not "comfortably above" as I'd first claimed
    # in the print statement. That's a real, useful thing to have found:
    # log-scale deviations compose ADDITIVELY and a moderate-looking patient
    # deviation plus a moderate delta*surp term can land much closer to the
    # numerical floor than either term alone suggests. Worth carrying into
    # prior design for the real PyMC model later -- delta's own prior needs
    # to be tight enough, or surp's range understood well enough, that the
    # composed lengthscale doesn't routinely wander near MIN_SCALE during
    # ordinary sampling; this floor is a safety net for pathological
    # corners, not a substitute for sensible priors reaching those corners
    # rarely in the first place.
    params_large = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={0: -3.0}, a_p_sigma={0: -3.0}, b_e_ell={}, b_e_sigma={},
        delta=-0.3,
    )
    ell_large, sigma_large = effective_hyperparams(params_large, 0, 0, surp=6.0)
    assert ell_large > 100 * MIN_SCALE, (
        f"expected a realistic-extreme ell well clear of the floor (>100x), got {ell_large} "
        f"({ell_large / MIN_SCALE:.1f}x the floor)"
    )
    assert sigma_large > 100 * MIN_SCALE, f"expected a realistic-extreme sigma well clear of the floor, got {sigma_large}"
    print(f"hierarchical_kernel.py: positivity holds under realistic large deviations "
          f"(ell={ell_large:.4f}, {ell_large/MIN_SCALE:.0f}x the floor; "
          f"sigma={sigma_large:.4f}, {sigma_large/MIN_SCALE:.0f}x the floor).")

    # --- Test 5b: the underflow floor itself, deliberately engaged ---------
    # This is the genuinely pathological case (not something a real prior
    # would produce) that first caught the float64 underflow issue: without
    # the MIN_SCALE clip in effective_hyperparams, this would silently
    # return exactly 0.0, which then produces NaN/Inf the moment a kernel
    # divides by length_scale. Confirms the floor actually engages, and
    # confirms a kernel built from the floored value stays well-behaved
    # rather than merely "not crashing on this one assertion."
    params_pathological = HierarchicalKernelParams(
        log_ell_0=np.log(30.0), log_sigma_0=np.log(2.0),
        a_p_ell={0: -50.0}, a_p_sigma={0: -50.0}, b_e_ell={}, b_e_sigma={},
        delta=-100.0,
    )
    ell_floor, sigma_floor = effective_hyperparams(params_pathological, 0, 0, surp=10.0)
    assert ell_floor == MIN_SCALE, f"expected the floor to engage exactly at MIN_SCALE, got {ell_floor}"
    assert sigma_floor == MIN_SCALE, f"expected the floor to engage exactly at MIN_SCALE, got {sigma_floor}"
    # and confirm a kernel built from the floored value doesn't itself blow up
    K_floored = Matern52(output_scale=sigma_floor, length_scale=ell_floor)(
        np.array([0.0, 1.0]), np.array([0.0, 1.0])
    )
    assert np.all(np.isfinite(K_floored)), "kernel built from the floored hyperparameters produced non-finite values"
    print("hierarchical_kernel.py: underflow floor engages correctly under pathological input "
          "(would have underflowed to exactly 0.0 without it), and the resulting kernel stays finite.")

    # --- Test 6: build_kernel dispatches correctly, fails loudly on typos --
    k_matern = build_kernel("matern52", length_scale=30.0, output_scale=2.0)
    k_se = build_kernel("squared_exponential", length_scale=30.0, output_scale=2.0)
    assert isinstance(k_matern, Matern52), "build_kernel('matern52') returned the wrong type"
    assert isinstance(k_se, SquaredExponential), "build_kernel('squared_exponential') returned the wrong type"
    try:
        build_kernel("maternn52", 30.0, 2.0)  # deliberate typo
        raise AssertionError("build_kernel should have raised ValueError on an unknown kernel_type")
    except ValueError:
        pass  # expected
    print("hierarchical_kernel.py: build_kernel dispatches correctly and rejects unknown kernel_type.")

    # --- Test 7: effective_covariance produces a valid covariance matrix ---
    # for both kernel types, confirming the full pipeline (not just the
    # hyperparameter arithmetic) holds together.
    taus = np.linspace(0, 200, 101)
    for kernel_type in ["matern52", "squared_exponential"]:
        K = effective_covariance(params_dev, patient=5, electrode=0, surp=2.0, taus=taus, kernel_type=kernel_type)
        assert_valid_covariance(K, f"effective_covariance({kernel_type})")
    print("hierarchical_kernel.py: effective_covariance produces a valid covariance matrix "
          "for both kernel types.")

    print("hierarchical_kernel.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()
