"""
kernels.py
==========

Covariance kernels for Model B (the hierarchical Gaussian Process TRF,
see equations doc Section 7). Each kernel is a callable object with
signature k(taus1, taus2) -> covariance matrix of shape
(len(taus1), len(taus2)), following the same interface so kernel
choice is a swappable configuration parameter, not a code fork --
this is what lets bayesian_gp_trf.py be parameterised by
kernel_type without four separate model files (see equations doc
Section 8 / meeting-prep notes on the kernel-choice comparison).

Two substantive kernels, plus two structural helpers:

  SquaredExponential -- infinitely differentiable, unbounded memory.
    Standard default in many GP applications (including Balthasar's
    own work), but per Stein (1999) and the brain-specific evidence
    in Section 7.2 of the equations doc (Siden et al. 2021, Haak
    et al. 2018, Eklund et al. 2016), often more smoothness than
    real biological signals actually have.

  Matern52 -- finite, interpretable memory (an exact third-order
    Markov process, Hartikainen & Sarkka 2010). The working default
    per the equations doc, on the strength of that brain-specific
    evidence -- but Section 7.2 is explicit this should be settled
    by comparison (WAIC/LOO), not asserted, which is the whole
    reason both kernels need to exist side by side with a shared
    interface.

  WhiteNoise -- a diagonal nugget, used additively for numerical
    stability (a covariance matrix built from a smooth kernel alone
    can be near-singular; adding a small independent-noise term
    keeps Cholesky decomposition well-conditioned) and, at larger
    scale, as the degenerate zero-memory limit for sanity checks
    (mixed_trf.py's ridge fit is exactly a GP with a WhiteNoise
    kernel -- verified numerically elsewhere in this repo).

  AdditiveKernel -- combines two kernels by summing their covariance
    matrices (e.g. Matern52 + WhiteNoise for a smooth process plus a
    numerical-stability nugget). Sum of two valid covariance
    functions is itself a valid covariance function, so this
    composition is safe by construction.

This module deliberately has no dependency on anything else in the
project -- it's pure math, same role basis_functions.py plays for
Model A, and gets the same amount of direct testing for the same
reason: everything downstream (gp_mtrf.py, hierarchical_kernel.py,
bayesian_gp_trf.py) trusts these functions completely, so bugs here
need to be caught here, not three modules downstream in a posterior
that looks merely "a bit odd."
"""

from __future__ import annotations

import numpy as np


class SquaredExponential:
    """
    k(tau, tau') = output_scale * exp(-(tau-tau')^2 / (2*length_scale^2))

    Stationary, infinitely mean-square differentiable -- the
    nu -> infinity limit of the Matern family (Rasmussen & Williams,
    2006). No finite state-space representation exists: no finite
    window of history is ever a sufficient statistic, which is the
    concrete meaning of "unbounded memory" used in the equations doc.
    """

    def __init__(self, output_scale: float = 1.0, length_scale: float = 50.0):
        self.output_scale = output_scale
        self.length_scale = length_scale

    def __call__(self, taus1: np.ndarray, taus2: np.ndarray) -> np.ndarray:
        diff = taus1[:, None] - taus2[None, :]
        return self.output_scale * np.exp(-diff**2 / (2 * self.length_scale**2))


class Matern52:
    """
    k(tau, tau') = output_scale * (1 + sqrt(5)*r + (5/3)*r^2) * exp(-sqrt(5)*r)
    where r = |tau - tau'| / length_scale.

    Stationary, twice mean-square differentiable, with an exponential
    envelope but NOT purely exponential (that's nu=1/2, the
    Ornstein-Uhlenbeck kernel, memoryless / zero-order Markov).
    Matern 5/2 is a finite-order (third-order) Markov process -- an
    exact finite-dimensional state-space representation exists
    (Hartikainen & Sarkka, 2010), so it has a controllable, finite
    amount of memory rather than none (nu=1/2) or infinite (SE).
    """

    def __init__(self, output_scale: float = 1.0, length_scale: float = 50.0):
        self.output_scale = output_scale
        self.length_scale = length_scale

    def __call__(self, taus1: np.ndarray, taus2: np.ndarray) -> np.ndarray:
        diff = taus1[:, None] - taus2[None, :]
        r = np.abs(diff) / self.length_scale
        sqrt5_r = np.sqrt(5) * r
        return self.output_scale * (1 + sqrt5_r + (5 / 3) * r**2) * np.exp(-sqrt5_r)


class WhiteNoise:
    """
    k(tau, tau') = output_scale if tau == tau' else 0.

    A diagonal nugget. Used additively (via AdditiveKernel) on top of
    a smooth kernel for numerical stability during Cholesky
    decomposition, and on its own as the degenerate zero-memory
    baseline case (ridge regression with regularisation alpha is
    exactly a GP with a WhiteNoise kernel of matching variance --
    already verified numerically to floating-point precision
    elsewhere in this repo).
    """

    def __init__(self, output_scale: float = 1.0):
        self.output_scale = output_scale

    def __call__(self, taus1: np.ndarray, taus2: np.ndarray) -> np.ndarray:
        diff = taus1[:, None] - taus2[None, :]
        return self.output_scale * (diff == 0).astype(float)


class AdditiveKernel:
    """
    k(tau, tau') = kernel1(tau, tau') + kernel2(tau, tau').

    Sum of two valid covariance functions is itself a valid
    covariance function (sum of two PSD matrices is PSD), so this
    composition never needs its own validity proof -- only its own
    test that it actually does the addition correctly.
    """

    def __init__(self, kernel1, kernel2):
        self.kernel1 = kernel1
        self.kernel2 = kernel2

    def __call__(self, taus1: np.ndarray, taus2: np.ndarray) -> np.ndarray:
        return self.kernel1(taus1, taus2) + self.kernel2(taus1, taus2)


# --------------------------------------------------------------------------
# Self-tests. Run this file directly: `python -m utils.kernels`
# --------------------------------------------------------------------------
def assert_symmetric(K: np.ndarray, name: str, atol: float = 1e-10) -> None:
    """Public (importable) validity check: every module that builds a
    covariance matrix downstream of a kernel -- hierarchical_kernel.py,
    ground_truth_gp.py, bayesian_gp_trf.py -- reuses this rather than
    duplicating the same two checks in every file's self-test."""
    assert np.allclose(K, K.T, atol=atol), f"{name}: covariance matrix not symmetric"


def assert_psd(K: np.ndarray, name: str, atol: float = 1e-8) -> None:
    """Public (importable) validity check, see assert_symmetric docstring."""
    eigvals = np.linalg.eigvalsh(K)
    assert eigvals.min() >= -atol, (
        f"{name}: not positive semi-definite, min eigenvalue = {eigvals.min():.2e}"
    )


def assert_valid_covariance(K: np.ndarray, name: str) -> None:
    """Convenience wrapper: the two checks together, since a covariance
    matrix that's symmetric but not PSD (or vice versa) is invalid either
    way -- callers usually want both, not one or the other."""
    assert_symmetric(K, name)
    assert_psd(K, name)


def _self_test() -> None:
    taus = np.linspace(0, 200, 101)  # 0..200ms lag grid, matches typical usage

    # --- SquaredExponential ------------------------------------------------
    se = SquaredExponential(output_scale=2.0, length_scale=30.0)
    K_se = se(taus, taus)

    assert_valid_covariance(K_se, "SquaredExponential")
    # diagonal must equal output_scale exactly (k(tau,tau) = output_scale * exp(0))
    assert np.allclose(np.diag(K_se), 2.0), "SquaredExponential: diagonal != output_scale"
    # must decay monotonically away from the diagonal for a stationary kernel
    row0 = K_se[0, :]
    assert np.all(np.diff(row0) <= 1e-12), "SquaredExponential: not monotonically decaying"
    print("kernels.py: SquaredExponential -- symmetric, PSD, correct diagonal, decays monotonically.")

    # --- Matern52 ------------------------------------------------------------
    m52 = Matern52(output_scale=2.0, length_scale=30.0)
    K_m52 = m52(taus, taus)

    assert_valid_covariance(K_m52, "Matern52")
    assert np.allclose(np.diag(K_m52), 2.0), "Matern52: diagonal != output_scale"
    row0_m = K_m52[0, :]
    assert np.all(np.diff(row0_m) <= 1e-12), "Matern52: not monotonically decaying"
    print("kernels.py: Matern52 -- symmetric, PSD, correct diagonal, decays monotonically.")

    # --- THE key comparative test: SE produces smoother sample paths than
    # Matern 5/2 at matched length_scale. This is the numerical backbone of
    # the "SE is too smooth" claim (Stein 1999; Siden et al. 2021; Haak
    # et al. 2018) in the equations doc Section 7.2 -- checked here via
    # actual sample-path roughness, not asserted.
    #
    # NOTE on a mistake caught while writing this test: my first version of
    # this test compared raw correlation magnitude at a fixed lag (e.g. "does
    # SE retain more correlation than Matern5/2 at 3 lengthscales out?") and
    # got it backwards -- SE decays like exp(-r^2) (Gaussian tail), Matern5/2
    # like exp(-r) (exponential tail), and exponential tails are FATTER than
    # Gaussian tails asymptotically, so Matern5/2 actually retains MORE raw
    # correlation at long lag, not less. That's a real, correct fact about
    # these kernels -- it just isn't what "SE = unbounded memory" in the
    # equations doc refers to. The doc's claim is about the STATE-SPACE /
    # Markov-order property (no finite window of history is ever a
    # sufficient statistic for SE -- Hartikainen & Sarkka 2010), which is a
    # different property from tail-decay speed. The correct numerically
    # testable proxy for "SE is too smooth" is sample-path SMOOTHNESS
    # (SE is infinitely differentiable, Matern5/2 only twice), not tail mass.
    ell = 30.0
    fine_taus = np.linspace(0, 200, 400)  # finer grid to resolve path roughness
    se_matched = SquaredExponential(output_scale=1.0, length_scale=ell)
    m52_matched = Matern52(output_scale=1.0, length_scale=ell)

    nugget = WhiteNoise(output_scale=1e-8)  # numerical-stability jitter for Cholesky
    K_se_fine = AdditiveKernel(se_matched, nugget)(fine_taus, fine_taus)
    K_m52_fine = AdditiveKernel(m52_matched, nugget)(fine_taus, fine_taus)

    L_se = np.linalg.cholesky(K_se_fine)
    L_m52 = np.linalg.cholesky(K_m52_fine)

    # same underlying random draw fed through each kernel's Cholesky factor,
    # so any difference in path roughness is attributable to the kernel's
    # smoothness, not to different random noise realisations.
    rng = np.random.default_rng(0)
    z = rng.normal(size=fine_taus.shape[0])
    path_se = L_se @ z
    path_m52 = L_m52 @ z

    # discrete second difference as a roughness proxy: a smoother path has
    # smaller local curvature at every point.
    roughness_se = np.sum(np.diff(path_se, n=2) ** 2)
    roughness_m52 = np.sum(np.diff(path_m52, n=2) ** 2)

    assert roughness_se < roughness_m52, (
        f"expected SE sample paths to be smoother (lower roughness) than Matern 5/2 "
        f"at matched length_scale -- got SE roughness={roughness_se:.4f}, "
        f"Matern5/2 roughness={roughness_m52:.4f}"
    )
    print(f"kernels.py: sample-path roughness at matched length_scale -- "
          f"SE={roughness_se:.4f}, Matern5/2={roughness_m52:.4f} "
          f"(SE smoother, confirms the differentiability claim numerically, "
          f"NOT a claim about which kernel retains more tail correlation).")

    # --- WhiteNoise ------------------------------------------------------------
    wn = WhiteNoise(output_scale=0.5)
    K_wn = wn(taus, taus)

    assert_valid_covariance(K_wn, "WhiteNoise")
    assert np.allclose(np.diag(K_wn), 0.5), "WhiteNoise: diagonal != output_scale"
    off_diag_mask = ~np.eye(len(taus), dtype=bool)
    assert np.allclose(K_wn[off_diag_mask], 0.0), "WhiteNoise: off-diagonal entries should be zero"
    print("kernels.py: WhiteNoise -- symmetric, PSD, diagonal-only as expected.")

    # --- AdditiveKernel: smooth kernel + nugget, the pattern used to keep ---
    # Cholesky decomposition well-conditioned in bayesian_gp_trf.py later.
    combo = AdditiveKernel(Matern52(output_scale=1.0, length_scale=30.0), WhiteNoise(output_scale=1e-6))
    K_combo = combo(taus, taus)

    assert_valid_covariance(K_combo, "AdditiveKernel(Matern52, WhiteNoise)")
    # combo diagonal should be Matern52's diagonal (1.0) plus the nugget (1e-6)
    assert np.allclose(np.diag(K_combo), 1.0 + 1e-6), "AdditiveKernel: diagonal doesn't match sum"
    # off-diagonal should match Matern52 alone (WhiteNoise contributes nothing off-diagonal)
    K_m52_alone = Matern52(output_scale=1.0, length_scale=30.0)(taus, taus)
    off_diag_mask = ~np.eye(len(taus), dtype=bool)
    assert np.allclose(K_combo[off_diag_mask], K_m52_alone[off_diag_mask]), (
        "AdditiveKernel: off-diagonal should be unaffected by the WhiteNoise nugget"
    )
    # the actual practical point of the nugget: confirm it measurably improves
    # conditioning relative to the smooth kernel alone (this is *why* it's used
    # ahead of Cholesky decomposition in the hierarchical GP model).
    cond_without_nugget = np.linalg.cond(K_m52_alone)
    cond_with_nugget = np.linalg.cond(K_combo)
    assert cond_with_nugget <= cond_without_nugget, (
        "AdditiveKernel: nugget should not worsen conditioning"
    )
    print(f"kernels.py: AdditiveKernel -- correct sum, nugget improves condition number "
          f"({cond_without_nugget:.2e} -> {cond_with_nugget:.2e}).")

    # --- Cross-kernel shape/dtype sanity, cheap but catches silly bugs ------
    taus_a = np.array([0.0, 10.0, 20.0])
    taus_b = np.array([5.0, 15.0])
    for kernel, name in [
        (SquaredExponential(), "SquaredExponential"),
        (Matern52(), "Matern52"),
        (WhiteNoise(), "WhiteNoise"),
    ]:
        K_rect = kernel(taus_a, taus_b)
        assert K_rect.shape == (3, 2), f"{name}: wrong shape for non-square input, got {K_rect.shape}"
    print("kernels.py: all kernels handle non-square (taus1 != taus2) inputs with correct shape.")

    print("kernels.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()
