"""
ground_truth_gp.py
===================

GP-sampled synthetic ground truth for Model B, mirroring ground_truth.py's
role for Model A: generate a "true" h(tau|surp) I already know the answer
for, so I can check whether a fitted model recovers it. Same four-stage
philosophy (static -> amplitude-only -> full modulation -> multi-patient),
same reasoning for why the stages matter (equations doc Section 1's
validation plan) -- just h(tau|surp) = h0(tau) + h1(tau)*surp is now built
from GP draws (equations doc Section 7.3) instead of a basis-weight sum.

A REAL CONCEPTUAL ≠ FROM ground_truth.py, in Model A,
patient/electrode ≠ are a deterministic offset on top of a
shared basis-weight vector (mu + u_p), so two patients with u_p=0 get
IDENTICAL ground truth. In Model B, h0/h1 are draws from a distribution
over functions -- so even two patients with IDENTICAL kernel
hyperparameters (no hierarchical deviation at all) still get DIFFERENT
sampled shapes, because a GP is a distribution, not a single function
plus a knob. This means "no hierarchy" doesn't mean "identical ground
truth across units" the way it did for Model A -- it means "same
DISTRIBUTION governing each unit's own independent draw." I think this
is the honest behaviour to have, not a bug: it's exactly what a GP
prior over the TRF *means*.

WHAT'S DELIBERATELY NOT HERE YET: B2 ground truth (kernel hyperparam
genuinely varying with surprisal, equations doc Section 7.4, Eq 17-18).
The equations doc gives the hyperparam equations for B2 but doesn't
yet fully re-derive how that feeds back into the h0 + h1*surp generative
form Eq 15 relies on -- multiplying a FIXED h1 draw by surp doesn't
obviously square with h1's own kernel changing at every surp value. I
don't want to guess at a generative structure the doc itself hasn't
pinned down, so this file only implements B1-style ground truth (kernel
hyperparameters fixed w.r.t. surp) for now. Building B2 ground truth
properly is follow-up work once that generative question is actually
resolved, not something to paper over here.

Depends on: utils/kernels.py, utils/hierarchical_kernel.py
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field

from utils.hierarchical_kernel import HierarchicalKernelParams, effective_covariance


@dataclass
class GroundTruthGP:
    """
    One synthetic Model-B dataset's worth of ground truth.

    h1_mode controls how h1 relates to h0, matching the four stages:
      "zero"          -- h1 is exactly the zero vector (stage 1: no
                          surprisal modulation at all).
      "proportional"  -- h1 = h1_gain * h0, deterministically tied to
                          whatever h0 was drawn (stage 2: amplitude-only
                          / negative control -- the whole shape scales
                          by one factor at every lag, exactly mirroring
                          Model A's make_stage2_amplitude_only).
      "independent"   -- h1 is its own independent GP draw, from its
                          own (typically shorter-lengthscale) kernel
                          (stage 3: full modulation / positive control
                          -- genuinely different shape and timescale
                          from h0, not just a rescaled copy).

    Samples are drawn lazily and CACHED per (patient, electrode) the
    first time shape_for() is called for that unit, then reused on
    every subsequent call -- this is what makes it "ground truth"
    rather than fresh noise every time something asks for it. Without
    this cache, a train/test split of the same synthetic recording
    would be comparing against two different "true" answers, which
    would make validation meaningless.
    """

    taus: np.ndarray
    kernel_type: str
    params_h0: HierarchicalKernelParams
    params_h1: HierarchicalKernelParams | None
    h1_mode: str  # "zero" | "proportional" | "independent"
    h1_gain: float | None = None
    _h0_cache: dict = field(default_factory=dict)
    _h1_cache: dict = field(default_factory=dict)

    def shape_for(
        self, patient: int, electrode: int, rng: np.random.Generator, surp: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (h0, h1) for this patient/electrode -- drawn once, cached
        forever after for this GroundTruthGP instance. `rng` is only
        consumed the first time a given (patient, electrode) is
        requested; subsequent calls ignore it and return the cached draw.

        `surp` : the surprisal value h1's covariance is conditioned on
        when params_h1.delta/gamma are set (B2 ground truth -- kernel
        hyperparameters genuinely shift with surprisal, equations doc
        Section 7.4). When params_h1.delta is None (B1), effective_
        hyperparams ignores surp entirely (see hierarchical_kernel.py),
        so passing a value here is always safe and never changes B1
        behaviour -- one code path serves both, same as everywhere else
        B1/B2 is handled as one orthogonal axis rather than a fork.
        Matches bayesian_gp_trf.py's own B2 simplification (Design
        decision #2 in that file's docstring): one representative
        surprisal value per unit, not a continuously time-varying kernel.
        """
        key = (patient, electrode)

        if key not in self._h0_cache:
            K0 = effective_covariance(
                self.params_h0, patient, electrode, surp=0.0, taus=self.taus, kernel_type=self.kernel_type
            )
            self._h0_cache[key] = rng.multivariate_normal(np.zeros(len(self.taus)), K0)

        h0 = self._h0_cache[key]

        if key not in self._h1_cache:
            if self.h1_mode == "zero":
                h1 = np.zeros(len(self.taus))
            elif self.h1_mode == "proportional":
                h1 = self.h1_gain * h0
            elif self.h1_mode == "independent":
                K1 = effective_covariance(
                    self.params_h1, patient, electrode, surp=surp, taus=self.taus, kernel_type=self.kernel_type
                )
                h1 = rng.multivariate_normal(np.zeros(len(self.taus)), K1)
            else:
                raise ValueError(f"unknown h1_mode {self.h1_mode!r}")
            self._h1_cache[key] = h1

        return h0, self._h1_cache[key]


def _flat_params(ell0: float, sigma0: float) -> HierarchicalKernelParams:
    """No patient/electrode deviations, no surprisal-in-kernel term --
    just a global (ell0, sigma0) pair. Shared helper since stages 1-3
    all start from this before stage 4 adds deviations on top."""
    return HierarchicalKernelParams(
        log_ell_0=np.log(ell0), log_sigma_0=np.log(sigma0),
        a_p_ell={}, a_p_sigma={}, b_e_ell={}, b_e_sigma={},
    )


def make_stage1_static(
    kernel_type: str, taus: np.ndarray, ell0: float = 100.0, sigma0: float = 2.0
) -> GroundTruthGP:
    """Stage 1: static TRF, no surprisal modulation at all -- h1 exactly
    zero, matching ground_truth.py's make_stage1_static exactly in spirit."""
    return GroundTruthGP(
        taus=taus, kernel_type=kernel_type,
        params_h0=_flat_params(ell0, sigma0), params_h1=None, h1_mode="zero",
    )


def make_stage2_amplitude_only(
    kernel_type: str, taus: np.ndarray, ell0: float = 100.0, sigma0: float = 2.0, gain: float = -0.3
) -> GroundTruthGP:
    """Stage 2: amplitude-only modulation, the negative control. h1 is
    deterministically proportional to h0 -- the whole TRF scales by
    (1 + gain*surp) at every lag simultaneously, shape frozen. This is
    the GP-ground-truth analogue of Model A's beta_j = gain*mu_j, and
    of A-B1's kappa constraint (equations doc Section 2.4) -- same
    underlying "gain control" structure, expressed for whichever model
    family is being validated."""
    return GroundTruthGP(
        taus=taus, kernel_type=kernel_type,
        params_h0=_flat_params(ell0, sigma0), params_h1=None,
        h1_mode="proportional", h1_gain=gain,
    )


def make_stage3_full_modulation(
    kernel_type: str, taus: np.ndarray,
    ell0: float = 100.0, sigma0: float = 2.0,
    ell1: float = 30.0, sigma1: float = 1.5,
    delta: float | None = None, gamma: float | None = None,
) -> GroundTruthGP:
    """Stage 3: full modulation, the positive control. h1 is drawn
    independently from its OWN kernel, deliberately given a shorter
    lengthscale than h0's (30ms vs 100ms by default) -- genuinely
    different timescale, not just a rescaled copy, mirroring how
    Model A's stage 3 puts beta_j weight on different basis functions
    than mu_j to force a real shape difference rather than a trivial one.

    delta, gamma : None -> B1 ground truth (kernel hyperparameters fixed
    w.r.t. surprisal, the original behaviour of this function). Set
    either to a float -> B2 ground truth: h1's effective (length_scale,
    output_scale) genuinely shift with the unit's surprisal at draw
    time (equations doc Section 7.4, Eq 17-18), via shape_for's surp
    argument. This was the piece flagged as deliberately not built yet
    in this module's docstring -- now implemented as an optional
    extension of the existing stage-3 generator, not a separate
    generative structure, since B1 vs B2 is only ever a difference in
    whether delta/gamma are set (hierarchical_kernel.py's whole design)."""
    return GroundTruthGP(
        taus=taus, kernel_type=kernel_type,
        params_h0=_flat_params(ell0, sigma0),
        params_h1=HierarchicalKernelParams(
            log_ell_0=np.log(ell1), log_sigma_0=np.log(sigma1),
            a_p_ell={}, a_p_sigma={}, b_e_ell={}, b_e_sigma={},
            delta=delta, gamma=gamma,
        ),
        h1_mode="independent",
    )


def add_hierarchical_random_effects(
    base: GroundTruthGP,
    n_patients: int,
    n_electrodes: int,
    rng: np.random.Generator,
    patient_ell_sd: float = 0.3,
    patient_sigma_sd: float = 0.2,
    electrode_ell_sd: float = 0.15,
    electrode_sigma_sd: float = 0.1,
) -> GroundTruthGP:
    """
    Stage 4: attach patient- and electrode-level deviations to the
    KERNEL HYPERPARAMETERS (not to function values -- see the module
    docstring for why that distinction matters for Model B specifically).
    Draws one deviation per patient and per (patient, electrode) pair,
    on the log scale, matching HierarchicalKernelParams' own convention.

    Only applied to params_h0 always; only applied to params_h1 when
    base.h1_mode == "independent" -- under "zero" mode h1 stays exactly
    zero regardless of hierarchy (nothing to vary), and under
    "proportional" mode h1 = gain*h0 already inherits h0's per-unit
    variation automatically, so giving h1 its own separate hierarchy
    there would be double-counting.
    """

    def _deviate(params: HierarchicalKernelParams | None) -> HierarchicalKernelParams | None:
        if params is None:
            return None
        a_p_ell = {p: float(rng.normal(0, patient_ell_sd)) for p in range(n_patients)}
        a_p_sigma = {p: float(rng.normal(0, patient_sigma_sd)) for p in range(n_patients)}
        b_e_ell = {
            (p, e): float(rng.normal(0, electrode_ell_sd))
            for p in range(n_patients) for e in range(n_electrodes)
        }
        b_e_sigma = {
            (p, e): float(rng.normal(0, electrode_sigma_sd))
            for p in range(n_patients) for e in range(n_electrodes)
        }
        return HierarchicalKernelParams(
            log_ell_0=params.log_ell_0, log_sigma_0=params.log_sigma_0,
            a_p_ell=a_p_ell, a_p_sigma=a_p_sigma, b_e_ell=b_e_ell, b_e_sigma=b_e_sigma,
            delta=params.delta, gamma=params.gamma,
        )

    new_params_h0 = _deviate(base.params_h0)
    new_params_h1 = _deviate(base.params_h1) if base.h1_mode == "independent" else base.params_h1

    return GroundTruthGP(
        taus=base.taus, kernel_type=base.kernel_type,
        params_h0=new_params_h0, params_h1=new_params_h1,
        h1_mode=base.h1_mode, h1_gain=base.h1_gain,
    )


# --------------------------------------------------------------------------
# Self-tests. Run this file directly: `python -m ground_truth_gp`
# --------------------------------------------------------------------------
def _self_test() -> None:
    from utils.hierarchical_kernel import effective_hyperparams

    taus = np.linspace(0, 400, 81)

    for kernel_type in ["matern52", "squared_exponential"]:
        rng = np.random.default_rng(0)

        # --- Stage 1: h1 exactly zero, h0 nonzero and correct length -------
        gt1 = make_stage1_static(kernel_type, taus)
        h0_1, h1_1 = gt1.shape_for(patient=0, electrode=0, rng=rng)
        assert h0_1.shape == (81,), f"[{kernel_type}] stage1 h0 wrong shape"
        assert np.allclose(h1_1, 0.0), f"[{kernel_type}] stage1 h1 must be exactly zero"
        assert not np.allclose(h0_1, 0.0), f"[{kernel_type}] stage1 h0 should not be all zero"
        print(f"ground_truth_gp.py [{kernel_type}]: stage 1 (static) test passed.")

        # --- Stage 2: h1 exactly proportional to h0 -------------------------
        gt2 = make_stage2_amplitude_only(kernel_type, taus, gain=-0.3)
        h0_2, h1_2 = gt2.shape_for(patient=0, electrode=0, rng=rng)
        assert np.allclose(h1_2, -0.3 * h0_2), f"[{kernel_type}] stage2 h1 must equal gain*h0 exactly"
        print(f"ground_truth_gp.py [{kernel_type}]: stage 2 (amplitude-only) test passed.")

        # --- Stage 3: h1 independently drawn, genuinely different shape -----
        # from h0 -- reusing the same sample-path-roughness logic already
        # validated in kernels.py's self-test (SE vs Matern comparison),
        # since h1's shorter lengthscale should make it visibly rougher.
        gt3 = make_stage3_full_modulation(kernel_type, taus, ell0=100.0, ell1=15.0)
        h0_3, h1_3 = gt3.shape_for(patient=0, electrode=0, rng=rng)
        roughness_h0 = np.sum(np.diff(h0_3, n=2) ** 2)
        roughness_h1 = np.sum(np.diff(h1_3, n=2) ** 2)
        assert roughness_h1 > roughness_h0, (
            f"[{kernel_type}] expected h1 (shorter lengthscale, 15ms) to be rougher than "
            f"h0 (100ms) -- got roughness h0={roughness_h0:.4f}, h1={roughness_h1:.4f}"
        )
        # and confirm it's not accidentally proportional to h0 either
        ratio = h1_3 / (h0_3 + 1e-12)
        assert ratio.std() > 1e-3, f"[{kernel_type}] stage3 h1 should not be a constant multiple of h0"
        print(f"ground_truth_gp.py [{kernel_type}]: stage 3 (full modulation, genuinely "
              f"different shape) test passed -- roughness h0={roughness_h0:.4f}, h1={roughness_h1:.4f}.")

        # --- Stage 4: hierarchy actually changes effective hyperparameters -
        gt4 = add_hierarchical_random_effects(gt3, n_patients=4, n_electrodes=6, rng=rng)
        ell_p0, _ = effective_hyperparams(gt4.params_h0, patient=0, electrode=0, surp=0.0)
        ell_p1, _ = effective_hyperparams(gt4.params_h0, patient=1, electrode=0, surp=0.0)
        assert ell_p0 != ell_p1, (
            f"[{kernel_type}] expected different patients to get different effective "
            f"lengthscales under stage-4 hierarchy, got identical values"
        )
        # and confirm the actual sampled shapes differ between patients too
        h0_p0, _ = gt4.shape_for(patient=0, electrode=0, rng=rng)
        h0_p1, _ = gt4.shape_for(patient=1, electrode=0, rng=rng)
        assert not np.allclose(h0_p0, h0_p1), (
            f"[{kernel_type}] different patients should get different sampled shapes"
        )
        print(f"ground_truth_gp.py [{kernel_type}]: stage 4 (hierarchical) test passed "
              f"(patient 0 ell={ell_p0:.2f}, patient 1 ell={ell_p1:.2f}).")

        # --- Reproducibility: repeated calls return the SAME cached draw, --
        # not a fresh resample -- this is the whole point of caching, and
        # the property that makes this usable as "ground truth" at all.
        h0_again, h1_again = gt4.shape_for(patient=0, electrode=0, rng=rng)
        assert np.array_equal(h0_p0, h0_again), (
            f"[{kernel_type}] shape_for should return the cached draw on repeated calls, "
            "not resample -- otherwise this isn't a fixed ground truth"
        )
        print(f"ground_truth_gp.py [{kernel_type}]: reproducibility (cached, not resampled) test passed.")

        # --- Stage 3, B2 variant: h1's effective covariance actually --------
        # shifts with surp, and B1 (delta=None) stays exactly insensitive to
        # it -- the one thing that makes B2 ground truth genuinely different
        # ground truth, not just B1 with unused arguments plumbed through.
        gt3_b2 = make_stage3_full_modulation(
            kernel_type, taus, ell0=100.0, ell1=30.0, delta=-0.3, gamma=0.0
        )
        rng_b2 = np.random.default_rng(7)
        h0_lo, h1_lo = gt3_b2.shape_for(patient=0, electrode=0, rng=rng_b2, surp=0.0)
        rng_b2_hi = np.random.default_rng(7)  # fresh instance -> fresh cache, isolates the surp effect
        gt3_b2_hi = make_stage3_full_modulation(
            kernel_type, taus, ell0=100.0, ell1=30.0, delta=-0.3, gamma=0.0
        )
        h0_hi, h1_hi = gt3_b2_hi.shape_for(patient=0, electrode=0, rng=rng_b2_hi, surp=6.0)
        # same rng seed and same underlying standard-normal draw structure,
        # but a different covariance at high surp (delta<0 -> shorter ell1
        # -> rougher h1) should still show up as different roughness
        roughness_lo = np.sum(np.diff(h1_lo, n=2) ** 2)
        roughness_hi = np.sum(np.diff(h1_hi, n=2) ** 2)
        assert roughness_hi > roughness_lo, (
            f"[{kernel_type}] B2 ground truth: expected h1 at high surprisal (shorter "
            f"effective lengthscale, delta=-0.3) to be rougher than at surp=0, got "
            f"roughness_lo={roughness_lo:.4f}, roughness_hi={roughness_hi:.4f}"
        )
        print(f"ground_truth_gp.py [{kernel_type}]: stage 3 B2 (surprisal-in-kernel) test "
              f"passed -- roughness surp=0: {roughness_lo:.4f}, surp=6: {roughness_hi:.4f}.")

        # --- B1 sanity check: passing surp to a B1 (delta=None) generator ---
        # must be a complete no-op, confirming the unified shape_for signature
        # didn't accidentally change B1 behaviour.
        gt3_b1 = make_stage3_full_modulation(kernel_type, taus, ell0=100.0, ell1=30.0)
        rng_b1a = np.random.default_rng(3)
        _, h1_b1_surp0 = gt3_b1.shape_for(patient=0, electrode=0, rng=rng_b1a, surp=0.0)
        gt3_b1_check = make_stage3_full_modulation(kernel_type, taus, ell0=100.0, ell1=30.0)
        rng_b1b = np.random.default_rng(3)
        _, h1_b1_surp6 = gt3_b1_check.shape_for(patient=0, electrode=0, rng=rng_b1b, surp=6.0)
        assert np.array_equal(h1_b1_surp0, h1_b1_surp6), (
            f"[{kernel_type}] B1 (delta=None) ground truth must be completely insensitive "
            "to the surp argument -- passing surp=6 changed the draw"
        )
        print(f"ground_truth_gp.py [{kernel_type}]: B1 unaffected by surp argument (as expected).")

    print("ground_truth_gp.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()