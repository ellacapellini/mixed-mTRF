"""
generates known ground-truth TRF parameters (mu_j, beta_j, and
optionally patient/electrode random effects) for the four-stage
synthetic validation plan in Objective 1:

  stage 1: static TRF recovery            -> beta_j = 0, no random effects
  stage 2: amplitude-only modulation      -> beta_j = k * mu_j (negative
                                              control: same shape at every
                                              surprisal level, only scaled)
  stage 3: full amplitude-latency-scale   -> beta_j independent per basis,
            modulation (positive control)    reshapes the TRF, not just scales
  stage 4: multi-patient/electrode        -> stage 3 + random intercepts/
            hierarchical recovery            slopes per patient & electrode
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from basis.basis_functions import RaisedCosineBasis


@dataclass
class GroundTruthEffects:
    """
    ground-truth parameters for one synthetic dataset.
    """

    mu: np.ndarray
    beta: np.ndarray
    u_intercept: np.ndarray | None = None
    u_slope: np.ndarray | None = None
    v_intercept: np.ndarray | None = None
    v_slope: np.ndarray | None = None

    def weight_for(self, patient: int = 0, electrode: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        return the effective (mu_eff, beta_eff) for a given
        patient/electrode
        """
        mu_eff = self.mu.copy()
        beta_eff = self.beta.copy()
        if self.u_intercept is not None:
            mu_eff = mu_eff + self.u_intercept[patient]
        if self.u_slope is not None:
            beta_eff = beta_eff + self.u_slope[patient]
        if self.v_intercept is not None:
            mu_eff = mu_eff + self.v_intercept[patient, electrode]
        if self.v_slope is not None:
            beta_eff = beta_eff + self.v_slope[patient, electrode]
        return mu_eff, beta_eff


def make_stage1_static(basis: RaisedCosineBasis, rng: np.random.Generator) -> GroundTruthEffects:
    """
    Stage 1
    """
    n_basis = basis.n_basis
    mu = np.zeros(n_basis)
    #put weight on early-to-mid basis f with alternating sign, loosely mimicking an N1-P2-like waveform shape.
    peak_idx = n_basis // 3
    mu[peak_idx] = -2.0
    mu[peak_idx + 1] = 1.2
    mu += rng.normal(0, 0.05, size=n_basis)  # small jitter, not pure delta
    beta = np.zeros(n_basis)
    return GroundTruthEffects(mu=mu, beta=beta)


def make_stage2_amplitude_only(
    basis: RaisedCosineBasis, rng: np.random.Generator, gain: float = -0.3
) -> GroundTruthEffects:
    """
    Stage 2
    """
    base = make_stage1_static(basis, rng)
    beta = gain * base.mu
    return GroundTruthEffects(mu=base.mu, beta=beta)


def make_stage3_full_modulation(
    basis: RaisedCosineBasis, rng: np.random.Generator
) -> GroundTruthEffects:
    """
    Stage 3
    """
    base = make_stage1_static(basis, rng)
    n_basis = basis.n_basis
    peak_idx = n_basis // 3
    beta = np.zeros(n_basis)
    #neg beta at the early peak (higher surprisal -> smaller early bump), positive beta at later basis functions (higher surprisal ->response recruits later/broader basis functions -> looks later and broader in h(tau)).
    beta[peak_idx] = 0.6          # ≈ cancels the early bump
    beta[peak_idx + 1] = -0.3
    if peak_idx + 3 < n_basis:
        beta[peak_idx + 3] = -1.1  #recruits a later basis f
    return GroundTruthEffects(mu=base.mu, beta=beta)


def add_hierarchical_random_effects(
    base: GroundTruthEffects,
    n_patients: int,
    n_electrodes: int,
    rng: np.random.Generator,
    patient_intercept_sd: float = 0.3,
    patient_slope_sd: float = 0.15,
    electrode_intercept_sd: float = 0.15,
    electrode_slope_sd: float = 0.08,
) -> GroundTruthEffects:
    """
    Stage 4
    """
    n_basis = base.mu.shape[0]
    u_intercept = rng.normal(0, patient_intercept_sd, size=(n_patients, n_basis))
    u_slope = rng.normal(0, patient_slope_sd, size=(n_patients, n_basis))
    v_intercept = rng.normal(
        0, electrode_intercept_sd, size=(n_patients, n_electrodes, n_basis)
    )
    v_slope = rng.normal(0, electrode_slope_sd, size=(n_patients, n_electrodes, n_basis))

    return GroundTruthEffects(
        mu=base.mu,
        beta=base.beta,
        u_intercept=u_intercept,
        u_slope=u_slope,
        v_intercept=v_intercept,
        v_slope=v_slope,
    )


# Self-tests

def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis

    rng = np.random.default_rng(1)
    basis = make_raised_cosine_basis(n_basis=10, tau_max=800.0, c=5.0)

    # Stage 1: beta must be = zero
    gt1 = make_stage1_static(basis, rng)
    assert np.allclose(gt1.beta, 0.0), "stage 1 must have beta = 0"
    assert gt1.mu.shape == (10,)
    print("ground_truth.py: stage 1 (static) test passed.")

    # Stage 2: beta must be proportional to mu (amplitude-only).
    gt2 = make_stage2_amplitude_only(basis, rng, gain=-0.3)
    ratio = gt2.beta / (gt2.mu + 1e-12)
    nonzero = np.abs(gt2.mu) > 1e-6
    assert np.allclose(ratio[nonzero], -0.3, atol=1e-8), (
        "stage 2 beta must be a scalar multiple of mu (amplitude-only control)"
    )
    print("ground_truth.py: stage 2 (amplitude-only) test passed.")

    # Stage 3: beta must NOT be proportional to mu (this is the whole point it should be a genuinely ≠ *shape*, not just a rescaled mu).
    gt3 = make_stage3_full_modulation(basis, rng)
    nonzero3 = np.abs(gt3.mu) > 1e-6
    ratios3 = gt3.beta[nonzero3] / gt3.mu[nonzero3]
    assert ratios3.std() > 1e-3, (
        "stage 3 beta should NOT be a constant multiple of mu -- "
        "otherwise it's identical to stage 2 and doesn't test latency/scale"
    )
    print("ground_truth.py: stage 3 (full modulation, shape genuinely differs) test passed.")

    # Stage 4: random effects have correct shapes and roughly correct scale.
    gt4 = add_hierarchical_random_effects(gt3, n_patients=5, n_electrodes=8, rng=rng)
    assert gt4.u_intercept.shape == (5, 10)
    assert gt4.v_intercept.shape == (5, 8, 10)
    mu_eff, beta_eff = gt4.weight_for(patient=2, electrode=3)
    assert mu_eff.shape == (10,) and beta_eff.shape == (10,)
    # effective weight should ≠ from the pure fixed effect (random
    # effects are actually being applied, not silently ignored).
    assert not np.allclose(mu_eff, gt4.mu), "random intercepts not being applied"
    print("ground_truth.py: stage 4 (hierarchical random effects) test passed.")

    print("ground_truth.py: all self-tests passed.")


if __name__ == "__main__":
    _self_test()