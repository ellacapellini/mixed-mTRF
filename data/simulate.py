"""
generates a synthetic LFP-style recording from a known ground-truth
TRF, following the generative model in Section 5
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from basis.basis_functions import RaisedCosineBasis
from utils.design_matrix import build_design_matrix, expand_word_level_to_samples, DesignMatrix
from ground_truth import GroundTruthEffects


@dataclass
class SimulatedRecording:
    """
    1 simulated recording (single patient/electrode) + everything needed to fit a model on it and check recovery afterwards.
    """

    r: np.ndarray                 # (n_times,) the "observed" LFP
    design: DesignMatrix          # S0, S1, X used to generate it
    surp_at_t: np.ndarray         # (n_times,) sample-level surprisal
    mu_true: np.ndarray           # (n_basis,) ground truth used (incl. random effects)
    beta_true: np.ndarray         # (n_basis,)
    sigma_noise: float
    dt: float
    basis: RaisedCosineBasis


def make_word_stream(
    n_times: int,
    dt: float,
    words_per_second: float,
    rng: np.random.Generator,
    surprisal_dist: str = "exponential",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate a plausible word-onset / surprisal stream
    """
    total_time = n_times * dt
    mean_isi = 1000.0 / words_per_second  # ms between words
    onsets = [0.0]
    while onsets[-1] < total_time - mean_isi:
        isi = rng.gamma(shape=4.0, scale=mean_isi / 4.0)  # jittered, always positive
        onsets.append(onsets[-1] + isi)
    onsets = np.array(onsets[:-1])  #drop last (may exceed recording)

    if surprisal_dist == "exponential":
        surp = rng.exponential(scale=2.5, size=onsets.shape[0])
    else:
        raise ValueError(f"unknown surprisal_dist {surprisal_dist!r}")

    return onsets, surp


def simulate_recording(
    basis: RaisedCosineBasis,
    ground_truth: GroundTruthEffects,
    n_times: int,
    dt: float,
    rng: np.random.Generator,
    patient: int = 0,
    electrode: int = 0,
    words_per_second: float = 3.0,
    snr_target: float = 3.0,
    noise_type: str = "white",
) -> SimulatedRecording:
    """
    Simulate 1 recording for a given patient/electrode.
    """
    stimulus = rng.normal(0, 1, size=n_times)  # e.g. speech envelope proxy
    word_onsets, word_surp = make_word_stream(n_times, dt, words_per_second, rng)
    surp_at_t = expand_word_level_to_samples(n_times, dt, word_onsets, word_surp)

    design = build_design_matrix(stimulus, surp_at_t, basis, dt)

    mu_eff, beta_eff = ground_truth.weight_for(patient=patient, electrode=electrode)
    signal = design.S0 @ mu_eff + design.S1 @ beta_eff

    signal_sd = signal.std()

    if noise_type == "white":
        sigma_noise = signal_sd / snr_target if signal_sd > 1e-10 else 0.1
        noise = rng.normal(0, sigma_noise, size=n_times)
    elif noise_type == "pink":
        from neurodsp.sim import sim_powerlaw
        fs = 1000.0 / dt  # dt is in ms, sim_powerlaw wants Hz
        raw_pink = sim_powerlaw(
            n_seconds=n_times / fs, fs=fs, exponent=-1.0,
            **{"seed": int(rng.integers(0, 2**31 - 1))},
        )
        raw_pink = raw_pink[:n_times]  # guard off-by-one from rounding
        # sim_powerlaw returns unit-ish variance; rescale to hit the same
        # target SNR convention as the white-noise branch.
        target_sigma = signal_sd / snr_target if signal_sd > 1e-10 else 0.1
        noise = raw_pink * (target_sigma / raw_pink.std())
        sigma_noise = float(noise.std())
    else:
        raise ValueError(f"unknown noise_type {noise_type!r}, expected 'white' or 'pink'")

    r = signal + noise

    return SimulatedRecording(
        r=r,
        design=design,
        surp_at_t=surp_at_t,
        mu_true=mu_eff,
        beta_true=beta_eff,
        sigma_noise=sigma_noise,
        dt=dt,
        basis=basis,
    )


# Self-tests

def _self_test() -> None:
    from basis.basis_functions import make_raised_cosine_basis
    from ground_truth import make_stage1_static

    rng = np.random.default_rng(2)
    basis = make_raised_cosine_basis(n_basis=8, tau_max=400.0, c=5.0)
    gt = make_stage1_static(basis, rng)

    rec = simulate_recording(
        basis, gt, n_times=3000, dt=5.0, rng=rng, snr_target=5.0
    )

    # 1. Shapes.
    assert rec.r.shape == (3000,)
    assert rec.design.S0.shape == (3000, 8)

    # 2. Achieved SNR should be in the right ballpark
    signal = rec.design.S0 @ rec.mu_true + rec.design.S1 @ rec.beta_true
    achieved_snr = signal.std() / rec.sigma_noise
    assert 3.0 < achieved_snr < 7.0, f"achieved SNR {achieved_snr} far from target"

    # 3. Stage-1 ground truth has beta=0, so S1 term should contribute nothing to the signal regardless of S1's values.
    assert np.allclose(rec.beta_true, 0.0)
    contribution_from_S1 = rec.design.S1 @ rec.beta_true
    assert np.allclose(contribution_from_S1, 0.0)

    # 4. r should not = signal exactly (noise was actually added).
    assert not np.allclose(rec.r, signal)

    print("simulate.py: all self-tests passed.")
    print(f"  achieved SNR: {achieved_snr:.2f} (target 5.0)")
    print(f"  sigma_noise: {rec.sigma_noise:.4f}")


if __name__ == "__main__":
    _self_test()