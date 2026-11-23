"""
generates a synthetic LFP-style recording from a known ground-truth
TRF, following the generative model in Section 5
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from basis.basis_functions import RaisedCosineBasis
from utils.design_matrix import build_design_matrix, build_raw_lag_design_matrix, expand_word_level_to_samples, DesignMatrix
from ground_truth import GroundTruthEffects
from ground_truth_gp import GroundTruthGP


@dataclass
class SimulatedRecordingGP:
    """Model B's version of SimulatedRecording -- same idea, but the
    ground truth is (h0, h1) sampled on the raw lag grid rather than
    (mu, beta) basis weights, and the design matrix is
    build_raw_lag_design_matrix's raw-lag output rather than
    build_design_matrix's basis-projected one."""

    r: np.ndarray
    design: DesignMatrix
    surp_at_t: np.ndarray
    h0_true: np.ndarray       # (n_lags,) ground truth used (incl. hierarchical deviations)
    h1_true: np.ndarray       # (n_lags,)
    sigma_noise: float
    dt: float
    taus: np.ndarray


@dataclass
class SimulatedRecording:
    """
    One simulated recording (single patient/electrode) + everything needed to fit a model on it and check recovery afterwards.
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
    Simulate one recording for a given patient/electrode.

    Parameters
    ----------
    noise_type : {"white", "pink"}
        "white": i.i.d. Gaussian observation noise (default -- this is
        what all the existing stage 1-3 validation numbers were run
        against, kept as default so those results don't silently
        change under you).
        "pink": 1/f aperiodic noise via neurodsp's sim_powerlaw
        (Cole, Donoghue, Gao & Voytek, JOSS 2019), exponent=-1. Real
        LFP/iEEG background activity is dominated by 1/f aperiodic
        activity, not white noise -- this is the more realistic
        option, worth using once you're checking robustness rather
        than just recovering ground truth under the easiest case.
        NOTE: correlated noise violates the i.i.d. assumption ridge
        regression implicitly relies on for statistical efficiency.
        Point estimates should still be consistent, but don't trust
        naive standard errors under pink noise without checking.
    """
    stimulus = rng.normal(0, 1, size=n_times)  # e.g. speech envelope proxy
    word_onsets, word_surp = make_word_stream(n_times, dt, words_per_second, rng)
    surp_at_t = expand_word_level_to_samples(n_times, dt, word_onsets, word_surp)

    design = build_design_matrix(stimulus, surp_at_t, basis, dt)

    mu_eff, beta_eff = ground_truth.weight_for(patient=patient, electrode=electrode)
    signal = design.S0 @ mu_eff + design.S1 @ beta_eff

    noise, sigma_noise = _generate_noise(signal, n_times, dt, rng, snr_target, noise_type)
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


def _generate_noise(
    signal: np.ndarray, n_times: int, dt: float, rng: np.random.Generator,
    snr_target: float, noise_type: str,
) -> tuple[np.ndarray, float]:
    """
    Shared white/pink noise generation, factored out of
    simulate_recording so simulate_recording_gp doesn't duplicate this
    logic -- Model A and Model B differ in how the SIGNAL is built
    (basis-projected vs. raw-lag), but the noise model on top of it is
    identical between them, so it should live in exactly one place.
    Returns (noise, sigma_noise).
    """
    signal_sd = signal.std()

    if noise_type == "white":
        sigma_noise = signal_sd / snr_target if signal_sd > 1e-10 else 0.1
        noise = rng.normal(0, sigma_noise, size=n_times)
    elif noise_type == "pink":
        from neurodsp.sim import sim_powerlaw
        fs = 1000.0 / dt
        raw_pink = sim_powerlaw(
            n_seconds=n_times / fs, fs=fs, exponent=-1.0,
            **{"seed": int(rng.integers(0, 2**31 - 1))},
        )
        raw_pink = raw_pink[:n_times]
        target_sigma = signal_sd / snr_target if signal_sd > 1e-10 else 0.1
        noise = raw_pink * (target_sigma / raw_pink.std())
        sigma_noise = float(noise.std())
    else:
        raise ValueError(f"unknown noise_type {noise_type!r}, expected 'white' or 'pink'")

    return noise, sigma_noise


def simulate_recording_gp(
    taus: np.ndarray,
    ground_truth: GroundTruthGP,
    n_times: int,
    dt: float,
    rng: np.random.Generator,
    patient: int = 0,
    electrode: int = 0,
    words_per_second: float = 3.0,
    snr_target: float = 3.0,
    noise_type: str = "white",
) -> SimulatedRecordingGP:
    """
    Model B's version of simulate_recording. Same stimulus/word-stream
    generation, same noise model (via _generate_noise) -- the only
    real difference is the design matrix and the signal construction:
    build_raw_lag_design_matrix instead of build_design_matrix, and
    h0/h1 (raw lag-grid values) instead of mu/beta (basis weights).
    """
    stimulus = rng.normal(0, 1, size=n_times)
    word_onsets, word_surp = make_word_stream(n_times, dt, words_per_second, rng)
    surp_at_t = expand_word_level_to_samples(n_times, dt, word_onsets, word_surp)

    design = build_raw_lag_design_matrix(stimulus, surp_at_t, taus, dt)

    h0_true, h1_true = ground_truth.shape_for(patient=patient, electrode=electrode, rng=rng)
    signal = design.S0 @ h0_true + design.S1 @ h1_true

    noise, sigma_noise = _generate_noise(signal, n_times, dt, rng, snr_target, noise_type)
    r = signal + noise

    return SimulatedRecordingGP(
        r=r,
        design=design,
        surp_at_t=surp_at_t,
        h0_true=h0_true,
        h1_true=h1_true,
        sigma_noise=sigma_noise,
        dt=dt,
        taus=taus,
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

    # --- Pink noise branch: exercised here for the first time. Confirm it
    # (a) actually runs and hits target SNR like the white-noise branch,
    # and (b) is genuinely spectrally different from white noise -- i.e.
    # actually 1/f, not accidentally a no-op that just renames white noise.
    rng_pink = np.random.default_rng(4)
    rec_pink = simulate_recording(
        basis, gt, n_times=3000, dt=5.0, rng=rng_pink, snr_target=5.0, noise_type="pink"
    )
    signal_pink = rec_pink.design.S0 @ rec_pink.mu_true + rec_pink.design.S1 @ rec_pink.beta_true
    achieved_snr_pink = signal_pink.std() / rec_pink.sigma_noise
    assert 3.0 < achieved_snr_pink < 7.0, f"pink-noise SNR {achieved_snr_pink} far from target"

    # Reconstruct the raw noise (r - signal) and check its power spectrum
    # falls off with frequency (1/f-like), unlike white noise which is flat.
    # Compare low-frequency vs high-frequency power: pink noise should have
    # substantially more power at low frequencies.
    noise_realisation = rec_pink.r - signal_pink
    fft_power = np.abs(np.fft.rfft(noise_realisation)) ** 2
    freqs = np.fft.rfftfreq(len(noise_realisation), d=5.0 / 1000.0)  # dt=5ms -> Hz
    low_band = (freqs > 1) & (freqs < 5)
    high_band = (freqs > 40) & (freqs < 80)
    low_power = fft_power[low_band].mean()
    high_power = fft_power[high_band].mean()
    ratio = low_power / high_power
    assert ratio > 3.0, (
        f"pink noise doesn't show expected 1/f power falloff: "
        f"low-freq/high-freq power ratio = {ratio:.2f} (expected >> 1)"
    )
    print(f"simulate.py: pink-noise branch passed (SNR={achieved_snr_pink:.2f}, "
          f"low/high-freq power ratio={ratio:.1f}, confirms genuine 1/f structure).")

    # --- Model B: simulate_recording_gp, exercised for the first time ------
    from ground_truth_gp import make_stage1_static as make_stage1_static_gp
    from ground_truth_gp import make_stage3_full_modulation as make_stage3_full_modulation_gp

    taus_b = np.linspace(0, 400, 81)
    gt_b1 = make_stage1_static_gp("matern52", taus_b, ell0=100.0, sigma0=2.0)
    rng_b = np.random.default_rng(5)

    rec_b1 = simulate_recording_gp(taus_b, gt_b1, n_times=3000, dt=5.0, rng=rng_b, snr_target=5.0)
    assert rec_b1.r.shape == (3000,)
    assert rec_b1.design.S0.shape == (3000, 81), f"expected raw-lag S0 with 81 columns, got {rec_b1.design.S0.shape}"
    assert np.allclose(rec_b1.h1_true, 0.0), "stage-1 GP ground truth should have h1 exactly zero"
    signal_b1 = rec_b1.design.S0 @ rec_b1.h0_true + rec_b1.design.S1 @ rec_b1.h1_true
    achieved_snr_b1 = signal_b1.std() / rec_b1.sigma_noise
    assert 3.0 < achieved_snr_b1 < 7.0, f"Model B achieved SNR {achieved_snr_b1} far from target"
    print(f"simulate.py: simulate_recording_gp (Model B, stage 1) test passed "
          f"(SNR={achieved_snr_b1:.2f}, raw-lag design matrix shape {rec_b1.design.S0.shape}).")

    # confirm the pink-noise branch also works for Model B, since it's
    # meant to be shared via _generate_noise, not reimplemented
    gt_b3 = make_stage3_full_modulation_gp("matern52", taus_b, ell0=100.0, ell1=15.0)
    rec_b3_pink = simulate_recording_gp(
        taus_b, gt_b3, n_times=3000, dt=5.0, rng=np.random.default_rng(6),
        snr_target=5.0, noise_type="pink",
    )
    signal_b3 = rec_b3_pink.design.S0 @ rec_b3_pink.h0_true + rec_b3_pink.design.S1 @ rec_b3_pink.h1_true
    achieved_snr_b3 = signal_b3.std() / rec_b3_pink.sigma_noise
    assert 3.0 < achieved_snr_b3 < 7.0, f"Model B pink-noise SNR {achieved_snr_b3} far from target"
    print(f"simulate.py: simulate_recording_gp (Model B, stage 3, pink noise) test passed "
          f"(SNR={achieved_snr_b3:.2f}) -- shared noise helper works for both model families.")

    print("simulate.py: all self-tests passed, including Model B.")


if __name__ == "__main__":
    _self_test()