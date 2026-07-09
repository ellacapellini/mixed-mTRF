import numpy as np

def simulate_trf_data(
    n_seconds=120,
    sr=100,
    tmin=-0.1,
    tmax=0.8,
    noise_snr=2.0,
    n_features=1,
    seed=42
):
    """ Simulate synthetic stimulus + EEG data for TRF validation"""
    rng = np.random.default_rng(seed)
    n_times = n_seconds * sr
    lags = np.arange(int(tmin * sr), int(tmax * sr) + 1)
    lag_times = lags / sr
    n_lags = len(lags)

    feature_names = [f'feature_{i}' for i in range(n_features)]
    stimulus_features = {}
    true_trfs = {}
    eeg_clean = np.zeros(n_times)

    for name in feature_names:
        #sim sparse stim: random events at ~2Hz
        # (≈ word onsets continuous speech)
        stim = np.zeros(n_times)
        event_times = rng.choice(n_times, size=int(2 * n_seconds), replace=False)
        #random amplit (≈ surprisal — not all words equally surprising)
        stim[event_times] = rng.exponential(scale=1.0, size=len(event_times))
        stimulus_features[name] = stim
        #gorund truth TRF: N1 at ~100ms + P2 at ~200ms + N400 at ~400ms
        true_trf = _make_erp_trf(lag_times)
        true_trfs[name] = true_trf
        #convolve stim with TRF to get noise-free EEG contribution
        #use only posit lags for convolution (causal)
        from numpy import convolve
        conv = np.convolve(stim, true_trf, mode='full')[:n_times]
        eeg_clean += conv

    # + G noise scaled by SNR
    signal_std = np.std(eeg_clean)
    noise_std = signal_std / noise_snr
    noise = rng.normal(0, noise_std, size=n_times)
    eeg = eeg_clean + noise

    return stimulus_features, eeg, lags, true_trfs, lag_times


def _make_erp_trf(lag_times):
    """Construct a realistic ERP-like ground truth TRF.
    Components:
      - N1:  neg peak at 100ms, width ~40ms
      - P2:  pos peak at 200ms, width ~50ms  
      - N400: neg peak at 400ms, width ~100ms (smaller)
    All modelled as G for simplicity.
    """
    trf = np.zeros(len(lag_times))

    def gaussian(t, center, width, amplitude):
        return amplitude * np.exp(-0.5 * ((t - center) / width) ** 2)

    trf += gaussian(lag_times, center=0.10, width=0.04, amplitude=-1.5)  # N1
    trf += gaussian(lag_times, center=0.20, width=0.05, amplitude=1.0)   # P2
    trf += gaussian(lag_times, center=0.40, width=0.10, amplitude=-0.5)  # N400

    return trf