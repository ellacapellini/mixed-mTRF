"""Event streams: word-like natural timing, or an isochronous rhythm with occasional timing violations (oddball paradigm); plus 1/f noise."""
from __future__ import annotations

import numpy as np
from mixed_trf.simulation.templates import DT


def make_words(n_times, rng, rate=3.0, surp_mean=7.0, surp_sd=3.0, rho=0.0, *,
               rhythm="natural", period_ms=300.0, violation_p=0.15, jitter_ms=90.0):
    """Event onsets (samples) with a surprisal value per event.
    rhythm="natural" (default): word-like gamma-distributed inter-onset intervals and gamma-distributed surprisal (bits);
        surp_mean/sd are ASSUMPTIONS -- replace with your GPT-2 values.
    rhythm="isochronous": events on a strict grid every `period_ms`; with probability `violation_p` an event's onset is
        displaced by +/- 0.5-1.0 x `jitter_ms` (a TIMING VIOLATION). Surprisal is the ideal-observer value under the generating
        process: regular -log2(1-p) bits, violation -log2(p) bits (binary). Strict periodicity makes lagged design columns
        j and j+period identical, so the TRF is not identifiable without violations -- this scenario tests that."""
    if rhythm == "isochronous":
        period = int(round(period_ms / DT))
        n_ev = (n_times - 2) // period
        grid = np.arange(1, n_ev + 1) * period
        viol = rng.random(n_ev) < violation_p
        sign = np.where(rng.random(n_ev) < 0.5, -1, 1)
        mag = np.round((0.5 + 0.5 * rng.random(n_ev)) * jitter_ms / DT).astype(int)
        onsets_all = grid + viol * sign * mag
        keep = (onsets_all > 0) & (onsets_all < n_times - 1)
        onsets, viol = onsets_all[keep], viol[keep]
        p_ = float(violation_p)
        if p_ <= 0.0:
            surp = np.full(len(onsets), 0.0); z = np.zeros(len(onsets))
        else:
            s_reg, s_dev = -np.log2(1 - p_), -np.log2(p_)
            surp = np.where(viol, s_dev, s_reg)
            mean = (1 - p_) * s_reg + p_ * s_dev
            sd = np.sqrt(p_ * (1 - p_)) * (s_dev - s_reg)
            z = (surp - mean) / sd                       # centred/scaled by the analytic population constants
        conf = rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.normal(size=len(z))
        return {"onsets": onsets, "surp": surp, "z": z, "conf": conf}
    mean_isi = 1000.0 / rate
    isi = 150.0 + rng.gamma(4.0, (mean_isi - 150.0) / 4.0, size=int(n_times * DT / mean_isi * 2) + 10)
    onsets = np.round(np.cumsum(isi) / DT).astype(int)
    onsets = onsets[onsets < n_times - 1]
    k, scale = (surp_mean / surp_sd) ** 2, surp_sd ** 2 / surp_mean
    surp = rng.gamma(k, scale, size=len(onsets))
    z = (surp - surp_mean) / surp_sd                         # centred by a GLOBAL constant
    conf = rho * z + np.sqrt(max(1 - rho ** 2, 0.0)) * rng.normal(size=len(z))
    return {"onsets": onsets, "surp": surp, "z": z, "conf": conf}


def make_noise(n_times, rng, kind):
    x = rng.normal(size=n_times)
    if kind == "pink":
        f = np.fft.rfftfreq(n_times)
        f[0] = f[1]
        x = np.fft.irfft(np.fft.rfft(x) / np.sqrt(f), n=n_times)
    return (x - x.mean()) / x.std()
