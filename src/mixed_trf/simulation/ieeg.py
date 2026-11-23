"""iEEG-scale simulator: electrode geometry (sEEG shafts / grid), band-dependent spatially correlated noise, spatially clustered responsiveness; returns sufficient statistics only."""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy import stats
from scipy.spatial.distance import cdist
from mixed_trf.core.suffstats import make_blocks, unit_stats
from mixed_trf.simulation.events import make_words
from mixed_trf.simulation.templates import COMPONENTS, effect_for


BAND_LEN_MM = {"hg": 1.5, "lfp": 15.0}


def erp_words(taus, z, eff, amp, shift, wscale):
    """Vectorised twin of erp_testbed.erp for ALL words at once -> (n_words, L)."""
    tt = taus[None, :]
    out = 0.0
    for name, (t, sd, a) in COMPONENTS.items():
        if name == "N400":
            aa = a * np.maximum(1.0 + eff["gain"] * z, 0.05)[:, None]
            tc = (t + shift + eff["shift"] * z)[:, None]
            ss = (sd * wscale * np.exp(eff["width"] * z))[:, None]
            if eff.get("decay", 0.0) != 0.0:                       # asymmetric: only the right (decay) flank changes width
                ss = np.where(tt < tc, ss, ss * np.exp(eff["decay"] * z)[:, None])
        else:
            aa, tc, ss = a, t + shift, sd * wscale
        out = out + aa * np.exp(-0.5 * ((tt - tc) / ss) ** 2)
    if eff.get("late", 0.0) != 0.0:                                # additive late component, linear in z
        out = out + (eff["late"] * z)[:, None] * np.exp(-0.5 * ((tt - (550.0 + shift)) / 60.0) ** 2)
    return amp * out


def place(onsets, H, n_times):
    idx = onsets[:, None] + np.arange(H.shape[1])[None, :]
    ok = idx < n_times
    return np.bincount(idx[ok], weights=H[ok], minlength=n_times)[:n_times]


def fast_pink(n, rng):
    x = rng.normal(size=n)
    f = np.fft.rfftfreq(n); f[0] = f[1]
    x = np.fft.irfft(np.fft.rfft(x) / np.sqrt(f), n=n)
    return (x - x.mean()) / x.std()


def make_positions(E, geometry, spacing_mm, rng):
    """(E, 3) contact positions in mm. sEEG: shafts of 12 contacts at random places/directions in an ~80x80x60 mm
    volume; grid: 16-column lattice in a plane."""
    if geometry == "grid":
        cols = 16
        i = np.arange(E)
        return np.stack([(i // cols) * spacing_mm, (i % cols) * spacing_mm, np.zeros(E)], axis=1).astype(float)
    per = 12
    pos = []
    for _ in range(int(np.ceil(E / per))):
        o = rng.uniform([0, 0, 0], [80, 80, 60])
        v = rng.normal(size=3); v /= np.linalg.norm(v)
        pos += [o + k * spacing_mm * v for k in range(per)]
    return np.array(pos[:E])


def pink_rows(E, n, rng, chunk=32):
    """E independent unit-variance 1/f series, float32, generated in chunks to cap memory."""
    out = np.empty((E, n), dtype=np.float32)
    f = np.fft.rfftfreq(n); f[0] = f[1]
    shape = (1.0 / np.sqrt(f)).astype(np.float32)
    for a in range(0, E, chunk):
        b = min(E, a + chunk)
        x = rng.standard_normal((b - a, n)).astype(np.float32)
        x = sfft.irfft(sfft.rfft(x, axis=1, workers=1) * shape[None, :], n=n, axis=1, workers=1)
        x -= x.mean(axis=1, keepdims=True); x /= x.std(axis=1, keepdims=True)
        out[a:b] = x
    return out


def spatial_noise(pos, n_times, rng, length_mm, common, noise_kind):
    """Noise matrix (E, n_times), unit variance per row, covariance c*11' + (1-c)*exp(-d/L)."""
    E = len(pos)
    D = cdist(pos, pos)
    Sigma = common * np.ones((E, E)) + (1 - common) * np.exp(-D / length_mm) + 1e-6 * np.eye(E)
    Lc = np.linalg.cholesky(Sigma).astype(np.float32)
    if noise_kind == "pink":
        Z = pink_rows(E, n_times, rng)
    else:
        Z = rng.standard_normal((E, n_times)).astype(np.float32)
    return Lc @ Z, Sigma


def thresholded_field(D, length_mm, frac, rng):
    """Spatially smooth Gaussian field -> binary mask with expected share `frac` (clusters, not salt-and-pepper)."""
    R = np.exp(-D / length_mm) + 1e-6 * np.eye(len(D))
    g = np.linalg.cholesky(R) @ rng.standard_normal(len(D))
    return g > stats.norm.ppf(1 - frac), g


def generate(args, rng, taus, L):
    n_times = args.n_times
    words = make_words(n_times, rng, surp_mean=args.surp_mean, surp_sd=args.surp_sd, rhythm=args.rhythm,
                         period_ms=args.period_ms, violation_p=args.violation_p, jitter_ms=args.jitter_ms)
    bin_idx, bins = None, None
    if args.estimator == "binned":
        z = words["z"]
        uz = np.unique(np.round(z, 9))
        if len(uz) <= args.n_bins:                       # few distinct surprisal values (e.g. binary rhythm violations): one bin per value
            b = np.searchsorted(uz, np.round(z, 9)); K_ = len(uz)
        else:
            q = np.quantile(z, np.linspace(0, 1, args.n_bins + 1))
            b = np.clip(np.searchsorted(q[1:-1], z, side="right"), 0, args.n_bins - 1); K_ = args.n_bins
        bin_idx = [np.where(b == k)[0] for k in range(K_)]
        bins = {"centers": np.array([z[ix].mean() for ix in bin_idx]), "counts": np.array([len(ix) for ix in bin_idx], float)}
    edges, XtXb, XbT = make_blocks(words, n_times, L, args.dev_frac, bin_idx)
    # nominal noise level: SNR is defined against a unit-amplitude, effect-free responsive electrode
    null = effect_for("null")
    ref_var = place(words["onsets"], erp_words(taus, words["z"], null, 1.0, 0.0, 1.0), n_times).var()
    noise_sd = np.sqrt(ref_var / args.snr)
    eff_on = effect_for(args.mode, args.effect_scale)
    P, E = args.n_patients, args.electrodes
    L_noise = args.noise_len_mm if args.noise_len_mm > 0 else BAND_LEN_MM[args.band]
    patient, responsive, has_eff, Xty_all, yty_all, mean_corr = [], [], [], [], [], []
    for p in range(P):
        p_scale, p_shift = np.exp(rng.normal(0, 0.15)), rng.normal(0, 8.0)
        p_w = float(np.exp(rng.normal(0, args.patient_width_sd))) if args.patient_width_sd > 0 else 1.0
        pos = make_positions(E, args.geometry, args.spacing_mm, rng)
        D = cdist(pos, pos)
        resp_mask, _ = thresholded_field(D, args.resp_len_mm, args.responsive_frac, rng)
        eff_mask, _ = thresholded_field(D, args.resp_len_mm, args.effect_frac, rng)
        amp_field = np.linalg.cholesky(np.exp(-D / args.resp_len_mm) + 1e-6 * np.eye(E)) @ rng.standard_normal(E)
        N, Sigma = spatial_noise(pos, n_times, rng, L_noise, args.common_noise, args.noise)
        for e in range(E):
            resp = bool(resp_mask[e]); has = bool(resp and eff_mask[e])
            sig = 0.0
            if resp:
                eff = eff_on if has else null
                sig = place(words["onsets"], erp_words(taus, words["z"], eff, p_scale * np.exp(0.25 * amp_field[e]), p_shift, p_w), n_times)
            y = sig + noise_sd * N[e].astype(np.float64)
            Xty, yty = unit_stats(y, edges, XbT)
            patient.append(p); responsive.append(resp); has_eff.append(has)
            Xty_all.append(Xty); yty_all.append(yty)
        mean_corr.append(Sigma)
        del N
    corr_mats = mean_corr
    return {"patient": np.array(patient), "responsive": np.array(responsive), "has_eff": np.array(has_eff),
            "Xty": np.array(Xty_all), "yty": np.array(yty_all), "XtXb": XtXb, "Sigma": corr_mats, "bins": bins}
