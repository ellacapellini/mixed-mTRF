"""The binned parametric estimator: per-surprisal-bin waveforms -> direct gain/shift/dilation fit to a reference waveform (nonlinear least squares) -> per-SD slope across bins. Also returns a family-adequacy diagnostic."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares



def fit_transform(href, hk, taus, window, tc):
    """Gain/shift/dilation of the reference waveform that best reproduces bin waveform hk inside `window`:
       hk(tau) ~ (1+g) * href(tc + (tau - tc - d) / exp(w))      (direct nonlinear fit, no Taylor expansion)."""
    sel = (taus >= window[0]) & (taus <= window[1])
    ref = gaussian_filter1d(href, 1.0); tgt = gaussian_filter1d(hk, 1.0)[sel]
    f = lambda q: (1 + q[0]) * np.interp(tc + (taus[sel] - tc - q[1]) / np.exp(q[2]), taus, ref) - tgt
    return least_squares(f, x0=[0.0, 0.0, 0.0], bounds=([-0.95, -60.0, -0.9], [3.0, 60.0, 0.9])).x


def binned_patient_decomp(Wb, W0, patient, mask, taus, L, centers, counts, window):
    """Per patient: bin waveforms (mean over the patient's selected units) -> per-bin (g, d, w) -> count-weighted slope on bin z."""
    K = len(centers); out = []; expl = []
    for p in np.unique(patient):
        m = mask & (patient == p)
        if m.sum() == 0:
            continue
        hb = Wb[m].reshape(-1, K, L).mean(0); href = W0[m].mean(0)
        sel = (taus >= window[0]) & (taus <= window[1])
        tc = taus[sel][np.argmax(np.abs(gaussian_filter1d(href, 1.0)[sel]))]
        par = np.array([fit_transform(href, hb[k], taus, window, tc) for k in range(K)])      # (K, 3)
        # family-adequacy diagnostic: share of the surprisal-dependent waveform change that gain/shift/dilation reproduces
        ref_s = gaussian_filter1d(href, 1.0)
        sse = sst = 0.0
        for k in range(K):
            tgt = gaussian_filter1d(hb[k], 1.0)[sel]
            fit = (1 + par[k, 0]) * np.interp(tc + (taus[sel] - tc - par[k, 1]) / np.exp(par[k, 2]), taus, ref_s)
            sse += np.sum((fit - tgt) ** 2); sst += np.sum((tgt - ref_s[sel]) ** 2)
        expl.append(1.0 - sse / sst if sst > 0 else np.nan)
        w = counts / counts.sum(); zc = (w * centers).sum()
        var = (w * (centers - zc) ** 2).sum()
        out.append(((w * (centers - zc))[:, None] * (par - (w[:, None] * par).sum(0))).sum(0) / var)
    return np.array(out), float(np.nanmean(expl))
