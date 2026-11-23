"""First-order gain / shift / width decomposition of h1 relative to h0 (no argmax), and the argmax-latency summary (global or windowed)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d



def decompose(h0, h1, taus, window=(250.0, 600.0), center=None, smooth=1.0):
    """First-order gain / shift / width of h1 relative to h0, inside `window`.
    h(tau|z) ~ h0 + z*h1,  h1 ~ g*h0 + shift*(-h0') + omega*(-(tau-tc)*h0').
    Returns (g per z, shift in ms per z, omega = log-width change per z)."""
    h0s, h1s = gaussian_filter1d(h0, smooth), gaussian_filter1d(h1, smooth)
    d0 = np.gradient(h0s, taus)
    sel = (taus >= window[0]) & (taus <= window[1])
    tc = center if center is not None else taus[sel][np.argmax(np.abs(h0s[sel]))]
    B = np.stack([h0s, -d0, -(taus - tc) * d0], axis=1)[sel]
    coef, *_ = np.linalg.lstsq(B, h1s[sel], rcond=None)
    return float(coef[0]), float(coef[1]), float(coef[2]), float(tc)


def argmax_latency_shift(h0, h1, taus, window=None):
    """ms between the |h| peak at +1 SD and at -1 SD. window=None: global argmax over all lags
    (what Section 3 does). window=(lo, hi): argmax restricted to a pre-specified component window."""
    sel = np.ones_like(taus, bool) if window is None else (taus >= window[0]) & (taus <= window[1])
    lat = lambda s: taus[sel][np.argmax(np.abs((h0 + s * h1)[sel]))]
    return float(lat(+1.0) - lat(-1.0))
