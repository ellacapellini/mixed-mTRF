"""The first-order (linear-in-surprisal) estimator: per-patient decomposition of the mean TRFs h0, h1."""
from __future__ import annotations

import numpy as np
from mixed_trf.estimators.decomposition import decompose


def patient_decomp(W, patient, sel, taus, L, window=(250.0, 600.0)):
    out = []
    for p in np.unique(patient):
        m = sel & (patient == p)
        if m.sum() == 0:
            continue
        g, sh, om, _ = decompose(W[m, :L].mean(0), W[m, L:].mean(0), taus, window=window)
        out.append((g, sh, om))
    return np.array(out)
