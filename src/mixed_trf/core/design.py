"""Lagged event design matrices (sparse): column j = the value of the event whose onset is j samples earlier."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp



def lag_design(onsets, values, n_times, n_lags):
    """Lagged word-onset design as a SPARSE matrix: S[t, j] = value of the word whose onset is j samples before t."""
    rows, cols, vals = [], [], []
    for j in range(n_lags):
        idx = onsets + j
        ok = idx < n_times
        rows.append(idx[ok]); cols.append(np.full(int(ok.sum()), j)); vals.append(values[ok])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(n_times, n_lags))


def unit_design(u, n_times, n_lags, mods):
    w = u["words"]
    cols = [lag_design(w["onsets"], np.ones(len(w["onsets"])), n_times, n_lags)]
    for m in mods:
        cols.append(lag_design(w["onsets"], w[{"surp": "z", "conf": "conf"}[m]], n_times, n_lags))
    return sp.hstack(cols, format="csr")
