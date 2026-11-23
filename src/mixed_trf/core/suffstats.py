"""Sufficient statistics for a SHARED stimulus: one X'X per time block, and per unit only X'y and y'y. Memory per unit is a few KB regardless of recording length."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from mixed_trf.core.design import lag_design


N_DEV_BLOCKS = 5


def block_edges(n_times, dev_frac):
    n_dev = int(n_times * dev_frac)
    return list(np.linspace(0, n_dev, N_DEV_BLOCKS + 1).astype(int)) + [n_times]     # 6 blocks: 5 dev + 1 conf


def make_blocks(words, n_times, L, dev_frac, bin_idx=None):
    """X = [S0 | S1 | (optional) one lagged-impulse block per surprisal bin]. Column layout:
    0:L = all words, L:2L = surprisal-weighted, 2L + k*L : 2L + (k+1)*L = words in bin k."""
    on = words["onsets"]
    parts = [lag_design(on, np.ones(len(on)), n_times, L), lag_design(on, words["z"], n_times, L)]
    if bin_idx is not None:
        parts += [lag_design(on[ix], np.ones(len(ix)), n_times, L) for ix in bin_idx]
    X = sp.hstack(parts, format="csr")
    edges = block_edges(n_times, dev_frac)
    XtXb, XbT = [], []
    for b in range(6):
        Xb = X[edges[b]:edges[b + 1]]
        XtXb.append((Xb.T @ Xb).toarray())
        XbT.append(Xb.T.tocsr())
    return edges, np.array(XtXb), XbT


def unit_stats(y, edges, XbT):
    Xty = np.array([XbT[b] @ y[edges[b]:edges[b + 1]] for b in range(6)])
    yty = np.array([y[edges[b]:edges[b + 1]] @ y[edges[b]:edges[b + 1]] for b in range(6)])
    return Xty, yty
