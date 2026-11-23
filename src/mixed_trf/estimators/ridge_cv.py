"""Ridge regression from sufficient statistics: per-unit cross-validated strength on the development blocks, and the confirmatory fit."""
from __future__ import annotations

import numpy as np
from mixed_trf.core.suffstats import N_DEV_BLOCKS


CS = np.logspace(-3, 1, 9)            # relative ridge strengths (lambda = c * trace(X'X)/p)


def cv_ridge(XtXb, Xty, yty, cols, cs=CS):
    """Per-unit CV over the dev blocks. Returns best c index per unit and CV R^2 per unit."""
    idx = np.array(cols); p = len(idx); U = Xty.shape[0]
    A = [XtXb[b][np.ix_(idx, idx)] for b in range(N_DEV_BLOCKS)]
    tot = sum(A); scale = np.trace(tot) / p
    xty_tot = Xty[:, :N_DEV_BLOCKS, :][:, :, idx].sum(1)
    sse = np.zeros((len(cs), U))
    for ci, c in enumerate(cs):
        for b in range(N_DEV_BLOCKS):
            xb = Xty[:, b, idx]
            W = np.linalg.solve(tot - A[b] + c * scale * np.eye(p), (xty_tot - xb).T).T
            sse[ci] += yty[:, b] - 2 * np.einsum("up,up->u", W, xb) + np.einsum("up,pq,uq->u", W, A[b], W)
    best = sse.argmin(0)
    r2 = 1.0 - sse[best, np.arange(U)] / yty[:, :N_DEV_BLOCKS].sum(1)
    return best, r2


def conf_fit(XtXc, Xtyc, best, cols, cs=CS, c_mult=1.0):
    idx = np.array(cols); p = len(idx)
    A = XtXc[np.ix_(idx, idx)]; scale = np.trace(A) / p
    W = np.zeros((Xtyc.shape[0], p))
    for ci in np.unique(best):
        m = best == ci
        W[m] = np.linalg.solve(A + c_mult * cs[ci] * scale * np.eye(p), Xtyc[m][:, idx].T).T
    return W
