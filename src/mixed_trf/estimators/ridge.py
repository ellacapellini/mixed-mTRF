"""Small ridge helpers for the dense per-unit testbed (contiguous k-fold CV on sparse designs)."""
from __future__ import annotations

import numpy as np



def _blocks(n, k):
    return np.array_split(np.arange(n), k)


def ridge_cv_c(X, y, cs=np.logspace(-3, 1, 9), k=5):
    """Pick relative ridge strength c (lambda = c * tr(X'X)/p) by contiguous k-fold CV."""
    n, p = X.shape
    XtX, Xty = (X.T @ X).toarray(), X.T @ y
    scale = np.trace(XtX) / p
    folds = _blocks(n, k)
    blk = [((X[b[0]:b[-1] + 1].T @ X[b[0]:b[-1] + 1]).toarray(), X[b[0]:b[-1] + 1].T @ y[b], b) for b in folds]
    best, best_err = cs[0], np.inf
    for c in cs:
        err = 0.0
        for xx, xy, b in blk:
            w = np.linalg.solve(XtX - xx + c * scale * np.eye(p), Xty - xy)
            err += np.sum((y[b] - X[b[0]:b[-1] + 1] @ w) ** 2)
        if err < best_err:
            best, best_err = c, err
    return best


def ridge_fit(XtX, Xty, c):
    p = XtX.shape[0]
    return np.linalg.solve(XtX + c * np.trace(XtX) / p * np.eye(p), Xty)
