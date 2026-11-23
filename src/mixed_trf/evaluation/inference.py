"""Patient-level t-interval (the patient is the unit of inference) and leave-one-patient-out R^2 gain."""
from __future__ import annotations

import numpy as np
from scipy import stats



def t_interval(pd_, level=0.90):
    n = len(pd_)
    m = pd_.mean(0); se = pd_.std(0, ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.5 + level / 2, n - 1)
    return m, m - q * se, m + q * se


def lopo_gain(XtXc, Xtyc, yty_c, patient, sel, c_pop, L):
    """Leave-one-patient-out R^2 gain of M1 over M0 on the confirmatory block (selected units only)."""
    res = []
    for p in np.unique(patient):
        tr = sel & (patient != p); te = sel & (patient == p)
        if tr.sum() == 0 or te.sum() == 0:
            continue
        r2 = {}
        for name, cols in (("M0", np.arange(L)), ("M1", np.arange(2 * L))):
            A = XtXc[np.ix_(cols, cols)] * tr.sum()
            sc = np.trace(A) / len(cols)
            w = np.linalg.solve(A + c_pop * sc * np.eye(len(cols)), Xtyc[tr][:, cols].sum(0))
            sse = yty_c[te] - 2 * Xtyc[te][:, cols] @ w + w @ XtXc[np.ix_(cols, cols)] @ w
            r2[name] = np.mean(1 - sse / yty_c[te])
        res.append((r2["M0"], r2["M1"]))
    res = np.array(res)
    return res[:, 1] - res[:, 0], res[:, 0]
