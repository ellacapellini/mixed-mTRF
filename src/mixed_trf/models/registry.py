"""One interface for every model of the ladder.

    from mixed_trf.models.registry import MODELS, fit
    res = fit("B-pop-hyp", suffstats, patient_idx, taus, draws=500, tune=500, chains=2)

Models (each differs from the previous by exactly one ingredient):
    A-pop      raised-cosine basis (20 functions), weights pooled across patients/electrodes   (Model A)
    A-pop-10   same with 10 functions (original resolution; underfits N1/N400 peaks)
    B-pop-amp  GP prior on h(tau), amplitude pooled, ONE shared length scale
    B-pop-hyp  GP deviations with hierarchical length scale + amplitude         (Model B with population mean)
"""
from __future__ import annotations

MODELS = {
    "A-pop": dict(kind="raised_cosine", hierarchy="amplitude", n_basis=20),      # 20 functions: 10 is too coarse for sharp ERP peaks
    "A-pop-10": dict(kind="raised_cosine", hierarchy="amplitude", n_basis=10),   # the original Model A resolution, kept for the comparison
    "B-pop-amp": dict(kind="matern52", hierarchy="amplitude"),
    "B-pop-hyp": dict(kind="matern52", hierarchy="hyperparameter"),
    "B-pop-amp-se": dict(kind="squared_exponential", hierarchy="amplitude"),
    "B-pop-hyp-se": dict(kind="squared_exponential", hierarchy="hyperparameter"),
}


def fit(name, ss, patient_idx, taus, **kw):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    from .pop_trf import fit_pop_trf          # lazy: PyMC is only needed for Stage 2
    return fit_pop_trf(ss, patient_idx, taus, **{**MODELS[name], **kw})
