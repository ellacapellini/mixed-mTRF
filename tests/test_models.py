import numpy as np
import pytest

pm = pytest.importorskip("pymc")


def test_suffstat_likelihood_identity():
    from mixed_trf.models.pop_trf import loglik_from_suffstats, prepare_suffstats
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)); th = rng.normal(size=6); y = X @ th + 0.7 * rng.normal(size=400)
    ss = prepare_suffstats([X], [y]); ys = (y - y.mean()) / y.std(); s = 0.8
    Xc = X - X.mean(0)                                   # design columns are centred (free intercept)
    direct = np.sum(-0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * ((ys - Xc @ th) / s) ** 2)
    via = loglik_from_suffstats(th, ss["XtX"][0], ss["Xty"][0], ss["yty"][0], ss["n"][0], s)
    assert abs(direct - via) < 1e-8


def test_registry_lists_the_ladder():
    from mixed_trf.models.registry import MODELS
    assert {"A-pop", "B-pop-amp", "B-pop-hyp"} <= set(MODELS)


def test_basis_hyperparameter_combination_rejected():
    from mixed_trf.models.pop_trf import fit_pop_trf
    with pytest.raises(ValueError):
        fit_pop_trf({"XtX": np.zeros((1, 4, 4)), "Xty": np.zeros((1, 4)), "yty": np.ones(1), "n": np.ones(1)},
                    [0], np.arange(2.0), kind="raised_cosine", hierarchy="hyperparameter")


def test_intercept_does_not_bias_h0():
    """A response with a nonzero mean must not push a constant offset onto the recovered impulse response."""
    from mixed_trf.models.pop_trf import prepare_suffstats
    rng = np.random.default_rng(1); n, L = 20000, 12
    X = np.zeros((n, L)); on = np.sort(rng.choice(n - L, 900, replace=False))
    for o in on:
        X[o + np.arange(L), np.arange(L)] = 1.0
    h = -np.exp(-0.5 * ((np.arange(L) - 5) / 1.5) ** 2); y = X @ h + 0.3 * rng.normal(size=n)
    ss = prepare_suffstats([X], [y]); sd = y.std()
    th = np.linalg.solve(ss["XtX"][0] + 1e-6 * np.eye(L), ss["Xty"][0]) * sd
    assert abs(th[-1] - h[-1]) < 0.15 and abs(th[0] - h[0]) < 0.15       # tails near zero, not offset by +mean
