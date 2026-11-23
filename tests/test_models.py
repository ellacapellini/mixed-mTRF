import numpy as np
import pytest

pm = pytest.importorskip("pymc")


def test_suffstat_likelihood_identity():
    from mixed_trf.models.pop_trf import loglik_from_suffstats, prepare_suffstats
    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6)); th = rng.normal(size=6); y = X @ th + 0.7 * rng.normal(size=400)
    ss = prepare_suffstats([X], [y]); ys = (y - y.mean()) / y.std(); s = 0.8
    direct = np.sum(-0.5 * np.log(2 * np.pi * s ** 2) - 0.5 * ((ys - X @ th) / s) ** 2)
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
