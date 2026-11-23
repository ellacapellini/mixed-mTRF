"""
bayesian_pop_trf.py
===================
B-pop / A-pop: a hierarchical TRF WITH a population-level mean, fitted from
per-unit SUFFICIENT STATISTICS so the cost does not depend on recording length.

Why this exists (findings from the earlier work)
------------------------------------------------
1. bayesian_gp_trf.py has NO shared mean TRF: each unit's h is an independent
   zero-mean GP draw and only the kernel hyperparameters are pooled. A held-out
   patient can then only be predicted by the zero-mean prior, so
   leave-one-patient-out comparison with Model A (which has mu_j) is impossible.
2. It builds dense (units x time x lags) designs inside the likelihood: at 2 h,
   32 units, 81 lags that is ~30 GB and ~7.5 GFLOP per gradient. With a Gaussian
   likelihood none of that is needed:
       log p(y_i | theta_i, s_i) = -n_i/2 log(2 pi) - n_i log s_i
                                   - (y'y - 2 theta'X'y + theta'X'X theta) / (2 s_i^2)
   so each unit only contributes X'X (p x p), X'y (p), y'y, n.
   Cost per gradient ~ units x p^2, independent of n.
3. The GP draws are NON-CENTERED (h = B z, z ~ N(0, I)), which removed most of
   the sampling pathology in the ablation (signal R-hat 2.2 -> 1.03).

Model (k = 0: baseline TRF, k = 1: surprisal modulation; unit i in patient p)
-------------------------------------------------------------------------------
    h_k,i(tau) = h_k,pop(tau) + h_k,p(tau) + h_k,i(tau)            (functional mixed model)
    each term:  sd_level,k * (B_k z),  z ~ N(0, I)
        kind = "matern52" | "squared_exponential": B_k = chol(K_k), K_k = kernel(ell_k) (unit scale)
        kind = "raised_cosine":                      B_k = Phi (L x J raised-cosine basis)  -> Model A
    y_i = [S0 S1] [h_0,i ; h_1,i] + noise,  noise sd s_i,  log s_i ~ N(mu_s, tau_s)

Because kind="raised_cosine" is the same model with a low-rank B, Model A and
Model B differ ONLY in the level prior -- exactly the Section-4 equivalence,
now literal, and a fair comparison (identical hierarchy, priors and likelihood).

Conventions (explicit, because a sigma-vs-sigma^2 mismatch caused a real bug before):
    sd_*  are standard deviations; the level kernel is sd^2 * k_unit(ell).
    All y are standardised per unit by the caller (prepare_suffstats does it).
    The surprisal regressor must be centred/z-scored by a GLOBAL constant.

Three models, one code path (`hierarchy` / `kind`):
    A-pop      kind="raised_cosine", hierarchy="amplitude"  -- weights pooled (Model A)
    B-pop-amp  kind=GP,              hierarchy="amplitude"  -- GP prior, amplitude pooled, ONE shared length scale
    B-pop-hyp  kind=GP,              hierarchy="hyperparameter"  -- the ORIGINAL Model B idea, now WITH a mean:
        h_k,i = h_k,pop + d_k,i,   d_k,i ~ GP(0, k(ell_k,i, sd_k,i))
        log ell_k,i = log ell_k + a^ell_k,p(i) + b^ell_k,i        (patients and electrodes differ in SMOOTHNESS)
        log sd_k,i  = log sd_k  + a^sd_k,p(i)  + b^sd_k,i         (... and in amplitude of their deviation)
    The population mean h_k,pop (own length scale/amplitude) is what makes leave-one-patient-out possible;
    the deviation GP's hyperparameter hierarchy is what lets patients differ in smoothness.
    The ladder A-pop -> B-pop-amp -> B-pop-hyp isolates what each ingredient buys.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.sparse as sp

from models.bayesian_gp_trf import _build_kernel_pt


# ------------------------------------------------------------ sufficient stats
def prepare_suffstats(designs, ys):
    """designs: list of (n_i x p) arrays/sparse; ys: list of (n_i,). Standardises each y.
    Returns dict with XtX (U,p,p), Xty (U,p), yty (U,), n (U,), y_scale (U,)."""
    XtX, Xty, yty, n, scale = [], [], [], [], []
    for X, y in zip(designs, ys):
        y = np.asarray(y, float)
        sd = y.std()
        ysd = (y - y.mean()) / sd
        XtX.append(np.asarray((X.T @ X).todense()) if sp.issparse(X) else X.T @ X)
        Xty.append(np.asarray(X.T @ ysd).ravel())
        yty.append(float(ysd @ ysd)); n.append(len(ysd)); scale.append(sd)
    return {"XtX": np.array(XtX), "Xty": np.array(Xty), "yty": np.array(yty),
            "n": np.array(n, float), "y_scale": np.array(scale)}


def loglik_from_suffstats(theta, XtX, Xty, yty, n, sigma):
    """numpy reference for the likelihood used inside the model (used by the self-test)."""
    quad = yty - 2 * theta @ Xty + theta @ XtX @ theta
    return -0.5 * n * np.log(2 * np.pi) - n * np.log(sigma) - 0.5 * quad / sigma ** 2


# ----------------------------------------------------------------------- model
@dataclass
class PopTRFResult:
    trace: object
    taus: np.ndarray
    kind: str
    ll_unit: np.ndarray      # (chain, draw, unit): per-unit log-lik -> PSIS-LOO by unit/patient
    h_pop: np.ndarray        # (chain, draw, 2, L): population-level h0, h1


def fit_pop_trf(ss, patient_idx, taus, *, kind="matern52", hierarchy="amplitude", n_basis=10, basis_c=5.0,
                ell0=100.0, sd0_prior=(0.5, 0.1), tau_hyper_scale=0.3, draws=1000, tune=1500, chains=4,
                cores=None, target_accept=0.9, seed=0, jitter_rel=None, progressbar=False):
    if hierarchy not in ("amplitude", "hyperparameter"):
        raise ValueError("hierarchy must be 'amplitude' or 'hyperparameter'")
    if kind == "raised_cosine" and hierarchy == "hyperparameter":
        raise ValueError("a basis has no length scale: hyperparameter hierarchy needs a GP kind")
    if jitter_rel is None:
        jitter_rel = 1e-4 if kind == "squared_exponential" else 1e-6     # SE is far worse conditioned
    XtX, Xty, yty, n_obs = ss["XtX"], ss["Xty"], ss["yty"], ss["n"]
    U, p2, _ = XtX.shape
    L = len(taus)
    assert p2 == 2 * L, "design must be [S0 S1] with the same lag grid as taus"
    patient_idx = np.asarray(patient_idx, dtype="int64")
    P = int(patient_idx.max()) + 1
    taus_t = np.asarray(taus, dtype="float64")
    if kind == "raised_cosine":
        from basis.basis_functions import make_raised_cosine_basis
        Phi = make_raised_cosine_basis(n_basis=n_basis, tau_max=float(taus_t.max()), c=basis_c).eval(taus_t)
        D = Phi.shape[1]
    else:
        D = L

    with pm.Model() as model:
        H, Hpop = [], []
        if hierarchy == "amplitude":
            Bk, sd_pop, sd_pat, sd_el = [], [], [], []
            for k in (0, 1):
                m = sd0_prior[k]
                if kind == "raised_cosine":
                    Bk.append(pt.as_tensor_variable(Phi))
                else:
                    ell = pt.exp(pm.Normal(f"log_ell_{k}", np.log(ell0), 0.5))
                    K = _build_kernel_pt(kind, taus_t, ell, 1.0) + jitter_rel * np.eye(L)
                    Bk.append(pt.linalg.cholesky(K))
                sd_pop.append(pt.exp(pm.Normal(f"log_sd_pop_{k}", np.log(m), 1.0)))
                sd_pat.append(pm.HalfNormal(f"sd_pat_{k}", sigma=m))
                sd_el.append(pm.HalfNormal(f"sd_el_{k}", sigma=m))
            z_pop = pm.Normal("z_pop", 0.0, 1.0, shape=(2, D))
            z_pat = pm.Normal("z_pat", 0.0, 1.0, shape=(P, 2, D))
            z_el = pm.Normal("z_el", 0.0, 1.0, shape=(U, 2, D))
            for k in (0, 1):
                W = sd_pop[k] * z_pop[k][None, :] + sd_pat[k] * z_pat[patient_idx, k, :] + sd_el[k] * z_el[:, k, :]
                H.append(pt.dot(W, Bk[k].T))
                Hpop.append(pt.dot(Bk[k], sd_pop[k] * z_pop[k]))
        else:
            # B-pop-hyp: population mean GP + per-unit deviation GP whose (ell, sd) are hierarchical
            diff = np.abs(taus_t[:, None] - taus_t[None, :])                  # (L, L)
            z_pop = pm.Normal("z_pop", 0.0, 1.0, shape=(2, L))
            z_dev = pm.Normal("z_dev", 0.0, 1.0, shape=(U, 2, L))
            for k in (0, 1):
                m = sd0_prior[k]
                # population mean: own smoothness and amplitude
                ell_pop = pt.exp(pm.Normal(f"log_ell_pop_{k}", np.log(ell0), 0.5))
                Kp = _build_kernel_pt(kind, taus_t, ell_pop, 1.0) + jitter_rel * np.eye(L)
                sd_pop = pt.exp(pm.Normal(f"log_sd_pop_{k}", np.log(m), 1.0))
                Hpop.append(pt.dot(pt.linalg.cholesky(Kp), sd_pop * z_pop[k]))
                # deviation hyperparameters: global level + patient + electrode deviations (non-centred)
                log_ell0 = pm.Normal(f"log_ell_dev_{k}", np.log(ell0), 0.5)
                log_sd0 = pm.Normal(f"log_sd_dev_{k}", np.log(0.5 * m), 1.0)
                t_a_l = pm.HalfNormal(f"tau_pat_ell_{k}", tau_hyper_scale)
                t_a_s = pm.HalfNormal(f"tau_pat_sd_{k}", tau_hyper_scale)
                t_b_l = pm.HalfNormal(f"tau_el_ell_{k}", tau_hyper_scale)
                t_b_s = pm.HalfNormal(f"tau_el_sd_{k}", tau_hyper_scale)
                ra_l = pm.Normal(f"raw_pat_ell_{k}", 0.0, 1.0, shape=P)
                ra_s = pm.Normal(f"raw_pat_sd_{k}", 0.0, 1.0, shape=P)
                rb_l = pm.Normal(f"raw_el_ell_{k}", 0.0, 1.0, shape=U)
                rb_s = pm.Normal(f"raw_el_sd_{k}", 0.0, 1.0, shape=U)
                ell_i = pt.exp(log_ell0 + t_a_l * ra_l[patient_idx] + t_b_l * rb_l)      # (U,)
                sd_i = pt.exp(log_sd0 + t_a_s * ra_s[patient_idx] + t_b_s * rb_s)        # (U,)
                pm.Deterministic(f"ell_unit_{k}", ell_i)
                pm.Deterministic(f"sd_unit_{k}", sd_i)
                # batched unit-scale kernel (U, L, L); sd_i applied after the Cholesky
                r = diff[None, :, :] / ell_i[:, None, None]
                if kind == "matern52":
                    s5 = np.sqrt(5.0) * r
                    Kb = (1 + s5 + (5.0 / 3.0) * r ** 2) * pt.exp(-s5)
                else:
                    Kb = pt.exp(-0.5 * r ** 2)
                Kb = Kb + jitter_rel * np.eye(L)[None, :, :]
                Bb = pt.linalg.cholesky(Kb)                                              # (U, L, L)
                d = sd_i[:, None] * pt.matmul(Bb, z_dev[:, k, :, None])[..., 0]          # (U, L)
                H.append(Hpop[k][None, :] + d)
        theta = pt.concatenate(H, axis=1)                                  # (U, 2L)
        pm.Deterministic("h_pop", pt.stack(Hpop, axis=0))                  # (2, L)
        pm.Deterministic("h_unit0", H[0]); pm.Deterministic("h_unit1", H[1])

        mu_s = pm.Normal("mu_log_noise", 0.0, 0.5)
        tau_s = pm.HalfNormal("tau_log_noise", 0.5)
        z_s = pm.Normal("z_noise", 0.0, 1.0, shape=U)
        log_s = mu_s + tau_s * z_s
        s = pt.exp(log_s)

        quad = yty - 2 * pt.sum(theta * Xty, axis=1) \
            + pt.sum(theta * pt.matmul(XtX, theta[..., None])[..., 0], axis=1)
        ll = -0.5 * n_obs * np.log(2 * np.pi) - n_obs * log_s - 0.5 * quad / s ** 2
        pm.Deterministic("ll_unit", ll)
        pm.Potential("loglik", pt.sum(ll))

        trace = pm.sample(draws=draws, tune=tune, chains=chains,
                          cores=(cores if cores is not None else chains),
                          target_accept=target_accept, random_seed=seed, progressbar=progressbar)
    post = trace.posterior
    return PopTRFResult(trace=trace, taus=taus_t, kind=kind,
                        ll_unit=np.asarray(post["ll_unit"].values),
                        h_pop=np.asarray(post["h_pop"].values))


def predict_population(h_pop_mean, X):
    """Predict a (possibly held-out-patient) unit from the POPULATION mean TRF only.
    h_pop_mean: (2, L) = [h0, h1]; X: (n, 2L) design [S0 S1]."""
    return X @ np.concatenate([h_pop_mean[0], h_pop_mean[1]])


# -------------------------------------------------------------------- self-test
def _selftest():
    import time
    rng = np.random.default_rng(0)
    # 1) the sufficient-statistic likelihood equals the direct Gaussian log-density
    n, p = 400, 6
    X = rng.normal(size=(n, p)); theta = rng.normal(size=p); y = X @ theta + rng.normal(size=n) * 0.7
    ss = prepare_suffstats([X], [y])
    ys = (y - y.mean()) / y.std()
    sigma = 0.8
    direct = float(np.sum(-0.5 * np.log(2 * np.pi * sigma ** 2) - 0.5 * ((ys - X @ theta) / sigma) ** 2))
    viass = float(loglik_from_suffstats(theta, ss["XtX"][0], ss["Xty"][0], ss["yty"][0], ss["n"][0], sigma))
    print(f"[1] likelihood identity: direct {direct:.6f}  suffstats {viass:.6f}  |diff| {abs(direct - viass):.2e}")
    assert abs(direct - viass) < 1e-8

    # 2) tiny end-to-end fit on testbed data (gain effect) -- sampler runs, population mean recovered
    import argparse
    import erp_testbed as T
    a = argparse.Namespace(n_patients=4, n_electrodes=2, n_times=9000, tau_max=800.0, snr=0.2, noise="white",
                           common_noise=0.0, effect_frac=1.0, effect_on="surprisal", confound_rho=0.0,
                           shared_stimulus=False, surp_mean=7.0, surp_sd=3.0, center_ms=None, boot=10, seed=0)
    rng = np.random.default_rng(3)
    L = int(a.tau_max / T.DT) + 1; taus = np.arange(L) * T.DT
    units = T.make_dataset(rng, n_patients=a.n_patients, n_electrodes=a.n_electrodes, n_times=a.n_times, taus=taus,
                           mode="gain", effect_frac=1.0, effect_on="surprisal", shared=False, snr=a.snr,
                           noise="white", common_frac=0.0, rho=0.0, surp_mean=7.0, surp_sd=3.0)
    designs = [T.unit_design(u, a.n_times, L, ["surp"]) for u in units]
    ss = prepare_suffstats(designs, [u["r"] for u in units])
    pat = np.array([u["patient"] for u in units])
    t0 = time.time()
    res = fit_pop_trf(ss, pat, taus, kind="matern52", draws=150, tune=250, chains=2, cores=1, seed=1)
    import arviz as az
    hp = res.h_pop.mean(axis=(0, 1))                                   # (2, L)
    g, sh, om, _ = T.decompose(hp[0], hp[1], taus)
    div = int(res.trace.sample_stats["diverging"].sum())
    rh = az.rhat(res.trace, var_names=["h_pop"])["h_pop"].values
    print(f"[2] fit ok in {time.time() - t0:.0f}s  divergences {div}  max R-hat(h_pop) {np.nanmax(rh):.3f}")
    print(f"    decomposition from posterior-mean population TRF: gain {g:+.2f} (truth 0.30)  "
          f"shift {sh:+.1f} ms (0)  width {om:+.2f} (0)")


if __name__ == "__main__":
    _selftest()