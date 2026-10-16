"""
ablate_standard.py
==================
One-change-at-a-time ablations of Model B on the Standard (null) generator,
to discriminate between competing explanations of Section 10.3's R-hat/BFMI
failure. Drop this in the project root (next to recovery_grid.py) and run:

    python ablate_standard.py --variant asis --seed 0
    python ablate_standard.py --variant center --seed 0
    ... (one Slurm array task per (variant, seed))

Variants
--------
asis        : bayesian_gp_trf.py's model, re-implemented here verbatim
              (centered GP: h ~ MvNormal(0, L)), so every other variant
              differs from it by exactly one flag.
center      : surprisal centered per unit:  S1c = S1 - mean_surp * S0.
              (Tests: collinearity between h0 and h1*surp.)
no_h1       : drop h1 entirely.
              (Tests: is the problem localised to h1 / its coupling with h0?)
ncp         : non-centered GP draws: h = L z, z ~ N(0, I).
              (Tests: funnel between kernel hyperparameters and h.)
shared_mean : add ONE shared population mean function m0 (all units), and
              per-unit h0_i = m0 + GP deviation. (Tests: the fact that the
              current model has NO population-level mean TRF while the
              Standard generator gives every unit the SAME h(tau).)
fix_h1      : h1 hyperparameters pinned (not sampled) to their prior centre.
              (Tests: does h0's bimodality depend on h1's wandering?)

Reports max R-hat, fraction of parameters with R-hat > 1.01, per-chain
BFMI, divergences, and R-hat/ESS on FUNCTIONALS (the predicted signal and
h0_i), which are the quantities scientific claims are made about.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import arviz as az

import recovery_grid as rg
from basis.basis_functions import make_raised_cosine_basis
from models.bayesian_gp_trf import _build_kernel_pt


def build_and_sample(S0, S1, r, patient_idx, taus, kernel_type, *, ncp, use_h1,
                     fix_h1, shared_mean, draws, tune, chains, cores, target_accept, seed,
                     ell0_prior=50.0, sigma0_prior=1.0, jitter_rel=1e-6):
    n_units, n_times, n_lags = S0.shape
    n_patients = int(patient_idx.max()) + 1
    taus = np.asarray(taus, dtype="float64")

    def gp_draw(name, L):
        if ncp:
            z = pm.Normal(f"z_{name}", 0.0, 1.0, shape=n_lags)
            return pm.Deterministic(name, pt.dot(L, z))
        return pm.MvNormal(name, mu=np.zeros(n_lags), chol=L)

    with pm.Model() as model:
        log_ell0_h0 = pm.Normal("log_ell0_h0", np.log(ell0_prior), 0.3)
        log_sigma0_h0 = pm.Normal("log_sigma0_h0", np.log(sigma0_prior), 0.3)
        if use_h1:
            if fix_h1:
                log_ell0_h1 = pt.as_tensor_variable(np.log(ell0_prior))
                log_sigma0_h1 = pt.as_tensor_variable(np.log(sigma0_prior))
            else:
                log_ell0_h1 = pm.Normal("log_ell0_h1", np.log(ell0_prior), 0.3)
                log_sigma0_h1 = pm.Normal("log_sigma0_h1", np.log(sigma0_prior), 0.3)

        def dev(level, name, n):
            tau = pm.HalfCauchy(f"tau_{level}_{name}", beta=0.3)
            raw = pm.Normal(f"raw_{level}_{name}", 0.0, 1.0, shape=n)
            return tau * raw

        a_ell_h0, a_sig_h0 = dev("p", "ell_h0", n_patients), dev("p", "sigma_h0", n_patients)
        b_ell_h0, b_sig_h0 = dev("e", "ell_h0", n_units), dev("e", "sigma_h0", n_units)
        if use_h1 and not fix_h1:
            a_ell_h1, a_sig_h1 = dev("p", "ell_h1", n_patients), dev("p", "sigma_h1", n_patients)
            b_ell_h1, b_sig_h1 = dev("e", "ell_h1", n_units), dev("e", "sigma_h1", n_units)

        if shared_mean:
            Kp = _build_kernel_pt(kernel_type, taus, ell0_prior, sigma0_prior) + 1e-6 * np.eye(n_lags)
            m0 = pm.MvNormal("m0", mu=np.zeros(n_lags), chol=pt.linalg.cholesky(Kp))

        h0s, h1s = [], []
        for i in range(n_units):
            p = int(patient_idx[i])
            ell = pt.exp(log_ell0_h0 + a_ell_h0[p] + b_ell_h0[i])
            sig = pt.exp(log_sigma0_h0 + a_sig_h0[p] + b_sig_h0[i])
            K = _build_kernel_pt(kernel_type, taus, ell, sig)
            K = K + jitter_rel * pt.mean(pt.diag(K)) * np.eye(n_lags)
            h = gp_draw(f"h0_{i}", pt.linalg.cholesky(K))
            h0s.append(m0 + h if shared_mean else h)
            if use_h1:
                if fix_h1:
                    ell1, sig1 = pt.exp(log_ell0_h1), pt.exp(log_sigma0_h1)
                else:
                    ell1 = pt.exp(log_ell0_h1 + a_ell_h1[p] + b_ell_h1[i])
                    sig1 = pt.exp(log_sigma0_h1 + a_sig_h1[p] + b_sig_h1[i])
                K1 = _build_kernel_pt(kernel_type, taus, ell1, sig1)
                K1 = K1 + jitter_rel * pt.mean(pt.diag(K1)) * np.eye(n_lags)
                h1s.append(gp_draw(f"h1_{i}", pt.linalg.cholesky(K1)))

        h0_stack = pt.stack(h0s, axis=0)
        signal = pt.sum(S0 * h0_stack[:, None, :], axis=-1)
        if use_h1:
            signal = signal + pt.sum(S1 * pt.stack(h1s, axis=0)[:, None, :], axis=-1)
        pm.Deterministic("signal_unit0", signal[0])          # functional for R-hat/ESS
        pm.Deterministic("h0_unit0", h0_stack[0])
        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)
        pm.Normal("obs", mu=signal, sigma=sigma_obs, observed=r)
        trace = pm.sample(draws=draws, tune=tune, chains=chains, cores=cores,
                          target_accept=target_accept, random_seed=seed, progressbar=False)
    return trace


def summarize(trace):
    s = az.summary(trace, round_to=4)
    hyper = [v for v in s.index if not v.startswith(("raw_", "z_", "h0_", "h1_", "signal_", "m0"))]
    func = [v for v in s.index if v.startswith(("signal_unit0", "h0_unit0"))]
    # BFMI computed directly from the energy trace: version-proof (az.bfmi's
    # return type differs across ArviZ versions).
    en = np.asarray(trace.sample_stats["energy"].values)          # (chain, draw)
    bfmi = [float(np.sum(np.diff(e) ** 2) / len(e) / np.var(e)) for e in en]
    div = int(trace.sample_stats["diverging"].sum())
    out = {
        "n_params": int(len(s)),
        "max_rhat_all": float(s["r_hat"].max()),
        "frac_rhat_gt_1.01_all": float((s["r_hat"] > 1.01).mean()),
        "max_rhat_hyper": float(s.loc[hyper, "r_hat"].max()),
        "max_rhat_functionals": float(s.loc[func, "r_hat"].max()),
        "min_ess_bulk_functionals": float(s.loc[func, "ess_bulk"].min()),
        "bfmi_per_chain": bfmi,
        "n_divergences": div,
    }
    # Where do divergences live? Median of each tau_* at divergent vs other draws.
    div_mask = np.asarray(trace.sample_stats["diverging"].values).astype(bool)   # (chain, draw)
    loc = {}
    if div_mask.any():
        for v in trace.posterior.data_vars:
            if str(v).startswith("tau_"):
                x = np.asarray(trace.posterior[v].values)
                loc[str(v)] = {"median_divergent": float(np.median(x[div_mask])),
                               "median_other": float(np.median(x[~div_mask])) if (~div_mask).any() else None}
    out["tau_at_divergences"] = loc
    if "log_ell0_h0" in trace.posterior:
        out["log_ell0_h0_chain_means"] = trace.posterior["log_ell0_h0"].mean(("draw")).values.tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True,
                    choices=["asis", "center", "no_h1", "ncp", "shared_mean", "fix_h1", "ncp_fix_h1", "ncp_no_h1"])
    ap.add_argument("--kernel", default="matern52")
    ap.add_argument("--generator", default="Standard")
    ap.add_argument("--n_patients", type=int, default=2)
    ap.add_argument("--n_electrodes", type=int, default=2)
    ap.add_argument("--n_times", type=int, default=1000)
    ap.add_argument("--tau_max", type=float, default=200.0)
    ap.add_argument("--snr", type=float, default=5.0)
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--cores", type=int, default=None)
    ap.add_argument("--target_accept", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dt = 10.0
    n_lags = int(np.floor(a.tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt
    basis = make_raised_cosine_basis(n_basis=6, tau_max=a.tau_max, c=5.0)
    data = rg.build_dataset_from_generator(a.generator, basis, taus, a.n_patients, a.n_electrodes,
                                           a.n_times, dt, a.snr, a.seed)
    S0, S1, r = data["S0_raw"], data["S1_raw"], data["r"]
    if a.variant == "center":
        S1 = S1 - data["unit_mean_surp"][:, None, None] * S0   # (surp - m)*s = S1 - m*S0

    t0 = time.time()
    trace = build_and_sample(
        S0, S1, r, data["patient_idx"], taus, a.kernel,
        ncp=a.variant.startswith("ncp"), use_h1=(not a.variant.endswith("no_h1")),
        fix_h1=a.variant.endswith("fix_h1"), shared_mean=(a.variant == "shared_mean"),
        draws=a.draws, tune=a.tune, chains=a.chains,
        cores=(a.cores if a.cores is not None else a.chains),
        target_accept=a.target_accept, seed=a.seed,
    )
    res = summarize(trace)
    res.update(variant=a.variant, kernel=a.kernel, generator=a.generator, seed=a.seed,
               snr=a.snr, n_units=a.n_patients * a.n_electrodes, elapsed_s=time.time() - t0)
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()