"""
bayesian_gp_trf.py
===================

Models 4-7 of the seven-model comparison -- Model B's hierarchical
Bayesian GP fit via NUTS in PyMC, parameterised by kernel_type
("matern52" / "squared_exponential") and surprisal_in_kernel
(False=B1, True=B2). One implementation, four configurations, same
"orthogonal axis, not code fork" principle used everywhere else in
this project (kernels.py, hierarchical_kernel.py).

DESIGN DECISIONS MADE UNDER TIME PRESSURE, flagged explicitly rather
than hidden, given the 3 Sept presentation deadline:

1. NO LKJ correlation between a hyperparameter's patient- and
   electrode-level deviations. The equations doc doesn't specify LKJ
   here at all (unlike Model A's random slopes), but I want to note
   why I'm not adding it even as an option: bayesian_mixed_trf.py
   showed LKJ-correlated random effects need real replication to be
   identifiable and caused serious convergence problems with only a
   handful of patients/electrodes. Independent (non-centered) log-
   scale deviations, matching hierarchical_kernel.py's own numpy
   design exactly, avoids that risk entirely.

2. B2's kernel-hyperparameter surprisal-dependence uses each unit's
   MEAN surprisal as a single representative scalar, not the full
   per-timepoint surprisal trace. The equations doc (Section 7.4)
   writes log ell(surp) as a function of a scalar surp -- but
   surprisal genuinely varies within a unit's own recording. Handling
   that properly (kernel hyperparameters that vary continuously
   within one recording) is a real, unresolved modelling question
   flagged already in ground_truth_gp.py's docstring. Given the
   deadline, using each unit's mean surprisal is a defensible,
   explainable simplification that still produces a genuine B1-vs-B2
   distinction -- not a permanent design choice.

3. A separate PyTensor implementation of the Matern52/SquaredExponential
   formulas, not an import from kernels.py -- kernels.py's versions are
   plain NumPy and don't work inside a PyMC model (need PyTensor ops for
   autodiff). This is the exact "two parallel implementations of the
   same formula" situation flagged in hierarchical_kernel.py's docstring.
   A parity test against kernels.py's NumPy versions is included below
   specifically to catch the two implementations silently drifting apart.

BUG FOUND AND FIXED (recovery-grid investigation, after the diagonal
cells showed squared-exponential specifically failing to recover its
OWN known ground truth even at chains=4): item 3's parity test caught
that the two parallel kernel implementations HAD silently drifted
apart, just not in the kernel FORMULA itself. _matern52_pt and
_squared_exponential_pt used to square their `sigma` argument before
using it as the variance multiplier (`sigma**2 * exp(...)`), while
kernels.py's Matern52/SquaredExponential and hierarchical_kernel.py's
effective_covariance (which ground_truth_gp.py calls to build the
synthetic ground truth these models are scored against) both use their
`output_scale` argument DIRECTLY, unsquared. Since log_sigma0_h0's own
prior here is centred at np.log(sigma0_prior), mirroring hierarchical_
kernel.py's log_sigma_0 composition exactly, exp(log_sigma0_h0 + ...)
was always meant to BE output_scale directly, not a standard deviation
needing a further squaring -- so this fit was implicitly modelling
variance = output_scale^2 as if it were output_scale itself, a genuine
factor that compounds with any hyperparameter deviation. The self-test
below had actually already caught the mismatch (note the pre-
compensating sqrt(2.0) it fed in) but that compensation was never
propagated to the actual hierarchical prior it sits next to, so the
parity check passed while the production model stayed wrong -- a good
reminder that a passing unit test only covers what it explicitly
checks. This bug applied identically to BOTH kernel types, so it does
NOT by itself explain why SE specifically failed self-recovery while
Matern didn't -- but an incorrect, systematically inflated implied
variance is exactly the kind of thing that would push an already-
fragile kernel (SE, near-singular at short lengthscales on this grid)
further into bad conditioning while a more robust one (Matern) mostly
shrugs it off. Worth re-running the diagonal cells after this fix
before concluding anything further about SE specifically.
"""

from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from dataclasses import dataclass


def _matern52_pt(taus, ell, sigma):
    """sigma is output_scale (variance multiplier) DIRECTLY, matching
    kernels.py's Matern52/SquaredExponential convention and
    hierarchical_kernel.py's own documented semantics (log_sigma_0 ->
    exp -> used as output_scale, no further squaring). NOT a standard
    deviation to be squared -- see module docstring "Design decision"
    note on the sigma/output_scale units bug this replaces."""
    diff = taus[:, None] - taus[None, :]
    r = pt.abs(diff) / ell
    sqrt5_r = pt.sqrt(5.0) * r
    return sigma * (1 + sqrt5_r + (5.0 / 3.0) * r**2) * pt.exp(-sqrt5_r)


def _squared_exponential_pt(taus, ell, sigma):
    """See _matern52_pt's docstring: sigma is output_scale directly."""
    diff = taus[:, None] - taus[None, :]
    return sigma * pt.exp(-(diff**2) / (2 * ell**2))


def _build_kernel_pt(kernel_type, taus, ell, sigma):
    if kernel_type == "matern52":
        return _matern52_pt(taus, ell, sigma)
    elif kernel_type == "squared_exponential":
        return _squared_exponential_pt(taus, ell, sigma)
    else:
        raise ValueError(f"unknown kernel_type {kernel_type!r}")


@dataclass
class BayesianGPTRFResult:
    trace: object
    kernel_type: str
    surprisal_in_kernel: bool
    n_units: int
    model: pm.Model


def fit_bayesian_gp_trf(
    S0_all: np.ndarray,
    S1_all: np.ndarray,
    r_all: np.ndarray,
    patient_idx: np.ndarray,
    taus: np.ndarray,
    unit_mean_surp: np.ndarray,
    kernel_type: str = "matern52",
    surprisal_in_kernel: bool = False,
    ell0_prior: float = 50.0,
    sigma0_prior: float = 1.0,
    draws: int = 200,
    tune: int = 300,
    chains: int = 2,
    cores: int | None = None,
    target_accept: float = 0.9,
    max_treedepth: int = 10,
    seed: int = 0,
    jitter_rel: float = 1e-6,
) -> BayesianGPTRFResult:
    """
    S0_all, S1_all : (n_units, n_times, n_lags) raw-lag design matrices
        (build_raw_lag_design_matrix's output), NOT basis-projected.
    r_all : (n_units, n_times)
    patient_idx : (n_units,) patient id per unit
    taus : (n_lags,) the lag grid.
    unit_mean_surp : (n_units,) mean surprisal for each unit's
        recording -- used only when surprisal_in_kernel=True (B2).
    surprisal_in_kernel : False = B1 (kernel fixed w.r.t. surprisal),
        True = B2 (kernel hyperparameters shift with unit_mean_surp).
    jitter_rel : nugget added to each unit's covariance diagonal,
        expressed as a FRACTION of that unit's own kernel diagonal
        (= output_scale, since both kernels are stationary), not an
        absolute constant. A fixed nugget is negligible for a unit
        with large fitted sigma^2 and disproportionately large for a
        unit with small sigma^2 -- scaling to each unit's own diagonal
        keeps the relative conditioning improvement comparable across
        units/patients regardless of their fitted output scale.
    cores : number of chains to run in PARALLEL. None (default) ->
        cores=chains, i.e. every chain gets its own process. This
        matters a lot on a shared multi-core node (e.g. ETH's 128-CPU
        large-memory node) -- the previous hardcoded cores=1 forced
        every chain to run sequentially even with dozens of idle
        cores sitting right there, silently wasting almost the entire
        compute allocation on every single fit.
    """
    patient_idx = np.asarray(patient_idx, dtype="int64")
    n_units, n_times, n_lags = S0_all.shape
    n_patients = int(patient_idx.max()) + 1
    taus_arr = np.asarray(taus, dtype="float64")

    with pm.Model() as model:
        # --- global (population-level) hyperparameters, one pair each
        # for h0 and h1's kernel ---
        log_ell0_h0 = pm.Normal("log_ell0_h0", np.log(ell0_prior), 0.3)
        log_sigma0_h0 = pm.Normal("log_sigma0_h0", np.log(sigma0_prior), 0.3)
        log_ell0_h1 = pm.Normal("log_ell0_h1", np.log(ell0_prior), 0.3)
        log_sigma0_h1 = pm.Normal("log_sigma0_h1", np.log(sigma0_prior), 0.3)

        if surprisal_in_kernel:
            delta = pm.Normal("delta", 0.0, 0.3)   # B2: surprisal shifts h1's lengthscale
            gamma = pm.Normal("gamma", 0.0, 0.3)   # B2: surprisal shifts h1's output scale

        # --- patient-level deviations, independent (no LKJ), non-centered,
        # one pair per hyperparameter, matching hierarchical_kernel.py's
        # own diagonal design ---
        def patient_deviation(name):
            tau = pm.HalfCauchy(f"tau_p_{name}", beta=0.3)
            raw = pm.Normal(f"raw_p_{name}", 0.0, 1.0, shape=n_patients)
            return tau * raw

        a_ell_h0 = patient_deviation("ell_h0")
        a_sigma_h0 = patient_deviation("sigma_h0")
        a_ell_h1 = patient_deviation("ell_h1")
        a_sigma_h1 = patient_deviation("sigma_h1")

        # --- electrode-level deviations, same pattern, one per unit ---
        def electrode_deviation(name):
            tau = pm.HalfCauchy(f"tau_e_{name}", beta=0.3)
            raw = pm.Normal(f"raw_e_{name}", 0.0, 1.0, shape=n_units)
            return tau * raw

        b_ell_h0 = electrode_deviation("ell_h0")
        b_sigma_h0 = electrode_deviation("sigma_h0")
        b_ell_h1 = electrode_deviation("ell_h1")
        b_sigma_h1 = electrode_deviation("sigma_h1")

        # --- per-unit effective hyperparameters, then per-unit GP draws.
        # Genuinely different covariance matrix per unit, so this has to
        # be a Python loop over units, not a vectorised batch operation --
        # each unit needs its own Cholesky decomposition. ---
        h0_list, h1_list = [], []
        for i in range(n_units):
            p = int(patient_idx[i])

            ell_h0_i = pt.exp(log_ell0_h0 + a_ell_h0[p] + b_ell_h0[i])
            sigma_h0_i = pt.exp(log_sigma0_h0 + a_sigma_h0[p] + b_sigma_h0[i])
            K0_i = _build_kernel_pt(kernel_type, taus_arr, ell_h0_i, sigma_h0_i)
            jitter0_i = jitter_rel * pt.mean(pt.diag(K0_i))
            K0_i = K0_i + jitter0_i * np.eye(n_lags)
            L0_i = pt.linalg.cholesky(K0_i)
            h0_i = pm.MvNormal(f"h0_{i}", mu=np.zeros(n_lags), chol=L0_i)
            h0_list.append(h0_i)

            log_ell_h1_i = log_ell0_h1 + a_ell_h1[p] + b_ell_h1[i]
            log_sigma_h1_i = log_sigma0_h1 + a_sigma_h1[p] + b_sigma_h1[i]
            if surprisal_in_kernel:
                log_ell_h1_i = log_ell_h1_i + delta * unit_mean_surp[i]
                log_sigma_h1_i = log_sigma_h1_i + gamma * unit_mean_surp[i]
            ell_h1_i = pt.exp(log_ell_h1_i)
            sigma_h1_i = pt.exp(log_sigma_h1_i)
            K1_i = _build_kernel_pt(kernel_type, taus_arr, ell_h1_i, sigma_h1_i)
            jitter1_i = jitter_rel * pt.mean(pt.diag(K1_i))
            K1_i = K1_i + jitter1_i * np.eye(n_lags)
            L1_i = pt.linalg.cholesky(K1_i)
            h1_i = pm.MvNormal(f"h1_{i}", mu=np.zeros(n_lags), chol=L1_i)
            h1_list.append(h1_i)

        h0_stack = pt.stack(h0_list, axis=0)  # (n_units, n_lags)
        h1_stack = pt.stack(h1_list, axis=0)

        signal = pt.sum(S0_all * h0_stack[:, None, :], axis=-1) \
            + pt.sum(S1_all * h1_stack[:, None, :], axis=-1)   # (n_units, n_times)

        sigma_obs = pm.HalfCauchy("sigma_obs", beta=1.0)
        pm.Normal("obs", mu=signal, sigma=sigma_obs, observed=r_all)

        trace = pm.sample(
            draws=draws, tune=tune, chains=chains, cores=(cores if cores is not None else chains),
            target_accept=target_accept,
            max_treedepth=max_treedepth, random_seed=seed, progressbar=False,
        )
        # NOTE: log-likelihood is computed per (unit, timepoint) here, but
        # WAIC/LOO must NOT be run on that directly -- see model_selection.py's
        # module docstring for why (temporal autocorrelation within a unit
        # violates LOO's exchangeability assumption). Use
        # model_selection.loo_by_unit(trace) instead of az.loo(trace) directly.
        pm.compute_log_likelihood(trace, model=model)

    return BayesianGPTRFResult(
        trace=trace, kernel_type=kernel_type, surprisal_in_kernel=surprisal_in_kernel,
        n_units=n_units, model=model,
    )


# --------------------------------------------------------------------------
# Self-tests. Run this file directly: `python -m models.bayesian_gp_trf`
# Deliberately small/fast -- presentation-track check, not exhaustive
# validation. Gates on r_hat, the more meaningful diagnostic, same
# calibration decision made in bayesian_mixed_trf.py.
# --------------------------------------------------------------------------
def _self_test() -> None:
    import arviz as az
    from utils.kernels import Matern52 as Matern52_np
    from utils.design_matrix import build_raw_lag_design_matrix, expand_word_level_to_samples
    from data.simulate import make_word_stream

    # --- Parity check: PyTensor kernel matches kernels.py's NumPy version,
    # so the two parallel implementations haven't silently drifted apart ---
    taus_check = np.linspace(0, 100, 10)
    K_np = Matern52_np(output_scale=2.0, length_scale=30.0)(taus_check, taus_check)
    ell_pt = pt.as_tensor_variable(30.0)
    sigma_pt = pt.as_tensor_variable(2.0)  # output_scale directly, matches kernels.py -- no squaring anymore
    K_pt = _matern52_pt(taus_check, ell_pt, sigma_pt).eval()
    assert np.allclose(K_np, K_pt, atol=1e-8), "PyTensor Matern52 doesn't match kernels.py's NumPy version"
    print("bayesian_gp_trf.py: PyTensor/NumPy kernel parity check passed.")

    # --- Same parity check for SquaredExponential -- this one was MISSING
    # entirely until now, which is exactly how the sigma-squaring bug above
    # went unnoticed for this kernel specifically: only Matern52 had ever
    # been checked against kernels.py's NumPy version ---
    from utils.kernels import SquaredExponential as SquaredExponential_np
    K_np_se = SquaredExponential_np(output_scale=2.0, length_scale=30.0)(taus_check, taus_check)
    K_pt_se = _squared_exponential_pt(taus_check, ell_pt, sigma_pt).eval()
    assert np.allclose(K_np_se, K_pt_se, atol=1e-8), "PyTensor SquaredExponential doesn't match kernels.py's NumPy version"
    print("bayesian_gp_trf.py: PyTensor/NumPy SquaredExponential parity check passed (previously untested).")

    # --- Small synthetic fit: 3 patients x 2 electrodes, tiny lag grid ---
    rng = np.random.default_rng(0)
    n_lags = 10
    taus = np.linspace(0, 90, n_lags)
    dt = 10.0
    n_times = 600
    n_patients, n_electrodes = 3, 2
    n_units = n_patients * n_electrodes

    true_kernel = Matern52_np(output_scale=1.5, length_scale=25.0)
    K_true = true_kernel(taus, taus) + 1e-6 * np.eye(n_lags)

    S0_all, S1_all, r_all, patient_idx, unit_mean_surp = [], [], [], [], []
    for p in range(n_patients):
        for e in range(n_electrodes):
            stimulus = rng.normal(0, 1, size=n_times)
            word_onsets, word_surp = make_word_stream(n_times, dt, words_per_second=3.0, rng=rng)
            surp_at_t = expand_word_level_to_samples(n_times, dt, word_onsets, word_surp)
            design = build_raw_lag_design_matrix(stimulus, surp_at_t, taus, dt)

            h0_true = rng.multivariate_normal(np.zeros(n_lags), K_true)
            h1_true = np.zeros(n_lags)  # stage-1-style: no surprisal effect, simplest recoverable case
            signal = design.S0 @ h0_true + design.S1 @ h1_true
            noise = rng.normal(0, signal.std() / 6.0, size=n_times)

            S0_all.append(design.S0)
            S1_all.append(design.S1)
            r_all.append(signal + noise)
            patient_idx.append(p)
            unit_mean_surp.append(float(word_surp.mean()))

    S0_all, S1_all, r_all = np.array(S0_all), np.array(S1_all), np.array(r_all)
    patient_idx = np.array(patient_idx)
    unit_mean_surp = np.array(unit_mean_surp)

    for kernel_type in ["matern52", "squared_exponential"]:
        for surprisal_in_kernel, label in [(False, "B1"), (True, "B2")]:
            result = fit_bayesian_gp_trf(
                S0_all, S1_all, r_all, patient_idx, taus, unit_mean_surp,
                kernel_type=kernel_type, surprisal_in_kernel=surprisal_in_kernel,
                draws=100, tune=150, chains=2, target_accept=0.9, max_treedepth=10, seed=1,
            )
            var_names = ["log_ell0_h0", "log_sigma0_h0", "log_ell0_h1", "log_sigma0_h1", "sigma_obs"]
            if surprisal_in_kernel:
                var_names += ["delta", "gamma"]
            summary = az.summary(result.trace, var_names=var_names)
            max_r_hat = float(summary["r_hat"].max())
            assert max_r_hat < 1.1, (
                f"[{kernel_type}, {label}] r_hat too high: {max_r_hat:.3f}\n{summary}"
            )
            print(f"bayesian_gp_trf.py [{kernel_type}, {label}]: fit completed, "
                  f"max r_hat={max_r_hat:.3f}.")

    print("bayesian_gp_trf.py: all self-tests passed (models 4-7 of 7).")
    print("NOTE: self-test uses small draws/tune/chains/lags for speed. "
          "A real presentation run should use more of each -- see "
          "fit_bayesian_gp_trf's default arguments.")


if __name__ == "__main__":
    _self_test()