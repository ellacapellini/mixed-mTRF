"""
identifiability_study.py
=========================

Answers the actual scientific question Section 6 of the equations doc
needs before the real-data comparison can be trusted: not "which model
wins on this one dataset" but "at what noise level and trial count does
the DESIGN even carry enough information to distinguish these models at
all" -- a proper simulation-based identifiability map, in the sense of
Wilson & Collins (2019) "model recovery": generate from model X many
times under a fixed experimental condition, fit every candidate model
to each simulated dataset, apply ONE consistent selection criterion
across candidates, and report P(the criterion picks X) as an actual
Monte Carlo proportion with a confidence interval -- not a single r
number from one lucky or unlucky draw.

THIS FILE covers the Matern-vs-SE (kernel family) question specifically,
swept over SNR x n_trials, since that's the comparison the target
result explicitly names ("Matern vs SE becomes reliably identifiable
above approximately X SNR and Y trials"). The C1-vs-A-B2 (restricted
vs flexible surprisal) question needs the SAME machinery with a
different generator pair and a different swept axis (effect size,
i.e. how far kappa/beta_j is from the null) -- see
"EXTENDING TO THE C1/C2 QUESTION" at the bottom of this docstring for
exactly what to change; it's a small diff, not a rewrite.

DECISION STATISTIC: by-unit LOO (model_selection.loo_by_unit), computed
from a REAL NUTS posterior -- this is the fit_mode="nuts" path, and it
is what the actual published numbers must come from, run at ETH scale
(fit_mode="nuts", full draws/tune/chains, many repeats). There is also
a fit_mode="point_estimate" path using GPmTRF's non-hierarchical MLE
point estimate and a held-out Gaussian log-likelihood proxy
(model_selection.held_out_gaussian_loglik_by_unit) -- this exists
SOLELY to validate that this harness itself (data generation, fitting
dispatch, win-tallying, checkpointing, aggregation, plotting) has no
bugs, using cheap/fast settings on a laptop. A result computed only in
point_estimate mode is not a publishable claim about identifiability;
it is a claim about "this code ran without crashing."

CLUSTER-READY BY DESIGN: every (true_kernel, snr, n_times, repeat)
combination is one independent task. enumerate_tasks() returns the
full flat list; --task-id/--n-tasks (or the task_id/n_tasks kwargs)
select task_id, task_id+n_tasks, task_id+2*n_tasks, ... from that list,
so this maps directly onto a Slurm/SGE array job: submit with
--n-tasks equal to the array size, one job per --task-id. Every
completed task is checkpointed to its own line in results_path
(JSON-lines, not one big JSON dict) specifically so concurrent array-
job workers can append without a shared-file write race; aggregation
reads the whole file back in at the end.

PARAMETER TRADE-OFF / "SUCCESSFUL DISGUISE" DIAGNOSTIC: for every task,
in addition to the win/lose tally, this also records both fitted
kernels' recovered (length_scale, output_scale) AND the compensation
score (model_selection.compensation_score) between each fitted kernel's
implied covariance curve and the TRUE generator's curve. A kernel can
"win" the LOO comparison by genuinely being the better-fitting kernel,
or by successfully mimicking the true kernel's covariance at the lags
actually present in the data despite substantially different (ell,
sigma) individually -- these are different findings and the aggregate
script reports both, not just the win rate.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass, asdict

import numpy as np

from basis.basis_functions import make_raised_cosine_basis
from recovery_grid import build_dataset_from_generator, split_train_test
from models.gp_mtrf import GPmTRF
from utils.kernels import Matern52, SquaredExponential
from models.bayesian_gp_trf import fit_bayesian_gp_trf
from model_selection import (
    loo_by_unit, held_out_gaussian_loglik_by_unit, covariance_curve, compensation_score,
)

KERNEL_TYPES = ["matern52", "squared_exponential"]
_KERNEL_CLS = {"matern52": Matern52, "squared_exponential": SquaredExponential}


# ============================================================================
# Task enumeration -- the cluster-array-job interface
# ============================================================================

@dataclass(frozen=True)
class Task:
    true_kernel: str
    snr_target: float
    n_times: int
    repeat: int

    @property
    def key(self) -> str:
        return f"kernel={self.true_kernel}|snr={self.snr_target}|n_times={self.n_times}|rep={self.repeat}"


def enumerate_tasks(snr_grid: list[float], n_times_grid: list[int], n_repeats: int) -> list[Task]:
    tasks = []
    for true_kernel in KERNEL_TYPES:
        for snr in snr_grid:
            for n_times in n_times_grid:
                for rep in range(n_repeats):
                    tasks.append(Task(true_kernel, snr, n_times, rep))
    return tasks


def select_tasks_for_this_job(tasks: list[Task], task_id: int | None, n_tasks: int | None) -> list[Task]:
    """Slurm/SGE array-job slicing: this worker (task_id of n_tasks) handles
    tasks[task_id::n_tasks]. Both None (the default) -> run everything, for
    single-machine / local runs."""
    if task_id is None or n_tasks is None:
        return tasks
    if not (0 <= task_id < n_tasks):
        raise ValueError(f"task_id ({task_id}) must be in [0, n_tasks={n_tasks})")
    return tasks[task_id::n_tasks]


# ============================================================================
# One task: generate from `true_kernel`, fit BOTH candidate kernels, compare
# ============================================================================

def run_one_task(
    task: Task,
    fit_mode: str,                  # "nuts" (real, ETH-scale) or "point_estimate" (fast local debug only)
    n_patients: int,
    n_electrodes: int,
    n_basis: int,
    tau_max: float,
    dt: float,
    nuts_kwargs: dict | None = None,
) -> dict:
    if fit_mode not in ("nuts", "point_estimate"):
        raise ValueError(f"fit_mode must be 'nuts' or 'point_estimate', got {fit_mode!r}")

    basis = make_raised_cosine_basis(n_basis=n_basis, tau_max=tau_max, c=5.0)
    n_lags = int(np.floor(tau_max / dt)) + 1
    taus = np.arange(n_lags) * dt

    generator_name = f"B-{'Matern' if task.true_kernel == 'matern52' else 'SE'}-B1"
    data = build_dataset_from_generator(
        generator_name, basis, taus, n_patients, n_electrodes, task.n_times, dt,
        snr_target=task.snr_target,
        # seed keyed on the task itself, not on list position -- see
        # recovery_grid.py's own seed-offset bug (fixed after being caught)
        # for why position-based seeding is a real, previously-hit hazard.
        seed=hash(task.key) % (2**31),
    )
    train, test = split_train_test(data)
    patient_idx = data["patient_idx"]
    unit_mean_surp = data["unit_mean_surp"]

    result = {"task": asdict(task), "fit_mode": fit_mode, "per_kernel": {}}

    for candidate_kernel in KERNEL_TYPES:
        t0 = time.time()
        if fit_mode == "point_estimate":
            elpd, recovered = _fit_point_estimate(candidate_kernel, train, test, taus)
        else:
            elpd, recovered = _fit_nuts(
                candidate_kernel, train, test, patient_idx, taus, unit_mean_surp,
                nuts_kwargs or {},
            )
        elapsed = time.time() - t0

        # compensation diagnostic: does this kernel's fitted covariance curve
        # resemble the TRUE generator's curve, regardless of whether its own
        # (ell, sigma) match the true ones?
        true_curve = covariance_curve(task.true_kernel, ell=100.0, sigma=2.0, taus=taus)  # h0's true params, ground_truth_gp.py's make_stage3_full_modulation defaults
        fit_curve = covariance_curve(candidate_kernel, ell=recovered["ell"], sigma=recovered["sigma"], taus=taus)
        comp_score = compensation_score(true_curve, fit_curve)

        result["per_kernel"][candidate_kernel] = {
            "elpd": elpd, "recovered": recovered,
            "compensation_score": comp_score, "elapsed_s": elapsed,
        }

    elpds = {k: v["elpd"] for k, v in result["per_kernel"].items()}
    result["winner"] = max(elpds, key=elpds.get)
    result["correct"] = result["winner"] == task.true_kernel
    return result


def _fit_point_estimate(candidate_kernel: str, train, test, taus):
    """FAST LOCAL DEBUG PATH ONLY -- see module docstring. Fits each unit's
    GP hyperparameters independently via empirical-Bayes MLE (GPmTRF), then
    scores held-out data with a homoskedastic Gaussian likelihood, summed
    by-unit (same exchangeability logic as loo_by_unit, just without a
    posterior)."""
    n_units = train["S0_raw"].shape[0]
    kernel_cls = _KERNEL_CLS[candidate_kernel]
    y_true_by_unit, y_pred_by_unit = [], []
    ells, sigmas, noise_vars = [], [], []

    for i in range(n_units):
        X_train = np.concatenate([train["S0_raw"][i], train["S1_raw"][i]], axis=1)
        y_train = train["r"][i]
        X_test = np.concatenate([test["S0_raw"][i], test["S1_raw"][i]], axis=1)
        y_test = test["r"][i]

        k0, k1 = kernel_cls(1.0, 50.0), kernel_cls(1.0, 50.0)
        gp = GPmTRF(lags=taus, kernels=[k0, k1], noise_var=1.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.optimise_hyperparams(X_train, y_train, verbose=False)

        y_pred = gp.predict(X_test)
        y_true_by_unit.append(y_test)
        y_pred_by_unit.append(y_pred)
        ells.append(k0.length_scale)
        sigmas.append(k0.output_scale)
        noise_vars.append(gp.noise_var)

    mean_noise_var = float(np.mean(noise_vars))
    elpd = held_out_gaussian_loglik_by_unit(y_true_by_unit, y_pred_by_unit, mean_noise_var)
    recovered = {"ell": float(np.mean(ells)), "sigma": float(np.mean(sigmas))}
    return elpd, recovered


def _fit_nuts(candidate_kernel: str, train, test, patient_idx, taus, unit_mean_surp, nuts_kwargs):
    """THE REAL PATH -- fits the full hierarchical Bayesian GP model via
    NUTS (bayesian_gp_trf.py) and scores it with the correct by-unit LOO
    (model_selection.loo_by_unit), not held-out r. This is what must run
    at ETH scale for the published identifiability map."""
    S0_all = np.concatenate([train["S0_raw"], test["S0_raw"]], axis=1)
    S1_all = np.concatenate([train["S1_raw"], test["S1_raw"]], axis=1)
    r_all = np.concatenate([train["r"], test["r"]], axis=1)
    # NOTE: LOO needs held-in-sample log-likelihood (it estimates held-out
    # performance FROM the full posterior via importance sampling), so this
    # fits on the FULL (train+test) data, unlike the r-based comparisons
    # elsewhere in this codebase which need a genuine temporal holdout.
    # Both are legitimate; they answer different questions (LOO: "would a
    # fair comparison pick this model", r: "how well does it predict new
    # data") and this file is answering the first one.

    result = fit_bayesian_gp_trf(
        S0_all, S1_all, r_all, patient_idx, taus, unit_mean_surp,
        kernel_type=candidate_kernel, surprisal_in_kernel=False,
        **nuts_kwargs,
    )
    loo_result = loo_by_unit(result.trace)

    ell_mean = float(np.exp(result.trace.posterior["log_ell0_h0"]).mean())
    sigma_mean = float(np.exp(result.trace.posterior["log_sigma0_h0"]).mean())
    recovered = {
        "ell": ell_mean, "sigma": sigma_mean,
        "max_pareto_k": loo_result.max_pareto_k, "n_bad_pareto_k": loo_result.n_bad_pareto_k,
        "elpd_waic": loo_result.elpd_waic,
    }
    return loo_result.elpd_loo, recovered


# ============================================================================
# Runner with checkpointing (JSON-lines, append-only -- safe for concurrent
# array-job workers writing to the SAME results_path)
# ============================================================================

def run_identifiability_sweep(
    snr_grid: list[float],
    n_times_grid: list[int],
    n_repeats: int,
    fit_mode: str = "point_estimate",
    n_patients: int = 3,
    n_electrodes: int = 3,
    n_basis: int = 6,
    tau_max: float = 100.0,
    dt: float = 10.0,
    nuts_kwargs: dict | None = None,
    results_path: str = "figures/identifiability_kernel_family.jsonl",
    task_id: int | None = None,
    n_tasks: int | None = None,
    progress_only: bool = False,
):
    all_tasks = enumerate_tasks(snr_grid, n_times_grid, n_repeats)
    my_tasks = select_tasks_for_this_job(all_tasks, task_id, n_tasks)

    done_keys = set()
    if os.path.exists(results_path):
        with open(results_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                t = rec["task"]
                done_keys.add(Task(t["true_kernel"], t["snr_target"], t["n_times"], t["repeat"]).key)

    if progress_only:
        total = len(all_tasks)
        done_in_full_grid = sum(1 for t in all_tasks if t.key in done_keys)
        print(f"Progress: {done_in_full_grid}/{total} tasks completed "
              f"({100*done_in_full_grid/total:.1f}%) in {results_path}")
        pending = [t for t in all_tasks if t.key not in done_keys]
        if pending:
            print(f"Next {min(5, len(pending))} pending task(s):")
            for t in pending[:5]:
                print(f"  {t.key}")
        return results_path

    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    print(f"Identifiability sweep (kernel family): {len(my_tasks)}/{len(all_tasks)} tasks assigned "
          f"to this worker (task_id={task_id}, n_tasks={n_tasks}), fit_mode={fit_mode!r}")
    print(f"Model size: n_patients={n_patients}, n_electrodes={n_electrodes} "
          f"({n_patients*n_electrodes} units), n_basis={n_basis}, tau_max={tau_max}")
    print(f"{len(done_keys)} already completed (any worker) -- skipping those.")

    n_run = 0
    with open(results_path, "a") as f:
        for i, task in enumerate(my_tasks):
            if task.key in done_keys:
                continue
            print(f"[{i+1}/{len(my_tasks)}] {task.key} ...", flush=True)
            t0 = time.time()
            result = run_one_task(
                task, fit_mode, n_patients, n_electrodes, n_basis, tau_max, dt, nuts_kwargs,
            )
            f.write(json.dumps(result) + "\n")
            f.flush()
            n_run += 1
            print(f"    winner={result['winner']} (true={task.true_kernel}, "
                  f"correct={result['correct']}), task took {time.time()-t0:.0f}s")

    print(f"\nDone. {n_run} task(s) run this session. Results in {results_path} (JSON-lines).")
    return results_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fit-mode", choices=["nuts", "point_estimate"], default="point_estimate")
    parser.add_argument("--snr-grid", type=float, nargs="+", default=[2.0, 5.0, 10.0])
    parser.add_argument("--n-times-grid", type=int, nargs="+", default=[300, 800])
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--n-patients", type=int, default=3)
    parser.add_argument("--n-electrodes", type=int, default=3)
    parser.add_argument("--n-basis", type=int, default=6)
    parser.add_argument("--tau-max", type=float, default=100.0)
    parser.add_argument("--results-path", default="figures/identifiability_kernel_family.jsonl")
    parser.add_argument("--task-id", type=int, default=None)
    parser.add_argument("--n-tasks", type=int, default=None)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--progress", action="store_true",
                         help="Print completion status against this grid and exit -- no fitting.")
    args = parser.parse_args()

    nuts_kwargs = dict(draws=args.draws, tune=args.tune, chains=args.chains,
                        target_accept=0.99, max_treedepth=14, seed=0)

    run_identifiability_sweep(
        snr_grid=args.snr_grid, n_times_grid=args.n_times_grid, n_repeats=args.n_repeats,
        fit_mode=args.fit_mode, n_patients=args.n_patients, n_electrodes=args.n_electrodes,
        n_basis=args.n_basis, tau_max=args.tau_max,
        nuts_kwargs=nuts_kwargs, results_path=args.results_path,
        task_id=args.task_id, n_tasks=args.n_tasks, progress_only=args.progress,
    )