# mixed-trf

Hierarchical (mixed) temporal-response-function models for how word **surprisal modulates neural responses**
(iEEG, Proix lab), plus the simulation and validation machinery used to decide what can be claimed.

    h(tau | surprisal) = h0(tau) + h1(tau) * surprisal

## Install
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[bayes,dev]"          # bayes = PyMC (Stage 2 only); Stage 1 needs only numpy/scipy/matplotlib

## One model at a time
    python -m mixed_trf models                                   # the ladder
    python -m mixed_trf stage2 --model A-pop     --mode gain --seed 0
    python -m mixed_trf stage2 --model B-pop-amp --mode gain --seed 0
    python -m mixed_trf stage2 --model B-pop-hyp --mode gain --seed 0 --patient_width_sd 0.45

| model | what it adds over the previous one |
|---|---|
| `A-pop` | raised-cosine basis; weights pooled over patients/electrodes (Model A) |
| `B-pop-amp` | GP prior on h(tau) instead of a basis; amplitude pooled; **one** shared length scale |
| `B-pop-hyp` | GP deviations whose length scale and amplitude are hierarchical (patients differ in smoothness) + population-mean GP |

All three share the same hierarchy, priors and **exact Gaussian likelihood from per-unit sufficient statistics**
(X'X, X'y, y'y), so cost does not depend on recording length. GP draws are non-centred (h = Lz).

## Two stages
* **Stage 1** (fast, no MCMC): simulate 100-256-electrode iEEG with spatial noise correlation, select responsive
  electrodes on dev data, estimate gain / shift / width of the surprisal effect (first-order or binned estimator),
  patient-level t-intervals, leave-one-patient-out. `python -m mixed_trf stage1 --help`
* **Stage 2** (Bayesian ladder above): convergence judged on *functionals* (population TRF), decomposition with
  posterior intervals, held-out-patient prediction, and learning-evolution outputs.

Also: `python -m mixed_trf testbed` (single-electrode-family ERP testbed), `summarize` (gates / verdicts),
`plot-learning` (figures below).

## Learning-evolution plots
    python -m mixed_trf plot-learning "results/*/learn*.jsonl" "results/*/ladder*.jsonl" --out figures
* `learning_convergence.png` - R-hat / ESS of h_pop vs fraction of draws (has the sampler converged?)
* `learning_data.png` - error and interval width vs minutes of data (data efficiency)
* `learning_sequential.png` - estimate and 90 % interval as patients are added (sequential updating)

## Running the big jobs
`run_local.py` is a resumable launcher for one machine (thermal guard, cooldown, nice, `--max-hours`); `scripts/slurm/*.sh`
are the Euler equivalents. **The order of work for the final run is in `docs/BIG_RUN.md`.**

    python run_local.py --status
    python run_local.py --only ieeg1 --seeds 0-9 --cpus 4 --max-hours 3
    python run_local.py --only ladder --seeds 0-4 --models A-pop,B-pop-amp,B-pop-hyp --stage2-args "--n_patients 8 --n_electrodes 4 --n_times 144000"
    python run_local.py --only learning --seeds 0-2 --models B-pop-amp

## Tests
    pytest        # stage-1 (first-order / binned / rhythm) and the summarizer reproduce the pre-refactor results bit-for-bit
                  # (timings excluded); Bayesian likelihood identity; registry checks

## Layout
    src/mixed_trf/   the package (models, estimators, simulation, core, evaluation, stage1, experiments, cli)
    run_local.py     resumable single-machine launcher          scripts/slurm/   Euler job scripts
    scripts/*.py     old script names as thin wrappers          tests/           regression tests + fixtures
    results/         stage1/ (jsonl per job), stage2/, summaries/ (text summaries), done/ (resume markers)
    legacy/          Sections 1-10 (recovery grid, identifiability, 7-model comparison, old Models A/B); run from inside legacy/
    docs/            analysis_plan.md (pre-specified gates), BIG_RUN.md (what to run, in order), FILES.md
