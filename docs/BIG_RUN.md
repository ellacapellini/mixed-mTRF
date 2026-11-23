# BIG FINAL run — order of work

Machine split: Mac (M5, 24 GB, overheats) takes Stage 1; Euler takes Stage 2 (long MCMC). Everything is resumable:
re-run the same command and finished jobs are skipped (`results/done/`). Check with `python run_local.py --status ...`.

## 0. Set up and verify (10 min)
    python3.12 -m venv .venv && source .venv/bin/activate
    pip install -e ".[bayes,dev]"
    pytest                                   # 7 passed = the package reproduces the old results bit-for-bit
    python run_local.py --only smoke         # Bayesian likelihood identity + tiny fit

## 1. DONE — Stage 1 main result (fresh seeds 100–199, binned estimator)
`results/stage1/ieeg1_fin_*` (2400 rows) and `results/summaries/summary_fin.txt`. Re-derive any time:
    python -m mixed_trf summarize --glob 'results/stage1/ieeg1_fin_*_seed*.jsonl' --table
(the refactored summarizer reproduces `summary_fin.txt` byte for byte.)

## 2. Stage 1 power map: effect size (failed before: the old script had no --effect_scale)
~1.3 min per job, 100 jobs per scale ≈ 45 min on 3 cores.
    python run_local.py --only ieeg1 --tag _es0.5 --seeds 200-249 --bands lfp,hg --snrs 0.03 --ieeg-args "--estimator binned --effect_scale 0.5" --cpus 3 --max-hours 2
    python run_local.py --only ieeg1 --tag _es2   --seeds 200-249 --bands lfp,hg --snrs 0.03 --ieeg-args "--estimator binned --effect_scale 2"   --cpus 3 --max-hours 2
    python run_local.py --only ieeg1 --tag _es3   --seeds 200-249 --bands lfp,hg --snrs 0.03 --ieeg-args "--estimator binned --effect_scale 3"   --cpus 3 --max-hours 2
    python -m mixed_trf summarize --glob 'results/stage1/ieeg1_es2_*_seed*.jsonl' --table

## 3. Stage 1 sensitivity: rhythm/oddball paradigm and 256 electrodes
    python run_local.py --only ieeg1 --tag _rhy  --seeds 300-319 --bands lfp --snrs 0.03,0.2 --ieeg-args "--estimator binned --rhythm isochronous --violation_p 0.1" --cpus 3
    python run_local.py --only ieeg1 --tag _e256 --seeds 300-319 --bands lfp,hg --snrs 0.03 --ieeg-mem-gb 6 --ieeg-args "--estimator binned --electrodes 256" --cpus 2

## 4. Stage 2 — Bayesian ladder (Euler)
Calibrate ONE job first, then scale; B-pop-hyp is ~7x slower per iteration than the amplitude models.
    sbatch --export=ALL,MODEL=A-pop,MODE=gain,SEED=0,EXTRA="--n_patients 8 --n_electrodes 4 --n_times 144000 --holdout 7" scripts/slurm/stage2.sh
    squeue -u $USER ; tail -f logs/stage2_<jobid>.out
Then models A-pop, B-pop-amp, B-pop-hyp x modes null,gain,latency,width x seeds 0–4 (loop the sbatch line, or on the Mac:
`python run_local.py --only ladder --seeds 0-4 --stage2-args "--n_patients 8 --n_electrodes 4 --n_times 144000"`).
Smoothness recovery (can the hierarchy find known patient differences?):
    python run_local.py --only smoothness
Learning-evolution inputs (data efficiency + sequential updating):
    python run_local.py --only learning --seeds 0-2 --models A-pop,B-pop-amp --stage2-args "--n_patients 8 --n_electrodes 4"
    python -m mixed_trf plot-learning "results/stage2/*.jsonl" --out results/figures --mode gain
Read first: divergences, R-hat/ESS of h_pop (not of raw parameters), then bias of est vs truth. B-pop-amp showed an upward gain
bias in the small sandbox test: check it at full size before drawing any conclusion about GP vs basis.

## 5. NOT IMPLEMENTED YET (needs data from you)
* ERP CORE positive control (MMN / P3 oddball, scalp EEG, OSF): needs the BIDS download + a loader in `src/mixed_trf/data/`.
* Proix iEEG with a dev/confirmatory split: needs the data format (sampling rate, word onsets, surprisal file). Pre-register
  the electrode selection on dev data exactly as in `docs/analysis_plan.md` before touching the confirmatory part.
