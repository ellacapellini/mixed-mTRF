#!/bin/bash
# submit_pilot_diagonal.sh
# =========================
# FIRST EULER TEST RUN -- not the real 7x7 grid, not the "full" scale.
#
# Goal: exercise the whole pipeline end to end as cheaply as possible --
# environment loads, PyMC/PyTensor actually compiles and runs on a Euler
# compute node, --task-id/--n-tasks array slicing does what recovery_grid.py
# says it does, each task checkpoints to its own file, and
# merge_recovery_results.py puts them back together -- BEFORE committing
# real compute to the full-scale 7x7 grid or the identifiability sweep.
#
# 7 array tasks, one per model's diagonal (self-recovery) cell, in
# GENERATOR_NAMES order (recovery_grid.py):
#   task 0: Standard        task 1: A-B1           task 2: A-B2
#   task 3: B-Matern-B1     task 4: B-Matern-B2
#   task 5: B-SE-B1         task 6: B-SE-B2
#
# At pilot scale (n_patients=2, n_electrodes=2, chains=2) these took
# 87-411s each on a laptop with the old cores=1 behaviour -- 30 min is
# generous padding for a first run on an unfamiliar node (cold PyTensor
# C-compile cache etc). Tighten --time once you've seen it run once.

#SBATCH --job-name=trf-pilot-diag
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2          # matches pilot preset's chains=2 -- one core per NUTS chain
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pilot_%A_%a.out
#SBATCH --error=logs/pilot_%A_%a.err
#
# --- ASK ABOUT THIS AT THE WORKSHOP --------------------------------------
# Euler uses the "cluster shareholder" model: a group's purchased nodes
# become part of the shared pool, and members get priority access to their
# share via a specific --account and/or --partition value, NOT by ending
# up there automatically. Find out NDlab's actual values and uncomment:
# #SBATCH --account=<ndlab_account_name>
# #SBATCH --partition=<ndlab_partition_name>
# --------------------------------------------------------------------------

set -euo pipefail

# --- Environment ----------------------------------------------------------
# Module names/versions below are illustrative (Euler's "new software stack"
# syntax) -- confirm current values with `module avail python` at the
# workshop, they do get updated over time.
module load stack/2024-06 gcc/12.2.0 python/3.11.6 eth_proxy

# Created once, ahead of time, e.g.:
#   python -m venv --system-site-packages /cluster/project/<group>/$USER/mixed-trf-venv
#   source /cluster/project/<group>/$USER/mixed-trf-venv/bin/activate
#   pip install -r requirements.txt
source "${HOME}/mixed-trf-venv/bin/activate"

# --- PyTensor compile cache -------------------------------------------
# PyTensor JIT-compiles C per process. Point its cache at this job's OWN
# local scratch ($TMPDIR, node-local disk), not the shared/networked
# $HOME -- with 7 array tasks compiling concurrently, sharing one cache
# directory risks lock contention / silent corruption, not just slowness.
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR}/pytensor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "${HOME}/mixed-mTRF"   # adjust to wherever the repo actually lives, e.g. /cluster/project/<group>/$USER/mixed-mTRF
mkdir -p logs figures

python recovery_grid.py \
    --scale pilot \
    --only-diagonal \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks 7 \
    --results-path "figures/pilot_diag_task${SLURM_ARRAY_TASK_ID}.json"

# After all 7 tasks finish (check with `sacct -j <jobid>`), merge on the
# login node:
#   python merge_recovery_results.py \
#       --glob 'figures/pilot_diag_task*.json' \
#       --output-path figures/pilot_diag_merged.json
