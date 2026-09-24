#!/bin/bash
# submit_pilot_diagonal.sh  (Euler tiny test run)
# ================================================
# Tiny end-to-end test BEFORE the big 7x7 run. 7 array tasks = the 7 diagonal
# (self-recovery) cells at pilot scale (2 patients x 2 electrodes, 400 samples,
# 2 chains, 150 draws / 300 tune).
#
# HOW TO SUBMIT (from the repo dir, and `logs/` must exist BEFORE sbatch --
# Slurm opens the --output file before this script starts, so a mkdir inside
# the script is too late and the logs would be silently lost):
#     cd $SCRATCH/mixed-mTRF && mkdir -p logs figures
#     sbatch --array=0,1 submit_pilot_diagonal.sh     # smoke test: Standard + A-B1
#     sbatch submit_pilot_diagonal.sh                 # then all 7 (array=0-6)
#
# RESOURCES (why these numbers)
#   cpus-per-task=2 : recovery_grid.py sets cores=chains=2 -> one process per NUTS chain
#   mem-per-cpu=3G  : data are tiny; memory is dominated by Python + PyTensor
#                     compile (~1-2 GB per chain process). 6 GB total is safe.
#   time=00:30:00   : laptop took 87-411 s/cell with cores=1; padding for a cold
#                     PyTensor compile cache. Jobs <=4h also go to Euler's
#                     fastest queue. Tighten after checking `sacct` (see bottom).

#SBATCH --job-name=trf-pilot-diag
#SBATCH --array=0-6
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=3G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pilot_%A_%a.out
#SBATCH --error=logs/pilot_%A_%a.err
#
# No --account / --partition on purpose: `my_share_info` says you are currently
# a GUEST user, i.e. not yet in the NDlab share. Jobs will run in the general
# pool (possibly longer queue wait). Once Tim/the lab adds you to the share,
# `my_share_info` will show it and you can add the right --account then.

set -eo pipefail   # NOT -u: Euler's `module` shell functions can trip on unset vars

# --- Environment -----------------------------------------------------------
# VERIFY these with `module avail python` / `module avail stack` -- Euler's
# stack changes. Your requirements (numpy 2.4 / scipy 1.18) want a recent
# Python (>=3.11, ideally 3.12 like your laptop venv).
module load stack/2024-06 gcc/12.2.0 python/3.11.6 eth_proxy

source "${HOME}/mixed-trf-venv/bin/activate"

# One BLAS thread per process: 2 chains x multithreaded BLAS would oversubscribe
# the 2 cores you were given.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

# PyTensor JIT cache on node-local disk, separate per array task (avoids lock
# contention on the shared filesystem).
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "${SCRATCH}/mixed-mTRF"
mkdir -p figures

echo "Host: $(hostname)  Task: ${SLURM_ARRAY_TASK_ID}  CPUs: ${SLURM_CPUS_PER_TASK}"
python --version

python recovery_grid.py \
    --scale pilot \
    --only-diagonal \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks 7 \
    --results-path "figures/pilot_diag_task${SLURM_ARRAY_TASK_ID}.json"

# After all tasks finish, on the login node:
#   sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem,TotalCPU
#   python merge_recovery_results.py --glob 'figures/pilot_diag_task*.json' \
#          --output-path figures/pilot_diag_merged.json