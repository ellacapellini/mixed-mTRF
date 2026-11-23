#!/bin/bash
# submit_full_grid.sh  (Euler: full-scale 7x7 recovery grid, 1 cell per task)
# ==========================================================================
# 49 array tasks = the 49 (generator -> fitter) cells, one cell per task.
# Task i runs cell i in generator-major order (recovery_grid.py):
#   i = 7*generator_index + fitter_index, order:
#   Standard, A-B1, A-B2, B-Matern-B1, B-Matern-B2, B-SE-B1, B-SE-B2
#   diagonal cells = tasks 0, 8, 16, 24, 32, 40, 48
#
# "full" scale preset: 4 patients x 3 electrodes, 1000 samples, 6 basis fns,
# tau_max=200 ms, 4 chains, 500 draws / 1000 tune.
#
# STEP 1 -- CALIBRATE FIRST (2 of the slowest diagonal cells, 4 h max):
#     cd ~/mixed-mTRF && mkdir -p logs figures
#     sbatch --array=24,48 submit_full_grid.sh
#   then read the real time/memory:
#     sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS,ReqMem,TotalCPU
#
# STEP 2 -- all 49, with --time set from the calibration (+~50% headroom):
#     sbatch --array=0-48 --time=12:00:00 submit_full_grid.sh
#   (a task that hits TIMEOUT loses its cell; re-submit just that task with
#    a longer --time, e.g. sbatch --array=<i> --time=24:00:00 ...)
#
# Resources: 4 CPUs = one per NUTS chain (cores=chains=4). 4G/CPU = 16G total
# is generous (pilot used ~1 GB with 2 chains); tighten after calibration.

#SBATCH --job-name=trf-full-grid
#SBATCH --array=0-48
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/full_%A_%a.out
#SBATCH --error=logs/full_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ecapellini@student.ethz.ch

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars

module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

cd "${HOME}/mixed-mTRF"
mkdir -p figures

echo "Host: $(hostname)  Task: ${SLURM_ARRAY_TASK_ID}  CPUs: ${SLURM_CPUS_PER_TASK}"
python --version

python recovery_grid.py \
    --scale full \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --n-tasks 49 \
    --results-path "figures/full_grid_task${SLURM_ARRAY_TASK_ID}.json"

# When everything is done (login node):
#   source ~/mixed-trf-venv/bin/activate
#   python merge_recovery_results.py --glob 'figures/full_grid_task*.json' \
#          --output-path figures/full_grid_merged.json