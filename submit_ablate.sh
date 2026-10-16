#!/bin/bash
# submit_ablate.sh -- Section-10 ablation array: variant x seed (x kernel).
#
#   sbatch submit_ablate.sh                       # Matern, all 7 variants x 5 seeds = 35 tasks
#   sbatch --export=ALL,KERNEL=squared_exponential submit_ablate.sh
#
# Task id -> (variant, seed). Each task appends one JSON line to
# figures/ablate_<kernel>.jsonl (append of a single short line is safe
# across concurrent array tasks on a shared filesystem in practice, but if
# you ever see a torn line, give each task its own file and cat them).

#SBATCH --job-name=ablate
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --array=0-34
#SBATCH --output=logs/ablate_%A_%a.out
#SBATCH --error=logs/ablate_%A_%a.err

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars

module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_ablate_${SLURM_JOB_ID}"

cd "${HOME}/mixed-mTRF"
mkdir -p logs figures

KERNEL="${KERNEL:-matern52}"
VARIANTS=(asis center ncp no_h1 shared_mean fix_h1 ncp_fix_h1)
N_SEEDS=5
v_idx=$(( SLURM_ARRAY_TASK_ID / N_SEEDS ))
seed=$(( SLURM_ARRAY_TASK_ID % N_SEEDS ))
variant="${VARIANTS[$v_idx]}"

echo "Host: $(hostname)  task=${SLURM_ARRAY_TASK_ID}  variant=${variant}  seed=${seed}  kernel=${KERNEL}"

# Realistic scale (matches recovery_grid 'full'): 4 patients x 3 electrodes, 21 lags.
python ablate_standard.py \
    --variant "${variant}" --kernel "${KERNEL}" --generator Standard \
    --n_patients 4 --n_electrodes 3 --n_times 1000 --tau_max 200 \
    --draws 1000 --tune 1500 --chains 4 --cores 4 --target_accept 0.95 \
    --seed "${seed}" --out "figures/ablate_${KERNEL}.jsonl"