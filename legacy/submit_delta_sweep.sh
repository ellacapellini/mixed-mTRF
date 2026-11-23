#!/bin/bash
# submit_delta_sweep.sh -- delta identifiability sweep (fit mode).
#   3 designs x 3 true deltas x 5 seeds = 45 tasks.
#   sbatch submit_delta_sweep.sh
#   sbatch --export=ALL,KERNEL=squared_exponential submit_delta_sweep.sh
#   sbatch --export=ALL,SHARED=1 submit_delta_sweep.sh    # realistic shared stimulus
#
# Designs (patients x electrodes = units): 2x2=4, 4x3=12, 6x4=24.
# True deltas: 0.0 (null: should NOT be confidently non-zero), -0.3, -0.8.
# 24-unit fits are the slow ones -- calibrate with one task first
# (sbatch --array=36 submit_delta_sweep.sh) and raise --time if needed.

#SBATCH --job-name=deltaid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --array=0-44
#SBATCH --output=logs/deltaid_%A_%a.out
#SBATCH --error=logs/deltaid_%A_%a.err

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars

module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_deltaid_${SLURM_JOB_ID}"

cd "${HOME}/mixed-mTRF"
mkdir -p logs figures

KERNEL="${KERNEL:-matern52}"
SHARED="${SHARED:-0}"          # SHARED=1 -> all units hear the same word stream
SHARED_FLAG=""; TAG=""
if [ "${SHARED}" = "1" ]; then SHARED_FLAG="--shared_stimulus"; TAG="_shared"; fi
DESIGNS=("2 2" "4 3" "6 4")
DELTAS=(0.0 -0.3 -0.8)
N_SEEDS=5

t=${SLURM_ARRAY_TASK_ID}
seed=$(( t % N_SEEDS ))
d_idx=$(( (t / N_SEEDS) % 3 ))
g_idx=$(( t / (N_SEEDS * 3) ))
read -r P E <<< "${DESIGNS[$g_idx]}"
delta="${DELTAS[$d_idx]}"

echo "Host: $(hostname) task=${t} design=${P}x${E} delta_true=${delta} seed=${seed} kernel=${KERNEL}"

python delta_identifiability.py --mode fit \
    --kernel "${KERNEL}" --n_patients "${P}" --n_electrodes "${E}" \
    --delta_true "${delta}" --seed "${seed}" \
    --chains 4 --cores 4 --draws 1000 --tune 1500 --target_accept 0.95 ${SHARED_FLAG} \
    --out "figures/delta_ident_${KERNEL}${TAG}_task${t}.jsonl"
