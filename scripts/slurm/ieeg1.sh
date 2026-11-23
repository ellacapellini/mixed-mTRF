#!/bin/bash
# Stage 1 on Euler: one array task per seed; each task loops bands x SNRs x 4 mechanisms (~5-8 min per band/SNR set).
#   sbatch --array=100-199 --export=ALL,TAG=_fin,EXTRA="--estimator binned" scripts/slurm/ieeg1.sh
#SBATCH --job-name=ieeg1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ieeg1_%A_%a.out
#SBATCH --error=logs/ieeg1_%A_%a.err
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${HOME}/mixed-mTRF"; mkdir -p logs results/stage1; export PYTHONPATH="$PWD/src"
TAG="${TAG:-}"; EXTRA="${EXTRA:-}"; BANDS="${BANDS:-lfp hg}"; SNRS="${SNRS:-0.005 0.03 0.2}"
for band in ${BANDS}; do for snr in ${SNRS}; do
  out="results/stage1/ieeg1${TAG}_${band}_snr${snr}_seed${SLURM_ARRAY_TASK_ID}.jsonl"
  [ -s "${out}" ] && { echo "skip ${out}"; continue; }
  python -m mixed_trf stage1 --band "${band}" --snr "${snr}" --seed "${SLURM_ARRAY_TASK_ID}" --out "${out}.partial" ${EXTRA}
  mv "${out}.partial" "${out}"
  mkdir -p results/done; date +%F_%T > "results/done/ieeg1${TAG}_${band}_snr${snr}_seed${SLURM_ARRAY_TASK_ID}.done"
done; done
