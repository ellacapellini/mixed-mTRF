#!/bin/bash
# Stage 2 (Bayesian ladder) on Euler: ONE model x ONE effect x ONE seed per call. Calibrate with a single task first.
#   sbatch --export=ALL,MODEL=B-pop-hyp,MODE=gain,SEED=0,EXTRA="--n_patients 8 --n_electrodes 4 --n_times 144000" scripts/slurm/stage2.sh
#SBATCH --job-name=stage2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/stage2_%j.out
#SBATCH --error=logs/stage2_%j.err
set -eo pipefail
module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${HOME}/mixed-mTRF"; mkdir -p logs results/stage2; export PYTHONPATH="$PWD/src"
out="results/stage2/ladder_${MODEL}_${MODE}_seed${SEED}${TAG:-}.jsonl"
python -m mixed_trf stage2 --model "${MODEL}" --mode "${MODE}" --seed "${SEED}" --out "${out}.partial" ${EXTRA:-}
mv "${out}.partial" "${out}"
