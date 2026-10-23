#!/bin/bash
# submit_testbed.sh -- Stage 1 of the plan: ridge testbed, 100 seeds, 8 patients x 4 electrodes, 2 h per unit.
#   sbatch submit_testbed.sh
# One task = one seed; it loops over 3 SNRs and 4 mechanisms (null, gain, latency, width).
# SNR 0.005 / 0.03 / 0.2 -> max attainable r ~ 0.07 / 0.17 / 0.41 (Gillis et al. report ~0.03 baseline r on scalp EEG).
# Output: one JSON line per (snr, mode) in figures/testbed_snr<snr>_task<seed>.jsonl

#SBATCH --job-name=testbed
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-99
#SBATCH --output=logs/testbed_%A_%a.out
#SBATCH --error=logs/testbed_%A_%a.err

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars
module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${HOME}/mixed-mTRF"; mkdir -p logs figures

for snr in 0.005 0.03 0.2; do
  echo "== seed ${SLURM_ARRAY_TASK_ID} snr ${snr}  $(date +%T)"
  python erp_testbed.py --modes null gain latency width --reps 1 --seed "${SLURM_ARRAY_TASK_ID}" \
      --n_patients 8 --n_electrodes 4 --n_times 720000 --snr "${snr}" \
      --noise pink --common_noise 0.3 --effect_frac 0.7 --shared_stimulus --boot 200 \
      --out "figures/testbed_snr${snr}_task${SLURM_ARRAY_TASK_ID}.jsonl"
done