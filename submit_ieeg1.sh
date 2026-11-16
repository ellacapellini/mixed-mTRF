#!/bin/bash
# submit_ieeg1.sh -- Stage 1 at iEEG scale (8 patients x 128 electrodes x 2 h), one array task per seed.
# Each task loops over 2 bands (lfp, hg) x 3 SNRs x 4 mechanisms (~1-1.5 h). Split the seeds with the Mac, e.g. Euler takes 30-99:
#   sbatch --array=30-99 submit_ieeg1.sh
# Output: figures/ieeg1<tag>_<band>_snr<snr>_seed<seed>.jsonl  (4 rows each)

#SBATCH --job-name=ieeg1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=6G
#SBATCH --time=08:00:00
#SBATCH --array=30-99
#SBATCH --output=logs/ieeg1_%A_%a.out
#SBATCH --error=logs/ieeg1_%A_%a.err

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars
module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "${HOME}/mixed-mTRF"; mkdir -p logs figures

TAG="${TAG:-}"; EXTRA="${EXTRA:-}"     # sensitivity variants: sbatch --export=ALL,TAG=_e256,EXTRA="--electrodes 256" ...
BANDS="${BANDS:-lfp hg}"
for band in ${BANDS}; do
for snr in 0.005 0.03 0.2; do
  out="figures/ieeg1${TAG}_${band}_snr${snr}_seed${SLURM_ARRAY_TASK_ID}.jsonl"
  [ -s "${out}" ] && { echo "skip ${out} (done)"; continue; }       # resumable
  echo "== seed ${SLURM_ARRAY_TASK_ID} band ${band} snr ${snr} $(date +%T)"
  python ieeg_stage1.py --band "${band}" --snr "${snr}" --seed "${SLURM_ARRAY_TASK_ID}" --out "${out}.partial" ${EXTRA}
  mv "${out}.partial" "${out}"
done
done