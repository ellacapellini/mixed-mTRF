#!/bin/bash
# submit_diag.sh -- run diagnose_cell.py as a proper Euler batch job.
#
# Unlike `sbatch --wrap="..."` (which runs the command through /bin/sh,
# where `source` doesn't exist -- that's what failed last time), this is a
# real script with a bash shebang, so `source` and everything else works
# exactly like your other submit scripts.
#
# All arguments after the script name are passed straight through to
# diagnose_cell.py, so this one script covers every case:
#
#   sbatch --time=04:00:00 --job-name=diag1 --output=logs/diag1.out \
#       submit_diag.sh --generator Standard --fitter B-Matern-B1
#
#   sbatch --time=04:00:00 --job-name=diag2 --output=logs/diag2.out \
#       submit_diag.sh --generator B-SE-B2 --fitter B-SE-B2
#
#   sbatch --time=12:00:00 --mem-per-cpu=6G --job-name=diag3 \
#       --output=logs/diag3.out submit_diag.sh --generator B-Matern-B1 \
#       --fitter B-Matern-B1 --n-patients 8 --n-electrodes 12

#SBATCH --job-name=diag
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/diag_%j.out
#SBATCH --error=logs/diag_%j.err

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars

module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_diag_${SLURM_JOB_ID}"

cd "${HOME}/mixed-mTRF"
mkdir -p logs figures

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  CPUs: ${SLURM_CPUS_PER_TASK}"
python --version
echo "Args: $@"

python diagnose_cell.py "$@"