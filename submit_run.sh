#!/bin/bash
# submit_run.sh -- general-purpose Euler job runner.
#
# Replaces submit_diag.sh (which only knew how to call diagnose_cell.py)
# with a single script that runs ANY Python script in this project, so a
# new analysis script never needs its own bespoke submit_*.sh.
#
# The first argument is the script to run; everything after it is passed
# straight through as that script's own arguments.
#
#   sbatch --time=00:10:00 --job-name=ov_std \
#       --output=logs/ov_std.out --error=logs/ov_std.err \
#       submit_run.sh same_shape_overlay.py --fitter Standard
#
#   sbatch --time=20:00:00 --job-name=ov_bm1 \
#       --output=logs/ov_bm1.out --error=logs/ov_bm1.err \
#       submit_run.sh same_shape_overlay.py --fitter B-Matern-B1
#
#   sbatch --time=00:10:00 --job-name=diag1 \
#       --output=logs/diag1.out --error=logs/diag1.err \
#       submit_run.sh diagnose_cell.py --generator Standard --fitter B-Matern-B1
#
# Resource flags (--time, --mem-per-cpu, --cpus-per-task, ...) are passed to
# sbatch itself, same as before -- size them for whatever script you're
# running, same as you already do.

#SBATCH --job-name=run
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_%j.out
#SBATCH --error=logs/run_%j.err

set -eo pipefail   # NOT -u: Euler's `module` functions can trip on unset vars

module load stack/2024-06 gcc/12.2.0 python/3.12.8 eth_proxy
source "${HOME}/mixed-trf-venv/bin/activate"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
export PYTENSOR_FLAGS="base_compiledir=${TMPDIR:-/tmp}/pytensor_run_${SLURM_JOB_ID}"

cd "${HOME}/mixed-mTRF"
mkdir -p logs figures

if [ "$#" -lt 1 ]; then
    echo "Usage: sbatch [sbatch options] submit_run.sh <script.py> [script args...]" >&2
    exit 1
fi

SCRIPT="$1"
shift

echo "Host: $(hostname)  Job: ${SLURM_JOB_ID}  CPUs: ${SLURM_CPUS_PER_TASK}"
python --version
echo "Running: python ${SCRIPT} $@"

python "${SCRIPT}" "$@"