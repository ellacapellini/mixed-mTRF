#!/bin/bash
# run_everything.sh -- submit every validation piece in one consolidated
# batch: the full 7x7 recovery grid, the diagonal-shapes time-series plot,
# the same-shape overlay (7 fitters), the two prior-posterior diagnostics,
# and the SNR sweep (5 points) + its aggregate plot.
#
# This is a PLAIN BASH SCRIPT run directly on the login node -- it only
# issues `sbatch` calls (fast) and exits; it is NOT itself a Slurm job and
# has no #SBATCH header. Run it with:
#
#     cd ~/mixed-mTRF && bash run_everything.sh
#
# Idempotent by design: before submitting each piece, it checks whether the
# piece's expected output already exists on disk, and skips submission if
# so (printing why). Delete the relevant output file/folder first if you
# genuinely want to force a fresh rerun of just that piece.
#
# Steps whose output depends on several earlier jobs (the full-grid merge
# and plot, the SNR sweep summary, the overlay plot) are submitted with
# --dependency=afterok:<jobids> so they wait for their inputs instead of
# racing them; Slurm queues them and runs them automatically once ready --
# nothing more to do on your end once this script returns.

set -eo pipefail
cd "$(dirname "$0")"
mkdir -p logs figures figures/diagnose figures/overlay

echo "=== 1. Full 7x7 recovery grid (49 cells, skips already-done cells) ==="
GRID_JOB=$(sbatch --parsable --array=0-48 --time=20:00:00 submit_full_grid.sh)
echo "Submitted full grid array job: ${GRID_JOB}"

echo ""
echo "=== 1b. Merge the grid, after it finishes ==="
MERGE_JOB=$(sbatch --parsable --dependency=afterok:${GRID_JOB} --time=00:10:00 --job-name=grid_merge \
    --output=logs/grid_merge_%j.out --error=logs/grid_merge_%j.err \
    submit_run.sh merge_recovery_results.py --glob 'figures/full_grid_task*.json' \
    --output-path figures/full_grid_merged.json)
echo "Queued merge, waiting on job ${GRID_JOB} -> merge job ${MERGE_JOB}"

echo ""
echo "=== 1c. Plot the grid, after the merge finishes ==="
sbatch --dependency=afterok:${MERGE_JOB} --time=00:10:00 --job-name=grid_plot \
    --output=logs/grid_plot_%j.out --error=logs/grid_plot_%j.err \
    submit_run.sh plot_recovery_grid.py --results-path figures/full_grid_merged.json \
    --output-path figures/full_grid.png
echo "Queued plot, waiting on merge job ${MERGE_JOB}"

echo ""
echo "=== 2. Diagonal shapes time-series plot ==="
if [ -f figures/diagonal_shapes_for_tim.png ]; then
    echo "SKIP: figures/diagonal_shapes_for_tim.png already exists"
else
    sbatch --time=02:00:00 --cpus-per-task=4 --mem-per-cpu=4G --job-name=diag_shapes \
        --output=logs/diag_shapes_%j.out --error=logs/diag_shapes_%j.err \
        submit_run.sh plot_diagonal_shapes.py
    echo "Submitted diagonal shapes plot"
fi

echo ""
echo "=== 3. Same-shape overlay (7 fitters, skip any already saved) ==="
OVERLAY_JOBS=()
declare -A OVERLAY_TIME=(
    ["Standard"]="00:15:00" ["A-B1"]="08:00:00" ["A-B2"]="08:00:00"
    ["B-Matern-B1"]="20:00:00" ["B-Matern-B2"]="20:00:00"
    ["B-SE-B1"]="20:00:00" ["B-SE-B2"]="20:00:00"
)
for FITTER in Standard A-B1 A-B2 B-Matern-B1 B-Matern-B2 B-SE-B1 B-SE-B2; do
    if [ -f "figures/overlay/${FITTER}.json" ]; then
        echo "SKIP: figures/overlay/${FITTER}.json already exists"
    else
        JID=$(sbatch --parsable --time="${OVERLAY_TIME[$FITTER]}" --job-name="ov_${FITTER}" \
            --output="logs/ov_${FITTER}_%j.out" --error="logs/ov_${FITTER}_%j.err" \
            submit_run.sh same_shape_overlay.py --fitter "${FITTER}")
        echo "Submitted overlay fitter ${FITTER}: job ${JID}"
        OVERLAY_JOBS+=("${JID}")
    fi
done

echo ""
if [ -f figures/overlay/same_shape_overlay.png ]; then
    echo "SKIP: figures/overlay/same_shape_overlay.png already exists"
elif [ ${#OVERLAY_JOBS[@]} -eq 0 ]; then
    echo "All 7 overlay fitters already had results on disk but the combined plot is missing -- plotting now."
    sbatch --time=00:10:00 --job-name=ov_plot --output=logs/ov_plot_%j.out --error=logs/ov_plot_%j.err \
        submit_run.sh plot_same_shape_overlay.py --generator A-B2 --seed 777
else
    DEP=$(IFS=:; echo "${OVERLAY_JOBS[*]}")
    sbatch --dependency=afterok:${DEP} --time=00:10:00 --job-name=ov_plot \
        --output=logs/ov_plot_%j.out --error=logs/ov_plot_%j.err \
        submit_run.sh plot_same_shape_overlay.py --generator A-B2 --seed 777
    echo "Queued overlay combine-plot, waiting on jobs: ${DEP}"
fi

echo ""
echo "=== 4. Prior-posterior diagnostics (diag1, diag2) ==="
if [ -f "figures/diagnose/Standard_B-Matern-B1/prior_posterior.png" ]; then
    echo "SKIP: diag1 (Standard -> B-Matern-B1) already has results"
else
    sbatch --time=20:00:00 --job-name=diag1 --output=logs/diag1_%j.out --error=logs/diag1_%j.err \
        submit_run.sh diagnose_cell.py --generator Standard --fitter B-Matern-B1
    echo "Submitted diag1 (Standard -> B-Matern-B1)"
fi
if [ -f "figures/diagnose/B-SE-B2_B-SE-B2/prior_posterior.png" ]; then
    echo "SKIP: diag2 (B-SE-B2 -> B-SE-B2) already has results"
else
    sbatch --time=20:00:00 --job-name=diag2 --output=logs/diag2_%j.out --error=logs/diag2_%j.err \
        submit_run.sh diagnose_cell.py --generator B-SE-B2 --fitter B-SE-B2
    echo "Submitted diag2 (B-SE-B2 -> B-SE-B2) -- this one never finished before, so this is a real rerun"
fi

echo ""
echo "=== 5. SNR sweep (5 points, skip any already saved) + aggregate plot ==="
SNR_JOBS=()
for SNR in 1 2 5 10 20; do
    SNR_SUFFIX=""
    if [ "${SNR}" != "5" ]; then SNR_SUFFIX="_snr${SNR}"; fi
    OUT_DIR="figures/diagnose/Standard_B-Matern-B1${SNR_SUFFIX}"
    if [ -f "${OUT_DIR}/summary.txt" ] && [ -f "${OUT_DIR}/bfmi.txt" ]; then
        echo "SKIP: SNR=${SNR} already has results in ${OUT_DIR}"
    else
        JID=$(sbatch --parsable --time=20:00:00 --job-name="snr${SNR}" \
            --output="logs/snr${SNR}_%j.out" --error="logs/snr${SNR}_%j.err" \
            submit_run.sh diagnose_cell.py --generator Standard --fitter B-Matern-B1 --snr-target "${SNR}")
        echo "Submitted SNR=${SNR}: job ${JID}"
        SNR_JOBS+=("${JID}")
    fi
done

echo ""
if [ -f "figures/diagnose/Standard_B-Matern-B1_snr_sweep/snr_sweep.png" ] && [ ${#SNR_JOBS[@]} -eq 0 ]; then
    echo "SKIP: figures/diagnose/Standard_B-Matern-B1_snr_sweep/snr_sweep.png already exists"
elif [ ${#SNR_JOBS[@]} -eq 0 ]; then
    echo "All 5 SNR points already had results but the aggregate plot is missing -- plotting now."
    sbatch --time=00:10:00 --job-name=snr_summary --output=logs/snr_summary_%j.out --error=logs/snr_summary_%j.err \
        submit_run.sh snr_sweep_summary.py --generator Standard --fitter B-Matern-B1 --snr-values 1 2 5 10 20
else
    DEP=$(IFS=:; echo "${SNR_JOBS[*]}")
    sbatch --dependency=afterok:${DEP} --time=00:10:00 --job-name=snr_summary \
        --output=logs/snr_summary_%j.out --error=logs/snr_summary_%j.err \
        submit_run.sh snr_sweep_summary.py --generator Standard --fitter B-Matern-B1 --snr-values 1 2 5 10 20
    echo "Queued SNR sweep summary, waiting on jobs: ${DEP}"
fi

echo ""
echo "=== All done submitting. Check status with: squeue --me ==="