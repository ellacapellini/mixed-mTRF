#!/bin/bash
# Stage 1 for the final run, on the Mac, in cool-down-friendly chunks. Resumable: re-run the same script after any stop
#   caffeinate -i bash scripts/run_mac_stage1.sh
set -u
cd "$(dirname "$0")/.."
C="--cpus 3 --nice 10 --cooldown-s 60 --max-hours 2"
for s in 2 0.5 3; do
  python run_local.py --only ieeg1 --tag _es$s --seeds 200-249 --bands lfp,hg --snrs 0.03 --ieeg-args "--estimator binned --effect_scale $s" $C
done
python run_local.py --only ieeg1 --tag _rhy  --seeds 300-319 --bands lfp --snrs 0.03,0.2 --ieeg-args "--estimator binned --rhythm isochronous --violation_p 0.1" $C
python run_local.py --only ieeg1 --tag _e256 --seeds 300-319 --bands lfp,hg --snrs 0.03 --ieeg-mem-gb 6 --ieeg-args "--estimator binned --electrodes 256" --cpus 2 --nice 10 --cooldown-s 60 --max-hours 2
echo "Stage 1 finished. Now: bash scripts/make_figures.sh"
