#!/bin/bash
# Build every figure from whatever results exist. Safe to run at any time; missing inputs are skipped with a message
cd "$(dirname "$0")/.."; export PYTHONPATH=src
F=results/figures; mkdir -p $F
python -m mixed_trf plot-stage1 --glob 'results/stage1/ieeg1_fin_*_seed*.jsonl' --out $F/stage1_main
python -m mixed_trf plot-stage1 --glob 'results/stage1/ieeg1_fin_*_seed*.jsonl' 'results/stage1/ieeg1_es*_seed*.jsonl' --out $F/power_map
python -m mixed_trf plot-stage1 --glob 'results/stage1/ieeg1_rhy_*_seed*.jsonl' --out $F/rhythm
python -m mixed_trf plot-stage1 --glob 'results/stage1/ieeg1_e256_*_seed*.jsonl' --out $F/electrodes256
python -m mixed_trf plot-stage2 --glob 'results/stage2/*.jsonl' --out $F/stage2
python -m mixed_trf plot-learning 'results/stage2/*.jsonl' --out $F/learning --mode gain
echo "figures in $F"
