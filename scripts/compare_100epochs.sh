#!/usr/bin/env bash
set -eu

mkdir -p out
LOG_FILE="out/compare_100epochs.log"
TS="$(date '+%Y-%m-%d %H:%M:%S')"

echo "[$TS] Running 100-epoch comparison (KDS vs MinK vs BCOS)..." | tee "$LOG_FILE"
CMD="python src/main.py --data wikimia --model mistral --target_num 700 --out_dir out/ --contamination 0.25 --sgd --lr 0.0001 --seed 0 --epochs 100"
echo "[$TS] Command: $CMD" | tee -a "$LOG_FILE"

# run and keep full console log
bash -lc "$CMD" 2>&1 | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Latest comparison rows:" | tee -a "$LOG_FILE"
tail -n 8 out/results.tsv | tee -a "$LOG_FILE"
tail -n 3 out/method_compare.tsv | tee -a "$LOG_FILE"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done. Log saved to $LOG_FILE" | tee -a "$LOG_FILE"
