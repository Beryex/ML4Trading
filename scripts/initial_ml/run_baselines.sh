#!/bin/bash
# Train and backtest the six baselines into OUT_DIR (one run directory each).
#   scripts/initial_ml/run_baselines.sh DATA_DIR OUT_DIR
# The report runs it twice: on the dataset (the NEW 552-symbol universe) into runs/report/new, and
# on a copy whose universe.json, dividends.parquet and MANIFEST.json are the 502-symbol release's
# (the OLD universe) into runs/report/old.
set -uo pipefail
DATA=${1:?usage: run_baselines.sh DATA_DIR OUT_DIR}
OUT=${2:?usage: run_baselines.sh DATA_DIR OUT_DIR}
PY=${PYTHON:-python}
mkdir -p "$OUT"
for spec in "voo investment" "qqq investment_tech" "daily_mom momentum --set K=390" \
            "daily_rev momentum --set K=390 --set model.reverse=true" \
            "weekly_mom momentum --set K=1950" "weekly_rev momentum --set K=1950 --set model.reverse=true"; do
  set -- $spec; name=$1; shift; method=$1; shift
  start=$(date +%s)
  $PY -m ml4trading.train --method "$method" --out "$OUT/$name" --data-dir "$DATA" "$@" > "$OUT/$name.train.log" 2>&1 || echo "$name: train FAILED"
  $PY -m ml4trading.backtest --run "$OUT/$name" --data-dir "$DATA" > "$OUT/$name.bt.log" 2>&1 || echo "$name: backtest FAILED"
  echo "$name done in $(( $(date +%s) - start ))s"
done
