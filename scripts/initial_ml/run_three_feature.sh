#!/bin/bash
# Train and backtest the two three-feature baselines into RUNS_DIR/report/{lin3_ols,lin3}.
#   scripts/initial_ml/run_three_feature.sh DATA_DIR RUNS_DIR
# lin3_ols: one ElasticNet at alpha_ratio 1e-6 (an OLS fit up to rounding); lin3: the linear
# family's per-fold grid (grids/linear.yaml). Both read the feature set three_feature.
set -uo pipefail
DATA=${1:?usage: run_three_feature.sh DATA_DIR RUNS_DIR}
OUT=${2:?usage: run_three_feature.sh DATA_DIR RUNS_DIR}/report
PY=${PYTHON:-python}
mkdir -p "$OUT"
$PY -m ml4trading.train --method linear --out "$OUT/lin3_ols" --data-dir "$DATA" \
  --set model.feature_set=three_feature --set model.alpha_ratio=1e-6 --set model.l1_ratio=0.5 \
  > "$OUT/lin3_ols.train.log" 2>&1 || echo "lin3_ols: train FAILED"
$PY -m ml4trading.backtest --run "$OUT/lin3_ols" --data-dir "$DATA" > "$OUT/lin3_ols.bt.log" 2>&1 || echo "lin3_ols: backtest FAILED"
$PY -m ml4trading.train --method linear --grid grids/linear.yaml --out "$OUT/lin3" --data-dir "$DATA" \
  --set model.feature_set=three_feature > "$OUT/lin3.train.log" 2>&1 || echo "lin3: train FAILED"
$PY -m ml4trading.backtest --run "$OUT/lin3" --data-dir "$DATA" > "$OUT/lin3.bt.log" 2>&1 || echo "lin3: backtest FAILED"
