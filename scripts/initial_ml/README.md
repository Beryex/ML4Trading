# Initial ML Solution: runs and tables

These commands rebuild every run and table of the Initial ML Solution from scratch. Run them from
the repository root with the dataset in `data/` (see `DATA.md`). The runs go to `runs/` and the
tables to `results/initial_ml/`. Both are git-ignored, like the data.

## 1. Train and backtest

```bash
DATA=data

# the six baselines, on the dataset's universe (552 symbols) ...
scripts/initial_ml/run_baselines.sh $DATA runs/report/new
# ... and on the 502-symbol universe: a copy of the dataset whose universe.json,
# dividends.parquet and MANIFEST.json are the 502-symbol release's (the bars are the same)
scripts/initial_ml/run_baselines.sh data_502 runs/report/old

# the three-feature baselines: an OLS-like fit and the per-fold grid
scripts/initial_ml/run_three_feature.sh $DATA runs

# the per-fold-tuned grids
python -m ml4trading.train --method linear    --grid grids/linear.yaml    --out runs/linear    --data-dir $DATA
python -m ml4trading.train --method nonlinear --grid grids/nonlinear.yaml --out runs/nonlinear --data-dir $DATA

# the frozen linear model (v29)
python -m ml4trading.train --method linear --out runs/v29 --data-dir $DATA \
  --set model.alpha_ratio=3e-5 --set model.l1_ratio=0.5 --set model.extra_features=true \
  --set model.feature_set=no_market_calendar_slow --set model.risk_adjust=vol --set model.vol_window=63

# the frozen nonlinear model (P3b): LightGBM boosted from v29's elastic net
python -m ml4trading.train --method nonlinear --out runs/p3b --data-dir $DATA \
  --set model.loss=mse --set model.stock_id=drop --set model.feature_set=no_market_calendar_slow \
  --set model.extra_features=true --set model.risk_adjust=vol --set model.num_leaves=7 \
  --set model.min_data_in_leaf=2000 --set model.learning_rate=0.05 --set model.n_estimators=30 \
  --set model.extra_trees=true --set model.feature_fraction=0.5 --set model.bagging_fraction=0.5 \
  --set model.bagging_freq=1 --set model.init_model=linear --set model.init_alpha_ratio=3e-5 \
  --set model.init_l1_ratio=0.5 --set model.init_feature_set=no_market_calendar_slow

# re-link the four trained runs for backtesting and build the two validation-selected splices
python scripts/initial_ml/make_report_runs.py --runs runs --linear-grid runs/linear \
  --nonlinear-grid runs/nonlinear --v29 runs/v29 --p3b runs/p3b
for r in linear_grid nonlinear_grid v29 p3b sel_grid sel_final; do
  python -m ml4trading.backtest --run runs/report/$r --data-dir $DATA
done
```

The reported runs were trained at earlier commits of this code:
- the grids at 01ea854;
- v29 at 78b6ca4;
- P3b at 5b59ac8.

Their saved predictions were backtested at this commit, and every metric matched the original
backtest exactly. Retraining them at this commit was not re-run.

On a 14-core laptop, the fold loop takes about:
- 55 min for the linear grid, 75 min for the nonlinear grid;
- 44 min for v29, 15 min for P3b;
- a few minutes for the three-feature fits;
- about 1 min per baseline.

## 2. Tables

```bash
python scripts/initial_ml/make_tables.py --runs runs --data-dir $DATA --out results/initial_ml
python scripts/initial_ml/make_registry.py --runs runs --out results/initial_ml/tables
python scripts/initial_ml/causal_check_real.py --runs runs --data-dir $DATA
```

`make_tables.py` writes:

| file | content |
|---|---|
| `B1`–`B3` | the baselines, 502 vs 552 symbols |
| `G1`–`G7` | every method on the same split and metric |
| `C3`, `C4` | test-path IC and sign accuracy; validation vs test |
| `F2` | the configuration chosen per fold |
| `D6` | training rows per fold |
| `A7`–`A7c` | the drop-outs in the fold universes |
| `daily_equity.csv` | the account value at each session's open, one column per method |

The other two scripts:
- `make_registry.py` lists every run directory with its configuration, folds, wall time, and
  validation and test Sharpe (`E1_run_registry.csv`).
- `causal_check_real.py` runs `ml4trading.testing.assert_causal` on the real data with the
  reported models' configurations.

Two more analyses:

```bash
# the development choice made causally: per fold, the configuration with the best mean
# validation Sharpe over the folds up to it (all candidates, and those scored before the
# frozen linear model's test)
python scripts/initial_ml/make_causal_selection.py build --runs runs
for r in causal_all causal_blind; do python -m ml4trading.backtest --run runs/report/$r --data-dir $DATA; done
python scripts/initial_ml/make_causal_selection.py tables --runs runs --data-dir $DATA --out results/initial_ml

# net P&L on US vs non-US companies; RANKING.csv is a companiesmarketcap.com ranking download
python scripts/initial_ml/make_geography.py --countries RANKING.csv --runs runs --data-dir $DATA
```

The features themselves are defined in `ml4trading/features/`: `daily.py`, `indicators.py` and
`build.py` for ML4Investment's columns, `slow.py` for the literature ones. The named subsets each
model trains on are in `ml4trading/features/feature_sets/`.
