# ML4Trading

Walk-forward training and integer-share backtesting for short-horizon US equity return
prediction. Course project for CMU 10-718 *Machine Learning in Practice*.

A model predicts each stock's return over the next decision period; a trader turns predictions
into a portfolio; a backtest buys and sells whole shares, pays commissions and spreads, and
reports what the strategy would have earned. Every model is refit monthly on a rolling window
and judged only on the month after its training data ends.

## Quick start

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"   # or: pip install -e ".[dev]"
# the nonlinear model's LightGBM needs OpenMP; on macOS: brew install libomp
# put the dataset under data/ (see DATA.md), then
python -m ml4trading.data verify
python -m ml4trading.train --method investment --out runs/investment
python -m ml4trading.backtest --run runs/investment
pytest
```

## How it works

```
30-min bars ──> periods ──> folds ──> universe ──> model ──> trader ──> shares ──> book ──> metrics
 (data.py)    (periods.py) (folds.py) (selector.py) (models/) (trader.py) (sizing.py) (book.py) (metrics.py)
```

- **Periods** (`periods.py`). Bars are aggregated into decision periods of `K` minutes
  (`K = 390`: one per session; `K = 390 x N`: one per N-session block of the exchange calendar,
  so `K = 1950` trades once a week). Each (period, symbol) row carries its realized label
  `y = exit / open - 1 (+ dividends)`: overnight the exit is the next period's open (positions
  are held across the close), intraday it is the period's close.
- **Folds** (`folds.py`). For every month `B` of the test path (2020-01 to 2026-08 by default):
  candidate configurations are fitted on 36 months of training data and scored on the 12
  validation months that follow; the best one is refitted on the 36 months before `B` and
  predicts month `B`. Fits stop 5 sessions before whatever they are scored on, because a label
  looks one period ahead. The test months, concatenated, are the result.
- **Frames and models** (`models/`). A run builds its method's frames once -- the labelled
  panel over the method's symbol pool and, for a learned model, its features (`features/`) row
  by row -- and every fold reaches the model only as a fit window and a predict window over them.
- **Universe** (`selector.py`). Per fold, symbols with enough labelled sessions and a median
  dollar volume of at least $1M, the 100 most liquid kept. Only they are traded; a learned model
  fits on every stock of its pool.
- **Trader** (`trader.py`). Per period, the top-N predictions by magnitude (long or short),
  weighted proportionally to the prediction, capped per name and in total, then a cost gate:
  a name is traded only if its predicted return beats its estimated round-trip cost.
- **Shares and book** (`sizing.py`, `book.py`). Weights become whole shares within the account's
  equity (gains are reinvested; `--set reinvest=false` caps every period at the starting
  capital instead), with a switching-friction rule that avoids trades whose cost exceeds
  the tracking error they remove. Entries are modelled as passive limit orders (no spread),
  exits as market orders (half the quoted spread); every order pays a per-share commission
  with a minimum; shorts pay a borrow fee and owe dividends. The defaults: $20,000 capital, a
  4 bp quoted spread, $0.0035/share with a $0.35 minimum, 1 %/year borrow.
- **Metrics** (`metrics.py`). Net and gross Sharpe (per-session returns, annualized by √252;
  per period for multi-session `K`), net profit, cumulative and annualized return, max
  drawdown, alpha and beta against VOO, costs paid, and trade statistics: a trade runs from a
  position's opening to its closing, and the trade hit rate is the share of completed trades
  that made money after their own costs.

All knobs and their defaults live in `ml4trading/config.py`; `--set key=value` overrides one
(`--set capital=10000`, `--set costs.spread_bps=2`).

## The baselines

Two buy-and-hold benchmarks a learned model has to beat: `investment` holds VOO (S&P 500) and
`investment_tech` QQQ (Nasdaq-100). Measured with the defaults above on the full dataset (test
path 2020-01-02 to 2026-08-28, 1,673 sessions -- the data's last session, 2026-08-31, has no
next open to exit at; `runs/<method>/backtest/metrics.json`):

| method | net Sharpe | net profit | cumulative return | max drawdown | alpha vs VOO | beta |
|---|---:|---:|---:|---:|---:|---:|
| `investment` (VOO) | 0.852 | $31,924 | +159.6 % | −31.9 % | −0.0 % | 0.99 |
| `investment_tech` (QQQ) | 0.895 | $49,006 | +245.0 % | −36.6 % | +2.5 % | 1.17 |

## The learned models

`linear` and `nonlinear` predict each stock's next open-to-open return (K = 390, held overnight)
from about 190 daily features, refitted every fold on every stock of the pool; the fold's
universe decides what is traded.

- **Features** (`ml4trading/features/`): the feature set of
  [ML4Investment](https://github.com/Beryex/ML4Investment) (MIT), computed on the 30-minute
  bars -- intraday statistics aggregated per session (returns, ranges, VWAP deviation,
  first- and last-hour moves, volume ratios), daily technical indicators (RSI, MACD,
  stochastics, ADX, CCI, MFI, Bollinger), lags, seven cross-sectional ranks, VOO's and QQQ's
  own features, calendar features and a stock code. The row of session *d* holds what was known
  at *d*'s open: the features of the stock's previous session. The package docstring lists every
  adaptation; the features are built once per run (about a minute and 8 GB of memory on the
  full dataset).
- **Split-adjusted levels are left out.** Prices and volumes are split-adjusted back from the
  end of the data, so a stock's past price or volume LEVEL reveals whether it split later --
  which companies do after their shares have risen. Every feature that moves when one stock's
  prices are multiplied and its volume divided by the same factor is dropped (58 of
  ML4Investment's per-stock features: price and volume levels, ATR and MACD in price units,
  OBV, ranges, ratios over such a level); a test holds the list equal to that rule.
- **`linear`**: an elastic net on the standardized features. A missing value is replaced by the
  median of its feature over the same session's stocks; the penalty is a fraction `alpha_ratio`
  of the smallest one that zeroes every coefficient (scikit-learn's `ElasticNetCV` convention).
  The stock code is not used.
- **`nonlinear`**: LightGBM, by default on the absolute error as in ML4Investment, with the stock
  code as a categorical feature; missing values go in as they are. Fits are deterministic (a
  fixed seed and thread count).
- **Grids** (`grids/linear.yaml`, `grids/nonlinear.yaml`) are searched per fold on validation net
  Sharpe like any method's. Each fold also records every candidate's out-of-sample R², against
  the constant fitted on its training rows (`candidate_val_r2` in `fold.json`); it is never
  ranked on.

```bash
python -m ml4trading.train --method linear --grid grids/linear.yaml --out runs/linear
python -m ml4trading.backtest --run runs/linear
```

### Validation diagnostics

A fold's validation Sharpe says how a candidate did, not why. For every fold, `train` also keeps
what the chosen candidate's validation pass produced (nothing is refitted):

- `val/predictions.parquet`: the scored validation rows -- `period, symbol, prediction`, the
  realized return `y` and the `label` the fit was trained on (rows without a realized return are
  not scored);
- `val_diagnostics` in `fold.json`: that validation book taken apart -- the per-period
  information coefficient (Spearman correlation of prediction and return) with its spread,
  t-statistic and hit rate; the long and short legs' P&L before and after costs; how often a held
  book is all one sign and its mean net and gross exposure; how many of the trader's top names
  clear its cost gate; the cross-sectional spread of the predictions next to the label's; and the
  out-of-sample R² under squared error (`r2_mse`). A value that is undefined for a fold, such as
  the IC of a constant forecast, is `null`;
- `val/importance.parquet`, with `--importance`, for a model with an estimator: one row per
  feature. The `nonlinear` fit adds its own account (LightGBM `gain`, `split` count and TreeSHAP
  `shap`); every fit gets permutation importance on the validation rows (Breiman 2001):
  `perm_mse`, the rise in squared error, and `perm_ic`, the fall in the mean per-period IC, when
  one feature's column is shuffled. A correlated twin can make a feature look unimportant, so
  read these as the cost of losing a column's values, not as causes.

`diagnose` reads those files (no dataset needed) and writes `<run>/diagnose/`:

```bash
python -m ml4trading.train --method nonlinear --out runs/nonlinear --importance
python -m ml4trading.diagnose --run runs/nonlinear
```

- `folds.csv`: one row per fold -- `val_net_sharpe` (the chosen candidate's score,
  `candidate_val_net_sharpe[chosen]`) and every `val_diagnostics` key; `by_test_year.csv`: their
  mean per test year, with the number of folds;
- `importance.csv`: per feature and measure, the mean over folds, the mean rank (1 = largest,
  ranked within each fold) and the share of folds in which it is in the top 20;
  `importance_stability.csv`: per measure, the mean Spearman correlation and top-20 overlap
  between consecutive folds' importance vectors. Folds without an importance table are left out;
- `summary.md`: the mean and median `val_net_sharpe`, the means of the headline diagnostics (a
  fold where one is undefined is not counted in its mean) and the 20 top-ranked features.

## Adding a model

1. Subclass `ml4trading.models.Model` and implement `fit_predict_fold(frames, fit_window,
   predict_window)` (the contract is in `ml4trading/models/base.py`). `frames`, built once per
   run by `build_frames`, holds the labelled panel over the model's pool and, for a model that
   needs them, features row-aligned with it; fit on the labelled rows inside `fit_window` and
   predict the periods of `predict_window`, where a prediction for period *t* may only use what
   was known at *t*'s open. A model on the features subclasses
   `ml4trading.models.learned.base.LearnedModel` and implements only `_fit`.
2. Register it with `@register_model("my_model")`, import its module in
   `ml4trading/models/__init__.py`, and add its defaults to `METHOD_DEFAULTS` in
   `ml4trading/config.py` (`K`, `model.hold_overnight`, and any trader overrides).
3. Check causality: `ml4trading.testing.assert_causal(model, bars, dividends, K, pool,
   fit_window, predict_window)` scrambles every bar from period *t* on, rebuilds the frames, fits
   and predicts again, and fails if the prediction for *t* changes.
4. Search hyperparameters per fold with a grid file — dotted `model.*` keys to lists:

   ```yaml
   model.alpha: [0.1, 1.0, 10.0]
   ```

   `python -m ml4trading.train --method my_model --grid grid.yaml --out runs/my_model`. A key
   the frames are built from (`Model.FRAME_PARAMS`, such as `model.hold_overnight`) is fixed for
   the run and cannot be searched.

## Repository layout

```
ml4trading/   the package (one module per stage above, models/ for the methods, features/ for
              the learned models' inputs)
grids/        the learned models' per-fold hyperparameter grids
tests/        unit and end-to-end tests on a synthetic dataset (no real data needed)
DATA.md       the dataset card: contents, construction, known limitations
```
