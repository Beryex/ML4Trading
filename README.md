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
  looks one period ahead. The test months, concatenated, are the result. The validation window
  ends the day before `B` and has no embargo after it, so its last period's overnight label
  exits at the first test session's open (plus a dividend ex-dated that day). That one price
  reaches the validation score and, under a hyperparameter search, the stopping round and the
  chosen configuration; nothing later in the test month does, and the deploy fit, which stops
  5 sessions before `B`, is unaffected.
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
- **Slow features** (`--set model.extra_features=true`; both models, off by default): thirteen
  cross-sectional features from the literature, built from the same bars
  (`ml4trading/features/slow.py`) and added after the others: `mom_12_1` and `mom_6_1`
  (Jegadeesh & Titman 1993), `mom_1m` (Jegadeesh 1990), `high_52w` (George & Hwang 2004),
  `max_21` (Bali, Cakici & Whitelaw 2011), `beta_252` (Frazzini & Pedersen 2014), `ivol_63`
  (Ang, Hodrick, Xing & Zhang 2006), `resid_mom_12_1` (Blitz, Huij & Martens 2011), `amihud_21`
  (Amihud 2002), `rskew_21` (Amaya, Christoffersen, Jacobs & Vasquez 2015), `overnight_21` and
  `intraday_21` (Lou, Polk & Skouras 2019), and `volatility_252`. Each is a ratio, a return or a
  dollar volume, so a split leaves it unchanged (a test holds it), and each is NaN until its
  window of the stock's own sessions is full -- the year-long ones for a stock's first 252
  sessions. The market-relative ones (`beta_252`, `ivol_63`, `resid_mom_12_1`) are measured
  against VOO on the sessions both have. They are columns like any other: `cs_rank` ranks them,
  and a feature set may name them (refused by name unless `extra_features` is on). The option
  is built into the run's frames, so it is fixed for a run and cannot be searched in a grid.
- **Volatility-scaled target** (`--set model.risk_adjust=vol` or `vol_return`, with
  `model.vol_window`, 63 sessions; both models, `none` by default): the trader ranks and weights
  names by |prediction|, so a model fitted on the raw return favours the volatile names. Here the
  fit's label is divided by the symbol's trailing volatility -- the standard deviation of its own
  one-session returns over `vol_window` rows, in the style of Moskowitz, Ooi & Pedersen (2012,
  scaling by ex-ante volatility) with a short trailing window (Ang, Hodrick, Xing & Zhang 2006).
  The window ends two rows before the row it scales: the previous row's return runs up to the
  row's own open, which a prediction may not read. `vol` multiplies the model's output by the
  median volatility of that session's stocks, so the ranking and the weights follow the
  risk-adjusted forecast and the units stay a return's; `vol_return` multiplies each stock by its
  own volatility, a per-stock return forecast. A row with fewer than 21 finite returns in its
  window has no volatility: it is not fitted on, and predicts nothing. The validation label, the
  early-stopping set and the R² are per unit of volatility. It is a fit-time option, so a grid
  may search it.
- **`linear`**: an elastic net on the standardized features. A missing value is replaced by the
  median of its feature over the same session's stocks; the penalty is a fraction `alpha_ratio`
  of the smallest one that zeroes every coefficient (scikit-learn's `ElasticNetCV` convention).
  The stock code is not used. `--set model.feature_set=<name>` (or a list in a grid) trains on
  the columns of a named subset instead of every one (`all`, the default); the format is in
  `ml4trading/features/feature_sets/FORMAT.md`. A set may list `stock_id`, which is left out
  all the same, and a column the features do not have is refused by name.
- **`nonlinear`**: LightGBM, by default on the absolute error as in ML4Investment, with the stock
  code as a categorical feature; missing values go in as they are. Fits are deterministic (a
  fixed seed and thread count). Its `model.*` options (`--set model.label=demean`, or a list in
  a grid), at their defaults in `ml4trading/config.py`; the defaults are the plain fit:
  - `label`: what the fit trains on in place of the raw next-open return. `raw` (default);
    `demean`, less the mean of the period's stocks, so the model learns which names beat the
    others, not where the market went; `winsor`, clipped to the 0.5 % and 99.5 % quantiles of the
    fold's fit labels (ML4Investment's `CLIP_LOWER_QUANTILE_RATIO` and
    `CLIP_UPPER_QUANTILE_RATIO`); `demean_winsor`, both, demeaning first. The validation scoring
    puts the realized returns through the same transform.
  - `feature_transform`: `none` (default), or `cs_rank`, which replaces each stock-specific
    feature, period by period, by its cross-sectional rank scaled to [-1, 1] (Gu, Kelly & Xiu 2020,
    "Empirical Asset Pricing via Machine Learning"). The stock code, VOO's and QQQ's columns and
    the calendar columns are left as they are. The ranks are built into the run's frames, so this
    one is fixed for a run and cannot be searched in a grid.
  - `stock_id`: `categorical` (default) gives LightGBM the stock code as a categorical feature;
    `drop` leaves it out.
  - `feature_set`: `all` (default) trains on every column; any other name is a file
    `ml4trading/features/feature_sets/<name>.txt` listing the columns to train on, one per line
    (the format is in `feature_sets/FORMAT.md`). A set has to list `stock_id` unless `stock_id`
    is `drop`, and a column the features do not have is refused by name.
  - `cat_smooth` (10.0), `min_data_per_group` (100), `max_cat_threshold` (32) and `cat_l2`
    (10.0): how LightGBM splits on the categorical stock code, at LightGBM's own defaults
    ([Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html), "Learning Control
    Parameters"). They have no effect under `stock_id=drop`.
  - `loss` (`l1` default, ML4Investment's `TRAIN_OBJECTIVE`; or `mse`), `n_estimators` (100) and
    LightGBM's `num_leaves`, `min_data_in_leaf`, `learning_rate`, `feature_fraction`,
    `bagging_fraction`, `bagging_freq`, `lambda_l1`, `lambda_l2`, `max_depth`,
    `min_gain_to_split`, `min_sum_hessian_in_leaf` and `max_bin`, all at LightGBM's defaults.
- **Grids** (`grids/linear.yaml`, `grids/nonlinear.yaml`) are searched per fold on validation net
  Sharpe like any method's. Each fold also records every candidate's out-of-sample R², against
  the constant fitted on its training rows (`candidate_val_r2` in `fold.json`); it is never
  ranked on. R² (and `r2_mse` below) is measured in each run's own label units, so it compares
  only across runs with the same `model.label` and `model.risk_adjust`; net Sharpe and IC compare across label
  variants. For `nonlinear`, a per-fold search can take a grid's place (see below).

```bash
python -m ml4trading.train --method linear --grid grids/linear.yaml --out runs/linear
python -m ml4trading.backtest --run runs/linear
```

### Hyperparameter search

`--hpo-hours H` searches `nonlinear`'s parameters per fold in place of a grid. Each fold runs an
[Optuna](https://optuna.org) study (the TPE sampler, seeded with 0) over its share of the budget:

```bash
python -m ml4trading.train --method nonlinear --out runs/nonlinear --hpo-hours 12
```

- **What is searched**, all on a log scale: `num_leaves` 15-255, `learning_rate` 0.005-0.5,
  `min_data_in_leaf` 20-5000, `lambda_l1` and `lambda_l2` 1e-8-1. These are ML4Investment's
  ranges (`model_training.py`), except `min_data_in_leaf`: it searches 20-100 on a linear scale,
  and the range is widened to suit a fold's fit window (LightGBM's "Parameters Tuning" guide:
  "hundreds or thousands is enough for a large dataset").
- **A trial** fits the model on the fold's training window and stops it early on the validation
  window: at most 1,000 rounds (ML4Investment's `NUM_ROUNDS`), stopping once the squared error on
  the universe's labelled validation rows has not improved for 100 rounds. The number of rounds
  is found by the fit, not searched. The trial's value is that squared error at the round the
  fit kept, so values compare across learning rates, and the winner is the fold's lowest-value
  trial (ranked on squared error, not on net Sharpe). The first trial is the run's own
  configuration (`--set model.*`, or the defaults), with a zero `lambda_l1` / `lambda_l2` raised
  to the lower bound 1e-8: every fold has a winner, and the search can only improve on it.
- **The sampler starts at random.** Optuna's TPE sampler draws its first 10 trials at random
  (its default `n_startup_trials`), and the enqueued baseline counts as one of them; TPE guides
  the search only after that. A trial may run up to 1,000 rounds, so it can cost several times a
  100-round default fit, and a fold's share of a small budget may end within the random phase.
  Each fold's realized trial count is `fold.json["hpo"]["n_trials"]` (also a column of
  `diagnose`'s `folds.csv`).
- **The winner is one candidate.** Its validation predictions are scored, kept and diagnosed
  like a grid winner's, so `candidate_val_net_sharpe` and `candidate_val_r2` have one entry and
  `chosen` is 0. The deploy fit trains on the deploy window with the winner's five parameters and
  `n_estimators` set to the round the winner kept.
- **The budget** is H hours for the whole run, split evenly across the folds: `--hpo-hours 12`
  over the default 80 folds gives each 9 minutes. A fold starts no trial once its share is
  spent, but the trial in progress finishes and the first trial always runs, so a fold takes at
  least one fit and may overrun its share by up to one.
- **Validation metrics are optimistic.** As for any grid winner, a fold's `val_net_sharpe`,
  `val_diagnostics` and R² for the HPO winner are measured on the window the winner was chosen
  on (and whose squared error stopped its rounds), and the importance tables come from that same
  window. They say how well the search fitted that window, not how the winner will do; the test
  months are the only out-of-sample result.
- **A time budget makes a run machine-dependent.** The study is seeded, so its sequence of
  points is fixed, but how many trials fit in a fold's share depends on the machine and its
  load. A fold that gets further can find a better point, and so a different winner, deploy fit
  and predictions. A re-run, or the same run on another machine, can therefore differ; read
  `n_trials` in each fold's record. `train(..., hpo_trials=N)` (Python only) runs N trials per
  fold instead, and that reproduces.
- **Record.** `run.json["hpo"]` holds the request -- `hours`, `trials` (null when not given) --
  and the per-fold `timeout_s` it came to; a run without a search has no such key.
  `fold.json["hpo"]` holds `timeout_s` (the fold's share, null when only `hpo_trials` was given),
  `n_trials`, `best_number` and `trials`: for each, its `number`, `params`, `value` (validation
  squared error), `best_iteration` and `seconds`. A fold with nothing to fit (an empty universe,
  or no fit window) is not searched: `n_trials` is 0 and it deploys the run's own configuration.
- HPO needs `--method nonlinear`, no `--grid` and a positive H; any other request is refused
  before the data is loaded.

### Validation diagnostics

A fold's validation Sharpe says how a candidate did, not why. For every fold, `train` also keeps
what the chosen candidate's validation pass produced (nothing is refitted):

- `val/predictions.parquet`: the scored validation rows -- `period, symbol, prediction`, the
  realized return `y` and the `label` the fit was trained on (rows without a realized return are
  not scored);
- `val/daily.parquet`: that same validation book valued once per session -- `session`, the net
  daily return `ret` and `gross_ret` (before the period's costs); `ret` is the series behind
  `val_diagnostics["net_sharpe"]`, so variants' validation returns can be correlated and
  blended. A fold whose validation book is empty keeps an empty frame with those columns;
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
  `candidate_val_net_sharpe[chosen]`) and every `val_diagnostics` key, then for a run with a
  hyperparameter search `hpo_n_trials` and `hpo_best_iteration` (the round the winning trial
  kept); `by_test_year.csv`: their mean per test year, with the number of folds;
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
   `ml4trading.models.learned.base.LearnedModel` and implements only `_fit(X_fit, y_fit)`. Its
   third parameter, `eval_set`, is optional: it is passed only when a fit asks to stop early
   (the hyperparameter search does), so a model that cannot may leave it out.
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
