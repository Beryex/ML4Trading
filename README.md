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
- **Universe** (`selector.py`). Per fold, symbols with enough labelled sessions and a median
  dollar volume of at least $1M, the 100 most liquid kept.
- **Trader** (`trader.py`). Per period, the top-N predictions by magnitude (long or short),
  weighted proportionally to the prediction, capped per name and in total, then a cost gate:
  a name is traded only if its predicted return beats its estimated round-trip cost.
- **Shares and book** (`sizing.py`, `book.py`). Weights become whole shares within the budget
  `min(capital, equity)`, with a switching-friction rule that avoids trades whose cost exceeds
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
| `investment` (VOO) | 0.772 | $20,616 | +103.1 % | −31.8 % | −1.0 % | 0.79 |
| `investment_tech` (QQQ) | 0.859 | $27,831 | +139.2 % | −26.9 % | +1.3 % | 0.82 |

Both size against `min(capital, equity)`, so once equity grows past the $20,000 basis the
excess stays in cash -- which is why the VOO book's beta to VOO is 0.79, not 1.

## Adding a model

1. Subclass `ml4trading.models.Model` and implement `fit(panel, window)` and
   `predict(panel, window)` (the contract is in `ml4trading/models/base.py`): a prediction for
   period *t* may only use data from periods before *t*.
2. Register it with `@register_model("my_model")`, import its module in
   `ml4trading/models/__init__.py`, and add its defaults to `METHOD_DEFAULTS` in
   `ml4trading/config.py` (`K`, `model.hold_overnight`, and any trader overrides).
3. Check causality: `ml4trading.testing.assert_causal(model, panel, window)` scrambles every row
   from period *t* on and fails if the prediction for *t* changes.
4. Search hyperparameters per fold with a grid file — dotted `model.*` keys to lists:

   ```yaml
   model.alpha: [0.1, 1.0, 10.0]
   ```

   `python -m ml4trading.train --method my_model --grid grid.yaml --out runs/my_model`

## Repository layout

```
ml4trading/   the package (one module per stage above, models/ for the methods)
tests/        unit and end-to-end tests on a synthetic dataset (no real data needed)
DATA.md       the dataset card: contents, construction, known limitations
```
