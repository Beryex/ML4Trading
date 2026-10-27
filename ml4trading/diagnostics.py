"""Validation diagnostics: why a model's validation Sharpe is what it is.

A Sharpe ratio alone cannot say whether a model ranks names badly, ranks them well but cannot pay
its costs, or earns on the market's drift instead of on its own forecasts. These pure functions
read what a fold scored on its validation window and break the result down:

* ``period_ic`` / ``ic_summary``: the per-period cross-sectional Spearman correlation between the
  prediction and the realized return (the information coefficient), and its mean, spread,
  t-statistic and hit rate. A period whose IC is undefined (one scored name, or constant
  predictions) is NaN and left out of the summary.
* ``leg_pnl``: the book's P&L split into its long and short legs, before and after costs.
* ``exposure_stats``: how often a held book is all one sign, and its mean net and gross exposure.
* ``gate_stats``: the trader's own selection and cost gate replayed on the scored rows -- how
  many of the top names pay their round-trip cost, and by what margin.
* ``dispersion``: the cross-sectional spread of the predictions next to that of the label.

``validation_diagnostics`` merges them (with the book's headline metrics and the out-of-sample
R² under squared error) into one JSON-safe dict.

A "scored" frame has the columns ``period, symbol, prediction, y, label, open_px``: ``y`` is the
raw realized return, ``label`` the label in the fit's units, ``open_px`` the entry price. The
book and positions frames are ``ml4trading.book.run_book``'s.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml4trading.book import POSITION_COLUMNS
from ml4trading.config import RunConfig
from ml4trading.models.learned.losses import oos_r2
from ml4trading.trader import apply_caps, cost_gate, decision_metric, select

#: The headline metrics ``validation_diagnostics`` takes from the book's metrics dict.
_METRIC_KEYS = ("net_sharpe", "gross_sharpe", "alpha", "beta", "alpha_t")

#: Every dollar cost a position row carries: the entry, exit and session-close spread and
#: commission, and the borrow fee.
_COST_COLUMNS = [
    c for c in POSITION_COLUMNS if c.endswith(("spread", "commission")) or c == "borrow"
]


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation over the rows where both are finite; NaN when fewer than two rows
    remain or either side is constant."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    rx, ry = pd.Series(x[ok]).rank().to_numpy(), pd.Series(y[ok]).rank().to_numpy()
    if rx.min() == rx.max() or ry.min() == ry.max():
        return float("nan")
    return float(np.clip(np.corrcoef(rx, ry)[0, 1], -1.0, 1.0))


def period_ic(scored: pd.DataFrame) -> pd.Series:
    """Per period, the Spearman correlation of ``prediction`` and ``y`` over the rows where both
    are finite; NaN when fewer than two such rows remain or either side is constant. Indexed by
    period, ascending; a period whose rows all drop out is still listed (as NaN)."""
    periods, values = [], []
    for period, g in scored.groupby("period", sort=True):
        periods.append(period)
        values.append(_spearman(g["prediction"].to_numpy(float), g["y"].to_numpy(float)))
    return pd.Series(values, index=pd.Index(periods, name="period"), dtype=float, name="ic")


def ic_summary(ic: pd.Series) -> dict:
    """Mean, std (ddof 1), t-statistic (mean / (std / sqrt(n))), share above zero and count of
    the defined periods; a value is None when it is undefined."""
    v = ic.dropna().to_numpy(float)
    n = len(v)
    out: dict = {
        "ic_mean": None,
        "ic_std": None,
        "ic_t": None,
        "ic_positive_share": None,
        "ic_periods": n,
    }
    if n:
        out["ic_mean"] = float(v.mean())
        out["ic_positive_share"] = float((v > 0).mean())
    if n > 1:
        std = float(v.std(ddof=1))
        out["ic_std"] = std
        if std > 0:
            out["ic_t"] = out["ic_mean"] / (std / math.sqrt(n))
    return out


def leg_pnl(positions: pd.DataFrame) -> dict:
    """The book's dollar P&L split by the side a row is on: gross, and net of every spread,
    commission and borrow column. A row belongs to the long leg when it ends long, to the short
    leg when it ends short, and -- when it ends flat -- to the side it just closed."""
    shares = positions["shares"].to_numpy(float)
    prev = positions["prev_shares"].to_numpy(float)
    side = np.where(shares != 0, np.sign(shares), np.sign(prev))
    gross = positions["gross"].to_numpy(float)
    net = gross - positions[_COST_COLUMNS].to_numpy(float).sum(axis=1)
    return {
        "long_gross": float(gross[side > 0].sum()),
        "long_net": float(net[side > 0].sum()),
        "short_gross": float(gross[side < 0].sum()),
        "short_net": float(net[side < 0].sum()),
    }


def exposure_stats(positions: pd.DataFrame, book: pd.DataFrame) -> dict:
    """Over the periods that hold at least one name (``shares != 0``): the share whose held
    names are all one sign, and the mean net (long minus short) and gross exposure,
    sum(shares x price) / ``equity_before``. A row with no price (a carried name) holds but adds
    no exposure. ``holding_share`` is the holding periods over all book periods; the other three
    are None when no period holds anything."""
    held = positions[positions["shares"].to_numpy(float) != 0]
    out: dict = {
        "same_sign_share": None,
        "mean_net_exposure": None,
        "mean_gross_exposure": None,
        "holding_share": None,
    }
    if len(book):
        out["holding_share"] = held["period"].nunique() / len(book)
    if held.empty:
        return out
    shares = held["shares"].to_numpy(float)
    value = shares * pd.to_numeric(held["price"], errors="coerce").to_numpy(float)
    rows = pd.DataFrame(
        {
            "period": held["period"].to_numpy(),
            "sign": np.sign(shares),
            "net": value,
            "gross": np.abs(value),
        }
    )
    by_period = rows.groupby("period")
    same_sign = (by_period["sign"].min() > 0) | (by_period["sign"].max() < 0)
    equity = book.set_index("period")["equity_before"].reindex(same_sign.index)
    out["same_sign_share"] = float(same_sign.mean())
    out["mean_net_exposure"] = float((by_period["net"].sum() / equity).mean())
    out["mean_gross_exposure"] = float((by_period["gross"].sum() / equity).mean())
    return out


def gate_stats(scored: pd.DataFrame, book: pd.DataFrame, cfg: RunConfig) -> dict:
    """The trader's steps replayed on each book period's scored rows: the decision metric, the
    selection, the capped proportional weights and the cost gate, sized against the period's
    equity when the run reinvests (else the capital), as ``run_book`` does.

    ``gate_selected_mean`` / ``gate_pass_mean``: the mean per period of the names selected and of
    those whose |prediction| exceeds their gate. ``gate_ratio_q10/50/90``: quantiles of
    |prediction| / gate over every selected name of every period (above 1 = pays its way).
    Periods the trader never trades -- no priced prediction, or no equity left -- do not count;
    every value is None when there is no such period."""
    trader, costs = cfg.trader, cfg.costs
    ok = np.isfinite(scored["prediction"].to_numpy(float)) & (scored["open_px"].to_numpy(float) > 0)
    rows = scored[ok]
    by_period = {p: g for p, g in rows.groupby("period", sort=False)}
    selected, passed, ratios = [], [], []
    for b in book.itertuples(index=False):
        g = by_period.get(b.period)
        if g is None or b.equity_before <= 0:
            continue
        g = g.sort_values("symbol", kind="stable")
        pred = g["prediction"].to_numpy(float)
        px = g["open_px"].to_numpy(float)
        metric = decision_metric(pred, cfg.hold_overnight, trader, costs)
        symbols = g["symbol"].to_numpy()
        pos = select(
            metric, symbols, trader.number_of_symbols_to_buy, trader.symbol_selection_strategy
        )
        selected.append(len(pos))
        if not len(pos):
            passed.append(0)
            continue
        absm = np.abs(metric[pos])
        w = apply_caps(absm / absm.sum(), trader.max_symbol_weight, trader.gross_exposure_cap)
        capital = b.equity_before if cfg.reinvest else cfg.capital
        gate = cost_gate(w, px[pos], trader=trader, costs=costs, capital=capital)
        passed.append(int((np.abs(pred[pos]) > gate).sum()))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios.extend(np.abs(pred[pos]) / gate)
    out: dict = dict.fromkeys(
        (
            "gate_selected_mean",
            "gate_pass_mean",
            "gate_ratio_q10",
            "gate_ratio_q50",
            "gate_ratio_q90",
        )
    )
    if selected:
        out["gate_selected_mean"] = float(np.mean(selected))
        out["gate_pass_mean"] = float(np.mean(passed))
    if ratios:
        q10, q50, q90 = np.quantile(ratios, [0.1, 0.5, 0.9])
        out.update(gate_ratio_q10=float(q10), gate_ratio_q50=float(q50), gate_ratio_q90=float(q90))
    return out


def dispersion(scored: pd.DataFrame) -> dict:
    """The median over periods of the cross-sectional std (ddof 0) of the prediction and of the
    label -- how far apart the model spreads its calls next to how far apart the outcomes are."""
    out: dict = {}
    for key, column in (("pred_xs_std", "prediction"), ("label_xs_std", "label")):
        values = scored[column].astype(float)
        values = values.where(np.isfinite(values))
        median = values.groupby(scored["period"]).std(ddof=0).median()
        out[key] = None if np.isnan(median) else float(median)
    return out


def _json_safe(value):
    """NaN and infinity become None, numpy scalars plain Python numbers."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def validation_diagnostics(
    scored: pd.DataFrame,
    book: pd.DataFrame,
    positions: pd.DataFrame,
    metrics: dict,
    cfg: RunConfig,
    fit_label_mean: float | None,
) -> dict:
    """Everything above in one flat dict, plus the book's headline ``metrics`` (None when
    absent) and ``r2_mse``: the out-of-sample R² of the predictions against the fit's mean label
    under squared error, over the rows with a finite label and prediction. Every NaN is None, so
    the dict is JSON-safe."""
    out: dict = {k: metrics.get(k) for k in _METRIC_KEYS}
    out.update(ic_summary(period_ic(scored)))
    out.update(leg_pnl(positions))
    out.update(exposure_stats(positions, book))
    out.update(gate_stats(scored, book, cfg))
    out.update(dispersion(scored))
    label, prediction = scored["label"].to_numpy(float), scored["prediction"].to_numpy(float)
    ok = np.isfinite(label) & np.isfinite(prediction)
    out["r2_mse"] = oos_r2(label[ok], prediction[ok], fit_label_mean, "mse")
    return _json_safe(out)
