"""The ``proportional`` trader: one period's predictions -> signed target weights.

1. Decision metric. Overnight: the prediction itself. Intraday (every period is a round trip):
   the prediction net of the gated spread cost, sign(p) * max(|p| - c, 0), so a name that
   cannot pay its own spread ranks at exactly 0.
2. Selection under the strategy -- BOTH (either sign), BUY_LONG (> 0), SELL_SHORT (< 0): the
   top ``number_of_symbols_to_buy`` eligible names (non-zero, non-NaN metric) by |metric|,
   descending, ties broken by symbol.
3. Weights proportional to |metric|, then the caps: scale the book down to
   ``gross_exposure_cap`` if it exceeds it, then water-fill ``max_symbol_weight`` (over-cap
   names pinned to the cap, the excess re-spread over the uncapped ones; whatever no name can
   absorb stays in cash).
4. Cost gate: a name survives only if |prediction| exceeds its round-trip cost estimate,
   ``cost_safety_factor`` x (round-trip spread fraction x spread + round-trip commission as a
   fraction of its estimated notional |w| x capital). Survivors are renormalized onto
   min(1, gross_exposure_cap) and capped again.
Long weights are positive, short weights negative.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import CostConfig, TraderConfig
from ml4trading.costs import commission, round_trip_spread_fraction

_EPS = 1e-12


def decision_metric(
    pred: np.ndarray, hold_overnight: bool, trader: TraderConfig, costs: CostConfig
):
    pred = np.asarray(pred, dtype=float)
    if hold_overnight:
        return pred
    gate = costs.spread * trader.cost_safety_factor * round_trip_spread_fraction(costs)
    net = np.abs(pred) - gate
    out = np.sign(pred) * np.maximum(net, 0.0)
    out[~(net > 0.0)] = 0.0
    return out


def select(metric: np.ndarray, symbols: np.ndarray, n: int, strategy: str) -> np.ndarray:
    """Positions of the selected rows, in rank order (|metric| desc, then symbol)."""
    ok = ~np.isnan(metric)
    if strategy == "BOTH":
        ok &= metric != 0.0
    elif strategy == "BUY_LONG":
        ok &= metric > 0.0
    elif strategy == "SELL_SHORT":
        ok &= metric < 0.0
    else:
        raise ValueError(f"unknown strategy {strategy!r}")
    pos = np.flatnonzero(ok)
    order = sorted(pos, key=lambda i: (-abs(metric[i]), str(symbols[i])))
    return np.asarray(order[:n], dtype=np.int64)


def apply_caps(weights: np.ndarray, max_symbol_weight: float, gross_exposure_cap: float):
    """Gross scale-down, then the per-name water-fill (module docstring, step 3)."""
    w = np.asarray(weights, dtype=float).copy()
    if w.size == 0:
        return w
    total = float(w.sum())
    if total > gross_exposure_cap + _EPS:
        w *= gross_exposure_cap / total
    capped = np.zeros(len(w), dtype=bool)
    for _ in range(len(w)):
        over = (w > max_symbol_weight + _EPS) & ~capped
        if not over.any():
            break
        excess = float((w[over] - max_symbol_weight).sum())
        w[over] = max_symbol_weight
        capped |= over
        free = ~capped
        free_sum = float(w[free].sum())
        if not free.any() or free_sum <= 0.0:
            break
        w[free] += excess * (w[free] / free_sum)
    return w


def target_weights(
    scores: pd.DataFrame,
    *,
    trader: TraderConfig,
    costs: CostConfig,
    capital: float,
    hold_overnight: bool,
) -> dict[str, float]:
    """``scores``: one period's rows with ``symbol``, ``prediction`` and ``open_px`` (the
    entry price). Returns {symbol: signed weight} for the names to hold."""
    if trader.name != "proportional":
        raise ValueError(f"unknown trader {trader.name!r}")
    if scores.empty:
        return {}
    symbols = scores["symbol"].to_numpy()
    pred = scores["prediction"].to_numpy(dtype=float)
    px = scores["open_px"].to_numpy(dtype=float)
    metric = decision_metric(pred, hold_overnight, trader, costs)
    pos = select(metric, symbols, trader.number_of_symbols_to_buy, trader.symbol_selection_strategy)
    if pos.size == 0:
        return {}
    absm = np.abs(metric[pos])
    w = apply_caps(absm / absm.sum(), trader.max_symbol_weight, trader.gross_exposure_cap)

    # cost gate
    k = trader.cost_safety_factor
    notional = w * float(capital)
    with np.errstate(divide="ignore", invalid="ignore"):
        est_shares = np.where(notional > 0.0, notional / px[pos], 0.0)
        comm = 2.0 * np.asarray(commission(est_shares, costs), dtype=float)
        comm_frac = np.where(notional > 0.0, comm / notional, np.inf)
    gate = k * round_trip_spread_fraction(costs) * costs.spread + k * comm_frac
    keep = np.abs(pred[pos]) > gate
    pos, w = pos[keep], w[keep]
    if pos.size == 0:
        return {}
    budget = min(1.0, trader.gross_exposure_cap)
    w = apply_caps(w / w.sum() * budget, trader.max_symbol_weight, trader.gross_exposure_cap)
    sign = np.where(metric[pos] > 0, 1.0, -1.0)
    return {
        str(symbols[i]): float(s * wi) for i, s, wi in zip(pos, sign, w, strict=True) if wi > 0.0
    }
