"""The integer-share book: predictions -> trades -> a dollar equity curve.

The walk visits the decision periods in order, carrying ``equity`` (starts at the capital basis)
and the previous share book. At period t:

1. TRADEABLE names are those with an entry price (open_px) and an exit price this period. The
   trader turns their predictions into target weights (``ml4trading.trader``).
2. Held names that are not tradeable this period are CARRIED (``ml4trading.sizing``).
3. Shares are sized against the budget min(capital, equity) at the entry prices.
4. Costs: each name's change of shares is split into entry/exit legs at its entry price
   (``ml4trading.costs``); short notional pays a borrow fee per period.
5. Gross P&L: shares x (exit - entry) + shares x entry x div_ret (longs receive dividends,
   shorts pay them). Overnight the exit is the next period's open, so a position held across
   periods telescopes exactly; intraday the exit is the period's close and the book is
   flattened at every session's last period (the flattening exit legs are charged there).
6. ret_t = net$_t / equity_before_t and equity grows by net$_t. Equity <= 0 is ruin: nothing is
   held afterwards.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import SESSION_MINUTES, TRADING_DAYS_PER_YEAR, RunConfig
from ml4trading.costs import leg_amounts, leg_cost
from ml4trading.sizing import size_shares
from ml4trading.trader import target_weights


def periods_per_year(k: int) -> float:
    """Decision periods per trading year: 252 x 390 / K up to one session, 252 / N for N-session
    periods."""
    if k > SESSION_MINUTES:
        return TRADING_DAYS_PER_YEAR / (k // SESSION_MINUTES)
    return TRADING_DAYS_PER_YEAR * SESSION_MINUTES / k


def run_book(predictions: pd.DataFrame, panel: pd.DataFrame, periods, cfg: RunConfig):
    """Walk ``periods`` (period keys, ascending). ``predictions``: period, symbol, prediction.
    ``panel``: the labelled panel rows (``ml4trading.periods``) of every symbol that can be
    predicted. Returns (per-period frame, long-form positions frame)."""
    periods = list(pd.DatetimeIndex(periods).sort_values())
    marks = panel[np.isfinite(panel["open_px"]) & (panel["open_px"] > 0)]
    marks = marks[np.isfinite(marks["exit_px"]) & (marks["exit_px"] > 0)]
    marks = marks[marks["period"].isin(set(periods))]
    marks_by_period = {p: g for p, g in marks.groupby("period", sort=False)}
    preds = predictions.dropna(subset=["prediction"])
    preds_by_period = {p: g for p, g in preds.groupby("period", sort=False)}
    session_of = {p: p.normalize() for p in periods}
    last_of_session = {}
    for p in periods:
        last_of_session[session_of[p]] = p

    costs, trader = cfg.costs, cfg.trader
    borrow_per_period = costs.borrow_rate_annual / periods_per_year(cfg.K)
    equity = float(cfg.capital)
    prev: dict = {}
    last_price: dict = {}
    rows, pos_rows = [], []
    prev_session = None
    for p in periods:
        equity_before = equity
        m = marks_by_period.get(p)
        if m is None:
            m = marks.iloc[:0]
        if not cfg.hold_overnight and session_of[p] != prev_session:
            prev = {}  # intraday: every session opens flat
        prev_session = session_of[p]
        prices = dict(zip(m["symbol"], m["open_px"].astype(float), strict=True))
        exits = dict(zip(m["symbol"], m["exit_px"].astype(float), strict=True))
        divs = dict(zip(m["symbol"], m["div_ret"].astype(float), strict=True))
        if equity <= 0:
            shares: dict = {}
        else:
            scored = preds_by_period.get(p)
            weights = {}
            if scored is not None:
                s = scored.merge(m[["symbol", "open_px"]], on="symbol", how="inner")
                weights = target_weights(
                    s.sort_values("symbol", kind="stable"),
                    trader=trader,
                    costs=costs,
                    capital=cfg.capital,
                    hold_overnight=cfg.hold_overnight,
                )
            carried = {sym: last_price.get(sym, np.nan) for sym in prev if sym not in prices}
            shares = size_shares(
                weights,
                prices,
                min(cfg.capital, equity),
                prev,
                carried,
                trader=trader,
                costs=costs,
            )

        spread_cost = commission = traded = 0.0
        for sym in sorted(set(prev) | set(shares)):
            if sym not in prices:
                continue  # carried: no trade
            entry, exit_ = leg_amounts(prev.get(sym, 0.0), shares.get(sym, 0.0))
            if entry == 0.0 and exit_ == 0.0:
                continue
            sc, cm = leg_cost(entry, exit_, prices[sym], costs)
            spread_cost += sc
            commission += cm
            traded += (entry + exit_) * prices[sym]

        gross = borrow = 0.0
        for sym, n in shares.items():
            if sym in prices:
                gross += n * (exits[sym] - prices[sym]) + n * prices[sym] * divs[sym]
                mark = prices[sym]
            else:
                mark = last_price.get(sym, np.nan)
            if n < 0 and mark == mark:
                borrow += abs(n) * mark * borrow_per_period

        if not cfg.hold_overnight and last_of_session[session_of[p]] == p:
            for sym, n in shares.items():  # the session-close flatten
                px = exits.get(sym, last_price.get(sym, np.nan))
                if px == px and n:
                    sc, cm = leg_cost(0.0, abs(n), px, costs)
                    spread_cost += sc
                    commission += cm
                    traded += abs(n) * px

        net = gross - spread_cost - commission - borrow
        equity = equity_before + net
        rows.append(
            {
                "period": p,
                "session": session_of[p],
                "equity_before": equity_before,
                "gross": gross,
                "spread_cost": spread_cost,
                "commission": commission,
                "borrow": borrow,
                "net": net,
                "ret": net / equity_before if equity_before > 0 else 0.0,
                "equity": equity,
                "n_positions": sum(1 for n in shares.values() if n),
                "traded_notional": traded,
            }
        )
        for sym, n in shares.items():
            pos_rows.append({"period": p, "symbol": sym, "shares": n, "price": prices.get(sym)})
        last_price.update(prices)
        prev = shares
    book = pd.DataFrame(rows)
    positions = pd.DataFrame(pos_rows, columns=["period", "symbol", "shares", "price"])
    return book, positions
