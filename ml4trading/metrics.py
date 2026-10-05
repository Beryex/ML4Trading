"""Performance metrics of a book (``ml4trading.book.run_book``'s per-period frame).

* Sharpe: per-session returns (the sum of that session's per-period returns), mean / std
  (population std), x sqrt(252); 0 when the std is 0. Net uses ``ret``; gross uses the gross
  P&L over the same equity.
* Max drawdown: the worst (wealth - running peak) / running peak along the per-period wealth
  path cumprod(1 + ret), the peak running from the first period's wealth; -1 if wealth ever
  reaches 0.
* Cumulative return: final equity / starting equity - 1.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import TRADING_DAYS_PER_YEAR


def sharpe(returns) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size and np.std(r) > 0:
        return float(np.mean(r) / np.std(r) * np.sqrt(TRADING_DAYS_PER_YEAR))
    return 0.0


def max_drawdown(wealth) -> float:
    w = np.asarray(wealth, dtype=float)
    if w.size == 0:
        return 0.0
    if np.any(w <= 0):
        return -1.0
    peak = np.maximum.accumulate(w)
    return float(np.min((w - peak) / peak))


def summarize(book: pd.DataFrame) -> dict:
    if book.empty:
        return {"n_sessions": 0}
    daily = book.groupby("session", sort=True)
    net_daily = daily["ret"].sum()
    gross_ret = book["gross"] / book["equity_before"].where(book["equity_before"] > 0)
    gross_daily = gross_ret.fillna(0.0).groupby(book["session"]).sum()
    wealth = np.cumprod(1.0 + book["ret"].to_numpy(dtype=float))
    start_equity = float(book["equity_before"].iloc[0])
    end_equity = float(book["equity"].iloc[-1])
    n_sessions = int(net_daily.size)
    years = n_sessions / TRADING_DAYS_PER_YEAR
    growth = end_equity / start_equity
    return {
        "start": str(book["session"].iloc[0].date()),
        "end": str(book["session"].iloc[-1].date()),
        "n_sessions": n_sessions,
        "net_sharpe": sharpe(net_daily),
        "gross_sharpe": sharpe(gross_daily),
        "max_drawdown": max_drawdown(wealth),
        "cumulative_return": growth - 1.0,
        "annualized_return": growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else -1.0,
        "annualized_volatility": float(np.std(net_daily) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_spread_cost": float(book["spread_cost"].sum()),
        "total_commission": float(book["commission"].sum()),
        "total_borrow": float(book["borrow"].sum()),
        "total_traded_notional": float(book["traded_notional"].sum()),
    }
