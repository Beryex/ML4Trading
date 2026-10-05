"""Performance metrics of a book (``ml4trading.book.run_book``'s per-period frame).

* Sharpe: returns per session (the sum of that session's per-period returns; for N-session
  periods, per period), mean / std (population std), x sqrt(sessions-or-periods per year: 252,
  or 252 / N); 0 when the std is 0. Net uses ``ret``; gross uses the gross P&L over the same
  equity.
* Max drawdown: the worst (wealth - running peak) / running peak along the per-period wealth
  path cumprod(1 + ret), the peak running from the first period's wealth; -1 if wealth ever
  reaches 0.
* Cumulative return: final equity / starting equity - 1; net profit: the same in dollars.
* Alpha / beta against a benchmark's returns over the same periods (VOO): the OLS of the
  book's Sharpe series on the benchmark's; alpha is the intercept x periods per year (the
  return left after the market exposure beta), with its t-statistic.
* Trades (``trade_stats``): a trade runs from the period a symbol's position opens from flat
  (or flips sign) to the period it is flat again (or flips); adds and trims inside it belong to
  it. Its P&L is its gross P&L minus every entry and exit cost and borrow fee it incurred. The
  hit rate is the share of COMPLETED trades with P&L > 0; positions still open at the end are
  counted separately (a buy-and-hold book has none completed, so no hit rate).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import SESSION_MINUTES, TRADING_DAYS_PER_YEAR


def sharpe(returns, per_year: float = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    if r.size and np.std(r) > 0:
        return float(np.mean(r) / np.std(r) * np.sqrt(per_year))
    return 0.0


def series_per_year(k: int) -> float:
    """How many of the Sharpe series' returns make a year: 252 sessions, or 252 / N blocks."""
    return TRADING_DAYS_PER_YEAR / (k // SESSION_MINUTES) if k > SESSION_MINUTES else 252.0


def max_drawdown(wealth) -> float:
    w = np.asarray(wealth, dtype=float)
    if w.size == 0:
        return 0.0
    if np.any(w <= 0):
        return -1.0
    peak = np.maximum.accumulate(w)
    return float(np.min((w - peak) / peak))


def alpha_beta(returns: pd.Series, benchmark: pd.Series, per_year: float) -> dict:
    """OLS of ``returns`` on ``benchmark`` over their common index (module docstring)."""
    joined = pd.concat([returns.rename("r"), benchmark.rename("b")], axis=1, join="inner").dropna()
    n = len(joined)
    if n < 3 or joined["b"].var(ddof=0) == 0:
        return {"alpha": None, "beta": None, "alpha_t": None, "n_benchmark_periods": n}
    r, b = joined["r"].to_numpy(), joined["b"].to_numpy()
    b_dev = b - b.mean()
    beta = float((b_dev * (r - r.mean())).sum() / (b_dev**2).sum())
    alpha = float(r.mean() - beta * b.mean())
    resid = r - alpha - beta * b
    s2 = float((resid**2).sum() / (n - 2))
    se = np.sqrt(s2 * (1.0 / n + b.mean() ** 2 / (b_dev**2).sum()))
    return {
        "alpha": alpha * per_year,
        "beta": beta,
        "alpha_t": float(alpha / se) if se > 0 else None,
        "n_benchmark_periods": n,
    }


def trade_stats(positions: pd.DataFrame) -> dict:
    """Completed-trade statistics from ``run_book``'s positions frame (module docstring)."""
    pnls, lengths, open_at_end = [], [], 0
    cols = ["prev_shares", "shares", "gross", "borrow"]
    for _, g in positions.sort_values(["symbol", "period"], kind="stable").groupby("symbol"):
        pnl, length, is_open = 0.0, 0, False
        entry_cost = (g["entry_spread"] + g["entry_commission"]).to_numpy()
        exit_cost = (g["exit_spread"] + g["exit_commission"]).to_numpy()
        close_cost = (g["close_spread"] + g["close_commission"]).to_numpy()
        session_end = g["session_end"].to_numpy(dtype=bool)
        for i, (before, after, gross, borrow) in enumerate(g[cols].itertuples(index=False)):
            if is_open:
                pnl -= exit_cost[i]  # trims and the closing leg belong to the open trade
                if after == 0 or before * after < 0:
                    pnls.append(pnl)
                    lengths.append(length)
                    is_open = False
            if after != 0:
                if not is_open:
                    pnl, length, is_open = 0.0, 0, True
                pnl += gross - borrow - entry_cost[i]
                length += 1
                if session_end[i]:  # intraday: flattened at the session close
                    pnls.append(pnl - close_cost[i])
                    lengths.append(length)
                    is_open = False
        open_at_end += int(is_open)
    n = len(pnls)
    return {
        "n_trades": n,
        "trade_hit_rate": float(np.mean(np.asarray(pnls) > 0)) if n else None,
        "mean_trade_pnl": float(np.mean(pnls)) if n else None,
        "median_trade_periods": float(np.median(lengths)) if n else None,
        "open_trades_at_end": open_at_end,
    }


def summarize(
    book: pd.DataFrame,
    k: int = SESSION_MINUTES,
    positions: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
) -> dict:
    """The metrics of a book whose periods are ``k`` minutes long; trade statistics when
    ``positions`` is given, alpha/beta when ``benchmark`` (returns keyed like the book's
    ``session`` column) is."""
    if book.empty:
        return {"n_periods": 0}
    per_year = series_per_year(k)
    daily = book.groupby("session", sort=True)
    net_daily = daily["ret"].sum()
    gross_ret = book["gross"] / book["equity_before"].where(book["equity_before"] > 0)
    gross_daily = gross_ret.fillna(0.0).groupby(book["session"]).sum()
    wealth = np.cumprod(1.0 + book["ret"].to_numpy(dtype=float))
    start_equity = float(book["equity_before"].iloc[0])
    end_equity = float(book["equity"].iloc[-1])
    n_periods = int(net_daily.size)
    years = n_periods / per_year
    growth = end_equity / start_equity
    out = {
        "start": str(book["session"].iloc[0].date()),
        "end": str(book["session"].iloc[-1].date()),
        "n_periods": n_periods,
        "net_sharpe": sharpe(net_daily, per_year),
        "gross_sharpe": sharpe(gross_daily, per_year),
        "max_drawdown": max_drawdown(wealth),
        "net_profit": end_equity - start_equity,
        "cumulative_return": growth - 1.0,
        "annualized_return": growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else -1.0,
        "annualized_volatility": float(np.std(net_daily) * np.sqrt(per_year)),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_spread_cost": float(book["spread_cost"].sum()),
        "total_commission": float(book["commission"].sum()),
        "total_borrow": float(book["borrow"].sum()),
        "total_traded_notional": float(book["traded_notional"].sum()),
    }
    if benchmark is not None:
        out.update(alpha_beta(net_daily, benchmark, per_year))
    if positions is not None:
        out.update(trade_stats(positions))
    return out
