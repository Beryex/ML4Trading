"""Performance metrics of a book (``ml4trading.book.run_book``'s per-period frame).

* Every return-based metric reads the account's DAILY value (``ml4trading.book.daily_values``),
  whatever the method's trading frequency, so all methods are measured the same way.
* Sharpe: daily returns, mean / std (population std) x sqrt(252), no risk-free rate; 0 when the
  std is 0. Net uses ``ret``; gross the same change before costs.
* Max drawdown: the worst (value - running peak) / running peak of the daily account value,
  the peak running from the first day's; -1 if the value ever reaches 0.
* Cumulative return: final equity / starting equity - 1; net profit: the same in dollars.
* Alpha / beta against the benchmark's daily returns (VOO, open to next open with dividends):
  the OLS of the account's daily returns on them; alpha is the intercept x 252 (the return left
  after the market exposure beta), with its t-statistic.
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


def trades(positions: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """(one row per COMPLETED trade -- symbol, start, end, periods, pnl --, the number still
    open at the end) from ``run_book``'s positions frame (module docstring). ``end`` is the
    period whose trades close it (the closing leg executes at that period's open)."""
    out, open_at_end = [], 0
    cols = ["period", "prev_shares", "shares", "gross", "borrow"]
    for sym, g in positions.sort_values(["symbol", "period"], kind="stable").groupby("symbol"):
        pnl, length, start, is_open = 0.0, 0, None, False
        entry_cost = (g["entry_spread"] + g["entry_commission"]).to_numpy()
        exit_cost = (g["exit_spread"] + g["exit_commission"]).to_numpy()
        close_cost = (g["close_spread"] + g["close_commission"]).to_numpy()
        session_end = g["session_end"].to_numpy(dtype=bool)
        for i, (period, before, after, gross, borrow) in enumerate(g[cols].itertuples(index=False)):
            if is_open:
                pnl -= exit_cost[i]  # trims and the closing leg belong to the open trade
                if after == 0 or before * after < 0:
                    out.append((sym, start, period, length, pnl))
                    is_open = False
            if after != 0:
                if not is_open:
                    pnl, length, start, is_open = 0.0, 0, period, True
                pnl += gross - borrow - entry_cost[i]
                length += 1
                if session_end[i]:  # intraday: flattened at the session close
                    out.append((sym, start, period, length, pnl - close_cost[i]))
                    is_open = False
        open_at_end += int(is_open)
    frame = pd.DataFrame(out, columns=["symbol", "start", "end", "periods", "pnl"])
    frame["start"] = pd.to_datetime(frame["start"])
    frame["end"] = pd.to_datetime(frame["end"])
    return frame, open_at_end


def _trade_summary(done: pd.DataFrame, open_at_end: int) -> dict:
    n = len(done)
    return {
        "n_trades": n,
        "trade_hit_rate": float((done["pnl"] > 0).mean()) if n else None,
        "mean_trade_pnl": float(done["pnl"].mean()) if n else None,
        "median_trade_periods": float(done["periods"].median()) if n else None,
        "open_trades_at_end": open_at_end,
    }


def trade_stats(positions: pd.DataFrame) -> dict:
    """Completed-trade statistics from ``run_book``'s positions frame (module docstring)."""
    return _trade_summary(*trades(positions))


def summarize(
    book: pd.DataFrame,
    k: int = SESSION_MINUTES,
    positions: pd.DataFrame | None = None,
    benchmark: pd.Series | None = None,
    daily: pd.DataFrame | None = None,
) -> dict:
    """The metrics of a book whose periods are ``k`` minutes long. Every return-based metric
    reads the DAILY account value (``ml4trading.book.daily_values``; derived from the book when
    a period is at most a session, required for longer periods), so methods that trade at
    different frequencies are measured the same way. Trade statistics when ``positions`` is
    given, alpha/beta when ``benchmark`` (daily returns keyed by session) is."""
    if book.empty:
        return {"n_periods": 0, "n_days": 0}
    if daily is None:
        if k > SESSION_MINUTES:
            raise ValueError(f"K={k}: a multi-session book needs its daily values")
        from ml4trading.book import daily_values

        daily = daily_values(book, positions, k)
    net = daily.set_index("session")["ret"]
    gross = daily.set_index("session")["gross_ret"]
    start_equity = float(daily["equity_before"].iloc[0])
    end_equity = float(daily["equity"].iloc[-1])
    wealth = daily["equity"].to_numpy(dtype=float) / start_equity
    n_days = int(len(daily))
    years = n_days / TRADING_DAYS_PER_YEAR
    growth = end_equity / start_equity
    out = {
        "start": str(daily["session"].iloc[0].date()),
        "end": str(daily["session"].iloc[-1].date()),
        "n_periods": int(len(book)),
        "n_days": n_days,
        "net_sharpe": sharpe(net),
        "gross_sharpe": sharpe(gross),
        "max_drawdown": max_drawdown(wealth),
        "net_profit": end_equity - start_equity,
        "cumulative_return": growth - 1.0,
        "annualized_return": growth ** (1.0 / years) - 1.0 if years > 0 and growth > 0 else -1.0,
        "annualized_volatility": float(np.std(net) * np.sqrt(TRADING_DAYS_PER_YEAR)),
        "start_equity": start_equity,
        "end_equity": end_equity,
        "total_spread_cost": float(book["spread_cost"].sum()),
        "total_commission": float(book["commission"].sum()),
        "total_borrow": float(book["borrow"].sum()),
        "total_traded_notional": float(book["traded_notional"].sum()),
        "max_daily_residual": float(daily["residual"].abs().max()),
    }
    if benchmark is not None:
        out.update(alpha_beta(net, benchmark, TRADING_DAYS_PER_YEAR))
    if positions is not None:
        out.update(trade_stats(positions))
    return out


def summarize_by_year(
    book: pd.DataFrame,
    k: int,
    positions: pd.DataFrame,
    benchmark: pd.Series | None = None,
    daily: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per calendar year: ``summarize`` over that year's days (its profit, Sharpe,
    drawdown, alpha/beta) and the costs of the periods that start in it, with the trade
    statistics of the trades that CLOSED in it, and ``full_year`` -- whether the account was
    valued on the year's first and last NYSE sessions."""
    from ml4trading.book import daily_values
    from ml4trading.periods import nyse_sessions

    if daily is None:
        daily = daily_values(book, positions, k)
    sessions = nyse_sessions()
    done, _ = trades(positions)
    first, last = daily["session"].min(), daily["session"].max()
    rows = []
    for year, days in daily.groupby(daily["session"].dt.year, sort=True):
        in_year = sessions[sessions.year == year]
        part = book[book["session"].dt.year == year].reset_index(drop=True)
        row = {"year": int(year)}
        row.update(summarize(part, k, None, benchmark, days.reset_index(drop=True)))
        row.update(_trade_summary(done[done["end"].dt.year == year], 0))
        del row["open_trades_at_end"]
        row["full_year"] = bool(first <= in_year[0] and in_year[-1] <= last)
        rows.append(row)
    return pd.DataFrame(rows)
