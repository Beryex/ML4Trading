"""The integer-share book: predictions -> trades -> a dollar equity curve.

The walk visits the decision periods in order, carrying ``equity`` (starts at the capital basis)
and the previous share book. At period t:

1. TRADEABLE names are those with an entry price (open_px) and an exit price this period. The
   trader turns their predictions into target weights (``ml4trading.trader``).
2. Held names that are not tradeable this period are CARRIED (``ml4trading.sizing``).
3. Shares are sized at the entry prices against the budget: the whole equity when the run
   reinvests (the default), else min(capital, equity) -- gains above the capital basis stay in
   cash.
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

#: One row per (period, symbol) held before or after the period's trades: the share change,
#: the symbol's own gross P&L and its costs, split by leg so a trade can be costed exactly
#: (``ml4trading.metrics.trade_stats``). ``close_*`` is the intraday session-close exit;
#: ``session_end`` marks the period it happens in.
POSITION_COLUMNS = [
    "period",
    "symbol",
    "prev_shares",
    "shares",
    "price",
    "exit_price",
    "gross",
    "entry_spread",
    "entry_commission",
    "exit_spread",
    "exit_commission",
    "close_spread",
    "close_commission",
    "borrow",
    "traded",
    "session_end",
]


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
        budget = equity if cfg.reinvest else min(cfg.capital, equity)
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
                    capital=budget if cfg.reinvest else cfg.capital,
                    hold_overnight=cfg.hold_overnight,
                )
            carried = {sym: last_price.get(sym, np.nan) for sym in prev if sym not in prices}
            shares = size_shares(
                weights,
                prices,
                budget,
                prev,
                carried,
                trader=trader,
                costs=costs,
            )

        # per-symbol accounting: every name held before or after this period's trades
        flatten = not cfg.hold_overnight and last_of_session[session_of[p]] == p
        recs = []
        for sym in sorted(set(prev) | set(shares)):
            before, after = prev.get(sym, 0.0), shares.get(sym, 0.0)
            px = prices.get(sym)
            r = {
                "period": p,
                "symbol": sym,
                "prev_shares": before,
                "shares": after,
                "price": px,
                "exit_price": exits.get(sym) if px is not None else None,
                "gross": 0.0,
                "entry_spread": 0.0,
                "entry_commission": 0.0,
                "exit_spread": 0.0,
                "exit_commission": 0.0,
                "close_spread": 0.0,
                "close_commission": 0.0,
                "borrow": 0.0,
                "traded": 0.0,
            }
            if px is not None:  # a carried name (no price) cannot trade
                entry, exit_ = leg_amounts(before, after)
                r["entry_spread"], r["entry_commission"] = leg_cost(entry, 0.0, px, costs)
                r["exit_spread"], r["exit_commission"] = leg_cost(0.0, exit_, px, costs)
                r["traded"] = (entry + exit_) * px
                r["gross"] = after * (exits[sym] - px) + after * px * divs[sym]
            mark = px if px is not None else last_price.get(sym, np.nan)
            if after < 0 and mark == mark:
                r["borrow"] = abs(after) * mark * borrow_per_period
            if flatten and after:  # intraday: the session-close exit of what is held
                close_px = exits.get(sym, last_price.get(sym, np.nan))
                if close_px == close_px:
                    r["close_spread"], r["close_commission"] = leg_cost(
                        0.0, abs(after), close_px, costs
                    )
                    r["traded"] += abs(after) * close_px
            recs.append(r)

        spread_cost = sum(r["entry_spread"] + r["exit_spread"] + r["close_spread"] for r in recs)
        commission = sum(
            r["entry_commission"] + r["exit_commission"] + r["close_commission"] for r in recs
        )
        gross = sum(r["gross"] for r in recs)
        borrow = sum(r["borrow"] for r in recs)
        traded = sum(r["traded"] for r in recs)
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
        for r in recs:
            r["session_end"] = flatten
        pos_rows.extend(recs)
        last_price.update(prices)
        prev = shares
    book = pd.DataFrame(rows)
    positions = pd.DataFrame(pos_rows, columns=POSITION_COLUMNS)
    return book, positions


def daily_values(
    book: pd.DataFrame,
    positions: pd.DataFrame,
    k: int,
    daily_panel: pd.DataFrame | None = None,
    dividends: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """The account valued once per trading day: one row per session with ``equity_before`` (at
    that session's open), ``equity`` (at the next session's open), ``ret`` and ``gross_ret``
    (the same change before the period's costs), and ``residual`` (see below).

    Up to one session per period (K <= 390) this is the book itself, summed per session.

    For N-session periods (K = 390 x N) the book only values the account when it trades, so
    each block is re-marked every day: every position bought at the block's open is valued at
    each later session's open (``daily_panel``: the K = 390 panel of the held names; a day
    without a bar keeps the last mark) and at the block's exit price on its last day; each
    dividend is credited on the day whose open-to-open span holds its ex-date, at the block's
    entry price, exactly as the block itself books it; the block's costs fall on its first day.
    The last day then lands on the book's own block-end equity; ``residual`` is whatever had to
    be added there to make it so (zero unless a name's data skips the block's boundaries)."""
    if k <= SESSION_MINUTES:
        g = book.groupby("session", sort=True)
        gross_ret = (book["gross"] / book["equity_before"].where(book["equity_before"] > 0)).fillna(
            0.0
        )
        return pd.DataFrame(
            {
                "session": list(g.groups),
                "equity_before": g["equity_before"].first().to_numpy(),
                "equity": g["equity"].last().to_numpy(),
                "ret": g["ret"].sum().to_numpy(),
                "gross_ret": gross_ret.groupby(book["session"]).sum().to_numpy(),
                "residual": 0.0,
            }
        )
    from ml4trading.periods import dividend_returns, nyse_sessions

    sessions = nyse_sessions()
    n = k // SESSION_MINUTES
    opens = {s: g.set_index("session")["open_px"] for s, g in daily_panel.groupby("symbol")}
    divs = {
        s: g
        for s, g in (
            dividends
            if dividends is not None
            else pd.DataFrame(columns=["symbol", "ex_date", "div_ret"])
        ).groupby("symbol")
    }
    held_by_period = {
        p: g[(g["shares"] != 0) & g["price"].notna()] for p, g in positions.groupby("period")
    }
    rows = []
    for b in book.itertuples(index=False):
        pos = sessions.get_loc(b.period)
        days = sessions[pos : pos + n]
        bounds = sessions[pos + 1 : pos + n + 1]  # each day's open-to-open end
        costs = b.spread_cost + b.commission + b.borrow
        change = np.zeros(len(days))  # cumulative mark-to-market P&L at each day's END
        held = held_by_period.get(b.period)
        for h in [] if held is None else held.itertuples(index=False):
            px = opens.get(h.symbol, pd.Series(dtype=float))
            marks = px.reindex(bounds[:-1]).ffill().fillna(h.price).to_numpy(dtype=float)
            marks = np.append(marks, h.exit_price)
            d = divs.get(h.symbol)
            div = np.zeros(len(days))
            if d is not None:
                div = dividend_returns(
                    days.to_numpy(), bounds.to_numpy(), d["ex_date"].to_numpy(), d["div_ret"]
                )
            change += h.shares * (marks - h.price) + h.shares * h.price * np.cumsum(div)
        values = b.equity_before - costs + change
        residual = b.equity - values[-1]
        values[-1] = b.equity
        before = np.concatenate(([b.equity_before], values[:-1]))
        gross_step = np.diff(np.concatenate(([b.equity_before], values)))
        gross_step[0] += costs
        with np.errstate(divide="ignore", invalid="ignore"):
            ret = np.where(before > 0, values / before - 1.0, 0.0)
            gross_ret = np.where(before > 0, gross_step / before, 0.0)
        for j, day in enumerate(days):
            rows.append(
                {
                    "session": day,
                    "equity_before": before[j],
                    "equity": values[j],
                    "ret": ret[j],
                    "gross_ret": gross_ret[j],
                    "residual": residual if j == len(days) - 1 else 0.0,
                }
            )
    return pd.DataFrame(rows)
