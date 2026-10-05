"""30-minute bars -> K-minute decision periods, and the realized label of each period.

A PERIOD is the unit of decision: positions are set at the period's open and valued at its exit.
Its key is a tz-naive New York wall-clock timestamp: K = 390 -> the session date (one period per
session, half days included); K < 390 -> the bar start floored to K minutes.

The panel built here has one row per (period, symbol) with a traded bar, and columns
    period, session, symbol, open, high, low, close, volume   -- the aggregated bar
    open_px   entry price: the period's open
    exit_px   exit price: overnight, the symbol's NEXT period's open; intraday, this close
    div_ret   dividends earned by holding from this open to the exit (overnight only)
    y         exit_px / open_px - 1 + div_ret: the realized return a model is scored against
A symbol's last period has no next open, so overnight its exit_px and y are NaN; it can be
predicted but not traded or fitted on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import BAR_MINUTES, SESSION_MINUTES, TIMEZONE

PANEL_COLUMNS = [
    "period",
    "session",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_px",
    "exit_px",
    "div_ret",
    "y",
]


def period_keys(ts: pd.Series, k: int) -> pd.Series:
    """Each bar's period key (module docstring) from its tz-aware start ``ts``."""
    if k % BAR_MINUTES or k <= 0 or k > SESSION_MINUTES:
        raise ValueError(f"K={k}: must be a positive multiple of {BAR_MINUTES} up to 390")
    naive = ts.dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    if k == SESSION_MINUTES:
        return naive.dt.normalize()
    return naive.dt.floor(f"{k}min")


def period_bars(bars: pd.DataFrame, k: int) -> pd.DataFrame:
    """Aggregate long-form bars (``ml4trading.data.load_bars``) into one bar per
    (symbol, period): first open, max high, min low, last close, summed volume. Sorted by
    (symbol, period)."""
    if bars.empty:
        return pd.DataFrame(columns=["period", "symbol", "open", "high", "low", "close", "volume"])
    frame = bars.sort_values(["symbol", "ts"], kind="stable")
    keys = period_keys(frame["ts"], k)
    agg = (
        frame.assign(period=keys.to_numpy())
        .groupby(["symbol", "period"], sort=True, observed=True)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .reset_index()
    )
    agg = agg.dropna(subset=["open", "close"])
    agg = agg[agg["open"] != 0]
    return agg[["period", "symbol", "open", "high", "low", "close", "volume"]].reset_index(
        drop=True
    )


def dividend_returns(
    entry_sessions: np.ndarray, exit_sessions: np.ndarray, ex_dates: np.ndarray, amounts
) -> np.ndarray:
    """Per row, the sum of dividends whose ex-date lies in (entry session, exit session]: a
    holder through the ex-date's open receives it, a buyer at that open does not."""
    if len(ex_dates) == 0:
        return np.zeros(len(entry_sessions))
    entry_sessions = np.asarray(entry_sessions, dtype="datetime64[ns]")
    exit_sessions = np.asarray(exit_sessions, dtype="datetime64[ns]")
    ex_dates = np.asarray(ex_dates, dtype="datetime64[ns]")
    order = np.argsort(ex_dates, kind="stable")
    dates = ex_dates[order]
    csum = np.concatenate(([0.0], np.cumsum(np.asarray(amounts, dtype=float)[order])))
    lo = np.searchsorted(dates, entry_sessions, side="right")
    hi = np.searchsorted(dates, exit_sessions, side="right")
    return csum[hi] - csum[lo]


def build_panel(
    bars: pd.DataFrame, k: int, hold_overnight: bool, dividends: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The labelled panel (module docstring) from long-form 30-minute bars."""
    pb = period_bars(bars, k)
    pb["session"] = pb["period"].dt.normalize()
    pb["open_px"] = pb["open"]
    if hold_overnight:
        pb["exit_px"] = pb.groupby("symbol", observed=True)["open"].shift(-1)
    else:
        pb["exit_px"] = pb["close"]
    pb["div_ret"] = 0.0
    if hold_overnight and dividends is not None and not dividends.empty:
        exit_session = pb.groupby("symbol", observed=True)["session"].shift(-1)
        exit_session = exit_session.fillna(pb["session"])  # last row: no exit, earns nothing
        div_ret = np.zeros(len(pb))
        by_symbol = {s: g for s, g in dividends.groupby("symbol", observed=True)}
        for sym, pos in pb.groupby("symbol", observed=True).indices.items():
            d = by_symbol.get(sym)
            if d is None:
                continue
            div_ret[pos] = dividend_returns(
                pb["session"].to_numpy()[pos],
                exit_session.to_numpy()[pos],
                pd.to_datetime(d["ex_date"]).to_numpy(),
                d["div_ret"].to_numpy(),
            )
        pb["div_ret"] = div_ret
    pb["y"] = (pb["exit_px"] - pb["open_px"]) / pb["open_px"] + pb["div_ret"]
    return pb[PANEL_COLUMNS]
