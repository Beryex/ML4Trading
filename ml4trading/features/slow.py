"""Slow cross-sectional features from the published literature, computed on the dataset's own bars.

ML4Investment's features are fast: they look back days. These are the monthly-to-yearly signals
that the asset-pricing literature finds in the cross-section of returns, each computed per symbol
from its daily series (and VOO's, for the market-relative ones) and shifted one session like the
rest, so the row of session d holds what was known at d's open. Every window is in sessions (252
a year, 126 half a year, 63 a quarter, 21 a month).

The features, with their sources:

- ``mom_12_1`` = close(d-21) / close(d-252) - 1 (Jegadeesh & Titman 1993);
- ``mom_6_1`` = close(d-21) / close(d-126) - 1 (Jegadeesh & Titman 1993);
- ``mom_1m`` = close(d) / close(d-21) - 1, the one-month reversal (Jegadeesh 1990);
- ``high_52w`` = close / the largest of the last 252 closes (George & Hwang 2004);
- ``max_21`` = the largest daily return of the last 21 sessions (Bali, Cakici & Whitelaw 2011);
- ``beta_252`` = cov / var of the daily returns against VOO's over 252 sessions (Frazzini &
  Pedersen 2014);
- ``ivol_63`` = the std of the residual of the daily returns regressed on VOO's over 63 sessions
  (Ang, Hodrick, Xing & Zhang 2006);
- ``resid_mom_12_1`` = the sum of the residual returns against VOO over the sessions d-252..d-21
  (beta from the same window), divided by their std (Blitz, Huij & Martens 2011);
- ``amihud_21`` = the mean of abs(return) / dollar volume over 21 sessions (Amihud 2002);
- ``rskew_21`` = the skewness of the 30-minute log returns of the last 21 sessions (Amaya,
  Christoffersen, Jacobs & Vasquez 2015);
- ``overnight_21`` = the sum over 21 sessions of log(open / previous close), and
- ``intraday_21`` = the sum over 21 sessions of log(close / open) (Lou, Polk & Skouras 2019);
- ``volatility_252`` = the std of the daily returns over 252 sessions, the scale of several of
  the above.

Every feature is a ratio, a return or a dollar volume, so none moves when one stock's prices are
multiplied and its volume divided by the same factor (``tests/test_slow_features.py`` holds it):
the bars are split-adjusted back from the end of the data, and a level would tell which companies
split later (see ``ml4trading.features.build.SCALE_DEPENDENT``).

Conventions, all in the symbol's own rows:

- A session is a New York date, as ``daily.py`` groups the bars: its open is its first bar's
  open, its close its last bar's close, its volume the sum of its bars'. A daily return is
  close / previous close - 1, the previous close the symbol's previous session's.
- A window is that many of the symbol's own sessions, the current one included, and is NaN unless
  it is full: no feature is ever computed from a short history, and a gap in the data makes a
  window longer in calendar time, not shorter in sessions.
- The market-relative features (beta, idiosyncratic volatility, residual momentum) use the
  sessions the symbol and VOO both have. Their returns run between consecutive such sessions, so
  a session one of them lacks never pairs a one-day return with a longer one, and a symbol's
  windows never reach into the history VOO has and it has not. A session VOO lacks has no
  market-relative value.
- Standard deviations and variances are the sample ones (ddof 1), as pandas' ``std`` is.

Choices where the definition leaves one open:

- ``resid_mom_12_1``: the residual is ``r - beta * r_market``, with the beta fitted on that same
  window of sessions d-252..d-21. A regression with an intercept leaves residuals that sum to
  exactly zero over the window it was fitted on, so the intercept is left out of the residual:
  the sum is then the window's accumulated alpha, which is what residual momentum ranks. The std
  that scales it is the std of those residuals. The window is the 232 returns of the sessions
  d-252..d-21, so the feature has a value from the 254th session (``mom_12_1``'s from the 253rd).
- ``rskew_21``: the third central moment of the pooled 30-minute log returns over the second to
  the power 3/2 (Fisher-Pearson, no bias correction, as ``scipy.stats.skew`` with its defaults).
  A session's returns are its intraday ones: the first bar's from the session's open to its
  close, each later bar's from the previous bar's close. The overnight gap is left out -- Amaya
  et al. use intraday returns only, and ``overnight_21`` carries it.
- ``amihud_21``: dollar volume is the session's close times its volume (Amihud's CRSP price times
  shares). A session of no volume has no ratio, so a window holding one is NaN.

Speed: the real data is about 2,700 sessions of 552 symbols, so the features are computed on wide
frames, one vectorized rolling operation for every symbol. The frames are indexed by POSITION, not
by date: column j holds symbol j's own sessions from position 0 on (NaN after its last), so a
window of n rows is n of its own sessions, as the conventions above say, whatever dates it lacks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import TIMEZONE

#: The columns of ``slow_features``, in order.
SLOW_FEATURES = (
    "mom_12_1",
    "mom_6_1",
    "mom_1m",
    "high_52w",
    "max_21",
    "beta_252",
    "ivol_63",
    "resid_mom_12_1",
    "amihud_21",
    "rskew_21",
    "overnight_21",
    "intraday_21",
    "volatility_252",
)

#: Window lengths, in sessions. The values are those of the sources: Jegadeesh & Titman (1993)
#: rank on 6 and 12 months skipping the last; George & Hwang (2004) use the 52-week high; the
#: remaining windows are the table's, 21 sessions to the month.
YEAR, HALF_YEAR, QUARTER, MONTH = 252, 126, 63, 21
#: The residual momentum window: the sessions d-252..d-21, inclusive (Blitz, Huij & Martens 2011
#: skip the last month of a year's residuals).
RESIDUAL_WINDOW = YEAR - MONTH + 1

#: A window whose variance is below this fraction of its mean square has none (a constant price).
_FLAT = 1e-9

#: The columns of ``_session_series`` that the features read.
_SESSION_COLUMNS = ("open", "close", "volume", "s1", "s2", "s3", "n")


def _finite(values):
    """``values`` (a Series or DataFrame) with each infinite value (a ratio with a zero) made
    missing."""
    return values.where(np.isfinite(values))


def _returns(close: np.ndarray) -> np.ndarray:
    """Close / previous close - 1 of consecutive entries; the first has no previous, and a ratio
    with a zero is missing."""
    out = np.full(len(close), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        out[1:] = close[1:] / close[:-1] - 1.0
    out[~np.isfinite(out)] = np.nan
    return out


def _session_series(bars: pd.DataFrame) -> pd.DataFrame:
    """One symbol's sessions, indexed by the tz-naive New York session date: ``open``, ``close``,
    ``volume``, and the sums of the session's 30-minute log returns' first three powers (``s1``,
    ``s2``, ``s3``) with their count (``n``), from which ``rskew_21`` is pooled."""
    rows = bars.sort_values("ts", kind="stable")
    session = pd.DatetimeIndex(rows["ts"]).tz_convert(TIMEZONE).tz_localize(None).normalize()
    session = session.rename("session")
    open_ = rows["open"].to_numpy(dtype=float)
    close = rows["close"].to_numpy(dtype=float)
    first_bar = np.concatenate(([True], session.values[1:] != session.values[:-1]))
    with np.errstate(divide="ignore", invalid="ignore"):
        log_close = np.log(close)
        # a bar's return runs from the previous bar's close, or from the open for a session's first
        step = log_close - np.where(first_bar, np.log(open_), np.roll(log_close, 1))
    step[~np.isfinite(step)] = np.nan
    frame = pd.DataFrame(
        {
            "open": open_,
            "close": close,
            "volume": rows["volume"].to_numpy(dtype=float),
            "s1": step,
            "s2": step**2,
            "s3": step**3,
            "n": (~np.isnan(step)).astype(float),
        },
        index=session,
    )
    # first / last skip a missing value, as the daily aggregation of ``daily.py`` does
    return frame.groupby(level=0, sort=True).agg(
        {
            "open": "first",
            "close": "last",
            "volume": "sum",
            "s1": "sum",
            "s2": "sum",
            "s3": "sum",
            "n": "sum",
        }
    )


def _stacked(columns: list[np.ndarray]) -> pd.DataFrame:
    """The arrays as the columns of one frame, each from position 0 and NaN after its end."""
    out = np.full((max((len(c) for c in columns), default=0), len(columns)), np.nan)
    for j, column in enumerate(columns):
        out[: len(column), j] = column
    return pd.DataFrame(out)


def _rolling_regression(y: pd.DataFrame, x: pd.DataFrame, window: int) -> tuple:
    """Over the ``window`` observations ending at each row: the sum of ``x``, the sum of ``y``,
    the slope of ``y`` on ``x`` (cov / var) and the sum of squared residuals of that line with an
    intercept, from rolling sums of x, y, x*x, x*y and y*y. NaN unless the window is full."""
    sx, sy = x.rolling(window).sum(), y.rolling(window).sum()
    ss_x = (x * x).rolling(window).sum() - sx**2 / window
    ss_y = (y * y).rolling(window).sum() - sy**2 / window
    s_xy = (x * y).rolling(window).sum() - sx * sy / window
    slope = s_xy / ss_x
    return sx, sy, slope, (ss_y - s_xy * slope).clip(lower=0.0)


def _market_relative(days: list[pd.DataFrame], market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """``beta_252``, ``ivol_63`` and ``resid_mom_12_1`` of each symbol (unshifted), on the
    symbol's own positions: its returns and the market's between consecutive sessions of the two
    of them, regressed over their common sessions, and NaN on a session the market lacks."""
    at_own, own_returns, market_returns = [], [], []
    for day in days:
        shared = day.index.intersection(market.index)
        at_own.append(day.index.get_indexer(shared))
        own_returns.append(_returns(day["close"].to_numpy()[at_own[-1]]))
        market_returns.append(
            _returns(market["close"].to_numpy()[market.index.get_indexer(shared)])
        )
    y, x = _stacked(own_returns), _stacked(market_returns)
    _, _, beta, _ = _rolling_regression(y, x, YEAR)
    _, _, _, ss_e = _rolling_regression(y, x, QUARTER)
    sx, sy, slope, ss_res = _rolling_regression(y, x, RESIDUAL_WINDOW)
    # the sum of r - slope * r_market over the window: no intercept (see the module)
    resid_mom = ((sy - slope * sx) / np.sqrt(ss_res / (RESIDUAL_WINDOW - 1))).shift(MONTH)
    on_common = {
        "beta_252": beta,
        "ivol_63": np.sqrt(ss_e / (QUARTER - 1)),
        "resid_mom_12_1": resid_mom,
    }
    # back from the common sessions' positions to the symbol's own
    length = max((len(day) for day in days), default=0)
    out = {name: np.full((length, len(days)), np.nan) for name in on_common}
    for name, frame in on_common.items():
        values = frame.to_numpy()
        for j, positions in enumerate(at_own):
            out[name][positions, j] = values[: len(positions), j]
    return {name: pd.DataFrame(values) for name, values in out.items()}


def _own_features(days: list[pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """The features of the symbols' own sessions (unshifted), by name, each a frame of the
    symbols' positions (``days`` are their ``_session_series``)."""
    wide = {name: _stacked([day[name].to_numpy() for day in days]) for name in _SESSION_COLUMNS}
    close, open_ = wide["close"], wide["open"]
    ret = _finite(close / close.shift(1) - 1)
    dollar_volume = close * wide["volume"]
    dollar_volume = dollar_volume.where(dollar_volume > 0)
    pooled = {name: wide[name].rolling(MONTH).sum() for name in ("s1", "s2", "s3", "n")}
    mean, mean_square = pooled["s1"] / pooled["n"], pooled["s2"] / pooled["n"]
    m2 = mean_square - mean**2
    m3 = pooled["s3"] / pooled["n"] - 3 * mean * mean_square + 2 * mean**3
    return {
        "mom_12_1": close.shift(MONTH) / close.shift(YEAR) - 1,
        "mom_6_1": close.shift(MONTH) / close.shift(HALF_YEAR) - 1,
        "mom_1m": close / close.shift(MONTH) - 1,
        "high_52w": close / close.rolling(YEAR).max(),
        "max_21": ret.rolling(MONTH).max(),
        "amihud_21": (ret.abs() / dollar_volume).rolling(MONTH).mean(),
        "rskew_21": m3 / m2.where(m2 > _FLAT * mean_square) ** 1.5,
        "overnight_21": _finite(np.log(open_ / close.shift(1))).rolling(MONTH).sum(),
        "intraday_21": _finite(np.log(close / open_)).rolling(MONTH).sum(),
        "volatility_252": ret.rolling(YEAR).std(),
    }


def slow_features(bars: pd.DataFrame, pool: list[str], market: str = "VOO") -> pd.DataFrame:
    """The pool's slow features: one float32 row per (session, symbol) the symbol has bars on,
    holding what was known at that session's open -- the features of the symbol's PREVIOUS
    session (the first session of a symbol is all NaN) -- with the columns ``SLOW_FEATURES``.
    ``bars`` holds the pool and ``market`` (VOO); the market has no rows of its own unless it is
    in the pool. The result does not depend on the row order of ``bars`` or of ``pool``."""
    if market not in set(bars["symbol"].unique()):
        raise ValueError(f"slow_features: no bars for the market {market!r}")
    rows = bars.groupby("symbol", sort=True).indices
    symbols = [s for s in sorted(set(pool)) if s in rows]
    days = [_session_series(bars.iloc[rows[s]]) for s in symbols]
    if not days:
        return pd.DataFrame(
            columns=list(SLOW_FEATURES),
            index=pd.MultiIndex.from_arrays([[], []], names=["session", "symbol"]),
            dtype=np.float32,
        )
    wide = _own_features(days)
    wide |= _market_relative(days, _session_series(bars.iloc[rows[market]]))
    values = [wide.pop(name).to_numpy() for name in SLOW_FEATURES]  # one pass per column
    frames = []
    for j, (symbol, day) in enumerate(zip(symbols, days, strict=True)):
        n = len(day)
        block = np.full((n, len(SLOW_FEATURES)), np.nan, dtype=np.float32)
        for k, column in enumerate(values):
            block[1:, k] = column[: n - 1, j]  # the shift: row d holds session d-1's value
        block[np.isinf(block)] = np.nan
        frame = pd.DataFrame(block, index=day.index, columns=list(SLOW_FEATURES))
        frame["symbol"] = symbol
        frames.append(frame.set_index("symbol", append=True))
    return pd.concat(frames)
