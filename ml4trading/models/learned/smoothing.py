"""Prediction smoothing: ``smooth_halflife`` averages a symbol's forecasts over its recent rows.

A one-session forecast moves with the day's noise, so the book that follows it trades names in
and out, and trading costs make that expensive: with them the optimal position moves only part
of the way toward its forecast (Gârleanu & Pedersen 2013, "Dynamic Trading with Predictable
Returns and Transaction Costs"), and costs weigh most on the anomalies that turn over most
(Novy-Marx & Velikov 2016, "A Taxonomy of Anomalies and Their Trading Costs"). With
``smooth_halflife`` ``H`` > 0 (sessions; 0 is off) the model's traded prediction is instead the
exponential average of the symbol's own predictions, the newest weighing most:

    s[t] = (1 - alpha) * s[t-1] + alpha * x[t],    alpha = 1 - 0.5 ** (1 / H),

so a prediction ``H`` rows old carries half the weight of the newest one (the half-life as
pandas defines it: ``Series.ewm(halflife=...)``). The average is seeded at the symbol's first
prediction (``ewm(adjust=False)``).

The average needs the rows before the predict window, so a fold predicts ``LOOKBACK_HALFLIVES``
x ``ceil(H)`` sessions before the window's first session as well (fewer where the data starts
sooner) and returns only the window's rows. The lookback rows are scaled like the window's
(``ml4trading.models.learned.risk_adjust``) before they are averaged, and their predictions are
the fitted model's own on earlier sessions, which a prediction may read. For a symbol with a row
on every lookback session the seed's weight at the window's first row is 0.5 ** (3 ceil(H) / H),
at most 12.5 %; a symbol with fewer rows there (it starts inside the lookback, or has gaps) has
fewer steps to forget the seed, and so more.

The average runs over the symbol's consecutive panel rows, not over calendar sessions: a session
the symbol did not trade has no row, and the average steps from the row before the gap to the
row after it (``ml4trading.models.learned.horizon`` reads a symbol's rows the same way). A
symbol whose first row is inside the window is seeded there. A row with no prediction (NaN) stays
without one -- the name is not traded that period -- and counts as a session that passed for the
average (pandas' ``ignore_na=False``): the next prediction weighs the older ones by that much
less.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ml4trading.models.learned.horizon import symbol_codes

#: How many half-lives of rows before the predict window are read. This project's own choice (no
#: public source fixes one): three half-lives leave the seed 0.5 ** 3 = 12.5 % of the weight at
#: the window's first row at most (see the module), enough for the average to forget where it
#: started, for the cost of three half-lives of extra predictions per fold.
LOOKBACK_HALFLIVES = 3


def ema_alpha(halflife: float) -> float:
    """The weight of the newest prediction: ``1 - 0.5 ** (1 / halflife)``."""
    return 1.0 - 0.5 ** (1.0 / halflife)


def lookback_sessions(halflife: float) -> int:
    """How many sessions before the predict window's first session the average reads."""
    return LOOKBACK_HALFLIVES * math.ceil(halflife)


def lookback_start(panel: pd.DataFrame, first_session, halflife: float) -> pd.Timestamp:
    """The first session the average of a window starting at ``first_session`` reads:
    ``lookback_sessions`` panel sessions before it, clipped to the data's first session. The
    sessions are those of the panel's pool, in which a symbol's own missing session is just
    absent."""
    sessions = pd.DatetimeIndex(panel["session"].unique()).sort_values()
    at = int(sessions.searchsorted(pd.Timestamp(first_session)))
    return sessions[max(0, at - lookback_sessions(halflife))]


def smooth_predictions(predicted: np.ndarray, rows: pd.DataFrame, halflife: float) -> np.ndarray:
    """``predicted`` (row-aligned with the panel ``rows``, which must be in (symbol, period)
    order as ``build_panel`` returns them) as each symbol's exponential average up to the row;
    NaN where ``predicted`` is NaN. ``halflife`` must be > 0."""
    if not halflife > 0:
        raise ValueError(f"smooth_halflife must be > 0 to smooth, got {halflife!r}")
    values = np.where(np.isfinite(predicted), predicted, np.nan)
    codes = symbol_codes(rows)  # also refuses a panel that is not in (symbol, period) order
    averaged = pd.Series(values).groupby(codes).ewm(alpha=ema_alpha(halflife), adjust=False).mean()
    # the grouped result is keyed by (symbol code, row): back to the rows' own order
    averaged = averaged.reset_index(level=0, drop=True).sort_index().to_numpy()
    return np.where(np.isfinite(values), averaged, np.nan)
