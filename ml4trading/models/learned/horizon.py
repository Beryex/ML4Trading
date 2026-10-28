"""Horizon labels: what a learned model trains and is scored on when it looks h sessions ahead.

The panel's label ``y`` of a row is the return from its open to its exit, the symbol's NEXT
row's open. The horizon label of row t combines the symbol's next ``h`` labels ``y[t..t+h-1]``:
their sum (``sum``: the return over h sessions) or their mean (``mean``: per-session units). It
is NaN unless all h labels exist and are finite, so a row near a symbol's data end, or with a
missing label inside its window, has no label -- never a partial sum. A symbol's rows are its
traded sessions in order, so a session it did not trade has no row, and the label spans the gap
(each ``y`` already runs to the symbol's next open).

Its exit is the open of the symbol's row t+h, the exit of row t+h-1. A row is *realized* by a
session when its exit is on or before it. A fit uses only rows realized by the first session
after its window (López de Prado 2018, "Advances in Financial Machine Learning", ch. 7: purge
the labels whose outcome overlaps the evaluation window), and with ``horizon`` > 1 the
validation scoring leaves a row out of its label-based metrics unless it is realized by the first
session after the last scored period -- so no selection reads more than the one open that
follows the window. At ``horizon`` 1 the scoring stays as it has always been: it does not look at
the exit.

Under ``hold_overnight`` false a label exits at the close of row t+h-1 instead; the open of row
t+h bounds that from above, so a purge on it is safe, never leaky.

``horizon: 1`` with ``sum`` is the panel's own label, returned as it is.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: ``sum``: the return over the horizon; ``mean``: its per-session average.
HORIZON_SCALES = ("sum", "mean")


def _check(h: int, scale: str) -> None:
    if scale not in HORIZON_SCALES:
        raise ValueError(f"unknown horizon scale {scale!r}; known: {HORIZON_SCALES}")
    if isinstance(h, bool) or not isinstance(h, int | np.integer) or h < 1:
        raise ValueError(f"horizon must be an integer >= 1, got {h!r}")


def _symbol_codes(panel: pd.DataFrame) -> np.ndarray:
    """A code per row for its symbol, refusing a panel that is not in (symbol, period) order:
    a horizon label reads the rows after a row, so a misordered panel would silently mix
    symbols or sessions."""
    codes = pd.factorize(panel["symbol"])[0]
    if len(codes) < 2:
        return codes
    same = codes[1:] == codes[:-1]
    period = panel["period"].to_numpy()
    if np.count_nonzero(~same) != codes.max() or not (period[1:][same] > period[:-1][same]).all():
        raise ValueError("the panel must be in (symbol, period) order, as build_panel returns it")
    return codes


def horizon_labels(panel: pd.DataFrame, h: int, scale: str) -> tuple[np.ndarray, np.ndarray]:
    """``(y_h, exit_session)``, row-aligned with ``panel`` (in (symbol, period) order).

    ``y_h[t]`` is the sum (``scale="sum"``) or the mean (``"mean"``) of the symbol's labels
    ``y[t..t+h-1]``, NaN unless all h exist and are finite. ``exit_session[t]`` is the session of
    the symbol's row t+h (NaT when it has none). For ``h == 1`` ``y_h`` holds the panel's ``y``
    values with no arithmetic done on them."""
    _check(h, scale)
    h = int(h)
    y = panel["y"].to_numpy(dtype=float)
    sessions = panel["session"].to_numpy()
    n = len(panel)
    codes = _symbol_codes(panel)
    # NumPy deprecates a NaT without a time unit, so the missing value takes the sessions' own
    missing = np.datetime64("NaT", np.datetime_data(sessions.dtype)[0])
    exit_session = np.full(n, missing, dtype=sessions.dtype)
    if h < n:
        reached = codes[h:] == codes[:-h]  # the symbol's rows are contiguous: t+h is its own
        exit_session[: n - h][reached] = sessions[h:][reached]
    if h == 1:
        return y.copy(), exit_session
    total = y.copy()
    valid = np.isfinite(y)
    with np.errstate(invalid="ignore"):  # an infinite label makes the sum NaN, then masked out
        for k in range(1, h):
            has_k = np.zeros(n, dtype=bool)
            if k < n:
                has_k[: n - k] = (codes[k:] == codes[:-k]) & np.isfinite(y[k:])
                total[: n - k] += y[k:]
            valid &= has_k
    if scale == "mean":
        total = total / h
    return np.where(valid, total, np.nan), exit_session


def realized_by(panel: pd.DataFrame, exit_session: np.ndarray, end) -> np.ndarray:
    """The rows whose label exit is on or before the first panel session after ``end``; a row
    with no exit (NaT) is not. When no session follows ``end`` every row qualifies: the labels
    that exist are all inside the data."""
    later = panel.loc[panel["session"] > pd.Timestamp(end), "session"]
    if later.empty:
        return np.ones(len(panel), dtype=bool)
    return np.asarray(pd.DatetimeIndex(exit_session) <= later.min())


def window_labels(panel: pd.DataFrame, periods, h: int, scale: str, transform) -> np.ndarray:
    """The label a fit is scored on, for every panel row: NaN outside ``periods`` and where the
    row has no label; otherwise the horizon label through the fit's ``transform`` (None: as it
    is). The transform is applied to every pool row of ``periods`` that has a label, so a
    period's mean is the pool's cross-section the fit demeaned over.

    For ``h > 1`` a row whose exit is after the first session following the last of ``periods``
    has no label either, and is left out of the transform's means too. At ``h == 1`` a label is
    not bounded by its exit: the window's scoring then stays what it has always been."""
    in_window = panel["period"].isin(set(periods)).to_numpy()
    label = np.full(len(panel), np.nan)
    if not in_window.any():
        return label
    if h == 1:
        y_h = panel["y"].to_numpy(dtype=float)
        usable = in_window & np.isfinite(y_h)
    else:
        y_h, exit_session = horizon_labels(panel, h, scale)
        last = panel.loc[in_window, "session"].max()
        usable = in_window & np.isfinite(y_h) & realized_by(panel, exit_session, last)
    period = panel["period"].to_numpy()[usable]
    label[usable] = y_h[usable] if transform is None else transform.apply(y_h[usable], period)
    return label
