"""Volatility-scaled targets: a learned model that ranks names by forecast risk-adjusted return.

The trader ranks and weights names by |prediction|. A model that fits the raw return forecasts
large moves for the volatile names, so those win the ranking whether or not their move is large
for their own volatility. ``risk_adjust`` changes what the model fits, the way Moskowitz, Ooi &
Pedersen (2012, "Time series momentum") scale each return by its ex-ante volatility:

- ``none``: the label as it is (``horizon.py``), the prediction as the model gives it.
- ``vol``: the fit's label is ``y_h / sigma``, the label divided by the row's sigma. The
  prediction is the model's output ``z`` times the median sigma of the rows predicted in the same
  period. Every name of a period shares that multiplier, so the ranking and the weights follow
  ``z`` -- the risk-adjusted forecast -- while the units stay those of a return.
- ``vol_return``: the same fit; the prediction is ``z`` times the row's own sigma, a per-name
  return forecast (a name with a large ``z`` but a calm history gets a small forecast).

``sigma`` is the standard deviation (ddof 1) of the symbol's own one-session labels ``y`` over a
trailing window of ``vol_window`` rows (a short window, as Ang, Hodrick, Xing & Zhang 2006
measure volatility). The window of row t is rows t-vol_window-1 .. t-2: the labels that were
realized by the open of row t-1. ``y[t-1]`` is left out although it is known at the open of row
t, because that open is row t's own bar, which a prediction for t may not read (the contract of
``ml4trading.models.base``, checked by ``ml4trading.testing.assert_causal``, which scrambles it);
overnight ``y[t-1]`` is the return up to that very open. ``y[t]`` or later is never read either.
A row with fewer than ``MIN_VOL_OBSERVATIONS`` finite labels in its window, or a window without
spread, has no sigma: it is left out of the fit, the early-stopping set and the label-based
scoring, and it predicts NaN, which the trainer drops (the name is not traded that period).

A symbol's rows are its traded sessions in order, so the window is the symbol's previous
``vol_window`` rows, not calendar sessions. The panel must be in (symbol, period) order, as
``build_panel`` returns it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.models.learned.horizon import symbol_codes

#: ``none``: the label and the prediction as they are; ``vol``: the label over sigma, the
#: prediction times the period's median sigma; ``vol_return``: times the row's own sigma.
RISK_ADJUSTS = ("none", "vol", "vol_return")
#: Fewest finite labels a sigma may rest on: a month of sessions (21 to the month, the unit of
#: Ang, Hodrick, Xing & Zhang 2006's one-month volatility). A shorter history has no sigma.
MIN_VOL_OBSERVATIONS = 21
#: How many rows before row t a sigma's window ends: its last label is ``y[t - LABEL_LAG]``.
LABEL_LAG = 2


def valid_vol_window(window) -> bool:
    """Whether ``window`` can be a ``vol_window``: an integer of at least
    ``MIN_VOL_OBSERVATIONS`` (a shorter window could never hold that many labels)."""
    return (
        not isinstance(window, bool)
        and isinstance(window, int | np.integer)
        and window >= MIN_VOL_OBSERVATIONS
    )


def trailing_volatility(panel: pd.DataFrame, window: int) -> np.ndarray:
    """Each panel row's sigma (see the module docstring), row-aligned with ``panel``: the std of
    the symbol's finite labels ``y`` over the ``window`` rows that end ``LABEL_LAG`` rows before
    it; NaN when there are fewer than ``MIN_VOL_OBSERVATIONS`` or they are all equal."""
    if not valid_vol_window(window):
        raise ValueError(
            f"vol_window must be an integer >= {MIN_VOL_OBSERVATIONS} (sessions), got {window!r}"
        )
    n = len(panel)
    if n == 0:
        return np.empty(0)
    codes = symbol_codes(panel)
    y = panel["y"].to_numpy(dtype=float)
    y = np.where(np.isfinite(y), y, np.nan)
    # the rolling std ending at row t - LABEL_LAG is row t's sigma
    ending_here = (
        pd.Series(y)
        .groupby(codes)
        .rolling(int(window), min_periods=MIN_VOL_OBSERVATIONS)
        .std()
        .to_numpy()
    )
    sigma = np.full(n, np.nan)
    sigma[LABEL_LAG:] = ending_here[:-LABEL_LAG]
    # a symbol's first rows have no window of their own: the one shifted in is its predecessor's
    sigma[pd.Series(codes).groupby(codes).cumcount().to_numpy() < LABEL_LAG] = np.nan
    return np.where(sigma > 0.0, sigma, np.nan)


def target_volatility(panel: pd.DataFrame, risk_adjust: str, window: int) -> np.ndarray | None:
    """The panel's sigma for a model that scales its target, None for one that does not (so
    that ``none`` computes nothing)."""
    _check(risk_adjust)
    return None if risk_adjust == "none" else trailing_volatility(panel, window)


def scale_predictions(
    raw: np.ndarray, sigma: np.ndarray, period: np.ndarray, risk_adjust: str
) -> np.ndarray:
    """A model's raw output for rows with sigma ``sigma`` and periods ``period`` (all three
    row-aligned) as the prediction the trader reads: ``vol`` multiplies by the median sigma of
    the rows given for the row's period, ``vol_return`` by the row's own sigma, ``none`` returns
    ``raw`` itself. A row without a sigma is NaN either way; the median is over the rows that
    have one, so a period with none has no multiplier."""
    _check(risk_adjust)
    if risk_adjust == "none":
        return raw
    return raw * prediction_multiplier(sigma, period, risk_adjust)


def prediction_multiplier(sigma: np.ndarray, period: np.ndarray, risk_adjust: str) -> np.ndarray:
    """What ``scale_predictions`` multiplies each row's raw output by: the row's own ``sigma``
    (``vol_return``), the median sigma of its period's rows that have one (``vol``; NaN for a row
    without its own), or 1 (``none``)."""
    _check(risk_adjust)
    if risk_adjust == "none":
        return np.ones(len(sigma))
    if risk_adjust == "vol_return":
        return sigma
    median = pd.Series(sigma).groupby(np.asarray(period)).transform("median").to_numpy()
    return np.where(np.isfinite(sigma), median, np.nan)


def _check(risk_adjust: str) -> None:
    if risk_adjust not in RISK_ADJUSTS:
        raise ValueError(f"risk_adjust must be one of {RISK_ADJUSTS}, got {risk_adjust!r}")
