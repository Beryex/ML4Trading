"""A check every model should pass: its prediction for period t does not change when every bar
from t's start onwards is scrambled (the decision is taken at t's open, before t's bar exists)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import TIMEZONE
from ml4trading.models.base import Model

_SCRAMBLED = ["open", "high", "low", "close", "volume"]


def assert_causal(
    model: Model,
    bars: pd.DataFrame,
    dividends: pd.DataFrame,
    K: int,
    pool: list[str],
    fit_window,
    predict_window,
    n_checks: int = 5,
    seed: int = 0,
) -> None:
    """Build ``model``'s frames from ``bars`` (of ``model.loaded_symbols(pool)``) and predict
    ``predict_window`` after fitting ``fit_window``. Then, for ``n_checks`` predicted periods t,
    replace every bar starting at or after t by noise, rebuild the frames, fit and predict again,
    and require the predictions for t to be unchanged. Raises AssertionError otherwise.

    ``fit_window`` must end at least two sessions before ``predict_window`` starts: the label of
    a period reads the next period's open, so a fit reaching the period just before t would
    legitimately move with t's bars."""
    clean = model.fit_predict_fold(
        model.build_frames(bars, dividends, K, pool), fit_window, predict_window
    ).predictions
    if clean.empty:
        return
    rng = np.random.default_rng(seed)
    starts = bars["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None)
    periods = np.sort(clean["period"].unique())
    for t in rng.choice(periods, size=min(n_checks, len(periods)), replace=False):
        t = pd.Timestamp(t)
        noisy = bars.copy()
        future = (starts >= t).to_numpy()
        for col in _SCRAMBLED:
            noisy.loc[future, col] = rng.lognormal(size=int(future.sum()))
        got = model.fit_predict_fold(
            model.build_frames(noisy, dividends, K, pool), fit_window, predict_window
        ).predictions
        want = clean[clean["period"] == t].sort_values("symbol")
        got = got[got["period"] == t].sort_values("symbol")
        if not (
            list(want["symbol"]) == list(got["symbol"])
            and np.allclose(want["prediction"], got["prediction"], equal_nan=True)
        ):
            raise AssertionError(f"{model.name}: the prediction for {t} reads data at or after {t}")
