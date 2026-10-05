"""A check every model should pass: its prediction for period t does not change when everything
from period t onwards is scrambled (the decision is taken at t's open, before t's bar exists)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.models.base import Model

_NUMERIC = ["open", "high", "low", "close", "volume", "open_px", "exit_px", "div_ret", "y"]


def assert_causal(model: Model, panel: pd.DataFrame, window, n_checks: int = 5, seed: int = 0):
    """``model`` is already fitted. For ``n_checks`` periods t drawn from ``window``, predict
    t's session on a copy of ``panel`` whose rows at or after t are replaced by noise, and
    require the predictions at t to equal the clean ones. Raises AssertionError otherwise."""
    clean = model.predict(panel, window)
    if clean.empty:
        return
    rng = np.random.default_rng(seed)
    periods = np.sort(clean["period"].unique())
    for t in rng.choice(periods, size=min(n_checks, len(periods)), replace=False):
        t = pd.Timestamp(t)
        noisy = panel.copy()
        future = noisy["period"] >= t
        for col in _NUMERIC:
            noisy.loc[future, col] = rng.lognormal(size=int(future.sum()))
        session = (t.normalize(), t.normalize())
        got = model.predict(noisy, session)
        want = clean[clean["period"] == t].sort_values("symbol")
        got = got[got["period"] == t].sort_values("symbol")
        if not (
            list(want["symbol"]) == list(got["symbol"])
            and np.allclose(want["prediction"], got["prediction"], equal_nan=True)
        ):
            raise AssertionError(f"{model.name}: the prediction for {t} reads data at or after {t}")
