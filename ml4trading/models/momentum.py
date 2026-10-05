"""Momentum: each symbol's predicted return for a period is its PREVIOUS period's
close-to-close return; ``reverse`` flips the sign (mean reversion). Nothing is fitted.

At K = 390 the prediction for session t is close(t-1) / close(t-2) - 1 -- yesterday's change,
known before t's open. Rows are consecutive per symbol, so across a gap in a symbol's data the
"previous period" is its last traded one.
"""

from __future__ import annotations

import pandas as pd

from ml4trading.models.base import Model, rows_in
from ml4trading.models.registry import register_model


@register_model("momentum")
class Momentum(Model):
    def __init__(self, hold_overnight: bool, reverse: bool = False):
        super().__init__(hold_overnight, reverse=reverse)
        self.reverse = bool(reverse)

    def fit(self, panel, window) -> None:
        return None

    def predict(self, panel: pd.DataFrame, window) -> pd.DataFrame:
        frame = panel.sort_values(["symbol", "period"], kind="stable")
        change = frame.groupby("symbol", observed=True)["close"].pct_change(fill_method=None)
        last_change = change.groupby(frame["symbol"], observed=True).shift(1)
        frame = frame.assign(prediction=(-1.0 if self.reverse else 1.0) * last_change)
        out = rows_in(frame, window)[["period", "symbol", "prediction"]]
        return out.dropna(subset=["prediction"]).reset_index(drop=True)
