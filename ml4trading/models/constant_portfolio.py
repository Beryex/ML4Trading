"""Constant portfolios: a fixed predicted return per symbol, nothing fitted.

``investment`` predicts +1 % for VOO every period. Under the ``proportional`` trader a single
positive name takes the whole (capped) book whatever the value, so the number only has to clear
the cost gate: the method is 100 % VOO, bought once and held -- the buy-and-hold baseline.
"""

from __future__ import annotations

import pandas as pd

from ml4trading.models.base import Model, rows_in
from ml4trading.models.registry import register_model


class ConstantPortfolio(Model):
    #: {symbol: constant predicted return}; a subclass declares it and nothing else.
    predictions: dict[str, float] = {}

    def symbol_pool(self, available: list[str]) -> list[str]:
        missing = sorted(set(self.predictions) - set(available))
        if missing:
            raise ValueError(f"{self.name}: symbols absent from the dataset: {missing}")
        return list(self.predictions)

    def fit(self, panel, window) -> None:
        return None

    def predict(self, panel: pd.DataFrame, window) -> pd.DataFrame:
        rows = rows_in(panel, window)
        rows = rows[rows["symbol"].isin(set(self.predictions))]
        out = rows[["period", "symbol"]].copy()
        out["prediction"] = out["symbol"].map(self.predictions).astype(float)
        return out.reset_index(drop=True)


@register_model("investment")
class Investment(ConstantPortfolio):
    predictions = {"VOO": 0.01}
