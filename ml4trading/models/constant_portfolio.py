"""Constant portfolios: a fixed predicted return per symbol, nothing fitted.

``investment`` predicts +1 % for VOO (S&P 500) every period, ``investment_tech`` the same for
QQQ (Nasdaq-100). Under the ``proportional`` trader a single positive name takes the whole
(capped) book whatever the value, so the number only has to clear the cost gate: each method is
100 % of its ETF, bought and held -- a buy-and-hold baseline.
"""

from __future__ import annotations

from ml4trading.models.base import FoldResult, Model, rows_in
from ml4trading.models.registry import register_model


class ConstantPortfolio(Model):
    #: {symbol: constant predicted return}; a subclass declares it and nothing else.
    predictions: dict[str, float] = {}

    def symbol_pool(self, available: list[str]) -> list[str]:
        missing = sorted(set(self.predictions) - set(available))
        if missing:
            raise ValueError(f"{self.name}: symbols absent from the dataset: {missing}")
        return list(self.predictions)

    def fit_predict_fold(self, frames, fit_window, predict_window) -> FoldResult:
        rows = rows_in(frames.panel, predict_window)
        rows = rows[rows["symbol"].isin(set(self.predictions))]
        out = rows[["period", "symbol"]].copy()
        out["prediction"] = out["symbol"].map(self.predictions).astype(float)
        return FoldResult(out.reset_index(drop=True), None, 0)


@register_model("investment")
class Investment(ConstantPortfolio):
    predictions = {"VOO": 0.01}


@register_model("investment_tech")
class InvestmentTech(ConstantPortfolio):
    predictions = {"QQQ": 0.01}
