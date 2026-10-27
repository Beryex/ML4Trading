"""The learned models' shared part: the features as frames, the fold's fit and predict.

A subclass declares its parameters (``PARAMS``) and its training loss, and implements ``_fit``,
which turns a fold's fit rows into a ``FitArtifact``; everything else -- which symbols, which
rows, the label, causality -- is here, so the methods cannot differ on it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from ml4trading.config import SESSION_MINUTES
from ml4trading.features import FEATURE_ETFS, build_features
from ml4trading.models.base import FoldResult, Frames, Model, empty_predictions, rows_in
from ml4trading.models.learned.losses import baseline_prediction


class FitArtifact(ABC):
    """A fold's fitted estimator. ``predict`` takes features indexed by (period, symbol) with
    (at least) ``feature_names()`` as columns; ``save`` writes ``FILENAME`` into a directory and
    ``load`` reads it back, predicting identically."""

    FILENAME: ClassVar[str]

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray: ...

    @abstractmethod
    def feature_names(self) -> list[str]: ...

    @abstractmethod
    def save(self, directory: Path) -> None: ...

    @classmethod
    @abstractmethod
    def load(cls, directory: Path) -> FitArtifact: ...


class LearnedModel(Model):
    """A model fitted per fold on ``ml4trading.features`` (K = 390 only: the features are
    daily). Its pool is every stock; VOO and QQQ are loaded for their feature columns only."""

    #: Every ``model.*`` key besides ``hold_overnight``; all are required.
    PARAMS: ClassVar[tuple[str, ...]] = ()

    def __init__(self, hold_overnight: bool, **params):
        unknown = sorted(set(params) - set(self.PARAMS))
        missing = sorted(set(self.PARAMS) - set(params))
        if unknown or missing:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {unknown}, missing parameter(s) {missing}"
            )
        super().__init__(hold_overnight, **params)

    @property
    @abstractmethod
    def loss(self) -> str:
        """The training loss (``ml4trading.models.learned.losses.LOSSES``)."""

    @abstractmethod
    def _fit(self, X_fit: pd.DataFrame, y_fit: np.ndarray) -> FitArtifact:
        """The estimator fitted on the fold's labelled fit rows (``X_fit`` indexed by
        (period, symbol))."""

    def symbol_pool(self, available: list[str]) -> list[str]:
        absent = [e for e in FEATURE_ETFS if e not in set(available)]
        if absent:
            raise ValueError(f"{self.name}: the feature ETF(s) {absent} are not in the dataset")
        return [s for s in available if s not in FEATURE_ETFS]

    def loaded_symbols(self, pool: list[str]) -> list[str]:
        return [*pool, *FEATURE_ETFS]

    def build_frames(self, bars, dividends, K, pool) -> Frames:
        if K != SESSION_MINUTES:
            raise ValueError(f"{self.name}: K={K}; the features are daily, so K must be 390")
        panel = super().build_frames(bars, dividends, K, pool).panel
        keys = pd.MultiIndex.from_arrays(
            [panel["session"], panel["symbol"]], names=["session", "symbol"]
        )
        X = build_features(bars, pool).reindex(keys)
        return Frames(panel, X.set_axis(panel.index))

    def fit_predict_fold(self, frames: Frames, fit_window, predict_window) -> FoldResult:
        panel = frames.panel
        X = frames.X.set_axis(
            pd.MultiIndex.from_arrays(
                [panel["period"], panel["symbol"]], names=["period", "symbol"]
            )
        )
        fit = np.zeros(len(panel), dtype=bool)
        if fit_window is not None:
            fit[panel.index.get_indexer(rows_in(panel, fit_window).index)] = True
            fit &= np.isfinite(panel["y"].to_numpy(dtype=float))
        if not fit.any():
            return FoldResult(empty_predictions(), None, 0, self.loss, None)
        y_fit = panel["y"].to_numpy(dtype=float)[fit]
        artifact = self._fit(X[fit], y_fit)

        rows = rows_in(panel, predict_window)
        at = panel.index.get_indexer(rows.index)
        out = pd.DataFrame(
            {
                "period": rows["period"].to_numpy(),
                "symbol": rows["symbol"].to_numpy(),
                "prediction": artifact.predict(X.iloc[at]),
            }
        )
        return FoldResult(
            out.dropna(subset=["prediction"]).reset_index(drop=True),
            artifact,
            int(fit.sum()),
            self.loss,
            baseline_prediction(y_fit, self.loss),
        )
