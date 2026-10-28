"""The learned models' shared part: the features as frames, the fold's fit and predict.

A subclass declares its parameters (``PARAMS``) and its training loss, and implements ``_fit``,
which turns a fold's fit rows into a ``FitArtifact``; everything else -- which symbols, which
rows, the label, causality -- is here, so the methods cannot differ on it.

A fit may also be given a validation evaluation set (``EarlyStop``): a booster then stops on its
squared error there and keeps the best round. The set is built here, in the units the fit trains
on, so that the label transform and the unlabelled rows are handled once for every method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from ml4trading.config import SESSION_MINUTES
from ml4trading.features import FEATURE_ETFS, build_features
from ml4trading.models.base import FoldResult, Frames, Model, empty_predictions, rows_in
from ml4trading.models.learned.labels import fit_label_transform
from ml4trading.models.learned.losses import baseline_prediction


@dataclass(frozen=True)
class EarlyStop:
    """Where a fit stops early: on the squared error over the panel rows of ``periods`` x
    ``symbols`` that have a label. It trains for at most ``max_rounds`` rounds and stops once the
    error has not improved for ``patience`` rounds."""

    periods: tuple[pd.Timestamp, ...]
    symbols: tuple[str, ...]
    max_rounds: int
    patience: int


class FitArtifact(ABC):
    """A fold's fitted estimator. ``predict`` takes features indexed by (period, symbol) with
    (at least) ``feature_names()`` as columns; ``save`` writes ``FILENAME`` into a directory and
    ``load`` reads it back, predicting identically."""

    FILENAME: ClassVar[str]
    #: Set by an artifact fitted with early stopping: the round it kept (1-based) and the
    #: squared error on the evaluation set at that round.
    best_iteration: int | None = None
    eval_mse: float | None = None

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

    @property
    def label_kind(self) -> str:
        """The label the fit trains on (``ml4trading.models.learned.labels.LABELS``): the raw
        return unless a model has a ``label`` parameter."""
        return "raw"

    @abstractmethod
    def _fit(
        self,
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray, EarlyStop] | None = None,
    ) -> FitArtifact:
        """The estimator fitted on the fold's labelled fit rows (``X_fit`` indexed by
        (period, symbol)); ``y_fit`` is their label after the model's ``label_kind``.
        ``eval_set`` is None, or the features and labels (in the same units as ``y_fit``) of the
        evaluation rows and the ``EarlyStop`` that chose them. It is passed only when a fit asks
        to stop early, so a model that cannot may leave the parameter out of its signature (a
        two-argument ``_fit``, which then fails with a TypeError if asked to); one that keeps it
        refuses a set."""

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
        X = X.set_axis(panel.index)
        return Frames(panel, self._transform_features(X, panel))

    def _transform_features(self, X: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        """The last step of ``build_frames``: the features the fits and predictions will read,
        from ``X`` (the build's own float32 frame, row-aligned with ``panel``). The identity by
        default. A transform may overwrite ``X`` and return it, so that it costs no second copy
        of a frame that is about a gigabyte on the full universe; it must keep the float32 dtype
        and the row order, and may use only the values of the row's own period."""
        return X

    def _realized_by(self, panel: pd.DataFrame, end: pd.Timestamp) -> np.ndarray:
        """Rows whose label is known by the first session after ``end``. Overnight, a label exits
        at the symbol's NEXT period's open, which a gap in its data can push past the embargo
        into the window the fit is scored on; intraday it exits at its own close. ``panel`` is
        in (symbol, period) order, as ``build_panel`` returns it."""
        if not self.hold_overnight:
            return np.ones(len(panel), dtype=bool)
        later = panel.loc[panel["session"] > end, "session"]
        if later.empty:
            return np.ones(len(panel), dtype=bool)
        exits = panel.groupby("symbol", observed=True, sort=False)["session"].shift(-1)
        return (exits <= later.min()).to_numpy()

    @staticmethod
    def _evaluation_rows(
        panel: pd.DataFrame, X: pd.DataFrame, transform, early_stop: EarlyStop
    ) -> tuple[pd.DataFrame, np.ndarray, EarlyStop]:
        """The early-stopping set: the rows of ``early_stop.periods`` x ``early_stop.symbols``
        with a finite ``y`` (a symbol's last overnight period has none), and their label through
        the fold's ``transform``. As in the validation scoring, the transform is applied to every
        pool row of those periods, so a period's mean is the pool's cross-section the fit
        demeaned over, and only then taken at the evaluation rows. ``X`` is indexed by
        (period, symbol), row-aligned with ``panel``."""
        y = panel["y"].to_numpy(dtype=float)
        labelled = panel["period"].isin(set(early_stop.periods)).to_numpy() & np.isfinite(y)
        label = np.full(len(panel), np.nan)
        label[labelled] = transform.apply(y[labelled], panel["period"].to_numpy()[labelled])
        rows = labelled & panel["symbol"].isin(set(early_stop.symbols)).to_numpy()
        if not rows.any():
            raise ValueError("early stopping: no labelled evaluation row")
        return X[rows], label[rows], early_stop

    def fit_predict_fold(
        self, frames: Frames, fit_window, predict_window, early_stop: EarlyStop | None = None
    ) -> FoldResult:
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
            fit &= self._realized_by(panel, pd.Timestamp(fit_window[1]))
        if not fit.any():
            return FoldResult(empty_predictions(), None, 0, self.loss, None)
        y_raw, period = panel["y"].to_numpy(dtype=float)[fit], panel["period"].to_numpy()[fit]
        transform = fit_label_transform(self.label_kind, y_raw, period)
        y_fit = transform.apply(y_raw, period)
        eval_set = (
            None if early_stop is None else self._evaluation_rows(panel, X, transform, early_stop)
        )
        # a model that cannot stop early may declare ``_fit`` without ``eval_set``
        artifact = (
            self._fit(X[fit], y_fit) if eval_set is None else self._fit(X[fit], y_fit, eval_set)
        )

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
            float(np.mean(y_fit)),
            transform,
            artifact.best_iteration,
            artifact.eval_mse,
        )
