"""The learned models' shared part: the features as frames, the fold's fit and predict.

A subclass declares its parameters (``PARAMS``) and its training loss, and implements ``_fit``,
which turns a fold's fit rows into a ``FitArtifact``; everything else -- which symbols, which
rows, the label, causality -- is here, so the methods cannot differ on it.

A fit may also be given a validation evaluation set (``EarlyStop``): a booster then stops on its
squared error there and keeps the best round. The set is built here, in the units the fit trains
on, so that the label transform and the unlabelled rows are handled once for every method.

The label is the horizon label (``ml4trading.models.learned.horizon``): the return over the next
``horizon`` sessions, or ``horizon: 1``, the next session's, exactly as before. A fit uses only
the rows whose label is realized by the first session after its window, and so does the
early-stopping set after the last of its periods.

The features are ML4Investment's (``ml4trading.features.build_features``); with ``extra_features``
the slow cross-sectional features of ``ml4trading.features.slow`` are joined on after them, before
the model's own transform of the features, so a transform sees the new columns too. The frames are
built once per run, so ``extra_features`` is a frame parameter: a grid may not search it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np
import pandas as pd

from ml4trading.config import SESSION_MINUTES
from ml4trading.features import FEATURE_ETFS, build_features, load_feature_set, slow_features
from ml4trading.models.base import FoldResult, Frames, Model, empty_predictions, rows_in
from ml4trading.models.learned.horizon import (
    HORIZON_SCALES,
    horizon_labels,
    realized_by,
    window_labels,
)
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
    #: The keys ``build_frames`` reads: ``extra_features`` adds the slow features to the frames,
    #: so it is fixed for a run (``Model.FRAME_PARAMS``). A subclass with frame parameters of its
    #: own extends this tuple.
    FRAME_PARAMS: ClassVar[tuple[str, ...]] = (*Model.FRAME_PARAMS, "extra_features")
    #: The keys every learned method takes, which its ``PARAMS`` includes: ``horizon`` and
    #: ``horizon_scale`` shape the label (``ml4trading.models.learned.horizon``);
    #: ``extra_features`` adds the slow features to the frames (``build_frames``);
    #: ``smooth_halflife`` is only checked here, nothing reads it yet.
    SHARED_PARAMS: ClassVar[tuple[str, ...]] = (
        "horizon",
        "horizon_scale",
        "smooth_halflife",
        "extra_features",
    )

    #: The columns of the ``feature_set`` parameter, set by ``_read_feature_set`` in a model that
    #: has one; None is every column.
    _feature_set: list[str] | None = None

    def __init__(self, hold_overnight: bool, **params):
        unknown = sorted(set(params) - set(self.PARAMS))
        missing = sorted(set(self.PARAMS) - set(params))
        if unknown or missing:
            raise ValueError(
                f"{self.name}: unknown parameter(s) {unknown}, missing parameter(s) {missing}"
            )
        super().__init__(hold_overnight, **params)
        self._check_shared_params()

    def _check_shared_params(self) -> None:
        """Refuse a bad value of a ``SHARED_PARAMS`` key the model has (a subclass whose
        ``PARAMS`` omits one gets its default)."""
        horizon, scale = self.params.get("horizon", 1), self.params.get("horizon_scale", "sum")
        halflife = self.params.get("smooth_halflife", 0)
        extra = self.params.get("extra_features", False)
        if isinstance(horizon, bool) or not isinstance(horizon, int | np.integer) or horizon < 1:
            raise ValueError(f"{self.name}: horizon must be an integer >= 1, got {horizon!r}")
        if scale not in HORIZON_SCALES:
            raise ValueError(
                f"{self.name}: horizon_scale must be one of {HORIZON_SCALES}, got {scale!r}"
            )
        if (
            isinstance(halflife, bool)
            or not isinstance(halflife, int | float | np.integer | np.floating)
            or not (np.isfinite(halflife) and halflife >= 0)
        ):
            raise ValueError(
                f"{self.name}: smooth_halflife must be a number >= 0 (sessions; 0 is off), "
                f"got {halflife!r}"
            )
        if not isinstance(extra, bool):
            raise ValueError(f"{self.name}: extra_features must be true or false, got {extra!r}")

    def _read_feature_set(self) -> None:
        """Read the ``feature_set`` parameter into ``_feature_set`` -- the set's columns, or None
        for ``all`` -- at construction, so that a misspelt set is refused before any data is
        loaded and every fold of a run reads the same columns."""
        self._feature_set = load_feature_set(self.params["feature_set"])

    def _feature_set_columns(self, available: pd.Index) -> list[str]:
        """The feature set's columns in the set's order (every column of ``available`` for
        ``all``). A set column the features lack is an error, named."""
        columns = list(available) if self._feature_set is None else self._feature_set
        absent = [c for c in columns if c not in available]
        if absent:
            raise ValueError(
                f"{self.name}: feature set {self.params['feature_set']!r} names column(s) the "
                f"features do not have: {absent}"
            )
        return columns

    @property
    def extra_features(self) -> bool:
        """Whether the frames also hold the slow features (``ml4trading.features.slow``)."""
        return bool(self.params.get("extra_features", False))

    @property
    def horizon(self) -> int:
        """How many sessions of returns the label sums (or averages): 1 is the next session's."""
        return int(self.params.get("horizon", 1))

    @property
    def horizon_scale(self) -> str:
        """``sum`` or ``mean`` (``ml4trading.models.learned.horizon.HORIZON_SCALES``)."""
        return str(self.params.get("horizon_scale", "sum"))

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
        X = build_features(bars, pool).reindex(keys).set_axis(panel.index)
        if self.extra_features:
            slow = slow_features(bars, pool).reindex(keys).set_axis(panel.index)
            # the frames share their (copy-on-write) columns: the join copies none of X's
            X = pd.concat([X, slow], axis=1)
        return Frames(panel, self._transform_features(X, panel))

    def _transform_features(self, X: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        """The last step of ``build_frames``: the features the fits and predictions will read,
        from ``X`` (the build's own float32 frame, row-aligned with ``panel``). The identity by
        default. A transform may overwrite ``X`` and return it, so that it costs no second copy
        of a frame that is about a gigabyte on the full universe; it must keep the float32 dtype
        and the row order, and may use only the values of the row's own period."""
        return X

    def _realized_by(
        self, panel: pd.DataFrame, end: pd.Timestamp, exit_session: np.ndarray
    ) -> np.ndarray:
        """Rows whose label is known by the first session after ``end``: the label's exit
        (``exit_session``, from ``horizon_labels``) is on or before it. Overnight, a label exits
        at the open of the symbol's row ``horizon`` ahead -- the next one at ``horizon: 1`` --
        which a gap in its data can push past the embargo into the window the fit is scored on.
        Intraday at ``horizon: 1`` it exits at its own close, inside the window; a longer
        intraday label exits at the close of the row before that open, which that open bounds, so
        the same rule keeps it. ``panel`` is in (symbol, period) order, as ``build_panel``
        returns it."""
        if not self.hold_overnight and self.horizon == 1:
            return np.ones(len(panel), dtype=bool)
        return realized_by(panel, exit_session, end)

    @staticmethod
    def _evaluation_rows(
        panel: pd.DataFrame,
        X: pd.DataFrame,
        transform,
        early_stop: EarlyStop,
        horizon: int = 1,
        horizon_scale: str = "sum",
    ) -> tuple[pd.DataFrame, np.ndarray, EarlyStop]:
        """The early-stopping set: the rows of ``early_stop.periods`` x ``early_stop.symbols``
        with a label (a symbol's last overnight period has none), and their label through the
        fold's ``transform`` (``ml4trading.models.learned.horizon.window_labels``). A
        ``horizon`` label that exits after the first session following the last of the periods
        is no label, as in the validation scoring. As there, the transform is applied to every
        pool row of those periods, so a period's mean is the pool's cross-section the fit
        demeaned over, and only then taken at the evaluation rows. ``X`` is indexed by
        (period, symbol), row-aligned with ``panel``."""
        label = window_labels(panel, early_stop.periods, horizon, horizon_scale, transform)
        rows = np.isfinite(label) & panel["symbol"].isin(set(early_stop.symbols)).to_numpy()
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
        y_h, exit_session = horizon_labels(panel, self.horizon, self.horizon_scale)
        fit = np.zeros(len(panel), dtype=bool)
        if fit_window is not None:
            fit[panel.index.get_indexer(rows_in(panel, fit_window).index)] = True
            fit &= np.isfinite(y_h)
            fit &= self._realized_by(panel, pd.Timestamp(fit_window[1]), exit_session)
        if not fit.any():
            return FoldResult(
                empty_predictions(),
                None,
                0,
                self.loss,
                None,
                horizon=self.horizon,
                horizon_scale=self.horizon_scale,
            )
        y_raw, period = y_h[fit], panel["period"].to_numpy()[fit]
        transform = fit_label_transform(self.label_kind, y_raw, period)
        y_fit = transform.apply(y_raw, period)
        eval_set = (
            None
            if early_stop is None
            else self._evaluation_rows(
                panel, X, transform, early_stop, self.horizon, self.horizon_scale
            )
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
            self.horizon,
            self.horizon_scale,
        )
