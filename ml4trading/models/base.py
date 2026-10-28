"""The model interface every method implements.

A run calls, once, on its base configuration:

    model  = ModelClass(**cfg.model)
    pool   = model.symbol_pool(all_symbols)          # the symbols it may trade
    frames = model.build_frames(bars, dividends, K, pool)   # bars of model.loaded_symbols(pool)

and then, per fold and per candidate configuration:

    result = ModelClass(**candidate.model).fit_predict_fold(frames, fit_window, predict_window)

``frames.panel`` is the labelled period panel (``ml4trading.periods``) over the pool for the
WHOLE dataset; a learned model adds ``frames.X``, its features, row-aligned with the panel. Every
frame is built once over the full history and a fold reaches the model only as row masks: a
recursive feature (an exponential average) computed on a trimmed history would shift. The
model is trusted to respect two causality rules, which ``ml4trading.testing.assert_causal``
checks:

* the fit may use only rows whose session lies inside ``fit_window`` (the trainer already ends
  it ``embargo_days`` sessions before whatever the fit is scored on, because a label ``y`` looks
  one period ahead); ``fit_window`` is None when nothing is fittable;
* the prediction for period t may use only information from before t's open -- t's own bar
  does not exist yet.

``FoldResult.predictions`` has columns ``period``, ``symbol``, ``prediction`` (a predicted
return of the period, same units as ``y``) over ``predict_window``; a missing (period, symbol)
is simply not traded. The trainer keeps only the fold's universe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import pandas as pd

from ml4trading.periods import build_panel


class Frames(NamedTuple):
    """What a run builds once: the labelled panel over the pool and, for a learned model, its
    float32 features (``X.index`` equals ``panel.index``; None for a model that fits nothing)."""

    panel: pd.DataFrame
    X: pd.DataFrame | None


@dataclass(frozen=True)
class FoldResult:
    """One fold's fit and predictions. ``artifact`` is the fitted estimator (a
    ``ml4trading.models.learned.base.FitArtifact``) or None; ``loss`` and
    ``baseline_prediction`` (the fit rows' mean under mse, median under l1) are set by a model
    that fits a loss; ``fit_label_mean`` (the mean label of the rows the fit trained on) is
    set by a model that fits, whatever its loss -- the reference of the validation diagnostics'
    squared-error R². ``label_transform`` (a ``ml4trading.models.learned.labels.LabelTransform``)
    is what the fit turned its labels into before training on them; the validation scoring
    applies it to the realized returns, so they are in the fit's units. A fit-free model leaves
    all four None; a fit with no rows, the last three. A fit that stopped early on an evaluation
    set (``ml4trading.models.learned.base.EarlyStop``) records ``best_iteration``, the round it
    kept, and ``eval_mse``, the squared error on that set at that round; otherwise both are
    None."""

    predictions: pd.DataFrame
    artifact: object | None
    n_fit_rows: int
    loss: str | None = None
    baseline_prediction: float | None = None
    fit_label_mean: float | None = None
    label_transform: object | None = None
    best_iteration: int | None = None
    eval_mse: float | None = None


class Model(ABC):
    #: Set by ``ml4trading.models.registry.register_model``.
    name: ClassVar[str] = ""
    #: The ``model.*`` keys ``build_frames`` reads: fixed for a run, so a grid may not search them.
    FRAME_PARAMS: ClassVar[tuple[str, ...]] = ("hold_overnight",)

    def __init__(self, hold_overnight: bool, **params):
        self.hold_overnight = bool(hold_overnight)
        self.params = dict(params)

    def symbol_pool(self, available: list[str]) -> list[str]:
        """The symbols this model may trade (default: all of them)."""
        return list(available)

    def loaded_symbols(self, pool: list[str]) -> list[str]:
        """The symbols whose bars ``build_frames`` reads (default: the pool)."""
        return list(pool)

    def build_frames(
        self, bars: pd.DataFrame, dividends: pd.DataFrame, K: int, pool: list[str]
    ) -> Frames:
        """The run's frames from the loaded bars and dividends (default: the pool's panel)."""
        keep = set(pool)
        panel = build_panel(
            bars[bars["symbol"].isin(keep)],
            K,
            self.hold_overnight,
            dividends[dividends["symbol"].isin(keep)],
        )
        return Frames(panel, None)

    @abstractmethod
    def fit_predict_fold(self, frames: Frames, fit_window, predict_window) -> FoldResult: ...


def rows_in(panel: pd.DataFrame, window) -> pd.DataFrame:
    """Panel rows whose session lies in the inclusive ``window``."""
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return panel[(panel["session"] >= lo) & (panel["session"] <= hi)]


def empty_predictions() -> pd.DataFrame:
    """A zero-row predictions frame with the real columns' dtypes."""
    return pd.DataFrame(
        {
            "period": pd.Series(dtype="datetime64[ns]"),
            "symbol": pd.Series(dtype="str"),
            "prediction": pd.Series(dtype="float64"),
        }
    )
