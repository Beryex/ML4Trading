"""The model interface every method implements.

Per fold the trainer calls, in order:

    model = ModelClass(**cfg.model)              # the run's model parameters
    pool  = model.symbol_pool(all_symbols)       # which symbols to load for it
    model.fit(panel, fit_window)                 # learn from panel rows inside fit_window
    preds = model.predict(panel, predict_window) # one prediction per (period, symbol)

``panel`` is the labelled period panel (``ml4trading.periods``) over the model's pool, for the
WHOLE dataset: the model is trusted to respect two causality rules, which
``ml4trading.testing.assert_causal`` checks:

* ``fit`` may use only rows whose session lies inside ``fit_window`` (the trainer already ends
  it ``embargo_days`` sessions before whatever the fit is scored on, because a label ``y`` looks
  one period ahead);
* the prediction for period t may use only rows of periods strictly BEFORE t -- the decision
  is taken at t's open, before t's bar exists.

``predict`` returns a frame with columns ``period``, ``symbol``, ``prediction`` (a predicted
return of the period, same units as ``y``); a missing (period, symbol) is simply not traded.
"""

from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import pandas as pd


class Model(ABC):
    #: Set by ``ml4trading.models.registry.register_model``.
    name: ClassVar[str] = ""

    def __init__(self, hold_overnight: bool, **params):
        self.hold_overnight = bool(hold_overnight)
        self.params = dict(params)

    def symbol_pool(self, available: list[str]) -> list[str]:
        """The symbols this model reads and may trade (default: all of them)."""
        return list(available)

    @abstractmethod
    def fit(self, panel: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp]) -> None: ...

    @abstractmethod
    def predict(
        self, panel: pd.DataFrame, window: tuple[pd.Timestamp, pd.Timestamp]
    ) -> pd.DataFrame: ...

    def save(self, directory: Path) -> None:
        """Persist the fitted model (default: pickle the whole object)."""
        Path(directory).mkdir(parents=True, exist_ok=True)
        with open(Path(directory) / "model.pkl", "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(directory: Path) -> Model:
        with open(Path(directory) / "model.pkl", "rb") as fh:
            return pickle.load(fh)


def rows_in(panel: pd.DataFrame, window) -> pd.DataFrame:
    """Panel rows whose session lies in the inclusive ``window``."""
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return panel[(panel["session"] >= lo) & (panel["session"] <= hi)]
