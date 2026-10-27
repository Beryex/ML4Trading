"""``nonlinear``: a LightGBM regressor on the features, ``stock_id`` as a categorical feature.

LightGBM splits on missing values itself, so the features go in as they are. A fit reproduces
exactly only with ``deterministic`` and ``force_row_wise`` (ML4Investment's fixed training
configuration) and the same thread count, which is therefore fixed here, not read from the
machine. LightGBM is imported only when a booster is trained or loaded: it needs an OpenMP
runtime (on macOS, ``brew install libomp``) that the other methods do not.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml4trading.models.learned.base import FitArtifact, LearnedModel
from ml4trading.models.learned.losses import LOSSES
from ml4trading.models.registry import register_model

if TYPE_CHECKING:
    import lightgbm as lgb

NUM_THREADS = 8
CATEGORICAL = "stock_id"

#: LightGBM's own parameter names, passed to it as they are.
BOOSTING_PARAMS = (
    "num_leaves",
    "min_data_in_leaf",
    "learning_rate",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
    "lambda_l1",
    "lambda_l2",
    "max_depth",
    "min_gain_to_split",
    "min_sum_hessian_in_leaf",
    "max_bin",
)
OBJECTIVES = {"mse": "regression", "l1": "regression_l1"}


class LGBMArtifact(FitArtifact):
    """A trained booster, saved as LightGBM's own text model."""

    FILENAME = "model.txt"

    booster: lgb.Booster

    def __init__(self, booster):
        self.booster = booster

    def feature_names(self) -> list[str]:
        return list(self.booster.feature_name())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.booster.predict(X[self.feature_names()], num_threads=NUM_THREADS), dtype=float
        )

    def save(self, directory: Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(Path(directory) / self.FILENAME))

    @classmethod
    def load(cls, directory: Path) -> LGBMArtifact:
        import lightgbm as lgb

        return cls(lgb.Booster(model_file=str(Path(directory) / cls.FILENAME)))


@register_model("nonlinear")
class NonlinearModel(LearnedModel):
    PARAMS = ("loss", "n_estimators", *BOOSTING_PARAMS)

    def __init__(self, hold_overnight: bool, **params):
        super().__init__(hold_overnight, **params)
        if params["loss"] not in LOSSES:
            raise ValueError(f"nonlinear: unknown loss {params['loss']!r}; known: {LOSSES}")

    @property
    def loss(self) -> str:
        return str(self.params["loss"])

    def _fit(self, X_fit: pd.DataFrame, y_fit: np.ndarray) -> LGBMArtifact:
        import lightgbm as lgb

        params = {
            "objective": OBJECTIVES[self.loss],
            **{k: self.params[k] for k in BOOSTING_PARAMS},
            "seed": 0,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": NUM_THREADS,
            "verbosity": -1,
        }
        data = lgb.Dataset(X_fit, label=y_fit, categorical_feature=[CATEGORICAL])
        booster = lgb.train(params, data, num_boost_round=int(self.params["n_estimators"]))
        return LGBMArtifact(booster)
