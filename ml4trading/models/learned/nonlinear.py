"""``nonlinear``: a LightGBM regressor on the features, ``stock_id`` as a categorical feature.

LightGBM splits on missing values itself, so the features go in as they are, unless
``feature_transform`` is ``cs_rank``: then each stock-specific feature is replaced, period by
period, by its cross-sectional rank (Gu, Kelly & Xiu 2020, "Empirical Asset Pricing via Machine
Learning": ranked within each month to [-1, 1]), so a fit sees where a stock stands among the
others of the day, not the level of a feature that drifts with the market. It is a frame
parameter: the frames are built once per run, with the ranks in.

Which columns a fit reads is a fit-time choice, so a grid may search it: ``feature_set`` names a
subset of the features (``ml4trading/features/feature_sets/``; ``all`` is every column), and
``stock_id`` is ``categorical`` (the code is a LightGBM categorical feature, split on as
``cat_smooth``, ``min_data_per_group``, ``max_cat_threshold`` and ``cat_l2`` say) or ``drop``
(the code is left out). With ``linear_tree`` each leaf of a tree holds a linear model on the
features its path splits on (Shi, Li & Li 2019, "Gradient Boosting With Piece-Wise Linear
Regression Trees"; ridge-penalised by ``linear_lambda``); a row that lacks one of those
features gets the leaf's constant output instead. LightGBM fits it on ``mse`` only.

A fit reproduces exactly only with ``deterministic`` and ``force_row_wise`` (ML4Investment's
fixed training configuration) and the same thread count, which is therefore fixed here, not read
from the machine. LightGBM is imported only when a booster is trained or loaded: it needs an
OpenMP runtime (on macOS, ``brew install libomp``) that the other methods do not.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml4trading.features import CALENDAR, FEATURE_ETFS
from ml4trading.models.learned.base import EarlyStop, FitArtifact, LearnedModel
from ml4trading.models.learned.labels import LABELS
from ml4trading.models.learned.losses import LOSSES
from ml4trading.models.registry import register_model

if TYPE_CHECKING:
    import lightgbm as lgb

NUM_THREADS = 8
CATEGORICAL = "stock_id"
#: ``cs_rank``: the per-period rank in [-1, 1] of Gu, Kelly & Xiu (2020); ``none``: as built.
FEATURE_TRANSFORMS = ("none", "cs_rank")
#: ``categorical``: ``stock_id`` is a LightGBM categorical feature; ``drop``: it is not used.
STOCK_ID_MODES = ("categorical", "drop")

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
    # how a categorical feature (``stock_id``) is split on; LightGBM 4's defaults, see
    # config.METHOD_DEFAULTS
    "cat_smooth",
    "min_data_per_group",
    "max_cat_threshold",
    "cat_l2",
    # piece-wise linear trees (Shi, Li & Li 2019): a linear model in each leaf, ridge-penalised
    # by ``linear_lambda``; LightGBM 4's defaults, see config.METHOD_DEFAULTS
    "linear_tree",
    "linear_lambda",
)
OBJECTIVES = {"mse": "regression", "l1": "regression_l1"}


def _is_ranked(column: str) -> bool:
    """Whether ``cs_rank`` ranks the column: not the stock code, the ETFs' columns (the same for
    every stock of a period) or the calendar's (a function of the date)."""
    return not (
        column == CATEGORICAL
        or column in CALENDAR
        or column.startswith(tuple(f"{etf}_" for etf in FEATURE_ETFS))
    )


def cs_rank_features(X: pd.DataFrame, period: np.ndarray) -> pd.DataFrame:
    """``X`` with each ranked column replaced, within each period, by ``2 * (r - 1) / (n - 1) -
    1``: ``r`` is the value's average-tie rank among the period's non-NaN values and ``n`` their
    count, so the smallest maps to -1 and the largest to 1 (equal values share their average
    rank). NaN stays NaN; a period with a single value maps it to 0.0, the middle of the range,
    and a period with none leaves the column NaN. The rank is taken over the rows a period has,
    i.e. the symbols with a bar in that session: the same convention as ``linear``'s per-period
    median imputation. ``period`` is row-aligned with ``X``.

    ``X`` is overwritten column by column and returned, so the working memory is a few columns
    of float64, not a second frame."""
    codes, periods = pd.factorize(np.asarray(period))
    for j, column in enumerate(X.columns):
        if not _is_ranked(column):
            continue
        values = X.iloc[:, j].to_numpy()
        known = ~np.isnan(values)
        rank = pd.Series(values).groupby(codes).rank(method="average").to_numpy()
        n = np.bincount(codes[known], minlength=len(periods))[codes].astype(float)
        scaled = 2.0 * (rank - 1.0) / np.maximum(n - 1.0, 1.0) - 1.0
        scaled[known & (n == 1.0)] = 0.0
        X.iloc[:, j] = scaled.astype(np.float32)
    return X


class LGBMArtifact(FitArtifact):
    """A trained booster, saved as LightGBM's own text model. After early stopping ``predict``
    reads only the first ``best_iteration`` rounds (``lgb.train`` already drops the later ones
    on return; the artifact does not rely on that), and a saved model keeps only the best
    rounds, so a loaded artifact has nothing left to cut (its ``best_iteration`` is None) and
    predicts the same."""

    FILENAME = "model.txt"

    booster: lgb.Booster

    def __init__(self, booster, best_iteration: int | None = None, eval_mse: float | None = None):
        self.booster = booster
        self.best_iteration = best_iteration
        self.eval_mse = eval_mse

    def feature_names(self) -> list[str]:
        return list(self.booster.feature_name())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            self.booster.predict(
                X[self.feature_names()],
                num_iteration=self.best_iteration,
                num_threads=NUM_THREADS,
            ),
            dtype=float,
        )

    def native_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """What the booster itself says about each feature, indexed by feature name: ``gain``
        and ``split`` (the total loss reduction and the number of splits on it, over all trees)
        and ``shap``, the mean absolute TreeSHAP contribution to ``X``'s rows (Lundberg & Lee
        2017). LightGBM appends the expected value as the last contribution column; it belongs
        to no feature and is dropped."""
        names = self.feature_names()
        contrib = np.asarray(
            self.booster.predict(X[names], pred_contrib=True, num_threads=NUM_THREADS), dtype=float
        )
        return pd.DataFrame(
            {
                "gain": self.booster.feature_importance("gain"),
                "split": self.booster.feature_importance("split"),
                "shap": np.abs(contrib[:, :-1]).mean(axis=0),
            },
            index=pd.Index(names, name="feature"),
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
    PARAMS = (
        *LearnedModel.SHARED_PARAMS,
        "loss",
        "label",
        "feature_transform",
        "stock_id",
        "feature_set",
        "n_estimators",
        *BOOSTING_PARAMS,
    )
    #: The features are built once per run, with ``extra_features`` joined and
    #: ``feature_transform`` applied.
    FRAME_PARAMS = (*LearnedModel.FRAME_PARAMS, "feature_transform")

    def __init__(self, hold_overnight: bool, **params):
        super().__init__(hold_overnight, **params)
        if params["loss"] not in LOSSES:
            raise ValueError(f"nonlinear: unknown loss {params['loss']!r}; known: {LOSSES}")
        if params["label"] not in LABELS:
            raise ValueError(f"nonlinear: unknown label {params['label']!r}; known: {LABELS}")
        if params["feature_transform"] not in FEATURE_TRANSFORMS:
            raise ValueError(
                f"nonlinear: unknown feature_transform {params['feature_transform']!r}; "
                f"known: {FEATURE_TRANSFORMS}"
            )
        if params["stock_id"] not in STOCK_ID_MODES:
            raise ValueError(
                f"nonlinear: unknown stock_id {params['stock_id']!r}; known: {STOCK_ID_MODES}"
            )
        if not isinstance(params["linear_tree"], bool):
            raise ValueError(
                f"nonlinear: linear_tree must be true or false, got {params['linear_tree']!r}"
            )
        if params["linear_tree"] and params["loss"] == "l1":
            raise ValueError(
                "nonlinear: linear_tree cannot be fitted on loss='l1' (LightGBM refuses "
                "regression_l1 with linear trees); use loss='mse'"
            )
        # read here, once: a misspelt set, or one that cannot serve a categorical stock_id, is
        # refused before any data is loaded, and every fold of the run reads the same columns
        self._read_feature_set()
        if (
            params["stock_id"] == "categorical"
            and self._feature_set is not None
            and CATEGORICAL not in self._feature_set
        ):
            raise ValueError(
                f"nonlinear: feature set {params['feature_set']!r} lacks {CATEGORICAL!r}, which "
                f"stock_id='categorical' needs; add it to the set or use stock_id='drop'"
            )

    @property
    def loss(self) -> str:
        return str(self.params["loss"])

    @property
    def label_kind(self) -> str:
        return str(self.params["label"])

    def _transform_features(self, X: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
        if self.params["feature_transform"] == "cs_rank":
            return cs_rank_features(X, panel["period"].to_numpy())
        return X

    def _fit_columns(self, available: pd.Index) -> list[str]:
        """The columns a fit trains on, in order: the feature set's (every column for ``all``),
        less ``stock_id`` under ``stock_id='drop'``. A set column the features lack is an error."""
        columns = self._feature_set_columns(available)
        if self.params["stock_id"] == "drop":
            columns = [c for c in columns if c != CATEGORICAL]
        return columns

    def _fit(
        self,
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray, EarlyStop] | None = None,
    ) -> LGBMArtifact:
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
        columns = self._fit_columns(X_fit.columns)
        if columns != list(X_fit.columns):  # the default fit hands LightGBM the frame itself
            X_fit = X_fit[columns]
        categorical = [CATEGORICAL] if CATEGORICAL in columns else []
        data = lgb.Dataset(X_fit, label=y_fit, categorical_feature=categorical)
        if eval_set is None:
            booster = lgb.train(params, data, num_boost_round=int(self.params["n_estimators"]))
            return LGBMArtifact(booster)
        X_eval, y_eval, stop = eval_set
        # the evaluation error is the squared one whatever the training loss (under ``l1`` the
        # objective's own default metric would be the absolute error)
        params["metric"] = "l2"
        booster = lgb.train(
            params,
            data,
            num_boost_round=stop.max_rounds,
            valid_sets=[lgb.Dataset(X_eval[columns], label=y_eval, reference=data)],
            callbacks=[lgb.early_stopping(stop.patience, verbose=False)],
        )
        return LGBMArtifact(booster, booster.best_iteration, booster.best_score["valid_0"]["l2"])
