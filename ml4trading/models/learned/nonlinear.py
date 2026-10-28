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
numerical features its path splits on (Shi, Li & Li 2019, "Gradient Boosting With Piece-Wise
Linear Regression Trees"; ridge-penalised by ``linear_lambda``); a row that lacks one of those
features gets the leaf's constant output instead. LightGBM fits it on ``mse`` only.

``monotone_constraints`` maps feature names to -1, 0 or 1: the booster's output is then
non-increasing, free or non-decreasing in that feature. It is a mapping, not a LightGBM parameter
as it stands: each fit turns it into LightGBM's list, one sign per column the fit trains on in the
fit's order (a column not named gets 0), and passes nothing when no sign is set. A name the fit
does not train on is refused; so is a constraint on the categorical ``stock_id`` (LightGBM aborts
the process on one) and one beside ``linear_tree`` (LightGBM enforces a constraint when it picks a
split, not in a leaf's linear model). LightGBM refuses a constraint on ``regression_l1`` too, so
a set one needs ``loss='mse'``, which is refused at construction as well.

With ``init_model='linear'`` the booster starts from an initial estimate F0 (Friedman 2001,
"Greedy Function Approximation: A Gradient Boosting Machine") in place of LightGBM's own constant
start: ``linear``'s elastic net, fitted first on the same fit rows and the same labels, in the
units the booster trains on (``init_alpha_ratio``, ``init_l1_ratio`` and ``init_feature_set`` are
its ``alpha_ratio``, ``l1_ratio`` and ``feature_set``). Its prediction is LightGBM's
``init_score`` (Dataset) on the fit rows and on the early-stopping rows, so the trees fit what the
net leaves and stop on the whole model's error. LightGBM's own prediction is the trees' alone, so
the fitted model (``HybridArtifact``) adds the net's back. A monotone constraint would bind the
trees, not the net, and is refused with it.

A fit reproduces exactly only with ``deterministic`` and ``force_row_wise`` (ML4Investment's
fixed training configuration) and the same thread count, which is therefore fixed here, not read
from the machine. LightGBM is imported only when a booster is trained or loaded: it needs an
OpenMP runtime (on macOS, ``brew install libomp``) that the other methods do not.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from ml4trading.config import METHOD_DEFAULTS
from ml4trading.features import CALENDAR, FEATURE_ETFS, load_feature_set
from ml4trading.models.learned.base import EarlyStop, FitArtifact, LearnedModel
from ml4trading.models.learned.labels import LABELS
from ml4trading.models.learned.linear import LinearArtifact, fit_elastic_net
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
#: The signs of ``monotone_constraints``: non-increasing, unconstrained, non-decreasing.
MONOTONE_SIGNS = (-1, 0, 1)
#: LightGBM's ``monotone_constraints_method`` values (Parameters, "Learning Control Parameters");
#: it takes any other string without complaint, so they are listed here.
MONOTONE_METHODS = ("basic", "intermediate", "advanced")
#: ``init_model``: ``none`` boosts from LightGBM's own constant start; ``linear`` from the
#: prediction of ``linear``'s elastic net, fitted on the same rows (Friedman 2001's F0).
INIT_MODELS = ("none", "linear")
#: The linear initial score's iteration cap: ``linear``'s own, a convergence cap rather than a
#: model choice (see its comment in ``config.METHOD_DEFAULTS``).
INIT_MAX_ITER = int(METHOD_DEFAULTS["linear"]["model"]["max_iter"])

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
    # noise-robust splits: extremely randomized trees (Geurts, Ernst & Wehenkel 2006), node
    # smoothing, a per-node feature draw; and how ``monotone_constraints`` is enforced. LightGBM 4's
    # defaults, see config.METHOD_DEFAULTS
    "extra_trees",
    "path_smooth",
    "feature_fraction_bynode",
    "monotone_constraints_method",
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
        to no feature and is dropped. LightGBM computes no TreeSHAP for linear trees
        (``linear_tree``; it raises), so for those ``shap`` is NaN; ``gain`` and ``split`` are
        still the booster's own."""
        names = self.feature_names()
        if self.booster.params.get("linear_tree", False):  # a loaded booster reads it from file
            shap = np.full(len(names), np.nan)
        else:
            contrib = np.asarray(
                self.booster.predict(X[names], pred_contrib=True, num_threads=NUM_THREADS),
                dtype=float,
            )
            shap = np.abs(contrib[:, :-1]).mean(axis=0)
        return pd.DataFrame(
            {
                "gain": self.booster.feature_importance("gain"),
                "split": self.booster.feature_importance("split"),
                "shap": shap,
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


class HybridArtifact(FitArtifact):
    """A booster boosted from a linear initial score (``init_model='linear'``): ``init``, the
    elastic net, and ``trees``, the booster trained from its prediction. LightGBM's own
    prediction is the trees' alone (an ``init_score`` belongs to the Dataset, not to the model),
    so ``predict`` adds the net's back. ``save`` writes the booster's ``model.txt`` (only its best
    rounds, as ``LGBMArtifact`` does) and the net's ``model.npz`` into the ``init``
    subdirectory, each part in its own format; a loaded artifact predicts the same.
    ``native_importance`` is the booster's alone."""

    FILENAME = LGBMArtifact.FILENAME
    #: The subdirectory the linear part is saved in.
    INIT_DIRECTORY = "init"

    def __init__(self, trees: LGBMArtifact, init: LinearArtifact):
        self.trees, self.init = trees, init
        self.best_iteration, self.eval_mse = trees.best_iteration, trees.eval_mse

    @property
    def booster(self) -> lgb.Booster:
        return self.trees.booster

    def feature_names(self) -> list[str]:
        """The booster's columns, then those only the linear part reads."""
        names = self.trees.feature_names()
        return names + [c for c in self.init.feature_names() if c not in names]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.init.predict(X) + self.trees.predict(X)

    def native_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """The booster's own measures (``LGBMArtifact.native_importance``) over every column the
        model reads. They describe the trees only: the TreeSHAP values are the trees'
        contributions, without the net's, and a column only the net reads has none (NaN)."""
        return self.trees.native_importance(X).reindex(
            pd.Index(self.feature_names(), name="feature")
        )

    def save(self, directory: Path) -> None:
        self.trees.save(directory)
        self.init.save(Path(directory) / self.INIT_DIRECTORY)

    @classmethod
    def load(cls, directory: Path) -> HybridArtifact:
        return cls(
            LGBMArtifact.load(directory), LinearArtifact.load(Path(directory) / cls.INIT_DIRECTORY)
        )


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
        "monotone_constraints",  # not passed as it is: see ``_monotone_signs``
        # the linear initial score (``_fit_init``); none reaches LightGBM as a parameter
        "init_model",
        "init_alpha_ratio",
        "init_l1_ratio",
        "init_feature_set",
    )
    #: The features are built once per run, with ``extra_features`` joined and
    #: ``feature_transform`` applied.
    FRAME_PARAMS = (*LearnedModel.FRAME_PARAMS, "feature_transform")
    #: The columns of ``init_feature_set``, read at construction; None is every column.
    _init_feature_set: list[str] | None = None

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
        self._check_init(params)
        self._check_monotone_constraints(params)
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

    def _check_init(self, params: dict) -> None:
        """Refuse a bad initial-score parameter at construction, before any data is loaded, and
        read ``init_feature_set`` once, as ``feature_set`` is read: a misspelt set is refused and
        every fold of a run reads the same columns. All four are checked whatever ``init_model``
        is, as ``linear`` checks its own."""
        if params["init_model"] not in INIT_MODELS:
            raise ValueError(
                f"nonlinear: unknown init_model {params['init_model']!r}; known: {INIT_MODELS}"
            )
        if not 0.0 < float(params["init_l1_ratio"]) <= 1.0:
            raise ValueError(
                f"nonlinear: init_l1_ratio must lie in (0, 1], got {params['init_l1_ratio']}"
            )
        if not float(params["init_alpha_ratio"]) > 0.0:
            raise ValueError(
                f"nonlinear: init_alpha_ratio must be positive, got {params['init_alpha_ratio']}"
            )
        self._init_feature_set = load_feature_set(params["init_feature_set"])
        if self._init_feature_set is not None and set(self._init_feature_set) <= {CATEGORICAL}:
            raise ValueError(
                f"nonlinear: init_feature_set {params['init_feature_set']!r} has no column but "
                f"{CATEGORICAL!r}, which the linear initial score does not use"
            )

    def _check_monotone_constraints(self, params: dict) -> None:
        """Refuse a bad ``monotone_constraints`` mapping or ``monotone_constraints_method`` at
        construction, before any data is loaded. Whether each name is a column the fit reads is
        known only once the features are built (``_monotone_signs``)."""
        constraints = params["monotone_constraints"]
        if not isinstance(constraints, Mapping):
            raise ValueError(
                f"nonlinear: monotone_constraints must be a mapping of feature name to "
                f"{MONOTONE_SIGNS}, got {constraints!r}"
            )
        for feature, sign in constraints.items():
            if not isinstance(feature, str):
                raise ValueError(
                    f"nonlinear: monotone_constraints keys must be feature names, got {feature!r}"
                )
            if isinstance(sign, bool) or not isinstance(sign, int | np.integer):
                sign = None  # not an integer: no sign, whatever it equals (1.0, True)
            if sign not in MONOTONE_SIGNS:
                raise ValueError(
                    f"nonlinear: monotone_constraints[{feature!r}] must be one of "
                    f"{MONOTONE_SIGNS}, got {constraints[feature]!r}"
                )
        if params["monotone_constraints_method"] not in MONOTONE_METHODS:
            raise ValueError(
                f"nonlinear: monotone_constraints_method must be one of {MONOTONE_METHODS}, "
                f"got {params['monotone_constraints_method']!r}"
            )
        constrained = [feature for feature, sign in constraints.items() if sign]
        if CATEGORICAL in constrained and params["stock_id"] == "categorical":
            raise ValueError(
                f"nonlinear: monotone_constraints cannot constrain {CATEGORICAL!r}, a categorical "
                f"feature under stock_id='categorical' (LightGBM aborts on one)"
            )
        if constrained and params["loss"] == "l1":
            raise ValueError(
                f"nonlinear: monotone_constraints {constrained} cannot be fitted on loss='l1' "
                f"(LightGBM refuses them with regression_l1); use loss='mse'"
            )
        if constrained and params["linear_tree"]:
            raise ValueError(
                f"nonlinear: monotone_constraints {constrained} cannot be used with linear_tree "
                f"(LightGBM enforces a constraint when it chooses a split, not in a leaf's linear "
                f"model, so the output would not be monotone)"
            )
        if constrained and params["init_model"] != "none":
            raise ValueError(
                f"nonlinear: monotone_constraints {constrained} cannot be used with "
                f"init_model={params['init_model']!r} (a constraint binds the trees, not the "
                f"linear initial score they are added to, so the output would not be monotone)"
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

    def _monotone_signs(self, columns: list[str]) -> list[int] | None:
        """``monotone_constraints`` as LightGBM's list: the sign of each of ``columns`` in order,
        0 for one the mapping does not name. None when no sign is set, so that a fit without
        constraints hands LightGBM no such parameter. A named feature that is not one of the
        columns -- a typo, or one the feature set or ``stock_id='drop'`` leaves out -- is an
        error, named, whatever its sign."""
        constraints = self.params["monotone_constraints"]
        absent = [feature for feature in constraints if feature not in columns]
        if absent:
            raise ValueError(
                f"nonlinear: monotone_constraints names feature(s) the fit does not train on: "
                f"{absent}"
            )
        signs = [int(constraints.get(column, 0)) for column in columns]
        return signs if any(signs) else None

    def _init_columns(self, available: pd.Index) -> list[str]:
        """The columns the linear initial score is fitted on, in order: ``init_feature_set``'s
        (every column for ``all``), less ``stock_id``, as ``linear`` reads its ``feature_set``.
        A set column the features lack is an error, named."""
        columns = list(available) if self._init_feature_set is None else self._init_feature_set
        absent = [c for c in columns if c not in available]
        if absent:
            raise ValueError(
                f"nonlinear: init_feature_set {self.params['init_feature_set']!r} names "
                f"column(s) the features do not have: {absent}"
            )
        return [c for c in columns if c != CATEGORICAL]

    def _fit_init(self, X_fit: pd.DataFrame, y_fit: np.ndarray) -> LinearArtifact | None:
        """The linear initial score under ``init_model='linear'``, None under ``none``:
        ``linear``'s elastic net on ``init_feature_set``'s columns of the fit rows, fitted on the
        labels the booster is given (``y_fit``, in the fit's units), with ``linear``'s own
        iteration cap."""
        if self.params["init_model"] == "none":
            return None
        return fit_elastic_net(
            X_fit[self._init_columns(X_fit.columns)],
            y_fit,
            float(self.params["init_alpha_ratio"]),
            float(self.params["init_l1_ratio"]),
            INIT_MAX_ITER,
        )

    def _fit(
        self,
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray, EarlyStop] | None = None,
    ) -> LGBMArtifact | HybridArtifact:
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
        # the linear initial score is fitted first, on its own columns of the fit rows; LightGBM
        # boosts from its prediction on them (init_score) in place of its own constant start.
        # Without one, the Dataset gets no init_score at all
        init = self._fit_init(X_fit, y_fit)
        start = {} if init is None else {"init_score": init.predict(X_fit)}
        if columns != list(X_fit.columns):  # the default fit hands LightGBM the frame itself
            X_fit = X_fit[columns]
        signs = self._monotone_signs(columns)
        if signs is not None:
            params["monotone_constraints"] = signs
        categorical = [CATEGORICAL] if CATEGORICAL in columns else []
        data = lgb.Dataset(X_fit, label=y_fit, categorical_feature=categorical, **start)
        if eval_set is None:
            booster = lgb.train(params, data, num_boost_round=int(self.params["n_estimators"]))
            trees = LGBMArtifact(booster)
            return trees if init is None else HybridArtifact(trees, init)
        X_eval, y_eval, stop = eval_set
        # the evaluation error is the squared one whatever the training loss (under ``l1`` the
        # objective's own default metric would be the absolute error)
        params["metric"] = "l2"
        # the evaluation rows start from the linear part's prediction on them as well, so the
        # error the fit stops on is the whole model's. The net imputes a missing value with the
        # median of the period's rows it is handed, as ``linear`` does: here the evaluation rows
        eval_start = {} if init is None else {"init_score": init.predict(X_eval)}
        booster = lgb.train(
            params,
            data,
            num_boost_round=stop.max_rounds,
            valid_sets=[lgb.Dataset(X_eval[columns], label=y_eval, reference=data, **eval_start)],
            callbacks=[lgb.early_stopping(stop.patience, verbose=False)],
        )
        trees = LGBMArtifact(booster, booster.best_iteration, booster.best_score["valid_0"]["l2"])
        return trees if init is None else HybridArtifact(trees, init)
