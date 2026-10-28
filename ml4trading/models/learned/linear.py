"""``linear``: an elastic net (squared loss) on the standardized features.

Missing values: each one is replaced by the median of its column over the same period's rows (a
cross-section the model would see at that open), then by 0.0 where the whole period is missing;
the columns are then standardized with the fit rows' mean and standard deviation. ``stock_id``
is a category code, meaningless on a line, and is not used.

Which columns a fit reads is a fit-time choice, so a grid may search it: ``feature_set`` names a
subset of the features (``ml4trading/features/feature_sets/``; ``all`` is every column). A set
that lists ``stock_id`` is fine -- it is left out, as it is under ``all``.

The penalty is set relative to the fit rows: ``alpha = alpha_ratio * alpha_max``, ``alpha_max``
being the smallest penalty at which every coefficient is zero -- scikit-learn's
``ElasticNetCV`` convention for its penalty path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet

from ml4trading.models.learned.base import EarlyStop, FitArtifact, LearnedModel
from ml4trading.models.registry import register_model

CATEGORICAL = "stock_id"


def impute_by_period(X: pd.DataFrame) -> pd.DataFrame:
    """``X`` (indexed by (period, symbol)) with each missing value replaced by its column's
    median over the same period's rows, then 0.0 where none is known."""
    medians = X.groupby(level="period").transform("median")
    return X.fillna(medians).fillna(0.0)


def scaling_of(X: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Each column's (mean, standard deviation with ddof 0); a NaN mean becomes 0 and a zero or
    NaN deviation 1, so a constant column standardizes to 0."""
    mean = X.mean().fillna(0.0)
    std = X.std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return mean, std


def alpha_max(Z: np.ndarray, y: np.ndarray, l1_ratio: float) -> float:
    """The smallest elastic-net penalty that zeroes every coefficient of a fit of ``y`` on the
    centred columns of ``Z``: ``max_j |z_j . (y - mean y)| / (n * l1_ratio)``."""
    return float(np.max(np.abs(Z.T @ (y - y.mean()))) / (len(y) * l1_ratio))


class LinearArtifact(FitArtifact):
    """Coefficients and intercept on the standardized features, with the standardizer."""

    FILENAME = "model.npz"

    def __init__(self, coef, intercept: float, mean: pd.Series, std: pd.Series):
        self.coef, self.intercept = np.array(coef, dtype=float), float(intercept)
        self.mean, self.std = mean, std

    def feature_names(self) -> list[str]:
        return [str(c) for c in self.mean.index]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        values = impute_by_period(X[self.feature_names()].astype(np.float64))
        return ((values - self.mean) / self.std).to_numpy() @ self.coef + self.intercept

    def save(self, directory: Path) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        names = self.feature_names()
        np.savez(
            Path(directory) / self.FILENAME,
            coef=self.coef,
            intercept=self.intercept,
            mean=self.mean[names].to_numpy(),
            std=self.std[names].to_numpy(),
            names=np.asarray(names),
        )

    @classmethod
    def load(cls, directory: Path) -> LinearArtifact:
        with np.load(Path(directory) / cls.FILENAME) as saved:
            names = [str(n) for n in saved["names"]]
            return cls(
                saved["coef"],
                float(saved["intercept"]),
                pd.Series(saved["mean"], index=names),
                pd.Series(saved["std"], index=names),
            )


def fit_elastic_net(
    X: pd.DataFrame, y: np.ndarray, alpha_ratio: float, l1_ratio: float, max_iter: int
) -> LinearArtifact:
    """``linear``'s fit of ``y`` on every column of ``X`` (indexed by (period, symbol)): the
    missing values imputed by period, the columns standardized over these rows, and an elastic
    net at ``alpha_ratio * alpha_max``. A constant ``y`` is fitted by its mean alone.
    ``nonlinear``'s linear initial score is this fit too."""
    values = impute_by_period(X.astype(np.float64))
    mean, std = scaling_of(values)
    Z = ((values - mean) / std).to_numpy()
    largest = alpha_max(Z, y, l1_ratio)
    if not largest > 0.0:  # a constant label: nothing to explain
        return LinearArtifact(np.zeros(Z.shape[1]), float(y.mean()), mean, std)
    net = ElasticNet(
        alpha=alpha_ratio * largest,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        precompute=True,
        selection="cyclic",
        random_state=0,
    )
    net.fit(Z, y)
    return LinearArtifact(net.coef_, float(net.intercept_), mean, std)


@register_model("linear")
class LinearModel(LearnedModel):
    PARAMS = (*LearnedModel.SHARED_PARAMS, "feature_set", "alpha_ratio", "l1_ratio", "max_iter")
    loss = "mse"

    def __init__(self, hold_overnight: bool, **params):
        super().__init__(hold_overnight, **params)
        if not 0.0 < float(params["l1_ratio"]) <= 1.0:
            raise ValueError(f"linear: l1_ratio must lie in (0, 1], got {params['l1_ratio']}")
        if not float(params["alpha_ratio"]) > 0.0:
            raise ValueError(f"linear: alpha_ratio must be positive, got {params['alpha_ratio']}")
        # read here, once: a misspelt set is refused before any data is loaded, and every fold of
        # the run reads the same columns
        self._read_feature_set()
        if self._feature_set is not None and set(self._feature_set) <= {CATEGORICAL}:
            raise ValueError(
                f"linear: feature set {params['feature_set']!r} has no column but {CATEGORICAL!r}, "
                f"which linear does not use"
            )

    def _fit_columns(self, available: pd.Index) -> list[str]:
        """The columns a fit trains on, in order: the feature set's (every column for ``all``),
        less ``stock_id``. A set column the features lack is an error."""
        return [c for c in self._feature_set_columns(available) if c != CATEGORICAL]

    def _fit(
        self,
        X_fit: pd.DataFrame,
        y_fit: np.ndarray,
        eval_set: tuple[pd.DataFrame, np.ndarray, EarlyStop] | None = None,
    ) -> LinearArtifact:
        if eval_set is not None:
            raise ValueError(
                "linear: early stopping needs a boosted model; an elastic net has none"
            )
        return fit_elastic_net(
            X_fit[self._fit_columns(X_fit.columns)],
            y_fit,
            float(self.params["alpha_ratio"]),
            float(self.params["l1_ratio"]),
            int(self.params["max_iter"]),
        )
