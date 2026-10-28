"""Feature importance of a fold's fit on the rows it was validated on.

Two kinds of measure, which answer different questions:

* A fit's own account of itself (``LGBMArtifact.native_importance``: the booster's gain, split
  count and TreeSHAP contribution) says what the trees use; it is read off the fit and says
  nothing about whether using it paid out of sample.
* ``permutation_importance`` shuffles one feature column at a time over the validation rows
  (Breiman 2001) and measures how much worse the fit's predictions get, so it works for any
  fit, and on rows the fit was not trained on (under HPO they also chose its stopping round and
  parameters). ``perm_mse`` is the rise in the mean squared error against the label; ``perm_ic``
  is the fall in the mean per-period information coefficient
  (``ml4trading.diagnostics.period_ic``) against the realized return. Both read the model's own
  raw output, in the units the fit trained on: under ``risk_adjust`` ``vol_return`` or with
  smoothing, the IC that ``perm_ic`` starts from therefore differs from
  ``val_diagnostics.ic_mean``, which is the traded prediction's.

``linear`` fills a missing value with its period's median, recomputed at prediction time over
the rows it is handed: here the scored rows only, not the fit's own.

A feature that is correlated with another can look unimportant, the twin carrying its signal once
it is shuffled; read the numbers as the cost of losing a column's own values, not as a ranking of
causes.

``importance_table`` joins the two for one fit; ``ml4trading.train`` writes it per fold with
``--importance``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.diagnostics import period_ic
from ml4trading.models.learned.base import FitArtifact


def _mean_ic(periods: pd.Index, prediction: np.ndarray, y: np.ndarray) -> float:
    """The mean of the defined per-period ICs (NaN when none is defined)."""
    frame = pd.DataFrame({"period": periods, "prediction": prediction, "y": y})
    return float(period_ic(frame).mean())


def permutation_importance(
    artifact: FitArtifact, X: pd.DataFrame, label: np.ndarray, y: np.ndarray, seed: int = 0
) -> pd.DataFrame:
    """Per feature of ``artifact``, how much its predictions on ``X`` lose when that column's
    values are shuffled across all rows: ``perm_mse`` = MSE(shuffled) - MSE(base) against
    ``label``, ``perm_ic`` = mean period IC(base) - mean period IC(shuffled) against ``y``.
    ``X`` is indexed by (period, symbol) as the models receive it; ``label`` and ``y`` are
    row-aligned with it. One generator seeded with ``seed`` draws every column's shuffle in turn,
    so a call repeats exactly. A positive value means the feature helps."""
    names = artifact.feature_names()
    label, y = np.asarray(label, dtype=float), np.asarray(y, dtype=float)
    periods = X.index.get_level_values("period")
    finite = np.isfinite(label)  # the rows the squared error is taken over

    def mse(prediction: np.ndarray) -> float:
        return float(np.mean((prediction[finite] - label[finite]) ** 2))

    base = np.asarray(artifact.predict(X), dtype=float)
    base_mse, base_ic = mse(base), _mean_ic(periods, base, y)
    draws = np.random.default_rng(seed)
    shuffled = X[names].copy()
    rows = []
    for name in names:
        original = X[name].to_numpy()
        shuffled[name] = original[draws.permutation(len(original))]
        prediction = np.asarray(artifact.predict(shuffled), dtype=float)
        shuffled[name] = original
        rows.append((mse(prediction) - base_mse, base_ic - _mean_ic(periods, prediction, y)))
    return pd.DataFrame(
        rows, columns=["perm_mse", "perm_ic"], index=pd.Index(names, name="feature"), dtype=float
    )


def importance_table(
    artifact: FitArtifact, X: pd.DataFrame, label: np.ndarray, y: np.ndarray
) -> pd.DataFrame:
    """One fit's importance on ``X``'s rows, indexed by ``feature``: the fit's own measures when
    it has them (``native_importance``), then the permutation measures."""
    permuted = permutation_importance(artifact, X, label, y)
    native = getattr(artifact, "native_importance", None)
    if native is None:
        return permuted
    return native(X).join(permuted)
