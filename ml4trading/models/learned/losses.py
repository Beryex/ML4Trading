"""Training losses and the out-of-sample R² a fold records.

``mse`` is the mean squared residual, ``l1`` the mean absolute residual. A fold's R² compares
its predictions with the constant that minimizes the loss on its fit rows (the mean under mse,
the median under l1): ``1 - loss(predictions) / loss(baseline)`` -- Campbell & Thompson (2008)'s
out-of-sample R² for mse, its absolute-error analogue for l1.
"""

from __future__ import annotations

import numpy as np

LOSSES = ("mse", "l1")


def _known(loss: str) -> None:
    if loss not in LOSSES:
        raise ValueError(f"unknown loss {loss!r}; known: {LOSSES}")


def loss_value(y, pred, loss: str) -> float | None:
    """The loss of ``pred`` against ``y``; None when there is no row or a residual is not
    finite. Arrays of different lengths are refused."""
    _known(loss)
    y, pred = np.asarray(y, dtype=float), np.asarray(pred, dtype=float)
    if y.shape != pred.shape:
        raise ValueError(f"{y.size} labels against {pred.size} predictions")
    residual = y - pred
    if residual.size == 0 or not np.isfinite(residual).all():
        return None
    return float(np.mean(residual**2) if loss == "mse" else np.mean(np.abs(residual)))


def baseline_prediction(y_fit, loss: str) -> float:
    """The constant prediction that minimizes ``loss`` on the fit labels."""
    _known(loss)
    y = np.asarray(y_fit, dtype=float)
    if y.size == 0:
        raise ValueError("baseline_prediction: no fit label")
    return float(np.mean(y) if loss == "mse" else np.median(y))


def oos_r2(y, pred, baseline: float | None, loss: str) -> float | None:
    """``1 - loss(pred) / loss(baseline)`` on ``y``; None when there is no baseline, either loss
    is None, or the baseline's loss is 0."""
    if baseline is None:
        return None
    y = np.asarray(y, dtype=float)
    model, constant = loss_value(y, pred, loss), loss_value(y, np.full(y.shape, baseline), loss)
    if model is None or constant is None or constant == 0.0:
        return None
    return 1.0 - model / constant
