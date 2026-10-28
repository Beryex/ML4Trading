"""Label transforms: what a learned model trains on in place of the raw next-period return.

``raw`` is the return as it is. ``demean`` subtracts each period's mean over the rows given, so
the model learns which names beat the others in a period, not where the market went; ``winsor``
clips the label to a lower and an upper quantile of the fit's labels, so a few extreme returns
cannot steer the fit; ``demean_winsor`` does both, demeaning first.

A transform is fitted on the fold's fit rows -- the clip bounds are quantiles of those labels --
and then applied to other rows by the same rule: the validation scoring reads it to put the
realized returns in the units the fit trained on. Per-period means are taken over the rows
handed to ``apply``, so they are the cross-section of the period itself: in the fit, labels the
fit already trains on, and in validation, labels used only to score, never to predict.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

LABELS = ("raw", "demean", "winsor", "demean_winsor")
#: The lower and upper quantile a winsorized label is clipped to: ML4Investment's
#: ``CLIP_LOWER_QUANTILE_RATIO`` and ``CLIP_UPPER_QUANTILE_RATIO``.
WINSOR_QUANTILES = (0.005, 0.995)


def _known(kind: str) -> None:
    if kind not in LABELS:
        raise ValueError(f"unknown label {kind!r}; known: {LABELS}")


def _demeaned(y: np.ndarray, period: np.ndarray) -> np.ndarray:
    """``y`` less the mean of its period's finite values (NaN where a period has none)."""
    y, period = np.asarray(y, dtype=float), np.asarray(period)
    if y.shape != period.shape:
        raise ValueError(f"{y.size} labels against {period.size} period rows")
    finite = pd.Series(np.where(np.isfinite(y), y, np.nan))
    return y - finite.groupby(period).transform("mean").to_numpy()


@dataclass(frozen=True)
class LabelTransform:
    """One of ``LABELS``; the winsor kinds carry their clip bounds ``lo`` and ``hi``."""

    kind: str
    lo: float | None = None
    hi: float | None = None

    def apply(self, y: np.ndarray, period: np.ndarray) -> np.ndarray:
        """The labels ``y`` of the rows whose periods are ``period``, transformed. NaN stays
        NaN and is left out of every mean; ``raw`` returns ``y`` itself, untouched."""
        _known(self.kind)
        if self.kind == "raw":
            return y
        out = _demeaned(y, period) if self.kind.startswith("demean") else np.asarray(y, float)
        if self.kind.endswith("winsor"):
            if self.lo is None or self.hi is None:
                raise ValueError(f"label {self.kind!r} needs its clip bounds lo and hi")
            out = np.clip(out, self.lo, self.hi)
        return out


def fit_label_transform(kind: str, y: np.ndarray, period: np.ndarray) -> LabelTransform:
    """The ``kind`` transform fitted on the fit labels ``y`` of periods ``period``: the winsor
    bounds are the ``WINSOR_QUANTILES`` of the finite labels, demeaned first for
    ``demean_winsor``."""
    _known(kind)
    if not kind.endswith("winsor"):
        return LabelTransform(kind)
    y = _demeaned(y, period) if kind.startswith("demean") else np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        raise ValueError(f"label {kind!r}: no finite fit label to take the quantiles of")
    lo, hi = np.quantile(y, WINSOR_QUANTILES)
    return LabelTransform(kind, float(lo), float(hi))
