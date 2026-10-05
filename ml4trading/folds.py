"""The walk-forward fold calendar.

A FOLD BOUNDARY B is a month start: the instant a (hyperparameters, universe, model) decision is
deployed. Its fold looks back ``val_months`` of validation and, before that, ``train_months`` of
training; it looks forward ``step_months`` (= 1) of test. Consecutive folds' test months tile
the path exactly, so every test day is scored once, by the fold deployed for it.

    train  [B - val - train, B - val)   fit + select candidate configurations
    val    [B - val, B)                 score candidates, pick the best
    deploy [B - train, B)               re-select the universe and re-fit the chosen config
    test   [B, B + 1 month)             predict; the concatenated test months are THE result

All windows are inclusive date pairs (start, end). A fold boundary is a REBALANCE in the
concatenated book, never a liquidation.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ml4trading.config import FoldConfig

_DAY = pd.Timedelta(days=1)


@dataclass(frozen=True)
class Fold:
    boundary: pd.Timestamp
    train: tuple[pd.Timestamp, pd.Timestamp]
    val: tuple[pd.Timestamp, pd.Timestamp]
    deploy: tuple[pd.Timestamp, pd.Timestamp]
    test: tuple[pd.Timestamp, pd.Timestamp]

    @property
    def name(self) -> str:
        return f"F{self.boundary.year}-{self.boundary.month:02d}"

    def to_dict(self) -> dict:
        def iso(w):
            return [str(w[0].date()), str(w[1].date())]

        return {
            "name": self.name,
            "boundary": str(self.boundary.date()),
            "train": iso(self.train),
            "val": iso(self.val),
            "deploy": iso(self.deploy),
            "test": iso(self.test),
        }


def fold_at(boundary, cfg: FoldConfig) -> Fold:
    b = pd.Timestamp(boundary)
    if b != b.normalize() or b.day != 1:
        raise ValueError(f"a fold boundary must be a month start, got {b}")
    months = pd.DateOffset
    val_start = b - months(months=cfg.val_months)
    train_start = val_start - months(months=cfg.train_months)
    return Fold(
        boundary=b,
        train=(train_start, val_start - _DAY),
        val=(val_start, b - _DAY),
        deploy=(b - months(months=cfg.train_months), b - _DAY),
        test=(b, b + months(months=cfg.step_months) - _DAY),
    )


def fold_boundaries(cfg: FoldConfig) -> list[pd.Timestamp]:
    """Every fold boundary of the test path: month starts from ``path_start`` whose whole test
    month ends on or before ``path_end``."""
    start = pd.Timestamp(cfg.path_start)
    start = pd.Timestamp(year=start.year, month=start.month, day=1)
    end = pd.Timestamp(cfg.path_end)
    out, b = [], start
    while fold_at(b, cfg).test[1] <= end:
        out.append(b)
        b = b + pd.DateOffset(months=cfg.step_months)
    return out


def embargoed_end(window_end: pd.Timestamp, sessions: list[pd.Timestamp], embargo: int):
    """The last session a fit may use when it is scored from the session after ``window_end``:
    the window's last session minus ``embargo`` sessions (labels look one period ahead, so the
    rows just before an evaluation window would otherwise leak it). None if nothing is left."""
    idx = pd.DatetimeIndex(sessions)
    inside = idx[idx <= window_end]
    if len(inside) <= embargo:
        return None
    return inside[len(inside) - 1 - embargo]
