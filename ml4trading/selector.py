"""Universe selection: which symbols a fold may trade.

``liquidity`` (the only selector): over the selection window, a symbol is a candidate iff
  (1) it has a realized label (finite ``y``) on at least max(min_coverage_sessions,
      n_sessions // 2) sessions, n_sessions = the distinct sessions present in the window, and
  (2) the median of close x volume over its periods in the window is >= min_dollar_volume;
the candidates are then ranked by that median dollar volume (descending, ties by symbol) and the
first ``selection_cap`` kept. The selection reads volume and labels only -- never costs or
predictions -- and is computed once per fold, then frozen for the fold's test month.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml4trading.config import SelectorConfig


def _window_rows(panel: pd.DataFrame, window) -> pd.DataFrame:
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    return panel[(panel["session"] >= lo) & (panel["session"] <= hi)]


def select_universe(
    panel: pd.DataFrame, window, cfg: SelectorConfig, pool: list[str] | None = None
) -> list[str]:
    """The fold's universe (module docstring), sorted alphabetically. ``pool`` restricts the
    candidates (a model's own symbol pool); None = every symbol in ``panel``."""
    if cfg.name != "liquidity":
        raise ValueError(f"unknown selector {cfg.name!r}")
    rows = _window_rows(panel, window)
    if pool is not None:
        rows = rows[rows["symbol"].isin(set(pool))]
    if rows.empty:
        return []
    n_sessions = _window_rows(panel, window)["session"].nunique()
    min_obs = max(cfg.min_coverage_sessions, n_sessions // 2)
    labelled = rows[np.isfinite(rows["y"].to_numpy(dtype=float))]
    coverage = labelled.groupby("symbol", observed=True)["session"].nunique()
    covered = set(coverage[coverage >= min_obs].index)
    dollar_volume = (rows["close"] * rows["volume"]).groupby(rows["symbol"], observed=True)
    medians = dollar_volume.median()
    liquid = medians[(medians >= cfg.min_dollar_volume) & medians.index.isin(covered)]
    ranked = sorted(liquid.items(), key=lambda kv: (-kv[1], kv[0]))[: cfg.selection_cap]
    return sorted(sym for sym, _ in ranked)
