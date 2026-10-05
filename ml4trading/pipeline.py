"""What training and backtesting share: loading a method's panel, scoring a window's book."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml4trading.book import daily_values, run_book
from ml4trading.config import BENCHMARK, SESSION_MINUTES, RunConfig
from ml4trading.data import all_symbols, load_bars, load_dividends
from ml4trading.metrics import summarize
from ml4trading.models import get_model
from ml4trading.periods import build_panel


def load_panel(cfg: RunConfig, data_dir: Path) -> tuple[list[str], pd.DataFrame]:
    """(the method's symbol pool, its labelled panel over the whole dataset)."""
    model = get_model(cfg.method)(**cfg.model)
    pool = model.symbol_pool(all_symbols(data_dir))
    bars = load_bars(data_dir, pool)
    divs = load_dividends(data_dir, pool)
    return pool, build_panel(bars, cfg.K, cfg.hold_overnight, divs)


def window_periods(panel: pd.DataFrame, window, skip_sessions: int = 0) -> list[pd.Timestamp]:
    """The book's periods in the inclusive ``window``: those where at least one symbol can be
    traded (a finite exit price -- overnight, the data's last session cannot), after dropping
    the window's first ``skip_sessions`` sessions (the evaluation embargo)."""
    lo, hi = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    inside = panel[(panel["session"] >= lo) & (panel["session"] <= hi)]
    inside = inside[np.isfinite(inside["exit_px"].to_numpy(dtype=float))]
    sessions = sorted(inside["session"].unique())
    if len(sessions) <= skip_sessions:
        return []
    first = sessions[skip_sessions]
    return sorted(inside.loc[inside["session"] >= first, "period"].unique())


def load_daily_inputs(cfg: RunConfig, data_dir: Path, symbols: list[str]):
    """What ``ml4trading.book.daily_values`` needs to value a multi-session book every day: the
    K = 390 panel of ``symbols`` and their dividends; None for books of a session or less."""
    if cfg.K <= SESSION_MINUTES:
        return None
    divs = load_dividends(data_dir, symbols)
    return build_panel(load_bars(data_dir, symbols), SESSION_MINUTES, True, divs), divs


def benchmark_returns(data_dir: Path) -> pd.Series:
    """The benchmark's daily buy-and-hold return (open to next open, dividends included),
    keyed by session -- what every book's daily returns are regressed on."""
    bars = load_bars(data_dir, [BENCHMARK])
    panel = build_panel(bars, SESSION_MINUTES, True, load_dividends(data_dir, [BENCHMARK]))
    return panel.set_index("session")["y"]


def score_book(
    predictions: pd.DataFrame,
    panel: pd.DataFrame,
    periods,
    cfg: RunConfig,
    benchmark: pd.Series | None = None,
    daily_inputs=None,
):
    """(book, positions, daily values, metrics) of ``predictions`` traded over ``periods``;
    ``daily_inputs`` (``load_daily_inputs``) is required for a multi-session K."""
    book, positions = run_book(predictions, panel, periods, cfg)
    daily = daily_values(book, positions, cfg.K, *(daily_inputs or (None, None)))
    return book, positions, daily, summarize(book, cfg.K, positions, benchmark, daily)
