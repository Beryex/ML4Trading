"""Replay a trained run: ONE continuous book over the concatenated test months.

    python -m ml4trading.backtest --run runs/investment

Every fold's predictions cover exactly its own test month, so concatenating them gives one
prediction per (period, symbol) along the whole path; the book trades them under the run's
configuration, starting flat at the capital basis. A fold boundary is a rebalance, not a
liquidation.

Writes ``<run>/backtest/``: metrics.json (the whole path), by_year.csv (each calendar year,
``full_year`` marking the years the path covers entirely), daily.csv (the account valued every
trading day, which every return metric reads), book.csv (one row per decision period) and
positions.csv (one row per symbol and period, with its P&L and costs).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml4trading.config import RunConfig
from ml4trading.data import default_data_dir
from ml4trading.metrics import summarize_by_year
from ml4trading.pipeline import (
    benchmark_returns,
    load_daily_inputs,
    load_panel,
    score_book,
    window_periods,
)


def backtest(run_dir: Path, data_dir: Path) -> dict:
    run_dir = Path(run_dir)
    run = json.loads((run_dir / "run.json").read_text())
    cfg = RunConfig.from_dict(run["config"])
    folds = [json.loads((run_dir / "folds" / f / "fold.json").read_text()) for f in run["folds"]]
    preds = pd.concat(
        [pd.read_parquet(run_dir / "folds" / f / "predictions.parquet") for f in run["folds"]],
        ignore_index=True,
    )
    pool, panel = load_panel(cfg, data_dir)
    path = (folds[0]["test"][0], folds[-1]["test"][1])
    periods = window_periods(panel, path)
    benchmark = benchmark_returns(data_dir)
    daily_inputs = load_daily_inputs(cfg, data_dir, pool)
    book, positions, daily, metrics = score_book(
        preds, panel, periods, cfg, benchmark, daily_inputs
    )

    out = run_dir / "backtest"
    out.mkdir(exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    by_year = summarize_by_year(book, cfg.K, positions, benchmark, daily)
    by_year.to_csv(out / "by_year.csv", index=False)
    daily.to_csv(out / "daily.csv", index=False)
    book.to_csv(out / "book.csv", index=False)
    positions.to_csv(out / "positions.csv", index=False)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ml4trading.backtest")
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = ap.parse_args()
    metrics = backtest(args.run, args.data_dir)
    for k, v in metrics.items():
        print(f"{k:>24}: {v:.6g}" if isinstance(v, float) else f"{k:>24}: {v}")


if __name__ == "__main__":
    main()
