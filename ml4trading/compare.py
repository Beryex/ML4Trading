"""Compare two backtested runs' Sharpe ratios.

    python -m ml4trading.compare --run runs/linear --run runs/investment_tech
    python -m ml4trading.compare --run runs/linear --run runs/investment_tech \
        --trials-sharpes trials.txt

Reads each run's ``backtest/daily.csv`` (the net daily return ``ret``) and prints the paired
test of equal Sharpe ratios of the first run against the second over the sessions they share
(``ml4trading.significance.sharpe_difference_test``). With a trials file -- the ANNUALIZED
Sharpe ratio of every strategy that was tried before the first run was picked (itself
included), one per line -- it also prints the first run's deflated Sharpe ratio: its Sharpe,
skewness and kurtosis come from its own full daily series. Nothing is written to either run.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ml4trading.config import TRADING_DAYS_PER_YEAR
from ml4trading.significance import deflated_sharpe, return_moments, sharpe_difference_test


def load_returns(run_dir: Path) -> pd.Series:
    """The run's net daily returns, indexed by session (``backtest/daily.csv``'s ``ret``)."""
    path = Path(run_dir) / "backtest" / "daily.csv"
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run python -m ml4trading.backtest first")
    return pd.read_csv(path, parse_dates=["session"]).set_index("session")["ret"]


def read_trials(path: Path) -> list[float]:
    """The annualized Sharpe ratios in a trials file: whitespace-separated numbers, one per line."""
    values = [float(token) for token in Path(path).read_text().split()]
    if not values:
        raise ValueError(f"{path} holds no Sharpe ratio")
    return values


def compare(run_a: Path, run_b: Path, trials_file: Path | None = None) -> dict:
    """``{"paired": sharpe_difference_test(A, B), "deflated": A's deflated Sharpe or None}``.

    The deflated block is annualized for reading (``sharpe``, ``expected_max_sharpe``); the
    statistic itself works per period, as the trials' standard deviation is taken per period."""
    returns_a, returns_b = load_returns(run_a), load_returns(run_b)
    out = {"paired": sharpe_difference_test(returns_a, returns_b), "deflated": None}
    if trials_file is not None:
        root = math.sqrt(TRADING_DAYS_PER_YEAR)
        trials = np.asarray(read_trials(trials_file)) / root  # per period
        sr, skew, kurt = return_moments(returns_a)
        n_trials = len(trials)
        sr_std = float(trials.std(ddof=1)) if n_trials > 1 else 0.0
        deflated = deflated_sharpe(sr, n_trials, sr_std, len(returns_a), skew, kurt)
        out["deflated"] = {
            "n_trials": n_trials,
            "n_obs": len(returns_a),
            "sharpe": sr * root,
            "skew": skew,
            "kurtosis": kurt,
            "expected_max_sharpe": deflated["expected_max_sr"] * root,
            "dsr": deflated["dsr"],
        }
    return out


def _print_block(title: str, values: dict) -> None:
    print(title)
    for k, v in values.items():
        print(f"{k:>24}: {v:.6g}" if isinstance(v, float) else f"{k:>24}: {v}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m ml4trading.compare")
    ap.add_argument("--run", type=Path, action="append", default=[], help="give it twice: A, B")
    ap.add_argument(
        "--trials-sharpes", type=Path, help="annualized Sharpe of every trial, per line"
    )
    args = ap.parse_args(argv)
    if len(args.run) != 2:
        ap.error("--run takes exactly two runs: A, then B")
    try:
        result = compare(args.run[0], args.run[1], args.trials_sharpes)
    except (FileNotFoundError, ValueError) as e:
        ap.error(str(e))
    print(f"1 = A = {args.run[0]}\n2 = B = {args.run[1]}")
    _print_block(
        "paired Sharpe test, A vs B (Jobson-Korkie, Memmel's correction)", result["paired"]
    )
    if result["deflated"] is not None:
        _print_block("deflated Sharpe ratio of A (Bailey-Lopez de Prado)", result["deflated"])


if __name__ == "__main__":
    main()
