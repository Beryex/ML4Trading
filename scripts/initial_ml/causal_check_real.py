"""``ml4trading.testing.assert_causal`` on the real data, with the reported models' exact configs.

    python scripts/initial_ml/causal_check_real.py [--runs runs] [--data-dir D]

A 30-stock pool (the first 28 names of universe.json plus CHTR and DD, two of the 50 drop-outs)
with VOO and QQQ for the market features; bars 2016-01-04 .. 2020-02-28. Fit 2018-01-02 ..
2019-12-20, predict January 2020; n_checks = 3 predicted periods, each with every bar from its
start onward replaced by noise, the frames rebuilt, refitted and predicted again: the prediction
for that period must not move. v29's and P3b's configurations are read from their trained runs'
run.json (``--runs``/report/v29, ``--runs``/report/p3b).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from ml4trading import config
from ml4trading.data import default_data_dir, load_bars, load_dividends, load_universe
from ml4trading.models import get_model
from ml4trading.testing import assert_causal

FIT = (pd.Timestamp("2018-01-02"), pd.Timestamp("2019-12-20"))
PREDICT = (pd.Timestamp("2020-01-02"), pd.Timestamp("2020-01-31"))
BARS_END = "2020-02-29"


def model_of(method: str, params: dict):
    overrides = {f"model.{k}": v for k, v in params.items() if k != "hold_overnight"}
    return get_model(method)(**config.resolve(method, overrides).model)


def main() -> None:
    ap = argparse.ArgumentParser(prog="python scripts/initial_ml/causal_check_real.py")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    args = ap.parse_args()
    symbols = [*load_universe(args.data_dir)["symbols"][:28], "CHTR", "DD", "VOO", "QQQ"]
    saved = {
        name: json.loads((args.runs / "report" / name / "run.json").read_text())["config"]["model"]
        for name in ("v29", "p3b")
    }
    three = {"feature_set": "three_feature"}
    cases = {
        "v29 (linear, frozen)": ("linear", saved["v29"]),
        "P3b (nonlinear hybrid, frozen)": ("nonlinear", saved["p3b"]),
        "3-feature EN grid, most-chosen point": (
            "linear",
            {**three, "alpha_ratio": 0.1, "l1_ratio": 0.1},
        ),
        "3-feature OLS (alpha_ratio 1e-6)": ("linear", {**three, "alpha_ratio": 1e-6}),
        "linear grid point (all features)": ("linear", {"alpha_ratio": 0.001, "l1_ratio": 0.5}),
        "nonlinear grid point (all features)": (
            "nonlinear",
            {"num_leaves": 63, "min_data_in_leaf": 200},
        ),
    }
    for name, (method, params) in cases.items():
        model = model_of(method, params)
        pool = model.symbol_pool(symbols)
        loaded = model.loaded_symbols(pool)
        bars = load_bars(args.data_dir, loaded, end=BARS_END)
        divs = load_dividends(args.data_dir, loaded)
        t0 = time.time()
        assert_causal(model, bars, divs, 390, pool, FIT, PREDICT, n_checks=3, seed=0)
        print(f"PASS {name}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
