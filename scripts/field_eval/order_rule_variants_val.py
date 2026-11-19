"""Validation scores of the order-rule variants (the development metric: fold-mean validation
net Sharpe over the 80 folds), on each frozen run's saved validation predictions.

    python scripts/field_eval/order_rule_variants_val.py v29|P3b
        # results/field_eval/order_rule_variants/val_<model>.csv

Per fold: the fold's saved validation predictions (folds/<F>/val/predictions.parquet, the chosen
candidate's, universe selected on the train window), traded from flat over the validation window
minus its first 5 sessions (the embargo), exactly as train.py scores candidates; only the trader
settings change. 'base' must reproduce fold.json's candidate_val_net_sharpe.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ml4trading.config import RunConfig
from ml4trading.pipeline import benchmark_returns, load_panel, score_book, window_periods

R = Path(__file__).resolve().parents[2]
VARIANTS = {
    "base": {},
    "long_only": {"symbol_selection_strategy": "BUY_LONG"},
    "short_only": {"symbol_selection_strategy": "SELL_SHORT"},
    "gate_x2": {"cost_safety_factor": 2.0},
    "gate_x3": {"cost_safety_factor": 3.0},
    "long_only_gate_x2": {"symbol_selection_strategy": "BUY_LONG", "cost_safety_factor": 2.0},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("model", choices=["v29", "P3b"])
    model = ap.parse_args().model
    src = R / "runs" / "report" / {"v29": "v29", "P3b": "p3b"}[model]
    run = json.loads((src / "run.json").read_text())
    base = RunConfig.from_dict(run["config"])
    data = R / "data"
    _, panel = load_panel(base, data)
    bench = benchmark_returns(data)
    rows = []
    for f in run["folds"]:
        fj = json.loads((src / "folds" / f / "fold.json").read_text())
        preds = pd.read_parquet(src / "folds" / f / "val" / "predictions.parquet")[
            ["period", "symbol", "prediction"]
        ]
        periods = window_periods(panel, fj["val"], skip_sessions=base.folds.embargo_days)
        row = {"fold": f, "recorded_base": fj["candidate_val_net_sharpe"][fj["chosen"]]}
        for name, over in VARIANTS.items():
            cfg = replace(base, trader=replace(base.trader, **over))
            _, _, _, m = score_book(preds, panel, periods, cfg, benchmark=bench)
            row[name] = float(m.get("net_sharpe", 0.0))
        rows.append(row)
        print(f, {k: round(v, 3) for k, v in row.items() if k != "fold"}, flush=True)
    out = pd.DataFrame(rows)
    dst = R / "results" / "field_eval" / "order_rule_variants"
    dst.mkdir(parents=True, exist_ok=True)
    out.to_csv(dst / f"val_{model}.csv", index=False)
    print("FOLD-MEAN", model, out.drop(columns="fold").mean().round(3).to_dict(), flush=True)


if __name__ == "__main__":
    main()
