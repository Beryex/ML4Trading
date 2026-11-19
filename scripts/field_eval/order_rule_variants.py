"""Order-rule variants of the frozen v29 / P3b runs, re-backtested from their saved test
predictions (no retraining).

    python scripts/field_eval/order_rule_variants.py

Writes runs/order_rule_variants/<model>_<variant>/ and
results/field_eval/order_rule_variants/summary.csv.

Variants (the TA's questions on the Initial ML Solution): the long/short P&L split of the reported
runs; long-only (BUY_LONG) and short-only books; a cost check with a margin (cost_safety_factor
2 and 3); long-only with a margin of 2. Each variant is a new run directory holding run.json (the
trader config edited) and every fold's fold.json and predictions.parquet (copied); runs/report is
never written. Data: data/ (the report's). The task metric is added: final account dollars minus
VOO buy-and-hold dollars, same days, same $20,000.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from ml4trading.backtest import backtest

R = Path(__file__).resolve().parents[2]
OUT = R / "runs" / "order_rule_variants"
SUM = R / "results" / "field_eval" / "order_rule_variants"
SUM.mkdir(parents=True, exist_ok=True)

VARIANTS = {
    "base": {},
    "long_only": {"symbol_selection_strategy": "BUY_LONG"},
    "short_only": {"symbol_selection_strategy": "SELL_SHORT"},
    "gate_x2": {"cost_safety_factor": 2.0},
    "gate_x3": {"cost_safety_factor": 3.0},
    "long_only_gate_x2": {"symbol_selection_strategy": "BUY_LONG", "cost_safety_factor": 2.0},
}


def make_variant(src: Path, dst: Path, trader_over: dict) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    run = json.loads((src / "run.json").read_text())
    run["config"]["trader"].update(trader_over)
    run["source_run"] = str(src.relative_to(R))
    (dst / "folds").mkdir(parents=True)
    for f in run["folds"]:
        (dst / "folds" / f).mkdir()
        for name in ("fold.json", "predictions.parquet"):
            shutil.copy2(src / "folds" / f / name, dst / "folds" / f / name)
    (dst / "run.json").write_text(json.dumps(run, indent=2) + "\n")


def side_split(pos_csv: Path) -> dict:
    p = pd.read_csv(pos_csv)
    cost = p[[c for c in p.columns if c.endswith(("_spread", "_commission")) or c == "borrow"]].sum(
        axis=1
    )
    held = np.where(
        p["shares"] != 0, p["shares"], p["prev_shares"]
    )  # the side the row's P&L belongs to
    side = np.where(held > 0, "long", np.where(held < 0, "short", "flat"))
    out = {}
    for s in ("long", "short"):
        m = side == s
        out[f"{s}_gross"] = float(p.loc[m, "gross"].sum())
        out[f"{s}_costs"] = float(cost[m].sum())
        out[f"{s}_net"] = out[f"{s}_gross"] - out[f"{s}_costs"]
        out[f"{s}_name_days"] = int(((p["shares"] != 0) & m).sum())
    return out


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    data = R / "data"
    voo_end = json.loads((R / "runs/report/new/voo/backtest/metrics.json").read_text())[
        "end_equity"
    ]
    rows = []
    sources = {"v29": R / "runs/report/v29", "P3b": R / "runs/report/p3b"}
    for model, src in sources.items():
        for name, over in VARIANTS.items():
            dst = OUT / f"{model}_{name}"
            make_variant(src, dst, over)
            m = backtest(dst, data)
            row = {
                "model": model,
                "variant": name,
                **{
                    k: m.get(k)
                    for k in (
                        "net_sharpe",
                        "gross_sharpe",
                        "net_profit",
                        "end_equity",
                        "annualized_return",
                        "max_drawdown",
                        "alpha",
                        "alpha_t",
                        "beta",
                        "trade_hit_rate",
                        "n_trades",
                    )
                },
            }
            row["costs"] = m["total_spread_cost"] + m["total_commission"] + m["total_borrow"]
            row["excess_dollars_vs_voo"] = m["end_equity"] - voo_end
            row.update(side_split(dst / "backtest" / "positions.csv"))
            rows.append(row)
            print(
                model,
                name,
                {
                    k: (round(v, 3) if isinstance(v, float) else v)
                    for k, v in row.items()
                    if k
                    in (
                        "net_sharpe",
                        "net_profit",
                        "excess_dollars_vs_voo",
                        "max_drawdown",
                        "long_net",
                        "short_net",
                    )
                },
                flush=True,
            )
    s = pd.DataFrame(rows)
    s.to_csv(SUM / "summary.csv", index=False)
    print("VOO end equity", voo_end)


if __name__ == "__main__":
    main()
