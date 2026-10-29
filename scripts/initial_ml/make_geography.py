"""Non-US vs US companies: their share of the pool and the fold universes, and each method's net
P&L on them.

    python scripts/initial_ml/make_geography.py --countries RANKING.csv [--runs runs]
        [--data-dir D] [--out results/initial_ml]

``--countries`` is a companiesmarketcap.com ranking download (columns ``Symbol`` and
``country``, the country of the company's headquarters). A symbol is matched as it is, then with
``.`` written as ``-``; an unmatched symbol is ``unknown``. "Non-US" is a company whose country is
not the United States, whatever its listing: an ADR, a foreign company's own US listing, or a
company with US operations domiciled abroad (e.g. Ireland). VOO and QQQ are ``ETF``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ml4trading.data import default_data_dir, load_universe

COST = [
    "entry_spread",
    "entry_commission",
    "exit_spread",
    "exit_commission",
    "close_spread",
    "close_commission",
    "borrow",
]
METHODS = {
    "Validation-selected: v29 vs P3b": "report/sel_final",
    "linear v29 (frozen)": "report/v29",
    "linear, all 188 features (per-fold grid)": "report/linear_grid",
    "Weekly momentum": "report/new/weekly_mom",
}


def countries(path: Path, symbols: list[str]) -> dict[str, str]:
    cmc = pd.read_csv(path).drop_duplicates("Symbol").set_index("Symbol")["country"]
    out = {}
    for s in symbols:
        c = cmc.get(s, cmc.get(s.replace(".", "-")))
        out[s] = (
            "unknown" if not isinstance(c, str) else ("US" if c == "United States" else "non-US")
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python scripts/initial_ml/make_geography.py")
    ap.add_argument("--countries", type=Path, required=True)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--out", type=Path, default=Path("results/initial_ml"))
    args = ap.parse_args()
    u = load_universe(args.data_dir)
    pool = list(u["symbols"])
    geo = {**countries(args.countries, pool), **dict.fromkeys(u["etfs"], "ETF")}
    tab = args.out / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    pd.Series({s: geo[s] for s in pool}, name="group").rename_axis("symbol").to_csv(
        tab / "R2a_company_country_group.csv"
    )
    rows = [{"set": "pool (550 stocks)", **pd.Series([geo[s] for s in pool]).value_counts()}]
    for name, rel in METHODS.items():
        d = args.runs / rel
        run = json.loads((d / "run.json").read_text())
        per_fold = [
            pd.Series(
                [
                    geo.get(s, "unknown")
                    for s in json.loads((d / "folds" / f / "universe.json").read_text())
                ]
            ).value_counts()
            for f in run["folds"]
        ]
        mean = pd.concat(per_fold, axis=1).fillna(0).mean(axis=1)
        rows.append({"set": f"fold universe of {name} (mean per fold)", **mean})
        pos = pd.read_csv(d / "backtest" / "positions.csv")
        pos["net"] = pos["gross"] - pos[COST].sum(axis=1)
        pos["group"] = pos["symbol"].map(lambda s: geo.get(s, "unknown"))
        held = pos[pos["shares"] != 0]
        for g in ("US", "non-US", "ETF", "unknown"):
            rows.append(
                {
                    "set": f"{name}: {g}",
                    "net_pnl": pos.loc[pos["group"] == g, "net"].sum(),
                    "share_of_name_periods_held_pct": 100 * float((held["group"] == g).mean()),
                }
            )
    pd.DataFrame(rows).to_csv(tab / "R2b_us_vs_non_us.csv", index=False, float_format="%.6g")
    print(pd.DataFrame(rows).round(2).to_string())


if __name__ == "__main__":
    main()
