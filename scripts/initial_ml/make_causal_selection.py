"""The development choice made causally: per fold B, the configuration with the best mean validation
net Sharpe over folds F2020-01 .. B, its test-month predictions spliced into one account.

    python scripts/initial_ml/make_causal_selection.py build [--runs runs]
    for r in causal_all causal_blind; do python -m ml4trading.backtest --run runs/report/$r; done
    python scripts/initial_ml/make_causal_selection.py tables [--runs runs] [--data-dir D]
        [--out results/initial_ml]

Candidates: every development run scored on all 80 folds (``--runs``/dev/, one configuration
each, the reproducibility and identity reruns excluded) and the two per-fold grids
(``--runs``/linear, ``--runs``/nonlinear, scored by their chosen candidate). Every one of fold
B's validation windows ends before B, so fold B's choice reads only the past. Ties go to the
earlier run (run.json's modification time). ``causal_all`` uses every candidate;
``causal_blind`` leaves out the ten configurations scored after the frozen linear model's test
result was known (n01-n04, p1a-p1d, p3a-p3b).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

#: Development runs that are not configurations of their own: reruns of another run.
RERUNS = ("v18b-", "v20b-", "linear", "linear_B", "linear_C", "idB", "idC", "probe-hpo")
#: Configurations scored after the frozen linear model's test result was known.
AFTER_TEST = ("n01-", "n02-", "n03-", "n04-", "p1a-", "p1b-", "p1c-", "p1d-", "p3a-", "p3b-")
SHARED = ("K", "selector", "trader", "folds", "costs", "capital", "reinvest")


def candidates(runs: Path) -> dict[str, Path]:
    """name -> run dir, in the order the runs were made."""
    found = {"linear grid": runs / "linear", "nonlinear grid": runs / "nonlinear"}
    for d in sorted((runs / "dev").iterdir()):
        if not d.is_dir() or d.name in RERUNS or d.name.startswith(RERUNS[:2]):
            continue
        if not (d / "run.json").exists():  # killed before the end: not scored on every fold
            continue
        run = json.loads((d / "run.json").read_text())
        if len(run["folds"]) == 80 and not run.get("grid") and "hpo" not in run:
            found[d.name] = d
    return dict(sorted(found.items(), key=lambda kv: (kv[1] / "run.json").stat().st_mtime))


def val_sharpes(d: Path, folds: list[str]) -> np.ndarray:
    out = []
    for f in folds:
        fj = json.loads((d / "folds" / f / "fold.json").read_text())
        out.append(fj["candidate_val_net_sharpe"][fj["chosen"]])
    return np.asarray(out, dtype=float)


def build(runs: Path) -> None:
    from ml4trading import config

    cands = candidates(runs)
    base = json.loads((runs / "linear" / "run.json").read_text())
    folds = base["folds"]
    missing = []
    for name, d in cands.items():
        run = json.loads((d / "run.json").read_text())
        if run["folds"] != folds:
            raise ValueError(f"{name}: different folds")
        for k in SHARED:
            if run["config"][k] != base["config"][k]:
                raise ValueError(f"{name}: differs in {k!r}")
        if any(not (d / "folds" / f / "predictions.parquet").exists() for f in folds):
            missing.append(name)
    for name in missing:
        cands.pop(name)
    scores = pd.DataFrame({name: val_sharpes(d, folds) for name, d in cands.items()}, index=folds)
    for variant, pool in (
        ("causal_all", list(cands)),
        ("causal_blind", [n for n in cands if not n.startswith(AFTER_TEST)]),
    ):
        out = runs / "report" / variant
        if out.exists():
            shutil.rmtree(out)
        (out / "folds").mkdir(parents=True)
        running = scores[pool].expanding().mean()  # row B: the mean over folds F2020-01 .. B
        choice = []
        for f in folds:
            row = running.loc[f]
            pick = row.index[int(np.argmax(row.to_numpy()))]  # the first best: the earlier run
            os.symlink((cands[pick] / "folds" / f).resolve(), out / "folds" / f)
            choice.append({"fold": f, "chosen": pick, "running_mean_val_sharpe": float(row[pick])})
        run = dict(base)
        run["config"]["model"] = {**config.resolve("linear").model, **run["config"]["model"]}
        run["splice"] = {
            "candidates": pool,
            "left_out_without_saved_predictions": missing,
            "rule": "per fold B, the candidate with the highest mean chosen validation net Sharpe "
            "over folds F2020-01..B; ties to the earlier run",
        }
        (out / "run.json").write_text(json.dumps(run, indent=2, default=str))
        (out / "choice.json").write_text(json.dumps(choice, indent=1))
        print(
            variant,
            len(pool),
            "candidates;",
            pd.Series([c["chosen"] for c in choice]).nunique(),
            "chosen",
        )


def tables(runs: Path, data_dir: Path, out: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))  # make_tables sits beside this file
    import make_tables as mt

    from ml4trading.metrics import sharpe

    mt.configure(runs, data_dir, out)
    tab = Path(out) / "tables"
    tab.mkdir(parents=True, exist_ok=True)
    names = {
        "Causal selection, all candidates": "causal_all",
        "Causal selection, blind (before the linear test)": "causal_blind",
    }
    base = {k: mt.load_run(v) for k, v in mt.NEW.items()}
    sel = {k: mt.load_run(runs / "report" / v) for k, v in names.items()}
    allr = {**base, **sel}
    pd.DataFrame({k: mt.whole_period(r) for k, r in sel.items()}).T.to_csv(
        tab / "R1a_causal_selection_whole_period.csv", float_format="%.6g"
    )
    by = (
        mt.by_year(sel, "net_sharpe")
        .join(mt.by_year(sel, "trade_hit_rate", 100).add_suffix(" hit %"))
        .join(mt.by_year(sel, "n_trades").add_suffix(" trades"))
    )
    by.to_csv(tab / "R1b_causal_selection_by_year.csv", float_format="%.6g")
    # the deployment bar against the best rule, as G2
    rules = ["Daily momentum", "Daily reversal", "Weekly momentum", "Weekly reversal"]
    best = max(rules, key=lambda k: base[k]["metrics"]["net_sharpe"])
    sh, ht = mt.by_year(allr, "net_sharpe"), mt.by_year(allr, "trade_hit_rate", 100)
    voo, qqq = base["Buy-and-hold VOO"]["metrics"], base["Buy-and-hold QQQ"]["metrics"]
    rows = []
    for k, r in sel.items():
        row, passed = {"model": k, "best_rule": best}, 0
        for yr in range(2020, 2026):
            ds, dh = sh.loc[k, yr] - sh.loc[best, yr], ht.loc[k, yr] - ht.loc[best, yr]
            ok = bool(ds >= 0.3 and dh >= 3.0)
            passed += ok
            row[f"{yr}_dSharpe"], row[f"{yr}_dHit_pts"], row[f"{yr}_pass"] = ds, dh, ok
        m = r["metrics"]
        row.update(
            years_passed_of_6=passed,
            maxdd_pct=100 * m["max_drawdown"],
            maxdd_within_voo=bool(m["max_drawdown"] >= voo["max_drawdown"]),
            whole_sharpe=m["net_sharpe"],
            beats_both_buy_and_holds=bool(
                m["net_sharpe"] > max(voo["net_sharpe"], qqq["net_sharpe"])
            ),
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        tab / "R1c_causal_selection_deployment_bar.csv", index=False, float_format="%.6g"
    )
    # months behind VOO
    mon = pd.DataFrame({k: mt.monthly(r) for k, r in allr.items()})
    rows = []
    for k in sel:
        below = mon[k] < mon["Buy-and-hold VOO"]
        row = {"model": k, "months_below_voo": int(below.sum()), "of": int(below.size)}
        for yr in mt.YEARS:
            s = below[below.index.year == yr]
            row[str(yr)] = f"{int(s.sum())}/{s.size}"
        rows.append(row)
    pd.DataFrame(rows).to_csv(tab / "R1d_causal_selection_months_below_voo.csv", index=False)
    # block bootstrap, as G7
    rng = np.random.default_rng(0)
    rows = []
    for a in sel:
        for b in ("Buy-and-hold VOO", "Buy-and-hold QQQ", "Weekly momentum"):
            da = allr[a]["daily"].set_index("session")["ret"]
            db = allr[b]["daily"].set_index("session")["ret"]
            j = pd.concat([da.rename("a"), db.rename("b")], axis=1, join="inner")
            months = j.index.to_period("M")
            blocks = [j[months == p].to_numpy() for p in months.unique()]
            diffs = []
            for _ in range(10000):
                x = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
                diffs.append(sharpe(x[:, 0]) - sharpe(x[:, 1]))
            rows.append(
                {
                    "model": a,
                    "vs": b,
                    "sharpe_diff": sharpe(j["a"]) - sharpe(j["b"]),
                    "ci95_lo": np.quantile(diffs, 0.025),
                    "ci95_hi": np.quantile(diffs, 0.975),
                    "share_boot_diff_gt0_pct": 100 * float(np.mean(np.asarray(diffs) > 0)),
                    "n_days": len(j),
                    "n_months": len(blocks),
                    "reps": 10000,
                }
            )
    pd.DataFrame(rows).to_csv(
        tab / "R1e_causal_selection_bootstrap.csv", index=False, float_format="%.6g"
    )
    # which configurations were chosen, by year
    frames = []
    for k, v in names.items():
        ch = pd.DataFrame(json.loads((runs / "report" / v / "choice.json").read_text()))
        ch["year"] = ch["fold"].str[1:5]
        t = ch.groupby(["chosen", "year"]).size().unstack(fill_value=0)
        t["total"] = t.sum(axis=1)
        t.insert(0, "model", k)
        frames.append(t.reset_index())
    pd.concat(frames).to_csv(tab / "R1f_causal_selection_choices.csv", index=False)
    print("wrote R1a-R1f")


def main() -> None:
    ap = argparse.ArgumentParser(prog="python scripts/initial_ml/make_causal_selection.py")
    ap.add_argument("step", choices=["build", "tables"])
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    from ml4trading.data import default_data_dir

    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--out", type=Path, default=Path("results/initial_ml"))
    args = ap.parse_args()
    if args.step == "build":
        build(args.runs)
    else:
        tables(args.runs, args.data_dir, args.out)


if __name__ == "__main__":
    main()
