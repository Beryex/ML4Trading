"""E1: a registry of every run of the Initial ML Solution, from the run directories and logs.

    python scripts/initial_ml/make_registry.py [--runs runs] [--identity DIR]
        [--out results/initial_ml/tables]

Reads ``--runs``/{superseded,linear,nonlinear,dev,report} (run.json, folds/*/fold.json,
backtest/metrics.json and the ``<run>.train.log`` beside each run) and, with ``--identity``, the
baseline identity-check runs kept outside ``--runs`` (one subdirectory per code checkpoint). Runs
killed before run.json was written are read from their folds' fold.json.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ml4trading import config

# set by main() from the command line: the directory the registry's paths are relative to
WT: Path = Path()
FOLD_LINE = re.compile(r"^F\d{4}-\d{2}: .* (\d+)s$")


def wall_seconds(log: Path) -> int | None:
    if not log.exists():
        return None
    last = None
    for line in log.read_text(errors="replace").splitlines():
        m = FOLD_LINE.match(line.strip())
        if m:
            last = int(m.group(1))
    return last


def non_defaults(method: str, model: dict) -> dict:
    try:
        base = config.resolve(method).model
    except ValueError:
        return model
    return {k: v for k, v in model.items() if base.get(k) != v and k != "hold_overnight"}


def entry(name: str, d: Path, log: Path | None, group: str, status: str = "") -> dict:
    rj = d / "run.json"
    fold_dirs = sorted((d / "folds").glob("F*/fold.json")) if (d / "folds").exists() else []
    row = {
        "run": name,
        "dir": str(d.relative_to(WT)) if d.is_relative_to(WT) else str(d.name),
        "group": group,
        "folds_completed": len(fold_dirs),
    }
    if rj.exists():
        run = json.loads(rj.read_text())
        method = run["method"]
        model = run["config"]["model"]
        row.update(
            method=method,
            K=run["config"]["K"],
            feature_set=model.get(
                "feature_set", "all" if method in ("linear", "nonlinear") else "n/a"
            ),
            non_default_params=json.dumps(non_defaults(method, model), default=str),
            grid=json.dumps(run.get("grid") or {}),
            n_candidates=int(np.prod([len(v) for v in (run.get("grid") or {}).values()]))
            if run.get("grid")
            else 1,
        )
        if "hpo" in run:
            row["n_candidates"] = f"HPO {run['hpo']}"
    elif fold_dirs:  # killed before run.json was written: each fold.json holds its params
        model = json.loads(fold_dirs[0].read_text())["model_params"]
        method = "nonlinear" if "loss" in model else "linear"
        row.update(
            method=method,
            K=390,
            feature_set=model.get("feature_set"),
            non_default_params=json.dumps(non_defaults(method, model), default=str),
            grid="{}",
            n_candidates=1,
        )
    else:
        row.update(method=None)
    vals = []
    for fj in fold_dirs:
        f = json.loads(fj.read_text())
        vals.append(f["candidate_val_net_sharpe"][f["chosen"]])
    if vals:
        row["val_net_sharpe_fold_mean"] = float(np.mean(vals))
        row["val_net_sharpe_fold_median"] = float(np.median(vals))
    bt = d / "backtest" / "metrics.json"
    if bt.exists():
        row["test_net_sharpe"] = json.loads(bt.read_text())["net_sharpe"]
    row["wall_s_folds"] = wall_seconds(log) if log else None
    row["status"] = status or (
        "finished" if len(fold_dirs) == 80 else f"partial ({len(fold_dirs)}/80)"
    )
    return row


def main() -> None:
    global WT
    ap = argparse.ArgumentParser(prog="python scripts/initial_ml/make_registry.py")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--identity", type=Path, help="baseline identity-check runs, if any")
    ap.add_argument("--out", type=Path, default=Path("results/initial_ml/tables"))
    args = ap.parse_args()
    WT = args.runs.resolve().parent
    runs = args.runs.resolve()
    rows = []
    for name in ("linear", "nonlinear"):
        rows.append(
            entry(
                f"superseded/{name}",
                runs / "superseded" / name,
                runs / "superseded" / f"{name}.train.log",
                "first ML runs, per-fold grid (superseded: before the label-embargo and "
                "convergence fixes; backtested on test)",
                "finished, superseded",
            )
        )
    for name in ("linear", "nonlinear"):
        rows.append(
            entry(
                name,
                runs / name,
                runs / f"{name}.train.log",
                "per-fold grid, default features (after the fixes, code 01ea854)",
            )
        )
    for d in sorted((runs / "dev").iterdir()):
        if not d.is_dir():
            continue
        group = "development (validation only)"
        status = ""
        if d.name.startswith(("idB", "idC")):
            group, status = "identity check (3 folds, code refactor)", "finished (identity check)"
        elif d.name in ("linear", "linear_B", "linear_C"):
            group, status = "identity check (rerun of runs/linear at later code)", ""
        elif d.name.startswith(("n05", "n06", "n07", "n08")):
            group, status = "development (aborted)", None
        elif d.name == "probe-hpo":
            group, status = "HPO timing probe (2 folds)", "finished (probe)"
        elif d.name in ("v20b-lin-slow-vol", "v18b-lin-slowonly-h20"):
            group = "reproducibility rerun (identical to v20 / v18)"
        r = entry(d.name, d, runs / "dev" / f"{d.name}.train.log", group, status or "")
        if status is None:
            r["status"] = (
                f"abandoned at {r['folds_completed']}/80 (stopped: a read-only analysis showed "
                "trees could not reproduce linear's signal)"
            )
        rows.append(r)
    for d in sorted((runs / "report").iterdir()):
        if d.is_dir() and d.name not in ("new", "old"):
            rows.append(entry(d.name, d, runs / "report" / f"{d.name}.train.log", "report run"))
    for u in ("new", "old"):
        for d in sorted((runs / "report" / u).iterdir()):
            if d.is_dir():
                rows.append(
                    entry(
                        f"{u}/{d.name}",
                        d,
                        runs / "report" / u / f"{d.name}.train.log",
                        f"baseline rerun, {u.upper()} universe",
                    )
                )
    for sub in sorted(args.identity.iterdir()) if args.identity else []:
        if sub.is_dir():
            for d in sorted(sub.iterdir()):
                if d.is_dir() and (d / "run.json").exists():
                    r = entry(
                        f"identity/{sub.name}/{d.name}",
                        d,
                        None,
                        "identity check of the four baselines",
                        "finished (identity check)",
                    )
                    r["dir"] = f"(identity check) {sub.name}/{d.name}"
                    rows.append(r)
    df = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "E1_run_registry.csv", index=False, float_format="%.4g")
    print(len(df), "runs")


if __name__ == "__main__":
    main()
