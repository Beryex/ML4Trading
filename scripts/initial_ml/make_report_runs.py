"""Build the report's backtest-only run directories under ``--runs``/report/ (no retraining).

    python scripts/initial_ml/make_report_runs.py [--runs runs] [--linear-grid runs/linear]
        [--nonlinear-grid runs/nonlinear] [--v29 runs/v29] [--p3b runs/p3b]
    for r in linear_grid nonlinear_grid v29 p3b sel_grid sel_final; do
        python -m ml4trading.backtest --run runs/report/$r
    done

* linear_grid, nonlinear_grid, v29, p3b: the trained runs' fold directories, symlinked, so the
  backtest is replayed at this code without touching the originals' backtest/ outputs. A run
  trained before the learned models' later keys gets those keys at their defaults in run.json
  (the backtest reads the model config only for the symbol pool).
* sel_grid, sel_final: the validation-selected splice. Per fold, the family whose chosen
  candidate has the higher validation net Sharpe in its own fold.json; ties go to the first
  family. That family's fold directory (its test-month predictions) is linked; the spliced
  predictions are then traded as one continuous account. choice.json records every fold.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from ml4trading import config

#: What a splice must share for its two families' predictions to be traded as one account.
SHARED = ("K", "selector", "trader", "folds", "costs", "capital", "reinvest")


def _fresh(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    (out / "folds").mkdir(parents=True)


def _filled(run: dict) -> dict:
    run["config"]["model"] = {**config.resolve(run["method"]).model, **run["config"]["model"]}
    return run


def relink(out: Path, src: Path) -> None:
    _fresh(out)
    run = _filled(json.loads((src / "run.json").read_text()))
    run["source_run"] = str(src)
    for f in run["folds"]:
        os.symlink((src / "folds" / f).resolve(), out / "folds" / f)
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))


def splice(out: Path, a: Path, b: Path, la: str, lb: str) -> None:
    _fresh(out)
    ra, rb = (json.loads((d / "run.json").read_text()) for d in (a, b))
    if ra["folds"] != rb["folds"]:
        raise ValueError(f"{a} and {b} cover different folds")
    for k in SHARED:
        if ra["config"][k] != rb["config"][k]:
            raise ValueError(f"{a} and {b} differ in {k!r}")
    choice = []
    for f in ra["folds"]:
        fa, fb = (json.loads((d / "folds" / f / "fold.json").read_text()) for d in (a, b))
        sa = fa["candidate_val_net_sharpe"][fa["chosen"]]
        sb = fb["candidate_val_net_sharpe"][fb["chosen"]]
        pick = a if sa >= sb else b
        os.symlink((pick / "folds" / f).resolve(), out / "folds" / f)
        choice.append(
            {
                "fold": f,
                f"val_sharpe_{la.replace(' ', '_')}": sa,
                f"val_sharpe_{lb.replace(' ', '_')}": sb,
                "chosen": la if pick == a else lb,
            }
        )
    run = _filled(dict(ra))
    run["splice"] = {
        "families": {la: str(a), lb: str(b)},
        "rule": "per fold, the family whose chosen candidate has the higher validation net "
        "Sharpe (fold.json); ties to the first",
    }
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str))
    (out / "choice.json").write_text(json.dumps(choice, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(prog="python scripts/initial_ml/make_report_runs.py")
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--linear-grid", type=Path, default=Path("runs/linear"))
    ap.add_argument("--nonlinear-grid", type=Path, default=Path("runs/nonlinear"))
    ap.add_argument("--v29", type=Path, default=Path("runs/v29"))
    ap.add_argument("--p3b", type=Path, default=Path("runs/p3b"))
    args = ap.parse_args()
    out = args.runs / "report"
    relink(out / "linear_grid", args.linear_grid)
    relink(out / "nonlinear_grid", args.nonlinear_grid)
    relink(out / "v29", args.v29)
    relink(out / "p3b", args.p3b)
    splice(out / "sel_grid", args.linear_grid, args.nonlinear_grid, "linear grid", "nonlinear grid")
    splice(out / "sel_final", args.v29, args.p3b, "linear v29", "nonlinear P3b")


if __name__ == "__main__":
    main()
