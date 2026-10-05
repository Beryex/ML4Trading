"""Walk-forward training: one model per fold boundary of the test path.

    python -m ml4trading.train --method investment --out runs/investment [--grid grid.yaml]

Per fold boundary B (calendar: ``ml4trading.folds``):

1. VALIDATION -- for every candidate configuration of the grid: select the universe on the
   train window, fit on the train window (its last ``embargo_days`` sessions dropped), predict
   the validation window, and trade it (validation window minus its first ``embargo_days``
   sessions) from flat; the candidate's score is that book's net Sharpe. The first best wins.
   With no grid there is one candidate, scored all the same.
2. DEPLOYMENT -- with the chosen configuration: select the universe on the deploy window (the
   trailing ``train_months`` before B), fit on it (embargoed the same way), and predict the
   test month.
3. The fold is written to ``<out>/folds/<FYYYY-MM>/``: fold.json (windows, candidate scores,
   the choice), universe.json, predictions.parquet and the fitted model.

The grid file maps dotted ``model.*`` keys to lists of values; only model parameters are
searched per fold -- the selector, the trader, the costs and K are fixed for the whole run.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ml4trading import config as config_mod
from ml4trading.config import RunConfig
from ml4trading.data import default_data_dir
from ml4trading.folds import embargoed_end, fold_at, fold_boundaries
from ml4trading.models import get_model
from ml4trading.pipeline import load_panel, score_book, window_periods
from ml4trading.selector import select_universe


def _fit_predict(cfg: RunConfig, panel, pool, select_window, fit_window_end, predict_window):
    """(universe, fitted model, predictions) for one configuration."""
    universe = select_universe(panel, select_window, cfg.selector, pool)
    sub = panel[panel["symbol"].isin(set(universe))]
    model = get_model(cfg.method)(**cfg.model)
    if universe and fit_window_end is not None:
        model.fit(sub, (pd.Timestamp(select_window[0]), fit_window_end))
    preds = model.predict(sub, predict_window) if universe else _no_predictions()
    return universe, model, preds


def _no_predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {"period": pd.Series(dtype="datetime64[ns]"), "symbol": [], "prediction": []}
    )


def train(
    method: str,
    out: Path,
    data_dir: Path,
    grid: dict | None = None,
    overrides: dict | None = None,
) -> Path:
    base = config_mod.resolve(method, overrides)
    for key in grid or {}:
        if not key.startswith("model."):
            raise ValueError(f"grid key {key!r}: only model.* parameters are searched per fold")
    candidates = config_mod.expand_grid(base, grid)
    pool, panel = load_panel(base, data_dir)
    sessions = sorted(panel["session"].unique())
    embargo = base.folds.embargo_days

    out = Path(out)
    (out / "folds").mkdir(parents=True, exist_ok=True)
    manifest = Path(data_dir) / "MANIFEST.json"
    run = {
        "method": method,
        "config": base.to_dict(),
        "grid": grid or {},
        "data_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        "folds": [],
    }
    t0 = time.time()
    for b in fold_boundaries(base.folds):
        fold = fold_at(b, base.folds)
        scores = []
        for cand in candidates:
            fit_end = embargoed_end(fold.train[1], sessions, embargo)
            _, _, preds = _fit_predict(cand, panel, pool, fold.train, fit_end, fold.val)
            periods = window_periods(panel, fold.val, skip_sessions=embargo)
            _, _, metrics = score_book(preds, panel, periods, cand)
            scores.append(float(metrics.get("net_sharpe", 0.0)))
        best = int(np.argmax(scores))
        chosen = candidates[best]
        fit_end = embargoed_end(fold.deploy[1], sessions, embargo)
        universe, model, preds = _fit_predict(chosen, panel, pool, fold.deploy, fit_end, fold.test)

        fdir = out / "folds" / fold.name
        fdir.mkdir(parents=True, exist_ok=True)
        meta = {
            **fold.to_dict(),
            "fit_end": None if fit_end is None else str(pd.Timestamp(fit_end).date()),
            "candidate_val_net_sharpe": scores,
            "chosen": best,
            "model_params": chosen.model,
            "n_universe": len(universe),
        }
        (fdir / "fold.json").write_text(json.dumps(meta, indent=2) + "\n")
        (fdir / "universe.json").write_text(json.dumps(universe) + "\n")
        preds.to_parquet(fdir / "predictions.parquet", index=False)
        model.save(fdir / "model")
        run["folds"].append(fold.name)
        print(
            f"{fold.name}: universe {len(universe)}, val net Sharpe {scores[best]:+.3f} "
            f"(candidate {best + 1}/{len(candidates)}), {time.time() - t0:.0f}s",
            flush=True,
        )
    (out / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="python -m ml4trading.train")
    ap.add_argument("--method", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=default_data_dir())
    ap.add_argument("--grid", type=Path, help="YAML: dotted model.* key -> list of values")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one config value, e.g. --set capital=10000 (value parsed as YAML)",
    )
    args = ap.parse_args()
    grid = yaml.safe_load(args.grid.read_text()) if args.grid else None
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    train(args.method, args.out, args.data_dir, grid=grid, overrides=overrides)


if __name__ == "__main__":
    main()
