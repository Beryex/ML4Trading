"""Walk-forward training: one model per fold boundary of the test path.

    python -m ml4trading.train --method investment --out runs/investment [--grid grid.yaml]

The method's frames (``ml4trading.models.base``) are built once for the whole run. Per fold
boundary B (calendar: ``ml4trading.folds``):

1. VALIDATION -- for every candidate configuration of the grid: select the universe on the
   train window, fit on the train window (its last ``embargo_days`` sessions dropped), predict
   the validation window, keep the universe's predictions and trade them (validation window
   minus its first ``embargo_days`` sessions) from flat; the candidate's score is that book's
   net Sharpe. The first best wins. With no grid there is one candidate, scored all the same.
2. DEPLOYMENT -- with the chosen configuration: select the universe on the deploy window (the
   trailing ``train_months`` before B), fit on it (embargoed the same way), and predict the
   test month, again keeping the universe's predictions.
3. The fold is written to ``<out>/folds/<FYYYY-MM>/``: fold.json (windows, candidate scores,
   the choice, and each candidate's out-of-sample R² on its scored validation rows -- recorded,
   never ranked on; null for a model that fits no loss), universe.json, predictions.parquet and,
   for a model that fits one, the fitted estimator under model/.

A model fits on every symbol of its pool; the universe only decides what is traded. The grid
file maps dotted ``model.*`` keys to lists of values; only model parameters are searched per
fold, and never one the frames are built from (``Model.FRAME_PARAMS``) -- the selector, the
trader, the costs and K are fixed for the whole run.
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
from ml4trading.models import FoldResult, Frames, get_model
from ml4trading.models.base import empty_predictions
from ml4trading.models.learned.losses import oos_r2
from ml4trading.pipeline import load_daily_inputs, load_frames, score_book, window_periods
from ml4trading.selector import select_universe


def _fit_predict(
    cfg: RunConfig, frames: Frames, pool, select_window, fit_window_end, predict_window
) -> tuple[list[str], FoldResult | None, pd.DataFrame]:
    """(universe, fold result, the universe's predictions) for one configuration; an empty
    universe fits nothing."""
    universe = select_universe(frames.panel, select_window, cfg.selector, pool)
    if not universe:
        return universe, None, empty_predictions()
    fit_window = (
        None if fit_window_end is None else (pd.Timestamp(select_window[0]), fit_window_end)
    )
    result = get_model(cfg.method)(**cfg.model).fit_predict_fold(frames, fit_window, predict_window)
    preds = result.predictions
    return universe, result, preds[preds["symbol"].isin(set(universe))].reset_index(drop=True)


def _val_r2(result: FoldResult | None, preds: pd.DataFrame, panel, periods) -> float | None:
    """The out-of-sample R² of the universe's predictions over the scored ``periods`` against
    the fit's baseline, under the fit's loss; None for a model that fits no loss."""
    if result is None or result.loss is None:
        return None
    scored = preds[preds["period"].isin(set(periods))].merge(
        panel[["period", "symbol", "y"]], on=["period", "symbol"]
    )
    scored = scored[np.isfinite(scored["y"].to_numpy(dtype=float))]
    return oos_r2(scored["y"], scored["prediction"], result.baseline_prediction, result.loss)


def _check_grid(method: str, grid: dict | None) -> None:
    frame_params = get_model(method).FRAME_PARAMS
    for key in grid or {}:
        if not key.startswith("model."):
            raise ValueError(f"grid key {key!r}: only model.* parameters are searched per fold")
        if key.removeprefix("model.") in frame_params:
            raise ValueError(
                f"grid key {key!r}: the run's frames are built from it once, so it cannot be "
                f"searched per fold"
            )


def train(
    method: str,
    out: Path,
    data_dir: Path,
    grid: dict | None = None,
    overrides: dict | None = None,
) -> Path:
    base = config_mod.resolve(method, overrides)
    _check_grid(method, grid)
    candidates = config_mod.expand_grid(base, grid)
    pool, frames = load_frames(base, data_dir)
    panel = frames.panel
    daily_inputs = load_daily_inputs(base, data_dir, pool)
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
        scores, r2s = [], []
        for cand in candidates:
            fit_end = embargoed_end(fold.train[1], sessions, embargo)
            _, result, preds = _fit_predict(cand, frames, pool, fold.train, fit_end, fold.val)
            periods = window_periods(panel, fold.val, skip_sessions=embargo)
            *_, metrics = score_book(preds, panel, periods, cand, daily_inputs=daily_inputs)
            scores.append(float(metrics.get("net_sharpe", 0.0)))
            r2s.append(_val_r2(result, preds, panel, periods))
        best = int(np.argmax(scores))
        chosen = candidates[best]
        fit_end = embargoed_end(fold.deploy[1], sessions, embargo)
        universe, result, preds = _fit_predict(
            chosen, frames, pool, fold.deploy, fit_end, fold.test
        )

        fdir = out / "folds" / fold.name
        fdir.mkdir(parents=True, exist_ok=True)
        meta = {
            **fold.to_dict(),
            "fit_end": None if fit_end is None else str(pd.Timestamp(fit_end).date()),
            "candidate_val_net_sharpe": scores,
            "candidate_val_r2": r2s,
            "chosen": best,
            "model_params": chosen.model,
            "n_universe": len(universe),
        }
        (fdir / "fold.json").write_text(json.dumps(meta, indent=2) + "\n")
        (fdir / "universe.json").write_text(json.dumps(universe) + "\n")
        preds.to_parquet(fdir / "predictions.parquet", index=False)
        if result is not None and result.artifact is not None:
            result.artifact.save(fdir / "model")
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
