"""Walk-forward training: one model per fold boundary of the test path.

    python -m ml4trading.train --method investment --out runs/investment [--grid grid.yaml]
        [--importance]
    python -m ml4trading.train --method nonlinear --out runs/nonlinear --hpo-hours 12

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
   the choice, each candidate's out-of-sample R² on its scored validation rows -- recorded,
   never ranked on; null for a model that fits no loss -- and ``val_diagnostics`` of the chosen
   candidate's validation book, ``ml4trading.diagnostics``), universe.json, predictions.parquet,
   val/predictions.parquet (the chosen candidate's scored validation rows, kept from the
   validation pass, never refitted) and, for a model that fits one, the fitted estimator under
   model/. With ``--importance`` that same validation fit is also scored for feature importance
   (``ml4trading.importance``) on those rows: val/importance.parquet, one row per feature. A
   fold's val/ and model/ directories are emptied before they are written, so a re-run into the
   same ``--out`` leaves no file of an earlier run.

With ``--hpo-hours H`` (``nonlinear`` only, in place of a grid) step 1 is a search instead
(``ml4trading.hpo``): the run's budget of H hours is split evenly across the folds, and each fold
searches its parameters for that long on the validation window. The winner is the single
candidate -- scored, kept and deployed like a grid's winner -- and fold.json gains an ``hpo``
block with every trial. run.json gains an ``hpo`` block too -- the request (``hours``,
``trials``) and the per-fold ``timeout_s`` it came to -- which a run without a search lacks.

A model fits on every symbol of its pool; the universe only decides what is traded. The grid
file maps dotted ``model.*`` keys to lists of values; only model parameters are searched per
fold, and never one the frames are built from (``Model.FRAME_PARAMS``) -- the selector, the
trader, the costs and K are fixed for the whole run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import yaml

from ml4trading import config as config_mod
from ml4trading import hpo as hpo_mod
from ml4trading.config import RunConfig
from ml4trading.data import default_data_dir
from ml4trading.diagnostics import validation_diagnostics
from ml4trading.folds import embargoed_end, fold_at, fold_boundaries
from ml4trading.importance import importance_table
from ml4trading.models import FoldResult, Frames, get_model
from ml4trading.models.base import empty_predictions
from ml4trading.models.learned.losses import oos_r2
from ml4trading.pipeline import (
    benchmark_returns,
    load_daily_inputs,
    load_frames,
    score_book,
    window_periods,
)
from ml4trading.selector import select_universe


def _fit_window(select_window, fit_window_end):
    """The window a fit may use: from the selection window's start to the embargoed end; None
    when nothing is left."""
    return None if fit_window_end is None else (pd.Timestamp(select_window[0]), fit_window_end)


def _gated(preds: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    """Only the universe's predictions: a model fits on its whole pool, the universe is traded."""
    return preds[preds["symbol"].isin(set(universe))].reset_index(drop=True)


def _fit_predict(
    cfg: RunConfig, frames: Frames, pool, select_window, fit_window_end, predict_window
) -> tuple[list[str], FoldResult | None, pd.DataFrame]:
    """(universe, fold result, the universe's predictions) for one configuration; an empty
    universe fits nothing."""
    universe = select_universe(frames.panel, select_window, cfg.selector, pool)
    if not universe:
        return universe, None, empty_predictions()
    fit_window = _fit_window(select_window, fit_window_end)
    result = get_model(cfg.method)(**cfg.model).fit_predict_fold(frames, fit_window, predict_window)
    return universe, result, _gated(result.predictions, universe)


_SCORED_COLUMNS = ["period", "symbol", "prediction", "y", "label", "open_px"]


def _scored_val(
    result: FoldResult | None, preds: pd.DataFrame, panel: pd.DataFrame, periods
) -> pd.DataFrame:
    """The validation rows a candidate is scored on: the universe's predictions in ``periods``
    next to the realized return ``y``, the entry price ``open_px`` and ``label``, the target in
    the fit's units: ``y`` through the fit's label transform (``result`` is the fit it will be
    read from), or ``y`` itself when it has none. The transform is applied to every pool row of
    ``periods``, not just the universe's, so a period's mean is the pool's cross-section the fit
    demeaned over. Only rows with a finite ``y`` count: a symbol's last overnight period has none
    (and so no exit price), and neither the IC, the R² nor the cost-gate replay can use it."""
    in_scope = panel[panel["period"].isin(set(periods))]
    pool = in_scope[np.isfinite(in_scope["y"].to_numpy(dtype=float))]
    transform = None if result is None else result.label_transform
    label = (
        pool["y"]
        if transform is None
        else transform.apply(pool["y"].to_numpy(dtype=float), pool["period"].to_numpy())
    )
    scored = preds[preds["period"].isin(set(periods))].merge(
        pool[["period", "symbol", "y", "open_px"]].assign(label=label),
        on=["period", "symbol"],
    )
    return scored[_SCORED_COLUMNS].reset_index(drop=True)


def _scored_features(frames: Frames, scored: pd.DataFrame) -> pd.DataFrame:
    """The features of the scored rows, in their order and indexed by (period, symbol) as a
    model receives them."""
    panel = frames.panel
    keys = pd.MultiIndex.from_arrays([panel["period"], panel["symbol"]], names=["period", "symbol"])
    wanted = pd.MultiIndex.from_frame(scored[["period", "symbol"]])
    return frames.X.iloc[keys.get_indexer(wanted)].set_axis(wanted)


def _val_r2(result: FoldResult | None, scored: pd.DataFrame) -> float | None:
    """The out-of-sample R² of the scored rows' predictions against the fit's baseline, under
    the fit's loss; None for a model that fits no loss."""
    if result is None or result.loss is None:
        return None
    return oos_r2(scored["label"], scored["prediction"], result.baseline_prediction, result.loss)


class _Fit(NamedTuple):
    """One validation fit: the configuration it was made with, its result (None when nothing was
    fitted) and the universe's predictions over the validation window."""

    cfg: RunConfig
    result: FoldResult | None
    preds: pd.DataFrame


class _Kept(NamedTuple):
    """The best candidate so far: its index among the fold's candidates and its configuration
    (what the fold deploys), and what its validation pass produced, kept so nothing is
    refitted."""

    index: int
    cfg: RunConfig
    result: FoldResult | None
    scored: pd.DataFrame
    book: pd.DataFrame
    positions: pd.DataFrame
    metrics: dict


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


def _check_candidates(candidates: list[RunConfig]) -> None:
    """Build every candidate's model once, so a misspelt or invalid ``model.*`` value is refused
    before any data is loaded."""
    for cand in candidates:
        try:
            get_model(cand.method)(**cand.model)
        except TypeError as err:  # a fixed-signature model given an unknown keyword
            raise ValueError(f"{cand.method}: {err}") from err


def _check_hpo(method: str, grid: dict | None, hpo_hours: float | None, hpo_trials: int | None):
    """Refuse an HPO request that cannot run, before any data is loaded."""
    if hpo_hours is None and hpo_trials is None:
        return
    if method != "nonlinear":
        raise ValueError(f"HPO searches nonlinear's parameters; {method!r} has none to search")
    if grid:
        raise ValueError("HPO takes the place of a grid: give one or the other")
    if hpo_hours is not None and not hpo_hours > 0:
        raise ValueError(f"hpo_hours={hpo_hours}: the time budget must be positive")
    if hpo_trials is not None and hpo_trials < 1:
        raise ValueError(f"hpo_trials={hpo_trials}: a fold's baseline trial always runs")


def _grid_fits(candidates: list[RunConfig], frames: Frames, pool, fold, fit_end) -> Iterator[_Fit]:
    """The fold's candidates fitted on its train window and predicting its validation window,
    one at a time."""
    for cand in candidates:
        _, result, preds = _fit_predict(cand, frames, pool, fold.train, fit_end, fold.val)
        yield _Fit(cand, result, preds)


def _hpo_fit(
    base: RunConfig,
    frames: Frames,
    pool,
    fold,
    fit_end,
    periods: list,
    *,
    timeout: float | None,
    n_trials: int | None,
) -> tuple[_Fit, dict]:
    """The fold's search (``ml4trading.hpo``) as the one validation fit it stands for, and the
    ``hpo`` block of fold.json. The search fits on the train window and early-stops on the
    universe's rows of the scored validation periods. The winner's configuration is the run's
    with the winner's five parameters and the round it kept as ``n_estimators``: the deploy fit
    stops there on its own window. A fold with nothing to fit -- no universe, or no fit window --
    is not searched and stands for the run's own configuration, as a grid's empty fold does."""
    record = {"timeout_s": timeout, "n_trials": 0, "best_number": None, "trials": []}
    universe = select_universe(frames.panel, fold.train, base.selector, pool)
    if not universe or fit_end is None:
        return _Fit(base, None, empty_predictions()), record
    found = hpo_mod.search_fold(
        base,
        frames,
        _fit_window(fold.train, fit_end),
        periods,
        universe,
        timeout=timeout,
        n_trials=n_trials,
    )
    best = found.best
    chosen = config_mod.apply_overrides(
        base,
        {f"model.{name}": value for name, value in best.params.items()}
        | {"model.n_estimators": best.best_iteration},
    )
    record.update(
        n_trials=len(found.trials),
        best_number=best.number,
        trials=[asdict(trial) for trial in found.trials],
    )
    return _Fit(chosen, found.result, _gated(found.result.predictions, universe)), record


def train(
    method: str,
    out: Path,
    data_dir: Path,
    grid: dict | None = None,
    overrides: dict | None = None,
    importance: bool = False,
    hpo_hours: float | None = None,
    hpo_trials: int | None = None,
) -> Path:
    """Train ``method`` over every fold of the test path into ``out``. ``hpo_hours`` (the CLI's
    ``--hpo-hours``) searches ``nonlinear``'s parameters per fold within that many hours in all;
    ``hpo_trials`` limits a fold's search to that many trials (the Python API, for tests)."""
    base = config_mod.resolve(method, overrides)
    _check_hpo(method, grid, hpo_hours, hpo_trials)
    _check_grid(method, grid)
    candidates = config_mod.expand_grid(base, grid)
    _check_candidates(candidates)
    pool, frames = load_frames(base, data_dir)
    panel = frames.panel
    daily_inputs = load_daily_inputs(base, data_dir, pool)
    benchmark = benchmark_returns(data_dir)
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
    boundaries = fold_boundaries(base.folds)
    searching = hpo_hours is not None or hpo_trials is not None
    timeout = hpo_hours * 3600 / len(boundaries) if hpo_hours and boundaries else None
    if searching:
        run["hpo"] = {"hours": hpo_hours, "trials": hpo_trials, "timeout_s": timeout}
    t0 = time.time()
    for b in boundaries:
        fold = fold_at(b, base.folds)
        val_fit_end = embargoed_end(fold.train[1], sessions, embargo)
        periods = window_periods(panel, fold.val, skip_sessions=embargo)
        hpo_record = None
        if searching:
            winner, hpo_record = _hpo_fit(
                base, frames, pool, fold, val_fit_end, periods, timeout=timeout, n_trials=hpo_trials
            )
            fits = [winner]
        else:
            fits = _grid_fits(candidates, frames, pool, fold, val_fit_end)
        scores, r2s = [], []
        kept = None
        for index, (cand, result, preds) in enumerate(fits):
            book, positions, _, metrics = score_book(
                preds, panel, periods, cand, benchmark=benchmark, daily_inputs=daily_inputs
            )
            scored = _scored_val(result, preds, panel, periods)
            scores.append(float(metrics.get("net_sharpe", 0.0)))
            r2s.append(_val_r2(result, scored))
            if kept is None or scores[-1] > scores[kept.index]:  # the first best wins
                kept = _Kept(index, cand, result, scored, book, positions, metrics)
        best, chosen = kept.index, kept.cfg
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
            "val_diagnostics": validation_diagnostics(
                kept.scored,
                kept.book,
                kept.positions,
                kept.metrics,
                chosen,
                None if kept.result is None else kept.result.fit_label_mean,
            ),
        }
        if hpo_record is not None:
            meta["hpo"] = hpo_record
        (fdir / "fold.json").write_text(json.dumps(meta, indent=2) + "\n")
        for stale in ("val", "model"):  # nothing of an earlier run into the same --out survives
            if (fdir / stale).exists():
                shutil.rmtree(fdir / stale)
        (fdir / "val").mkdir()
        kept.scored.drop(columns="open_px").to_parquet(
            fdir / "val" / "predictions.parquet", index=False
        )
        artifact = None if kept.result is None else kept.result.artifact
        if importance and artifact is not None and len(kept.scored):
            table = importance_table(
                artifact,
                _scored_features(frames, kept.scored),
                kept.scored["label"].to_numpy(),
                kept.scored["y"].to_numpy(),
            )
            table.reset_index().to_parquet(fdir / "val" / "importance.parquet", index=False)
        (fdir / "universe.json").write_text(json.dumps(universe) + "\n")
        preds.to_parquet(fdir / "predictions.parquet", index=False)
        if result is not None and result.artifact is not None:
            result.artifact.save(fdir / "model")
        run["folds"].append(fold.name)
        picked = (
            f"candidate {best + 1}/{len(candidates)}"
            if hpo_record is None
            else f"best of {hpo_record['n_trials']} HPO trials"
        )
        print(
            f"{fold.name}: universe {len(universe)}, val net Sharpe {scores[best]:+.3f} "
            f"({picked}), {time.time() - t0:.0f}s",
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
    ap.add_argument(
        "--importance",
        action="store_true",
        help="also score the chosen candidate's validation fit for feature importance "
        "(folds/<F>/val/importance.parquet); fits with no estimator write none",
    )
    ap.add_argument(
        "--hpo-hours",
        type=float,
        metavar="H",
        help="nonlinear only, instead of a grid: search its parameters per fold, H hours in all "
        "split evenly across the folds (the fold's fold.json records every trial)",
    )
    args = ap.parse_args()
    grid = yaml.safe_load(args.grid.read_text()) if args.grid else None
    overrides = {}
    for item in args.set:
        key, _, value = item.partition("=")
        overrides[key] = yaml.safe_load(value)
    train(
        args.method,
        args.out,
        args.data_dir,
        grid=grid,
        overrides=overrides,
        importance=args.importance,
        hpo_hours=args.hpo_hours,
    )


if __name__ == "__main__":
    main()
