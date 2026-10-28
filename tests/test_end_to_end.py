import json

import numpy as np
import pandas as pd
import pytest

from ml4trading import config
from ml4trading.backtest import backtest
from ml4trading.data import load_bars, load_dividends
from ml4trading.models import FoldResult, Model, register_model
from ml4trading.models.base import rows_in
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import train

SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-11-30",
}


@register_model("last_return_test")
class LastReturn(Model):
    """Predicts each symbol's previous-period return times ``scale`` (a test stand-in for a
    learned model: its fit counts the labelled rows of its window)."""

    def fit_predict_fold(self, frames, fit_window, predict_window):
        panel = frames.panel
        n_fit = 0 if fit_window is None else int(np.isfinite(rows_in(panel, fit_window)["y"]).sum())
        prev = panel.sort_values(["symbol", "period"]).copy()
        last = prev.groupby("symbol")["close"].pct_change()  # period t's own close: t-1 at best
        prev["prediction"] = last.groupby(prev["symbol"]).shift(1) * float(
            self.params.get("scale", 1.0)
        )
        out = rows_in(prev, predict_window)[["period", "symbol", "prediction"]]
        return FoldResult(out.dropna().reset_index(drop=True), None, n_fit)


FIT_WINDOW = (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))


def _inputs(data_dir):
    """(bars, dividends, K, pool) of the last_return_test method on ``data_dir``."""
    cfg = config.resolve("last_return_test")
    pool, _ = load_frames(cfg, data_dir)
    return load_bars(data_dir, pool), load_dividends(data_dir, pool), cfg.K, pool


config.METHOD_DEFAULTS["last_return_test"] = {
    "K": 390,
    "model": {"hold_overnight": True},
    "trader": {"number_of_symbols_to_buy": 2},
}


def test_investment_train_then_backtest(data_dir, tmp_path):
    run = train("investment", tmp_path / "run", data_dir, overrides=SHORT_FOLDS)
    run_meta = json.loads((run / "run.json").read_text())
    assert run_meta["folds"] == [f"F2021-{m:02d}" for m in range(6, 12)]
    assert "hpo" not in run_meta  # a baseline has no search to record
    fold = json.loads((run / "folds" / "F2021-06" / "fold.json").read_text())
    assert fold["test"] == ["2021-06-01", "2021-06-30"]
    assert json.loads((run / "folds" / "F2021-06" / "universe.json").read_text()) == ["VOO"]

    metrics = backtest(run, data_dir)
    assert metrics["start"] == "2021-06-01"  # the book trades from the path's first session
    assert metrics["end"] <= "2021-11-30"
    book = pd.read_csv(run / "backtest" / "book.csv")
    positions = pd.read_csv(run / "backtest" / "positions.csv")
    assert set(positions["symbol"]) == {"VOO"}
    assert (book["n_positions"] == 1).all()
    assert metrics["total_commission"] > 0


def test_a_learned_model_is_searched_per_fold_and_causal(data_dir, tmp_path):
    grid = {"model.scale": [-1.0, 1.0]}
    run = train("last_return_test", tmp_path / "run", data_dir, grid=grid, overrides=SHORT_FOLDS)
    fold = json.loads((run / "folds" / "F2021-07" / "fold.json").read_text())
    assert len(fold["candidate_val_net_sharpe"]) == 2
    assert fold["model_params"]["scale"] == grid["model.scale"][fold["chosen"]]
    preds = pd.read_parquet(run / "folds" / "F2021-07" / "predictions.parquet")
    assert preds["period"].min() >= pd.Timestamp("2021-07-01")
    assert preds["period"].max() <= pd.Timestamp("2021-07-31")
    metrics = backtest(run, data_dir)
    assert metrics["n_periods"] > 100

    model = LastReturn(hold_overnight=True, scale=1.0)
    assert_causal(model, *_inputs(data_dir), FIT_WINDOW, JUNE)


def test_a_fit_free_model_keeps_validation_too(data_dir, tmp_path):
    run = train("investment", tmp_path / "run", data_dir, overrides=SHORT_FOLDS)
    fdir = run / "folds" / "F2021-07"
    val = pd.read_parquet(fdir / "val" / "predictions.parquet")
    assert list(val.columns) == ["period", "symbol", "prediction", "y", "label"]
    assert len(val) > 0
    fold = json.loads((fdir / "fold.json").read_text())
    assert fold["val_diagnostics"]["ic_mean"] is None  # a constant forecast has no IC
    assert fold["val_diagnostics"]["net_sharpe"] == fold["candidate_val_net_sharpe"][0]


def test_an_empty_validation_universe_still_writes_its_outputs(data_dir, tmp_path):
    # momentum selects nobody on this data: no fit, no prediction, a book that never holds a name
    run = train("momentum", tmp_path / "run", data_dir, overrides=SHORT_FOLDS)
    fdir = run / "folds" / "F2021-07"
    assert json.loads((fdir / "universe.json").read_text()) == []
    val = pd.read_parquet(fdir / "val" / "predictions.parquet")
    assert list(val.columns) == ["period", "symbol", "prediction", "y", "label"]
    assert val.empty
    diagnostics = json.loads((fdir / "fold.json").read_text())["val_diagnostics"]
    assert diagnostics["ic_mean"] is None and diagnostics["r2_mse"] is None
    assert diagnostics["same_sign_share"] is None


def test_causality_check_catches_a_peeking_model(data_dir):
    class Peeking(LastReturn):
        def fit_predict_fold(self, frames, fit_window, predict_window):
            out = rows_in(frames.panel, predict_window)[["period", "symbol", "y"]]
            return FoldResult(out.rename(columns={"y": "prediction"}).dropna(), None, 0)

    with pytest.raises(AssertionError):
        assert_causal(Peeking(hold_overnight=True), *_inputs(data_dir), FIT_WINDOW, JUNE)


def test_grids_search_model_parameters_only(data_dir, tmp_path):
    with pytest.raises(ValueError):
        train(
            "investment",
            tmp_path / "run",
            data_dir,
            grid={"trader.number_of_symbols_to_buy": [1, 2]},
        )


def test_manifest_verification_detects_a_changed_file(data_dir):
    from ml4trading.data import verify

    assert verify(data_dir) == []
    path = data_dir / "bars_30min" / "AAA.parquet"
    path.write_bytes(path.read_bytes() + b"x")
    assert verify(data_dir) == ["bars_30min/AAA.parquet"]


@pytest.mark.parametrize("method, etf", [("investment", "VOO"), ("investment_tech", "QQQ")])
def test_each_single_etf_baseline_holds_only_its_etf(data_dir, tmp_path, method, etf):
    cfg = config.resolve(method)
    assert (cfg.K, cfg.hold_overnight, cfg.trader.number_of_symbols_to_buy) == (390, True, 1)
    run = train(method, tmp_path / "run", data_dir, overrides=SHORT_FOLDS)
    backtest(run, data_dir)
    positions = pd.read_csv(run / "backtest" / "positions.csv")
    assert set(positions["symbol"]) == {etf}


def test_the_account_value_chart_is_drawn(data_dir, tmp_path):
    pytest.importorskip("matplotlib")
    from ml4trading.plot import account_values, plot_runs

    run = train("investment", tmp_path / "run", data_dir, overrides=SHORT_FOLDS)
    backtest(run, data_dir)
    values = account_values(run)
    assert values.iloc[0] == pytest.approx(20_000.0)  # starts from the capital
    out = plot_runs([(run, "VOO"), (run, "VOO again")], tmp_path / "chart.png", "test")
    assert out.exists() and out.stat().st_size > 10_000
