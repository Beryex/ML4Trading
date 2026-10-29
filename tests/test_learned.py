import json
import tracemalloc
import warnings

import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset
from sklearn.exceptions import ConvergenceWarning

from ml4trading import config
from ml4trading.backtest import backtest
from ml4trading.config import TIMEZONE
from ml4trading.data import all_symbols, load_bars, load_dividends
from ml4trading.features import CALENDAR
from ml4trading.metrics import sharpe
from ml4trading.models import FoldResult, Frames, get_model
from ml4trading.models.base import empty_predictions, rows_in
from ml4trading.models.learned.base import EarlyStop, FitArtifact, LearnedModel
from ml4trading.models.learned.horizon import horizon_labels, realized_by
from ml4trading.models.learned.labels import LabelTransform
from ml4trading.models.learned.linear import impute_by_period
from ml4trading.models.learned.losses import baseline_prediction, loss_value, oos_r2
from ml4trading.models.learned.nonlinear import LGBMArtifact, NonlinearModel
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import _scored_val, _val_r2, train

METHODS = ["linear", "nonlinear"]
FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))
SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-08-31",
}


@pytest.fixture(scope="module")
def learned_data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("learned") / "data")


@pytest.fixture(scope="module")
def frames(learned_data):
    return {m: load_frames(config.resolve(m), learned_data)[1] for m in METHODS}


def _model(method, **params):
    cfg = config.resolve(method, {f"model.{k}": v for k, v in params.items()})
    return get_model(method)(**cfg.model)


def _keyed(frames, window):
    """The features of ``window``'s rows, indexed by (period, symbol) as a model hands them."""
    panel = frames.panel
    keys = pd.MultiIndex.from_arrays([panel["period"], panel["symbol"]], names=["period", "symbol"])
    rows = rows_in(panel, window).index
    return frames.X.set_axis(keys).iloc[panel.index.get_indexer(rows)]


def test_losses():
    assert loss_value([1.0, 2.0], [1.0, 4.0], "mse") == 2.0
    assert loss_value([1.0, 2.0], [1.0, 4.0], "l1") == 1.0
    assert loss_value([], [], "mse") is None
    assert loss_value([1.0], [np.nan], "mse") is None
    with pytest.raises(ValueError):
        loss_value([1.0], [1.0, 2.0], "mse")
    with pytest.raises(ValueError):
        loss_value([1.0], [1.0], "huber")
    assert baseline_prediction([1.0, 2.0, 9.0], "l1") == 2.0
    assert baseline_prediction([1.0, 2.0, 9.0], "mse") == 4.0
    with pytest.raises(ValueError):
        baseline_prediction([], "mse")
    y = np.array([0.01, -0.02, 0.03])
    for loss in ("mse", "l1"):
        b = baseline_prediction(y, loss)
        assert oos_r2(y, y, b, loss) == 1.0
        assert oos_r2(y, np.full(3, b), b, loss) == 0.0
    assert oos_r2(y, y, None, "mse") is None
    assert oos_r2(np.ones(3), np.ones(3), 1.0, "mse") is None  # the baseline loss is 0


def test_bad_configurations_are_refused(learned_data):
    with pytest.raises(ValueError, match="alpha_ratio"):
        get_model("linear")(hold_overnight=True, alpha=1.0, l1_ratio=0.5, max_iter=1000)
    with pytest.raises(ValueError, match="loss"):
        _model("nonlinear", loss="huber")
    with pytest.raises(ValueError, match="l1_ratio"):
        _model("linear", l1_ratio=0.0)
    with pytest.raises(ValueError, match="K=1950"):
        load_frames(config.resolve("linear", {"K": 1950}), learned_data)
    with pytest.raises(ValueError, match="VOO"):
        _model("linear").symbol_pool(["AAA", "QQQ"])


def test_the_pool_is_the_stocks_and_the_frames_align(frames):
    f = frames["linear"]
    assert set(f.panel["symbol"]) == {"AAA", "BBB", "CCC"}
    assert f.X.index.equals(f.panel.index)
    assert (f.X.dtypes == np.float32).all()


def test_alpha_ratio_one_zeroes_every_coefficient(frames):
    zero = _model("linear", alpha_ratio=1.0).fit_predict_fold(frames["linear"], FIT, JUNE)
    assert np.abs(zero.artifact.coef).max() <= 1e-12
    some = _model("linear", alpha_ratio=0.01).fit_predict_fold(frames["linear"], FIT, JUNE)
    assert (some.artifact.coef != 0).any()


def test_linear_ignores_stock_id(frames):
    f = frames["linear"]
    want = _model("linear").fit_predict_fold(f, FIT, JUNE)
    assert "stock_id" not in want.artifact.feature_names()
    X = f.X.copy()
    X["stock_id"] = X["stock_id"].to_numpy()[::-1]
    got = _model("linear").fit_predict_fold(Frames(f.panel, X), FIT, JUNE)
    pd.testing.assert_frame_equal(got.predictions, want.predictions)


def test_impute_by_period():
    d1, d2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    keys = [(d1, "A"), (d1, "B"), (d1, "C"), (d2, "A"), (d2, "B")]
    X = pd.DataFrame(
        {"f": [1.0, np.nan, 3.0, np.nan, np.nan]},
        index=pd.MultiIndex.from_tuples(keys, names=["period", "symbol"]),
    )
    assert impute_by_period(X)["f"].tolist() == [1.0, 2.0, 3.0, 0.0, 0.0]


def test_an_all_nan_column_keeps_every_prediction(frames):
    f = frames["linear"]
    X = f.X.copy()
    X["Return_1d"] = np.nan
    preds = _model("linear").fit_predict_fold(Frames(f.panel, X), FIT, JUNE).predictions
    assert len(preds) == len(rows_in(f.panel, JUNE))
    assert np.isfinite(preds["prediction"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_the_fit_reads_only_labelled_fit_rows(frames, method):
    f = frames[method]
    want = _model(method).fit_predict_fold(f, FIT, JUNE)
    panel = f.panel.copy()
    outside = (panel["session"] < FIT[0]) | (panel["session"] > FIT[1])
    panel.loc[outside, "y"] = panel.loc[outside, "y"] * -3.0 + 0.5
    got = _model(method).fit_predict_fold(Frames(panel, f.X), FIT, JUNE)
    pd.testing.assert_frame_equal(got.predictions, want.predictions)
    assert got.n_fit_rows == want.n_fit_rows == int(np.isfinite(rows_in(panel, FIT)["y"]).sum())


@pytest.mark.parametrize("method", METHODS)
def test_a_saved_fit_predicts_the_same(frames, method, tmp_path):
    f = frames[method]
    artifact = _model(method).fit_predict_fold(f, FIT, JUNE).artifact
    artifact.save(tmp_path / "model")
    loaded = type(artifact).load(tmp_path / "model")
    X = _keyed(f, JUNE)
    assert loaded.feature_names() == artifact.feature_names()
    assert np.array_equal(loaded.predict(X), artifact.predict(X))


@pytest.mark.parametrize("method", METHODS)
def test_two_fits_are_identical(frames, method):
    a = _model(method).fit_predict_fold(frames[method], FIT, JUNE).predictions
    b = _model(method).fit_predict_fold(frames[method], FIT, JUNE).predictions
    pd.testing.assert_frame_equal(a, b, check_exact=True)


@pytest.mark.parametrize(
    "method, params",
    [
        ("linear", {}),
        ("nonlinear", {"label": "raw"}),
        ("nonlinear", {"label": "demean_winsor"}),
        ("nonlinear", {"feature_transform": "cs_rank"}),
        ("linear", {"horizon": 5}),
        ("nonlinear", {"horizon": 5}),
        ("nonlinear", {"init_model": "linear"}),
        ("nonlinear", {"init_model": "linear", "label": "demean_winsor", "risk_adjust": "vol"}),
        (
            "nonlinear",
            {"init_model": "linear", "smooth_halflife": 3, "horizon": 5, "risk_adjust": "vol"},
        ),
    ],
)
def test_causal(learned_data, method, params):
    model = _model(method, **params)
    pool = model.symbol_pool(all_symbols(learned_data))
    loaded = model.loaded_symbols(pool)
    bars, divs = load_bars(learned_data, loaded), load_dividends(learned_data, loaded)
    assert_causal(model, bars, divs, 390, pool, FIT, JUNE, n_checks=3)


@pytest.mark.parametrize("method", METHODS)
def test_an_empty_fit_window_gives_an_empty_result(frames, method):
    early = (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30"))
    result = _model(method).fit_predict_fold(frames[method], early, JUNE)
    assert result.predictions.empty and result.n_fit_rows == 0 and result.artifact is None


@pytest.mark.parametrize(
    "method, grid, artifact",
    [
        ("linear", {"model.alpha_ratio": [0.01, 0.1]}, "model.npz"),
        ("nonlinear", {"model.num_leaves": [3, 7]}, "model.txt"),
    ],
)
def test_train_then_backtest(learned_data, tmp_path, method, grid, artifact):
    run = train(method, tmp_path / "run", learned_data, grid=grid, overrides=SHORT_FOLDS)
    fold = json.loads((run / "folds" / "F2021-07" / "fold.json").read_text())
    assert len(fold["candidate_val_r2"]) == 2
    assert all(r is None or np.isfinite(r) for r in fold["candidate_val_r2"])
    assert (run / "folds" / "F2021-07" / "model" / artifact).exists()
    assert backtest(run, learned_data)["n_periods"] > 0


def test_a_fit_free_model_records_no_r2(learned_data, tmp_path):
    run = train("investment", tmp_path / "run", learned_data, overrides=SHORT_FOLDS)
    fold = json.loads((run / "folds" / "F2021-07" / "fold.json").read_text())
    assert fold["candidate_val_r2"] == [None]
    assert not (run / "folds" / "F2021-07" / "model").exists()


def test_the_chosen_candidates_validation_is_kept(learned_data, tmp_path):
    grid = {"model.alpha_ratio": [0.01, 0.1]}
    run = train("linear", tmp_path / "run", learned_data, grid=grid, overrides=SHORT_FOLDS)
    for fdir in sorted((run / "folds").iterdir()):
        fold = json.loads((fdir / "fold.json").read_text())
        val = pd.read_parquet(fdir / "val" / "predictions.parquet")
        assert list(val.columns) == ["period", "symbol", "prediction", "y", "label"]
        assert len(val) > 0
        assert np.isfinite(val["y"]).all()
        assert set(val["symbol"]) <= set(json.loads((fdir / "universe.json").read_text()))
        diagnostics = fold["val_diagnostics"]
        keys = "net_sharpe beta ic_mean long_net same_sign_share gate_pass_mean".split()
        assert set(keys + ["pred_xs_std", "r2_mse"]) <= set(diagnostics)
        assert diagnostics["net_sharpe"] == fold["candidate_val_net_sharpe"][fold["chosen"]]


def test_the_kept_validation_is_the_chosen_candidates_not_the_last(learned_data, tmp_path):
    # alpha_ratio 0.1 wins every fold; listed first or last, its validation is what is kept
    runs = {}
    for name, ratios in (("last", [0.01, 0.1]), ("first", [0.1, 0.01])):
        grid = {"model.alpha_ratio": ratios}
        out = tmp_path / name
        runs[name] = train("linear", out, learned_data, grid=grid, overrides=SHORT_FOLDS)
    for fdir in sorted((runs["first"] / "folds").iterdir()):
        fold = json.loads((fdir / "fold.json").read_text())
        assert fold["chosen"] == 0
        assert fold["val_diagnostics"]["net_sharpe"] == max(fold["candidate_val_net_sharpe"])
        other = runs["last"] / "folds" / fdir.name
        kept_first = pd.read_parquet(fdir / "val" / "predictions.parquet")
        kept_last = pd.read_parquet(other / "val" / "predictions.parquet")
        assert kept_first.equals(kept_last)
        other_fold = json.loads((other / "fold.json").read_text())
        assert other_fold["val_diagnostics"] == fold["val_diagnostics"]


def test_validation_daily_returns_are_kept(learned_data, tmp_path):
    run = train("linear", tmp_path / "run", learned_data, overrides=SHORT_FOLDS)
    for fdir in sorted((run / "folds").iterdir()):
        fold = json.loads((fdir / "fold.json").read_text())
        daily = pd.read_parquet(fdir / "val" / "daily.parquet")
        assert list(daily.columns) == ["session", "ret", "gross_ret"]
        assert daily["session"].dtype.kind == "M" and daily["session"].is_monotonic_increasing
        assert daily["session"].is_unique and len(daily) > 0
        assert daily["ret"].dtype == daily["gross_ret"].dtype == np.float64
        assert np.isfinite(daily[["ret", "gross_ret"]].to_numpy()).all()
        # summarize computes net_sharpe from exactly this series
        net_sharpe = fold["val_diagnostics"]["net_sharpe"]
        assert sharpe(daily["ret"]) == pytest.approx(net_sharpe, abs=1e-12)


def test_the_kept_daily_returns_are_the_chosen_candidates_not_the_last(learned_data, tmp_path):
    # alpha_ratio 0.1 wins every fold; listed first or last, its daily returns are what is kept
    runs = {}
    for name, ratios in (("last", [0.01, 0.1]), ("first", [0.1, 0.01])):
        grid = {"model.alpha_ratio": ratios}
        runs[name] = train(
            "linear", tmp_path / name, learned_data, grid=grid, overrides=SHORT_FOLDS
        )
    for fdir in sorted((runs["first"] / "folds").iterdir()):
        fold = json.loads((fdir / "fold.json").read_text())
        kept_first = pd.read_parquet(fdir / "val" / "daily.parquet")
        kept_last = pd.read_parquet(runs["last"] / "folds" / fdir.name / "val" / "daily.parquet")
        assert kept_first.equals(kept_last)
        best = max(fold["candidate_val_net_sharpe"])
        assert sharpe(kept_first["ret"]) == pytest.approx(best, abs=1e-12)


def test_an_empty_validation_book_keeps_an_empty_daily_frame(learned_data, tmp_path):
    # momentum selects nobody on this data: the validation book is empty
    run = train("momentum", tmp_path / "run", learned_data, overrides=SHORT_FOLDS)
    for fdir in sorted((run / "folds").iterdir()):
        assert json.loads((fdir / "universe.json").read_text()) == []
        daily = pd.read_parquet(fdir / "val" / "daily.parquet")
        assert list(daily.columns) == ["session", "ret", "gross_ret"]
        assert daily.empty
        assert daily["session"].dtype.kind == "M"
        assert daily["ret"].dtype == daily["gross_ret"].dtype == np.float64


def test_scored_val_drops_unlabelled_rows(frames):
    # the data's last session has no next open, so overnight its rows have no label
    panel = frames["linear"].panel
    last_two = sorted(panel["session"].unique())[-2:]
    unlabelled = panel[panel["session"] == last_two[1]]
    assert unlabelled["y"].isna().all()
    rows = panel[panel["session"].isin(last_two)]
    preds = rows[["period", "symbol"]].assign(prediction=0.001)
    result = _model("linear").fit_predict_fold(frames["linear"], FIT, (last_two[0], last_two[1]))
    scored = _scored_val(result, preds, panel, sorted(rows["period"].unique()))
    assert list(scored.columns) == ["period", "symbol", "prediction", "y", "label", "open_px"]
    assert len(scored) > 0
    assert np.isfinite(scored["y"]).all() and np.isfinite(scored["open_px"]).all()
    assert set(scored["period"]) == set(panel.loc[panel["session"] == last_two[0], "period"])
    pd.testing.assert_series_equal(scored["label"], scored["y"], check_names=False)


def test_a_bad_grid_is_refused_before_any_data_is_loaded(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the grid was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    with pytest.raises(ValueError, match="alpha"):
        train("linear", tmp_path / "run", learned_data, grid={"model.alpha": [0.1]})
    with pytest.raises(ValueError, match="l1_ratio"):
        train("linear", tmp_path / "run", learned_data, grid={"model.l1_ratio": [0.0]})
    with pytest.raises(ValueError, match="revers"):
        train("momentum", tmp_path / "run", learned_data, grid={"model.revers": [True]})


@pytest.mark.parametrize("method", METHODS)
def test_a_label_realized_after_the_fit_window_is_not_fitted(learned_data, method):
    # BBB stops trading for 10 sessions right after the fit window: its last row inside the window
    # is labelled by the open after the gap, inside the period the fit is scored on
    model = _model(method)
    pool = model.symbol_pool(all_symbols(learned_data))
    loaded = model.loaded_symbols(pool)
    bars, divs = load_bars(learned_data, loaded), load_dividends(learned_data, loaded)
    sessions = bars["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    gap = sorted(sessions[sessions > FIT[1]].unique())[:10]
    frames = model.build_frames(
        bars[~((bars["symbol"] == "BBB") & sessions.isin(gap))], divs, 390, pool
    )
    want = model.fit_predict_fold(frames, FIT, JUNE)
    panel = frames.panel.copy()
    last = rows_in(panel[panel["symbol"] == "BBB"], FIT).index[-1]
    assert panel.loc[last, "session"] == FIT[1]
    panel.loc[last, "y"] += 0.5
    got = model.fit_predict_fold(Frames(panel, frames.X), FIT, JUNE)
    pd.testing.assert_frame_equal(got.predictions, want.predictions)
    assert got.n_fit_rows == want.n_fit_rows


def test_the_linear_fit_converges_across_the_grid(frames):
    # more fit rows than features, as on the real data (a short window here has fewer, and an
    # under-determined net may never converge)
    long_fit = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-11-22"))
    december = (pd.Timestamp("2021-12-01"), pd.Timestamp("2021-12-31"))
    for ratio in (0.001, 0.01, 0.1):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            _model("linear", alpha_ratio=ratio).fit_predict_fold(
                frames["linear"], long_fit, december
            )


def _fit_capturing(monkeypatch, frames, **params):
    """A nonlinear fold's result with the (X, y) its ``_fit`` was handed."""
    seen = {}
    real = NonlinearModel._fit

    def spy(self, X_fit, y_fit, eval_set=None):
        seen["X"], seen["y"] = X_fit, np.array(y_fit)
        return real(self, X_fit, y_fit, eval_set)

    monkeypatch.setattr(NonlinearModel, "_fit", spy)
    result = _model("nonlinear", **params).fit_predict_fold(frames, FIT, JUNE)
    return result, seen["X"], seen["y"]


def _raw_fit_labels(f):
    """The fit window's finite raw labels and their periods, in the order a fit sees them."""
    rows = rows_in(f.panel, FIT)
    ok = np.isfinite(rows["y"].to_numpy(dtype=float))
    return rows["y"].to_numpy(dtype=float)[ok], rows["period"].to_numpy()[ok]


def test_nonlinear_trains_on_the_demeaned_label(frames, monkeypatch):
    f = frames["nonlinear"]
    result, X_fit, y_fit = _fit_capturing(monkeypatch, f, label="demean")
    raw, period = _raw_fit_labels(f)
    assert len(y_fit) == len(raw) == result.n_fit_rows
    by_period = pd.Series(y_fit).groupby(X_fit.index.get_level_values("period")).mean()
    assert len(by_period) > 20 and (by_period.abs() < 1e-12).all()
    want = raw - pd.Series(raw).groupby(period).transform("mean").to_numpy()
    assert np.array_equal(y_fit, want)
    assert result.label_transform == LabelTransform("demean")


def test_nonlinear_trains_on_the_winsorized_label(frames, monkeypatch):
    f = frames["nonlinear"]
    result, _, y_fit = _fit_capturing(monkeypatch, f, label="winsor")
    raw, _ = _raw_fit_labels(f)
    lo, hi = np.quantile(raw, 0.005), np.quantile(raw, 0.995)
    assert (result.label_transform.lo, result.label_transform.hi) == (lo, hi)
    assert (raw < lo).any() and (raw > hi).any()  # the test has something to clip
    assert np.array_equal(y_fit, np.clip(raw, lo, hi))


def test_nonlinear_demeans_before_it_winsorizes(frames, monkeypatch):
    f = frames["nonlinear"]
    result, _, y_fit = _fit_capturing(monkeypatch, f, label="demean_winsor")
    raw, period = _raw_fit_labels(f)
    demeaned = raw - pd.Series(raw).groupby(period).transform("mean").to_numpy()
    lo, hi = np.quantile(demeaned, 0.005), np.quantile(demeaned, 0.995)
    assert (result.label_transform.lo, result.label_transform.hi) == (lo, hi)
    assert np.array_equal(y_fit, np.clip(demeaned, lo, hi))


@pytest.mark.parametrize("label", ["raw", "demean", "winsor", "demean_winsor"])
def test_fit_label_mean_is_the_mean_of_the_labels_the_fit_trained_on(frames, monkeypatch, label):
    f = frames["nonlinear"]
    result, _, y_fit = _fit_capturing(monkeypatch, f, label=label)
    assert result.fit_label_mean == float(np.mean(y_fit))
    raw, _ = _raw_fit_labels(f)
    if label == "raw":
        assert result.fit_label_mean == float(np.mean(raw))
    if label == "demean":
        assert abs(result.fit_label_mean) < 1e-12
    if label == "winsor":
        assert result.fit_label_mean != float(np.mean(raw))  # the clipped tails moved it
    # the baseline of the validation R² is a constant of the same labels
    assert result.baseline_prediction == float(np.median(y_fit))  # nonlinear's loss is l1


def test_linear_stays_raw(frames):
    f = frames["linear"]
    result = _model("linear").fit_predict_fold(f, FIT, JUNE)
    raw, _ = _raw_fit_labels(f)
    assert result.label_transform == LabelTransform("raw")
    assert result.fit_label_mean == float(np.mean(raw))
    assert result.baseline_prediction == float(np.mean(raw))  # linear's loss is mse


def test_bad_label_is_refused():
    with pytest.raises(ValueError, match="label"):
        _model("nonlinear", label="rank")
    with pytest.raises(ValueError, match="label"):
        _model("linear", label="demean")  # linear has no such parameter: it stays raw
    params = dict(config.resolve("nonlinear").model)
    del params["label"]
    with pytest.raises(ValueError, match="missing parameter.*label"):
        get_model("nonlinear")(**params)


def test_validation_label_uses_the_fits_transform(learned_data, tmp_path):
    # two of the three stocks are the universe, so the pool's cross-section is not the traded one
    overrides = {**SHORT_FOLDS, "model.label": "demean", "selector.selection_cap": 2}
    run = train("nonlinear", tmp_path / "run", learned_data, overrides=overrides)
    panel = load_frames(config.resolve("nonlinear"), learned_data)[1].panel
    labelled = panel[np.isfinite(panel["y"].to_numpy(dtype=float))]
    pool_mean = labelled.groupby("period")["y"].mean()
    for fdir in sorted((run / "folds").iterdir()):
        val = pd.read_parquet(fdir / "val" / "predictions.parquet")
        fold = json.loads((fdir / "fold.json").read_text())
        assert len(val) > 0 and val["symbol"].nunique() == 2
        assert (val["label"] != val["y"]).all()
        shift = val["period"].map(pool_mean).to_numpy()
        assert np.allclose(val["label"] + shift, val["y"], rtol=0, atol=1e-12)
        own = val.groupby("period")["y"].transform("mean").to_numpy()
        assert not np.allclose(val["label"] + own, val["y"], rtol=0, atol=1e-12)
        assert all(r is None or np.isfinite(r) for r in fold["candidate_val_r2"])


def test_scored_val_labels_the_universe_by_the_pools_cross_section():
    d1, d2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    panel = pd.DataFrame(
        {
            "period": [d1, d1, d1, d2, d2, d2],
            "symbol": ["A", "B", "C"] * 2,
            "y": [0.01, 0.03, 0.08, 0.02, np.nan, 0.04],  # B has no label in d2
            "open_px": 10.0,
        }
    )
    preds = pd.DataFrame(  # the universe is A and B; C is traded by nobody
        {"period": [d1, d1, d2, d2], "symbol": ["A", "B", "A", "B"], "prediction": 0.001}
    )
    fit = FoldResult(empty_predictions(), None, 1, "mse", 0.0, 0.0, LabelTransform("demean"))
    scored = _scored_val(fit, preds, panel, [d1, d2])
    # d1's mean is over A, B and C (0.04); d2's over its two labelled rows (0.03), not over B
    assert scored[["period", "symbol"]].values.tolist() == [[d1, "A"], [d1, "B"], [d2, "A"]]
    assert np.allclose(scored["label"], [-0.03, -0.01, -0.01], rtol=0, atol=1e-15)
    assert scored["y"].tolist() == [0.01, 0.03, 0.02]

    clipped = LabelTransform("demean_winsor", lo=-0.02, hi=0.02)
    scored = _scored_val(
        FoldResult(empty_predictions(), None, 1, "mse", 0.0, 0.0, clipped), preds, panel, [d1, d2]
    )
    assert np.allclose(scored["label"], [-0.02, -0.01, -0.01], rtol=0, atol=1e-15)

    for no_transform in (None, FoldResult(empty_predictions(), None, 0)):
        scored = _scored_val(no_transform, preds, panel, [d1, d2])
        assert scored["label"].tolist() == scored["y"].tolist() == [0.01, 0.03, 0.02]


@pytest.fixture(scope="module")
def ranked_frames(learned_data):
    cfg = config.resolve("nonlinear", {"model.feature_transform": "cs_rank"})
    return load_frames(cfg, learned_data)[1]


def _left_alone(columns):
    """The columns ``cs_rank`` does not rank: the stock code, the ETFs' and the calendar's."""
    return [
        c for c in columns if c == "stock_id" or c.startswith(("VOO_", "QQQ_")) or c in CALENDAR
    ]


def test_cs_rank_maps_each_period_onto_minus_one_to_one(frames, ranked_frames):
    raw, ranked = frames["nonlinear"], ranked_frames
    pd.testing.assert_frame_equal(raw.panel, ranked.panel)
    assert ranked.X.index.equals(raw.X.index) and list(ranked.X.columns) == list(raw.X.columns)
    assert (ranked.X.dtypes == np.float32).all()
    columns = [c for c in raw.X.columns if c not in _left_alone(raw.X.columns)]
    assert len(columns) > 100
    period = raw.panel["period"].to_numpy()
    seen = 0
    for col in columns:
        a, b = raw.X[col].to_numpy(), ranked.X[col].to_numpy()
        assert np.array_equal(np.isnan(a), np.isnan(b))  # NaN stays NaN, nothing else becomes one
        ok = ~np.isnan(a)
        seen += int(ok.sum())
        assert (np.abs(b[ok]) <= 1).all()
        # within a period the order is the raw column's: equal stays equal, larger never gets less
        d = pd.DataFrame({"p": period[ok], "a": a[ok], "b": b[ok]}).sort_values(
            ["p", "a"], kind="stable"
        )
        same_period = np.diff(d["p"].to_numpy().astype("int64")) == 0
        step = np.diff(d["b"].to_numpy(dtype=float))
        tied = np.diff(d["a"].to_numpy(dtype=float)) == 0
        assert (step[same_period] >= 0).all()
        assert (step[same_period & tied] == 0).all()
    assert seen > 100_000
    assert not np.array_equal(
        raw.X["Return_1d"].to_numpy(), ranked.X["Return_1d"].to_numpy(), equal_nan=True
    )


def test_cs_rank_three_distinct_values_give_minus_one_zero_one_and_ties_share_a_rank():
    d1, d2 = pd.Timestamp("2021-01-04"), pd.Timestamp("2021-01-05")
    panel = pd.DataFrame({"period": [d1, d1, d1, d2, d2, d2]})
    X = pd.DataFrame(
        {"f": [30.0, 10.0, 20.0, 5.0, 5.0, 9.0], "g": [1.0, 1.0, 1.0, 2.0, 1.0, 3.0]},
        dtype=np.float32,
    )
    got = _model("nonlinear", feature_transform="cs_rank")._transform_features(X, panel)
    assert got["f"].tolist() == [1.0, -1.0, 0.0, -0.5, -0.5, 1.0]  # d2: ranks 1.5, 1.5, 3
    assert got["g"].tolist() == [0.0, 0.0, 0.0, 0.0, -1.0, 1.0]  # d1 all tied: the middle rank
    assert set(got.dtypes) == {np.dtype("float32")}


def test_cs_rank_matches_the_definition_on_random_periods():
    rng = np.random.default_rng(seed=11)
    period = np.repeat(pd.date_range("2021-01-04", periods=80), 9)
    values = rng.integers(0, 5, size=period.size).astype(np.float32)  # many ties
    values[rng.random(period.size) < 0.65] = np.nan
    X = pd.DataFrame({"f": values})
    model = _model("nonlinear", feature_transform="cs_rank")
    got = model._transform_features(X, pd.DataFrame({"period": period}))["f"].to_numpy()
    want, counts = np.full(values.size, np.nan), []
    for p in np.unique(period):
        at = np.flatnonzero((period == p) & ~np.isnan(values))
        x = values[at]
        counts.append(len(x))
        # the average-tie rank by definition: the values below, plus the middle of the ties
        rank = np.array([(x < v).sum() + ((x == v).sum() + 1) / 2 for v in x])
        want[at] = 0.0 if len(x) == 1 else 2 * (rank - 1) / max(len(x) - 1, 1) - 1
    assert {0, 1} <= set(counts) and max(counts) > 5  # the empty and the lone periods occur
    np.testing.assert_allclose(got, want, rtol=0, atol=1e-7)


def test_cs_rank_leaves_market_calendar_and_id_columns(frames, ranked_frames):
    raw, ranked = frames["nonlinear"].X, ranked_frames.X
    kept = _left_alone(raw.columns)
    etf = [c for c in kept if c.startswith(("VOO_", "QQQ_"))]
    assert len(etf) == 2 * 7  # eight embedded features each, less the scale-dependent MACD_hist
    assert len(kept) == 1 + len(etf) + len(CALENDAR)
    assert set(CALENDAR) <= set(kept) and "stock_id" in kept
    assert raw[kept].notna().any().all()
    pd.testing.assert_frame_equal(ranked[kept], raw[kept], check_exact=True)


def test_cs_rank_edge_periods():
    """Review Focus 5: a column that is all NaN in a period stays NaN; a period with one value
    maps it to 0.0; neither divides by zero."""
    d = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06", "2021-01-07", "2021-01-08"])
    panel = pd.DataFrame({"period": [d[0]] * 3 + [d[1]] * 3 + [d[2]] * 3 + [d[3]] + [d[4]] * 2})
    nan = np.nan
    X = pd.DataFrame(
        {
            "stock_id": [0.0, 1.0, 2.0] * 3 + [0.0, 1.0, 2.0],
            #                one value  all NaN         all equal      alone  one value among NaN
            "Return_1d": [nan, 5.0, nan, nan, nan, nan, 7.0, 7.0, 7.0, 3.0, nan, 8.0],
            "Volume_MA_ratio": [1.0, 2.0, 3.0] + [nan] * 9,  # NaN but in its first period
            "RSI_14d": [nan] * 12,  # all NaN everywhere
            "VOO_Return_1d": [nan, 5.0, nan] * 4,
            "Weekday": [0.0] * 12,
        },
        dtype=np.float32,
    )
    before = X.copy()
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        got = _model("nonlinear", feature_transform="cs_rank")._transform_features(X, panel)
    want = {
        "Return_1d": [nan, 0, nan, nan, nan, nan, 0, 0, 0, 0, nan, 0],
        "Volume_MA_ratio": [-1, 0, 1] + [nan] * 9,
        "RSI_14d": [nan] * 12,
    }
    for col, values in want.items():
        pd.testing.assert_series_equal(got[col], pd.Series(values, dtype=np.float32, name=col))
    for col in ("stock_id", "VOO_Return_1d", "Weekday"):
        pd.testing.assert_series_equal(got[col], before[col])


def test_feature_transform_none_returns_the_frame_untouched(frames):
    f = frames["nonlinear"]
    for method in METHODS:
        assert _model(method)._transform_features(f.X, f.panel) is f.X


def test_cs_rank_does_not_copy_the_frame():
    rng = np.random.default_rng(0)
    periods, rows, columns = 400, 50, 200
    panel = pd.DataFrame({"period": np.repeat(pd.date_range("2020-01-01", periods=periods), rows)})
    X = pd.DataFrame(
        rng.normal(size=(periods * rows, columns)).astype(np.float32),
        columns=[f"f{i}" for i in range(columns)],
    )
    size = int(X.memory_usage(deep=True).sum())
    model = _model("nonlinear", feature_transform="cs_rank")
    tracemalloc.start()
    try:
        ranked = model._transform_features(X, panel)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert (ranked.dtypes == np.float32).all() and ranked.shape == (periods * rows, columns)
    assert peak < size / 4  # a second copy of the frame would be 1.0 x


def test_feature_transform_cannot_be_searched(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the grid was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    assert NonlinearModel.FRAME_PARAMS == ("hold_overnight", "extra_features", "feature_transform")
    with pytest.raises(ValueError, match="feature_transform.*cannot be searched"):
        train(
            "nonlinear",
            tmp_path / "run",
            learned_data,
            grid={"model.feature_transform": ["none", "cs_rank"]},
        )


def test_bad_feature_transform_is_refused():
    with pytest.raises(ValueError, match="feature_transform"):
        _model("nonlinear", feature_transform="zscore")
    with pytest.raises(ValueError, match="feature_transform"):
        _model("linear", feature_transform="cs_rank")  # linear has no such parameter
    params = dict(config.resolve("nonlinear").model)
    del params["feature_transform"]
    with pytest.raises(ValueError, match="missing parameter.*feature_transform"):
        get_model("nonlinear")(**params)


def _feature_sets(monkeypatch, tmp_path, **sets):
    """Point the named-set lookup at ``tmp_path`` holding one ``<name>.txt`` per keyword."""
    for name, text in sets.items():
        (tmp_path / f"{name}.txt").write_text(text)
    monkeypatch.setattr("ml4trading.features.build.FEATURE_SETS_DIR", tmp_path)


def test_stock_id_can_be_dropped(frames):
    f = frames["nonlinear"]
    result = _model("nonlinear", stock_id="drop").fit_predict_fold(f, FIT, JUNE)
    assert result.artifact.feature_names() == [c for c in f.X.columns if c != "stock_id"]
    assert len(result.predictions) > 0 and np.isfinite(result.predictions["prediction"]).all()
    kept = _model("nonlinear").fit_predict_fold(f, FIT, JUNE).artifact
    assert kept.feature_names() == list(f.X.columns)
    # the booster's own record: stock_id was a categorical feature in one fit, in no other
    assert "[categorical_feature: 0]" in kept.booster.model_to_string()
    assert "[categorical_feature: ]" in result.artifact.booster.model_to_string()


def test_bad_stock_id_is_refused():
    with pytest.raises(ValueError, match="stock_id.*numeric"):
        _model("nonlinear", stock_id="numeric")
    with pytest.raises(ValueError, match="unknown parameter.*stock_id"):
        _model("linear", stock_id="drop")  # linear has no such parameter


def test_categorical_parameters_reach_lightgbm(frames):
    booster = (
        _model("nonlinear", cat_smooth=5.0)
        .fit_predict_fold(frames["nonlinear"], FIT, JUNE)
        .artifact.booster
    )
    text = booster.model_to_string()
    assert "[cat_smooth: 5]" in text
    # the other three stay at LightGBM's defaults
    for line in ("min_data_per_group: 100", "max_cat_threshold: 32", "cat_l2: 10"):
        assert f"[{line}]" in text


def test_load_feature_set(monkeypatch, tmp_path):
    from ml4trading.features.build import load_feature_set

    _feature_sets(
        monkeypatch,
        tmp_path,
        tiny="# a comment\nstock_id\n\n  Returns_1h_mean  \n# another\nWeekday\n",
        empty="# nothing but a comment\n\n",
        twice="Weekday\nMonth\nWeekday\n",
    )
    assert load_feature_set("all") is None
    assert load_feature_set("tiny") == ["stock_id", "Returns_1h_mean", "Weekday"]
    with pytest.raises(ValueError, match=r"nope.*known.*all.*empty.*tiny.*twice"):
        load_feature_set("nope")
    with pytest.raises(ValueError, match="empty"):
        load_feature_set("empty")
    with pytest.raises(ValueError, match="Weekday.*twice|twice.*Weekday"):
        load_feature_set("twice")


def test_a_feature_set_limits_the_columns(frames, monkeypatch, tmp_path):
    f = frames["nonlinear"]
    chosen = ["Weekday", "stock_id", "Returns_1h_mean", "Volatility_5d"]
    _feature_sets(
        monkeypatch,
        tmp_path,
        tmp_test="# four columns, the stock code among them\n" + "\n".join(chosen) + "\n",
        no_code="Weekday\nReturns_1h_mean\n",
        unknown="Weekday\nstock_id\nNo_Such_Feature\n",
    )
    result = _model("nonlinear", feature_set="tmp_test").fit_predict_fold(f, FIT, JUNE)
    artifact = result.artifact
    assert artifact.feature_names() == chosen  # the set's own order
    assert "[categorical_feature: 1]" in artifact.booster.model_to_string()  # stock_id is 1
    assert len(result.predictions) > 0 and np.isfinite(result.predictions["prediction"]).all()
    # the artifact reads its columns out of the full frame, saved or not
    X = _keyed(f, JUNE)
    assert np.array_equal(artifact.predict(X), artifact.predict(X[chosen]))

    # stock_id="drop" removes the code from a set that lists it
    dropped = _model("nonlinear", feature_set="tmp_test", stock_id="drop")
    names = dropped.fit_predict_fold(f, FIT, JUNE).artifact.feature_names()
    assert names == [c for c in chosen if c != "stock_id"]

    # a column the features lack is refused, by name
    with pytest.raises(ValueError, match="No_Such_Feature"):
        _model("nonlinear", feature_set="unknown").fit_predict_fold(f, FIT, JUNE)
    # a categorical stock_id with a set that lacks it is refused, not silently dropped ...
    with pytest.raises(ValueError, match="stock_id"):
        _model("nonlinear", feature_set="no_code")
    # ... but a set without it is fine once the code is dropped
    ok = _model("nonlinear", feature_set="no_code", stock_id="drop")
    assert ok.fit_predict_fold(f, FIT, JUNE).artifact.feature_names() == [
        "Weekday",
        "Returns_1h_mean",
    ]
    with pytest.raises(ValueError, match=r"nope.*known"):
        _model("nonlinear", feature_set="nope")


def test_a_feature_set_is_refused_before_any_data_is_loaded(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    with pytest.raises(ValueError, match="nope"):
        train("nonlinear", tmp_path / "run", learned_data, overrides={"model.feature_set": "nope"})


def test_stock_id_and_feature_set_can_be_searched(learned_data, tmp_path, monkeypatch):
    _feature_sets(monkeypatch, tmp_path, small="stock_id\nReturns_1h_mean\nWeekday\n")
    grid = {"model.stock_id": ["categorical", "drop"], "model.feature_set": ["all", "small"]}
    out = tmp_path / "run"
    run = train("nonlinear", out, learned_data, grid=grid, overrides=SHORT_FOLDS)
    fold = json.loads((run / "folds" / "F2021-07" / "fold.json").read_text())
    assert len(fold["candidate_val_r2"]) == 4  # every pair of the two lists
    assert backtest(run, learned_data)["n_periods"] > 0


def test_linear_feature_set_limits_the_columns(frames, monkeypatch, tmp_path):
    f = frames["linear"]
    chosen = ["Weekday", "stock_id", "Returns_1h_mean", "Volatility_5d"]
    _feature_sets(
        monkeypatch,
        tmp_path,
        tmp_test="# four columns, the stock code among them\n" + "\n".join(chosen) + "\n",
        no_code="Weekday\nReturns_1h_mean\n",
        only_code="stock_id\n",
    )
    result = _model("linear", feature_set="tmp_test").fit_predict_fold(f, FIT, JUNE)
    artifact = result.artifact
    # the set's own order, less stock_id: linear never uses it, so a set naming it is fine
    assert artifact.feature_names() == [c for c in chosen if c != "stock_id"]
    assert len(artifact.coef) == 3
    assert len(result.predictions) > 0 and np.isfinite(result.predictions["prediction"]).all()
    # the artifact reads its columns out of the full frame, saved or not
    X = _keyed(f, JUNE)
    assert np.array_equal(artifact.predict(X), artifact.predict(X[chosen]))
    artifact.save(tmp_path / "model")
    loaded = type(artifact).load(tmp_path / "model")
    assert loaded.feature_names() == artifact.feature_names()
    assert np.array_equal(loaded.predict(X), artifact.predict(X))

    # a column outside the set has no say, in the fit or in the prediction
    other = f.X.copy()
    other["Returns_1h_std"] = other["Returns_1h_std"].to_numpy()[::-1]
    assert not np.array_equal(other["Returns_1h_std"], f.X["Returns_1h_std"], equal_nan=True)
    got = _model("linear", feature_set="tmp_test").fit_predict_fold(
        Frames(f.panel, other), FIT, JUNE
    )
    pd.testing.assert_frame_equal(got.predictions, result.predictions, check_exact=True)

    # a set without the code is the same fit as the set with it
    plain = _model("linear", feature_set="no_code").fit_predict_fold(f, FIT, JUNE)
    assert plain.artifact.feature_names() == ["Weekday", "Returns_1h_mean"]
    # a set that leaves nothing once stock_id is out is refused at construction
    with pytest.raises(ValueError, match="only_code.*stock_id|stock_id.*only_code"):
        _model("linear", feature_set="only_code")


def test_linear_unknown_set_is_refused_before_loading(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    with pytest.raises(ValueError, match=r"nope.*known"):
        _model("linear", feature_set="nope")
    with pytest.raises(ValueError, match="nope"):
        train("linear", tmp_path / "run", learned_data, overrides={"model.feature_set": "nope"})


def test_linear_missing_set_column_is_refused(frames, monkeypatch, tmp_path):
    _feature_sets(monkeypatch, tmp_path, unknown="Weekday\nstock_id\nNo_Such_Feature\n")
    with pytest.raises(ValueError, match="linear.*No_Such_Feature"):
        _model("linear", feature_set="unknown").fit_predict_fold(frames["linear"], FIT, JUNE)


def test_linear_default_set_is_bit_identical(frames, monkeypatch, tmp_path):
    f = frames["linear"]
    default = _model("linear").fit_predict_fold(f, FIT, JUNE)
    assert _model("linear").params["feature_set"] == "all"
    # the default fits every column but the code, in the frame's order, as it did before sets
    assert default.artifact.feature_names() == [c for c in f.X.columns if c != "stock_id"]
    # a set that lists every column (the code too) is the default fit, to the last bit
    _feature_sets(monkeypatch, tmp_path, everything="\n".join(f.X.columns) + "\n")
    listed = _model("linear", feature_set="everything").fit_predict_fold(f, FIT, JUNE)
    pd.testing.assert_frame_equal(default.predictions, listed.predictions, check_exact=True)
    assert np.array_equal(default.artifact.coef, listed.artifact.coef)
    assert default.artifact.intercept == listed.artifact.intercept
    assert (default.n_fit_rows, default.baseline_prediction, default.fit_label_mean) == (
        listed.n_fit_rows,
        listed.baseline_prediction,
        listed.fit_label_mean,
    )


def test_linear_feature_set_can_be_searched(learned_data, tmp_path, monkeypatch):
    _feature_sets(monkeypatch, tmp_path, small="stock_id\nReturns_1h_mean\nWeekday\n")
    grid = {"model.feature_set": ["all", "small"]}
    run = train("linear", tmp_path / "run", learned_data, grid=grid, overrides=SHORT_FOLDS)
    fold = json.loads((run / "folds" / "F2021-07" / "fold.json").read_text())
    assert len(fold["candidate_val_r2"]) == 2
    assert backtest(run, learned_data)["n_periods"] > 0


def test_default_added_parameters_change_nothing(frames, monkeypatch):
    import lightgbm as lgb

    real_dataset, real_train = lgb.Dataset, lgb.train
    seen = {}

    def spy_dataset(data, *args, **kwargs):
        seen["data"], seen["dataset_kwargs"] = data, kwargs
        return real_dataset(data, *args, **kwargs)

    def spy_train(params, train_set, *args, **kwargs):
        seen["params"], seen["train_kwargs"] = dict(params), kwargs
        return real_train(params, train_set, *args, **kwargs)

    monkeypatch.setattr(lgb, "Dataset", spy_dataset)
    monkeypatch.setattr(lgb, "train", spy_train)
    f = frames["nonlinear"]
    result = _model("nonlinear").fit_predict_fold(f, FIT, JUNE)

    before = {  # the parameters of a default fit before the added ones existed
        "objective": "regression_l1",
        "num_leaves": 31,
        "min_data_in_leaf": 20,
        "learning_rate": 0.1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "lambda_l1": 0.0,
        "lambda_l2": 0.0,
        "max_depth": -1,
        "min_gain_to_split": 0.0,
        "min_sum_hessian_in_leaf": 0.001,
        "max_bin": 255,
        "seed": 0,
        "deterministic": True,
        "force_row_wise": True,
        "num_threads": 8,
        "verbosity": -1,
    }
    categorical = {  # LightGBM 4's own defaults of its four categorical-split parameters
        "cat_smooth": 10.0,
        "min_data_per_group": 100,
        "max_cat_threshold": 32,
        "cat_l2": 10.0,
    }
    piece_wise_linear = {"linear_tree": False, "linear_lambda": 0.0}  # LightGBM 4's defaults
    noise_robust = {  # LightGBM 4's defaults of its four noise-robust split options
        "extra_trees": False,
        "path_smooth": 0.0,
        "feature_fraction_bynode": 1.0,
        "monotone_constraints_method": "basic",
    }
    # no constraint: the key itself is left out, not passed empty
    assert seen["params"] == {**before, **categorical, **piece_wise_linear, **noise_robust}
    assert "monotone_constraints" not in seen["params"]
    # init_model 'none' (and the other init_* keys) reach LightGBM in no form: no init_score, so
    # it boosts from its own start, and the artifact is the booster alone
    assert not any(key.startswith("init") for key in seen["params"])
    assert set(seen["dataset_kwargs"]) == {"label", "categorical_feature"}
    assert seen["dataset_kwargs"]["categorical_feature"] == ["stock_id"]
    assert seen["train_kwargs"] == {"num_boost_round": 100}
    assert type(result.artifact) is LGBMArtifact

    # the same fit without those ten keys (LightGBM fills in the same values) predicts the same
    data = seen["data"]
    plain = real_train(
        before,
        real_dataset(data, label=seen["dataset_kwargs"]["label"], categorical_feature=["stock_id"]),
        num_boost_round=100,
    )
    X = _keyed(f, JUNE)
    assert np.array_equal(
        plain.predict(X[plain.feature_name()], num_threads=8), result.artifact.predict(X)
    )
    assert plain.model_to_string() == result.artifact.booster.model_to_string()


POOL = ("AAA", "BBB", "CCC")


def _periods_of(f, window):
    return tuple(pd.Timestamp(p) for p in sorted(rows_in(f.panel, window)["period"].unique()))


def _early_stopping_capturing(monkeypatch, f, early_stop, **params):
    """A nonlinear fold's result under ``early_stop`` with the eval set its ``_fit`` was given."""
    seen = {}
    real = NonlinearModel._fit

    def spy(self, X_fit, y_fit, eval_set=None):
        seen["eval_set"] = eval_set
        return real(self, X_fit, y_fit, eval_set)

    monkeypatch.setattr(NonlinearModel, "_fit", spy)
    result = _model("nonlinear", **params).fit_predict_fold(f, FIT, JUNE, early_stop=early_stop)
    return result, seen["eval_set"]


@pytest.mark.parametrize(
    "label, symbols",
    [("raw", POOL), ("demean", ("AAA", "BBB"))],
)
def test_early_stopping_records_its_best_iteration(frames, label, symbols):
    f = frames["nonlinear"]
    periods = _periods_of(f, JUNE)
    stop = EarlyStop(periods, symbols, 50, 5)
    result = _model("nonlinear", label=label).fit_predict_fold(f, FIT, JUNE, early_stop=stop)
    assert 1 <= result.best_iteration <= 50
    assert result.best_iteration == result.artifact.best_iteration
    # the evaluation labels: demeaned over the pool's rows of the period (all three names), then
    # taken at the evaluation rows (here two of them)
    panel = f.panel
    pool = panel[panel["period"].isin(set(periods)) & np.isfinite(panel["y"].to_numpy(float))]
    label_ = (
        pool["y"] - pool.groupby("period")["y"].transform("mean")
        if label == "demean"
        else pool["y"]
    )
    rows = pool.assign(label=label_)[lambda d: d["symbol"].isin(symbols)]
    both = rows.merge(result.predictions, on=["period", "symbol"])
    assert len(both) == len(rows) > 0
    assert result.eval_mse == pytest.approx(
        np.mean((both["label"] - both["prediction"]) ** 2), rel=1e-6
    )


def test_early_stopping_drops_unlabelled_rows(frames, monkeypatch):
    f = frames["nonlinear"]
    last = f.panel["period"].max()
    assert f.panel.loc[f.panel["period"] == last, "y"].isna().all()  # no next open to exit at
    june = _periods_of(f, JUNE)
    result, eval_set = _early_stopping_capturing(
        monkeypatch, f, EarlyStop((*june, last), POOL, 50, 5)
    )
    X_eval, y_eval, _, _ = eval_set
    assert np.isfinite(y_eval).all()
    assert last not in set(X_eval.index.get_level_values("period"))
    assert np.isfinite(result.eval_mse)
    # the unlabelled period is as if it were not asked for
    without = _model("nonlinear").fit_predict_fold(f, FIT, JUNE, EarlyStop(june, POOL, 50, 5))
    assert (result.best_iteration, result.eval_mse) == (without.best_iteration, without.eval_mse)


@pytest.mark.parametrize(
    "periods, symbols",
    [
        ("last", POOL),  # the only period has no label
        ("june", ("ZZZ",)),  # no such name in the pool
        ("june", ()),
        ("none", POOL),
    ],
)
def test_no_evaluation_row_is_refused(frames, periods, symbols):
    f = frames["nonlinear"]
    chosen = {
        "last": (f.panel["period"].max(),),
        "june": _periods_of(f, JUNE),
        "none": (),
    }[periods]
    with pytest.raises(ValueError, match="early stopping: no labelled evaluation row"):
        _model("nonlinear").fit_predict_fold(f, FIT, JUNE, EarlyStop(chosen, symbols, 50, 5))


class _MeanArtifact(FitArtifact):
    FILENAME = "mean.txt"

    def __init__(self, mean: float, names: list[str]):
        self.mean, self.names = mean, names

    def predict(self, X):
        return np.full(len(X), self.mean)

    def feature_names(self):
        return self.names

    def save(self, directory):
        raise NotImplementedError

    @classmethod
    def load(cls, directory):
        raise NotImplementedError


class _TwoArgumentModel(LearnedModel):
    """A model written as the README's "Adding a model" has it: ``_fit`` takes two arguments."""

    loss = "mse"

    def _fit(self, X_fit, y_fit):
        return _MeanArtifact(float(np.mean(y_fit)), list(X_fit.columns))


def test_a_two_argument_fit_works_without_early_stopping(frames):
    f = frames["linear"]
    result = _TwoArgumentModel(hold_overnight=True).fit_predict_fold(f, FIT, JUNE)
    assert len(result.predictions) > 0
    assert result.predictions["prediction"].nunique() == 1
    # only a fit asked to stop early needs the third argument
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5)
    with pytest.raises(TypeError, match="positional argument"):
        _TwoArgumentModel(hold_overnight=True).fit_predict_fold(f, FIT, JUNE, early_stop=stop)


def test_linear_refuses_early_stopping(frames):
    f = frames["linear"]
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5)
    with pytest.raises(ValueError, match="linear.*early stopping"):
        _model("linear").fit_predict_fold(f, FIT, JUNE, early_stop=stop)


def test_without_early_stopping_nothing_changes(frames, monkeypatch):
    import lightgbm as lgb

    real_train = lgb.train
    calls = []

    def spy_train(params, train_set, *args, **kwargs):
        calls.append((dict(params), args, kwargs))
        return real_train(params, train_set, *args, **kwargs)

    monkeypatch.setattr(lgb, "train", spy_train)
    f = frames["nonlinear"]
    plain = _model("nonlinear").fit_predict_fold(f, FIT, JUNE)
    params, args, kwargs = calls[-1]
    assert "metric" not in params and args == ()
    assert kwargs == {"num_boost_round": 100}  # n_estimators; no valid_sets, no callbacks
    assert plain.best_iteration is None and plain.eval_mse is None
    assert plain.artifact.best_iteration is None

    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5)
    stopped = _model("nonlinear").fit_predict_fold(f, FIT, JUNE, early_stop=stop)
    params_es, args_es, kwargs_es = calls[-1]
    assert params_es == {**params, "metric": "l2"} and args_es == ()
    assert set(kwargs_es) == {"num_boost_round", "valid_sets", "callbacks"}
    assert kwargs_es["num_boost_round"] == 50 and len(kwargs_es["valid_sets"]) == 1
    assert len(kwargs_es["callbacks"]) == 1
    assert stopped.best_iteration is not None and stopped.eval_mse is not None


def test_early_stopping_reads_the_fit_columns(frames, monkeypatch):
    f = frames["nonlinear"]
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5)
    result, eval_set = _early_stopping_capturing(monkeypatch, f, stop, stock_id="drop")
    X_eval, _, given, cross_section = eval_set
    assert given is stop
    # the periods' whole cross-section is built only for a model that reads it (a hybrid)
    assert cross_section is None
    # the evaluation frame holds every feature column; LightGBM was handed only the fit's
    assert "stock_id" in X_eval.columns
    assert result.artifact.feature_names() == [c for c in f.X.columns if c != "stock_id"]
    assert result.best_iteration >= 1


@pytest.mark.parametrize("keep_training_booster", [False, True])
def test_a_saved_early_stopped_fit_predicts_the_same(
    frames, tmp_path, monkeypatch, keep_training_booster
):
    import lightgbm as lgb

    # lgb.train drops the rounds past the best one by default; kept, the booster holds them, and
    # the artifact itself must cut at the best round
    real_train = lgb.train
    monkeypatch.setattr(
        lgb,
        "train",
        lambda *a, **kw: real_train(*a, **kw, keep_training_booster=keep_training_booster),
    )
    f = frames["nonlinear"]
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5)
    artifact = _model("nonlinear").fit_predict_fold(f, FIT, JUNE, early_stop=stop).artifact
    best, X = artifact.best_iteration, _keyed(f, JUNE)
    assert (artifact.booster.num_trees() > best) == keep_training_booster
    assert np.array_equal(
        artifact.predict(X),
        np.asarray(artifact.booster.predict(X, num_iteration=best, num_threads=8)),
    )
    if keep_training_booster:  # the extra rounds would change the prediction
        every_round = artifact.booster.predict(X, num_iteration=-1, num_threads=8)
        assert not np.array_equal(artifact.predict(X), every_round)
    artifact.save(tmp_path / "model")
    loaded = type(artifact).load(tmp_path / "model")
    assert loaded.booster.num_trees() == best  # the file keeps the best rounds' trees only
    assert np.array_equal(loaded.predict(X), artifact.predict(X))


# ---- piece-wise linear trees (LightGBM's linear_tree, linear_lambda) ----

LINEAR_TREES = {"linear_tree": True, "loss": "mse", "num_leaves": 2, "n_estimators": 20}


def test_linear_trees_fit_a_linear_signal(frames):
    f = frames["nonlinear"]
    column, slope = "Volatility_5d", 0.5
    panel = f.panel.assign(y=slope * f.X[column].to_numpy())  # the label is linear in a feature
    signal = Frames(panel, f.X)
    X = _keyed(signal, JUNE)
    truth = slope * X[column].to_numpy()
    known = np.isfinite(truth)
    assert known.any()

    def rmse(**params):
        artifact = (
            _model("nonlinear", **{**LINEAR_TREES, **params})
            .fit_predict_fold(signal, FIT, JUNE)
            .artifact
        )
        predicted = artifact.predict(X)
        assert np.isfinite(predicted).all()
        return artifact.booster, np.sqrt(np.mean((predicted - truth)[known] ** 2))

    booster, linear = rmse()
    text = booster.model_to_string()
    assert "[linear_tree: 1]" in text and "is_linear=1" in text
    constant_booster, constant = rmse(linear_tree=False)
    assert "[linear_tree: 0]" in constant_booster.model_to_string()
    assert "is_linear=1" not in constant_booster.model_to_string()
    # two leaves per tree, 20 trees: a line per leaf follows the signal, a step per leaf cannot
    assert linear < constant / 2


def test_linear_lambda_reaches_lightgbm(frames):
    text = (
        _model("nonlinear", **LINEAR_TREES, linear_lambda=0.25)
        .fit_predict_fold(frames["nonlinear"], FIT, JUNE)
        .artifact.booster.model_to_string()
    )
    assert "[linear_lambda: 0.25]" in text


def test_linear_tree_with_l1_is_refused_before_loading(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    # the default loss is l1, which LightGBM cannot fit linear trees on
    with pytest.raises(ValueError, match="linear_tree.*l1"):
        _model("nonlinear", linear_tree=True)
    with pytest.raises(ValueError, match="linear_tree.*l1"):
        train("nonlinear", tmp_path / "run", learned_data, overrides={"model.linear_tree": True})
    # one candidate of a grid is enough
    with pytest.raises(ValueError, match="linear_tree.*l1"):
        train(
            "nonlinear", tmp_path / "run", learned_data, grid={"model.linear_tree": [False, True]}
        )
    # either alone is fine
    _model("nonlinear", linear_tree=True, loss="mse")
    _model("nonlinear", linear_tree=False, loss="l1")
    with pytest.raises(ValueError, match="linear_tree.*true or false"):
        _model("nonlinear", linear_tree="false", loss="mse")  # a string is truthy


def test_linear_tree_is_a_nonlinear_parameter_only():
    for name in ("linear_tree", "linear_lambda"):
        with pytest.raises(ValueError, match=f"unknown parameter.*{name}"):
            _model("linear", **{name: 0})


def test_nan_features_with_linear_trees(frames):
    f = frames["nonlinear"]
    # The natural NaNs are the rolling features' warm-up rows, which the fit window holds. Make
    # the frame harder: a third of every feature's values missing, and one feature missing
    # throughout. LightGBM predicts the leaf's constant output for a row that lacks a feature its
    # leaf's line uses, so such a row still gets a finite prediction.
    holes = np.random.default_rng(0).random(f.X.shape) < 0.3
    holes[:, list(f.X.columns).index("stock_id")] = False
    X = f.X.mask(holes)
    X["Return_20d"] = np.nan
    holed = Frames(f.panel, X)
    assert X.isna().any(axis=1).all()  # every row lacks something
    artifact = _model("nonlinear", **LINEAR_TREES).fit_predict_fold(holed, FIT, JUNE).artifact
    assert "is_linear=1" in artifact.booster.model_to_string()
    for frame in (holed, f):  # rows with missing features, then rows without
        june = _keyed(frame, JUNE)
        assert len(june) > 0 and np.isfinite(artifact.predict(june)).all()
    assert _keyed(f, JUNE).notna().all().all()  # the second set has none missing
    # the natural-NaN frame fits and predicts too, on its warm-up rows included
    natural = _model("nonlinear", **LINEAR_TREES).fit_predict_fold(f, FIT, JUNE)
    assert f.X.loc[rows_in(f.panel, FIT).index].isna().any(axis=1).any()
    assert len(natural.predictions) > 0 and np.isfinite(natural.predictions["prediction"]).all()


@pytest.mark.parametrize("early_stopping", [False, True])
def test_a_saved_linear_tree_fit_predicts_the_same(frames, tmp_path, early_stopping):
    f = frames["nonlinear"]
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5) if early_stopping else None
    artifact = (
        _model("nonlinear", **{**LINEAR_TREES, "linear_lambda": 0.1})
        .fit_predict_fold(f, FIT, JUNE, early_stop=stop)
        .artifact
    )
    # the early-stopping evaluation set is a second Dataset, built from the first's parameters
    assert (artifact.best_iteration is not None) == early_stopping
    assert "[linear_tree: 1]" in artifact.booster.model_to_string()
    X = _keyed(f, JUNE)
    artifact.save(tmp_path / "model")
    loaded = type(artifact).load(tmp_path / "model")
    assert "[linear_tree: 1]" in loaded.booster.model_to_string()
    assert loaded.feature_names() == artifact.feature_names()
    assert np.array_equal(loaded.predict(X), artifact.predict(X))


# ---- noise-robust splits and monotone constraints (LightGBM's extra_trees, path_smooth,
# feature_fraction_bynode, monotone_constraints, monotone_constraints_method) ----


def _model_text(frames, **params):
    return (
        _model("nonlinear", **params)
        .fit_predict_fold(frames["nonlinear"], FIT, JUNE)
        .artifact.booster.model_to_string()
    )


def test_extra_trees_and_bynode_reach_lightgbm(frames):
    text = _model_text(
        frames, extra_trees=True, feature_fraction_bynode=0.7, path_smooth=0.5, min_data_in_leaf=5
    )
    assert "[extra_trees: 1]" in text
    assert "[feature_fraction_bynode: 0.7]" in text
    assert "[path_smooth: 0.5]" in text
    default = _model_text(frames, min_data_in_leaf=5)
    assert "[extra_trees: 0]" in default and "[feature_fraction_bynode: 1]" in default
    assert "[path_smooth: 0]" in default and "[monotone_constraints_method: basic]" in default
    # the options are not only recorded: they grow other trees
    trees = lambda t: t[t.index("Tree=0") : t.index("end of trees")]  # noqa: E731
    assert trees(text) != trees(default)


@pytest.mark.parametrize("method", ["basic", "intermediate", "advanced"])
def test_monotone_constraints_method_reaches_lightgbm(frames, method):
    text = _model_text(
        frames,
        loss="mse",
        monotone_constraints={"Return_1d": 1},
        monotone_constraints_method=method,
    )
    assert f"[monotone_constraints_method: {method}]" in text


def _monotone_frames(frames):
    """The frames with a label that is a wave in two features, so that a fit left free is not
    monotone in either: ``Volatility_5d`` (the constraint +1) and ``Return_1d`` (-1)."""
    f = frames["nonlinear"]

    def wave(column, cycles):
        v = f.X[column].to_numpy(float)
        return np.sin(2 * np.pi * cycles * (v - np.nanmin(v)) / np.ptp(v[np.isfinite(v)]))

    return Frames(
        f.panel.assign(y=0.01 * wave("Volatility_5d", 2) + 0.01 * wave("Return_1d", 2)), f.X
    )


MONOTONE = {"loss": "mse", "num_leaves": 8, "min_data_in_leaf": 5, "n_estimators": 40}
SIGNS = {"Volatility_5d": 1, "Return_1d": -1}


def _along(artifact, X, column, n=60):
    """The predictions of ``X``'s rows with ``column`` alone set to each of ``n`` values over its
    range, one row of the result per base row."""
    values = X[column].dropna()
    grid = np.linspace(values.min(), values.max(), n)
    out = []
    for _, row in X.iloc[:8].iterrows():
        probe = pd.DataFrame([row] * n).reset_index(drop=True).astype(X.dtypes.to_dict())
        probe[column] = grid
        out.append(artifact.predict(probe))
    return np.array(out)


def test_monotone_constraints_are_aligned_with_the_fit_columns(frames, monkeypatch, tmp_path):
    # a set in an order of its own: the constraint list follows the fit's columns, not the frame's
    _feature_sets(
        monkeypatch, tmp_path, ordered="Volatility_5d\nstock_id\nReturn_1d\nVolume_MA_ratio\n"
    )
    f = _monotone_frames(frames)
    params = {**MONOTONE, "feature_set": "ordered"}
    artifact = (
        _model("nonlinear", **params, monotone_constraints=SIGNS)
        .fit_predict_fold(f, FIT, JUNE)
        .artifact
    )
    assert artifact.feature_names() == ["Volatility_5d", "stock_id", "Return_1d", "Volume_MA_ratio"]
    text = artifact.booster.model_to_string()
    assert "\nmonotone_constraints=1 0 -1 0\n" in text  # the booster's own, per column
    assert "[monotone_constraints: 1,0,-1,0]" in text
    # monotone in each constrained feature when it alone varies, and by the constraint's sign
    X = _keyed(f, JUNE)
    up = _along(artifact, X, "Volatility_5d")
    down = _along(artifact, X, "Return_1d")
    assert (np.diff(up, axis=1) >= -1e-12).all() and (np.diff(down, axis=1) <= 1e-12).all()
    assert np.ptp(up, axis=1).min() > 0 and np.ptp(down, axis=1).min() > 0
    # the same fit left free is not monotone in the first (so the check has power)
    free = _model("nonlinear", **params).fit_predict_fold(f, FIT, JUNE).artifact
    assert "monotone_constraints=" not in free.booster.model_to_string()
    steps = np.diff(_along(free, X, "Volatility_5d"), axis=1)
    assert (steps < -1e-6).any() and (steps > 1e-6).any()


def test_monotone_constraints_on_all_columns_follow_the_frame(frames):
    f = _monotone_frames(frames)
    artifact = (
        _model("nonlinear", **MONOTONE, monotone_constraints={"Return_1d": -1, "Weekday": 1})
        .fit_predict_fold(f, FIT, JUNE)
        .artifact
    )
    names = artifact.feature_names()
    signs = {"Return_1d": -1, "Weekday": 1}
    want = " ".join(str(signs.get(name, 0)) for name in names)
    assert f"\nmonotone_constraints={want}\n" in artifact.booster.model_to_string()
    assert len(names) == f.X.shape[1] and want.count("-1") == 1


def test_zero_signs_pass_no_constraint(frames):
    # 0 is "unconstrained": a mapping of zeros is the unconstrained fit, to the last bit
    f = frames["nonlinear"]
    free = _model("nonlinear").fit_predict_fold(f, FIT, JUNE)
    zeros = _model("nonlinear", monotone_constraints={"Return_1d": 0, "stock_id": 0})
    zeroed = zeros.fit_predict_fold(f, FIT, JUNE)
    assert zeroed.artifact.booster.model_to_string() == free.artifact.booster.model_to_string()
    pd.testing.assert_frame_equal(free.predictions, zeroed.predictions, check_exact=True)


def test_monotone_constraint_on_a_missing_feature_is_refused(frames, monkeypatch, tmp_path):
    f = frames["nonlinear"]
    mse = {"loss": "mse", "n_estimators": 5}  # the default l1 refuses any constraint earlier
    missing = "monotone_constraints.*does not train on.*'{}'"
    with pytest.raises(ValueError, match=missing.format("Return_1x")):
        _model(
            "nonlinear", **mse, monotone_constraints={"Return_1d": -1, "Return_1x": 1}
        ).fit_predict_fold(f, FIT, JUNE)
    # a typo is refused whatever its sign, since the sign 0 would hide it
    with pytest.raises(ValueError, match=missing.format("Return_1x")):
        _model("nonlinear", **mse, monotone_constraints={"Return_1x": 0}).fit_predict_fold(
            f, FIT, JUNE
        )
    # a real feature the fit does not read: outside its feature set, or the dropped code
    _feature_sets(monkeypatch, tmp_path, small="stock_id\nReturn_1d\n")
    with pytest.raises(ValueError, match=missing.format("Weekday")):
        _model(
            "nonlinear", **mse, feature_set="small", monotone_constraints={"Weekday": 1}
        ).fit_predict_fold(f, FIT, JUNE)
    with pytest.raises(ValueError, match=missing.format("stock_id")):
        _model(
            "nonlinear", **mse, stock_id="drop", monotone_constraints={"stock_id": 0}
        ).fit_predict_fold(f, FIT, JUNE)
    # the features the fit does train on are fine, set or not
    _model(
        "nonlinear", **mse, feature_set="small", monotone_constraints={"Return_1d": 1}
    ).fit_predict_fold(f, FIT, JUNE)


def test_bad_monotone_value_is_refused_before_loading(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    for bad in (2, -2, 0.5, 1.0, True, "1", None):
        with pytest.raises(ValueError, match="monotone_constraints.*Return_1d"):
            _model("nonlinear", monotone_constraints={"Return_1d": bad})
    for bad in ([("Return_1d", 1)], ["Return_1d"], "Return_1d", 1, None):
        with pytest.raises(ValueError, match="monotone_constraints must be a mapping"):
            _model("nonlinear", monotone_constraints=bad)
    with pytest.raises(ValueError, match="monotone_constraints.*feature names"):
        _model("nonlinear", monotone_constraints={1: -1})
    with pytest.raises(ValueError, match="monotone_constraints.*Return_1d"):
        train(
            "nonlinear",
            tmp_path / "run",
            learned_data,
            overrides={"model.monotone_constraints": {"Return_1d": 2}},
        )
    with pytest.raises(ValueError, match="monotone_constraints.*Return_1d"):  # one grid candidate
        train(
            "nonlinear",
            tmp_path / "run",
            learned_data,
            grid={"model.monotone_constraints": [{}, {"Return_1d": 2}]},
        )
    for good in ({}, {"Return_1d": -1}, {"Return_1d": 0, "Weekday": 1}):
        _model("nonlinear", loss="mse", monotone_constraints=good)
    # the method is one of LightGBM's three (it takes any string without complaint)
    with pytest.raises(ValueError, match="monotone_constraints_method.*'quick'"):
        _model("nonlinear", monotone_constraints_method="quick")


def test_monotone_constraints_cannot_cover_unsupported_fits(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    # LightGBM refuses a constraint on regression_l1, the default loss, once the fit starts
    with pytest.raises(ValueError, match="monotone_constraints.*loss='l1'"):
        _model("nonlinear", monotone_constraints={"Return_1d": 1})
    with pytest.raises(ValueError, match="monotone_constraints.*loss='l1'"):
        train(
            "nonlinear",
            tmp_path / "run",
            learned_data,
            overrides={"model.monotone_constraints": {"Return_1d": 1}},
        )
    _model("nonlinear", loss="mse", monotone_constraints={"Return_1d": 1})
    _model("nonlinear", loss="l1", monotone_constraints={"Return_1d": 0})  # no constraint
    # LightGBM aborts the process on a constraint on a categorical feature
    with pytest.raises(ValueError, match="stock_id.*categorical"):
        _model("nonlinear", loss="mse", monotone_constraints={"stock_id": 1})
    _model("nonlinear", stock_id="drop", monotone_constraints={})
    _model("nonlinear", monotone_constraints={"stock_id": 0})
    # LightGBM enforces the constraint when it chooses the splits, not in a leaf's linear model
    with pytest.raises(ValueError, match="monotone_constraints.*linear_tree"):
        _model("nonlinear", loss="mse", linear_tree=True, monotone_constraints={"Return_1d": 1})
    _model("nonlinear", loss="mse", linear_tree=True, monotone_constraints={"Return_1d": 0})


def test_monotone_constraints_are_a_nonlinear_parameter_only():
    for name in ("extra_trees", "path_smooth", "feature_fraction_bynode", "monotone_constraints"):
        with pytest.raises(ValueError, match=f"unknown parameter.*{name}"):
            _model("linear", **{name: 0})


def test_monotone_constraints_reach_the_run_files(learned_data, tmp_path, monkeypatch):
    import sys

    from ml4trading import train as train_module
    from ml4trading.config import RunConfig

    # --set parses its value as YAML: a flow mapping is a dict, of strings to integers
    seen = {}
    monkeypatch.setattr(train_module, "train", lambda *a, **kw: seen.update(kw))
    monkeypatch.setattr(
        sys,
        "argv",
        ["train", "--method", "nonlinear", "--out", str(tmp_path / "x"), "--set"]
        + ["model.monotone_constraints={Return_1d: -1, Weekday: +1}"],
    )
    train_module.main()
    constraints = {"Return_1d": -1, "Weekday": 1}
    assert seen["overrides"] == {"model.monotone_constraints": constraints}

    run = train(
        "nonlinear",
        tmp_path / "run",
        learned_data,
        grid={"model.monotone_constraints": [{}, constraints]},
        overrides={**seen["overrides"], **SHORT_FOLDS, "model.loss": "mse"},
    )
    meta = json.loads((run / "run.json").read_text())
    assert meta["config"]["model"]["monotone_constraints"] == constraints
    assert meta["grid"] == {"model.monotone_constraints": [{}, constraints]}
    assert RunConfig.from_dict(meta["config"]).model["monotone_constraints"] == constraints
    fold = json.loads((run / "folds" / "F2021-07" / "fold.json").read_text())
    assert fold["model_params"]["monotone_constraints"] == [{}, constraints][fold["chosen"]]
    assert len(fold["candidate_val_net_sharpe"]) == 2
    # the saved booster of the constrained candidate carries the constraints
    text = (run / "folds" / "F2021-07" / "model" / "model.txt").read_text()
    assert ("monotone_constraints=" in text) == (fold["chosen"] == 1)
    assert backtest(run, learned_data)["n_periods"] > 0


# ---- boosting from a linear initial score (init_model: Friedman 2001's initial estimate F0,
# handed to LightGBM as the Dataset's init_score) ----


def _hybrid(f, early_stop=None, **params):
    """A nonlinear fold on ``FIT`` -> ``JUNE``, boosted from the linear initial score."""
    model = _model("nonlinear", init_model="linear", **params)
    return model.fit_predict_fold(f, FIT, JUNE, early_stop=early_stop)


def _datasets_capturing(monkeypatch):
    """The keyword arguments of every ``lgb.Dataset`` a fit builds, in order."""
    import lightgbm as lgb

    real_dataset, made = lgb.Dataset, []

    def spy_dataset(data, *args, **kwargs):
        made.append(kwargs)
        return real_dataset(data, *args, **kwargs)

    monkeypatch.setattr(lgb, "Dataset", spy_dataset)
    return made


def _linear_signal(frames):
    """The frames with a label linear in ``Volatility_5d``, plus noise: the linear part explains
    most of it, so the trees boosted from it have little of their own to add."""
    f = frames["nonlinear"]
    noise = np.random.default_rng(0).normal(0.0, 0.002, len(f.panel))
    return Frames(f.panel.assign(y=0.5 * f.X["Volatility_5d"].to_numpy() + noise), f.X)


def test_hybrid_predicts_linear_plus_booster(frames, monkeypatch, tmp_path):
    from ml4trading.models.learned.nonlinear import HybridArtifact

    f = frames["nonlinear"]
    _feature_sets(monkeypatch, tmp_path, small="stock_id\nReturn_1d\nVolatility_5d\nWeekday\n")
    result = _hybrid(f, init_alpha_ratio=0.05, init_l1_ratio=0.3, init_feature_set="small")
    artifact = result.artifact
    assert isinstance(artifact, HybridArtifact)
    # the linear part is linear's own fit, with the same penalty and columns, on the same rows and
    # labels: an elastic net fitted separately here by the linear method
    net = _model("linear", alpha_ratio=0.05, l1_ratio=0.3, feature_set="small")
    linear = net.fit_predict_fold(f, FIT, JUNE).artifact
    assert artifact.init.feature_names() == linear.feature_names()
    assert linear.feature_names() == ["Return_1d", "Volatility_5d", "Weekday"]
    assert np.array_equal(artifact.init.coef, linear.coef) and (linear.coef != 0).any()
    assert artifact.init.intercept == linear.intercept
    # the prediction is that net's plus the trees' (LightGBM's own predict, which leaves the
    # init_score out), and the fold trades that sum
    X = _keyed(f, JUNE)
    trees = artifact.booster.predict(X[artifact.booster.feature_name()], num_threads=8)
    assert np.abs(artifact.predict(X) - (linear.predict(X) + trees)).max() <= 1e-12
    assert np.array_equal(result.predictions["prediction"].to_numpy(), artifact.predict(X))
    assert np.abs(trees).max() > 1e-6  # the trees do add something


def test_the_linear_part_fits_the_labels_the_booster_fits(frames, monkeypatch):
    from ml4trading.models.learned.linear import fit_elastic_net

    made = _datasets_capturing(monkeypatch)
    # fit units far from the raw return: demeaned, winsorized and per unit of volatility
    params = {"init_model": "linear", "label": "demean_winsor", "risk_adjust": "vol"}
    result, X_fit, y_fit = _fit_capturing(monkeypatch, frames["nonlinear"], **params)
    columns = [c for c in X_fit.columns if c != "stock_id"]
    max_iter = config.METHOD_DEFAULTS["linear"]["model"]["max_iter"]  # linear's own cap
    want = fit_elastic_net(X_fit[columns], y_fit, 0.01, 0.5, max_iter)
    init = result.artifact.init
    assert np.array_equal(init.coef, want.coef) and init.intercept == want.intercept
    # LightGBM starts from that net's in-sample prediction, on the same rows and labels
    (fitted,) = made
    assert np.array_equal(fitted["label"], y_fit)
    assert np.array_equal(fitted["init_score"], want.predict(X_fit))


@pytest.mark.parametrize("loss", ["l1", "mse"])
def test_hybrid_with_zero_rounds_equals_linear(frames, loss):
    # LightGBM needs a round and a learning rate above 0: one round at 1e-12 adds next to nothing,
    # and from an initial score LightGBM adds no starting constant of its own
    f = frames["nonlinear"]
    hybrid = _hybrid(f, loss=loss, n_estimators=1, learning_rate=1e-12).predictions
    linear = _model("linear").fit_predict_fold(f, FIT, JUNE).predictions
    assert len(hybrid) == len(linear) > 0
    pd.testing.assert_frame_equal(hybrid[["period", "symbol"]], linear[["period", "symbol"]])
    assert np.abs(hybrid["prediction"] - linear["prediction"]).max() <= 1e-9
    # the same round without the initial score starts from LightGBM's constant instead
    plain = _model("nonlinear", loss=loss, n_estimators=1, learning_rate=1e-12)
    constant = plain.fit_predict_fold(f, FIT, JUNE).predictions["prediction"]
    assert np.abs(constant - linear["prediction"]).max() > 1e-6


@pytest.mark.parametrize("early_stopping", [False, True])
def test_hybrid_save_load_round_trip(frames, tmp_path, early_stopping):
    from ml4trading.models.learned.linear import LinearArtifact
    from ml4trading.models.learned.nonlinear import HybridArtifact

    f = frames["nonlinear"]
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5) if early_stopping else None
    artifact = _hybrid(f, early_stop=stop).artifact
    assert (artifact.best_iteration is not None) == early_stopping
    saved = tmp_path / "model"
    artifact.save(saved)
    files = sorted(str(p.relative_to(saved)) for p in saved.rglob("*") if p.is_file())
    assert files == ["init/model.npz", "model.txt"]
    loaded = type(artifact).load(saved)
    assert isinstance(loaded, HybridArtifact) and loaded.best_iteration is None
    X = _keyed(f, JUNE)
    assert loaded.feature_names() == artifact.feature_names()
    assert np.array_equal(loaded.predict(X), artifact.predict(X))
    # each part in its own format: the booster's text (the trees alone) and linear's file
    trees, net = LGBMArtifact.load(saved), LinearArtifact.load(saved / "init")
    assert np.array_equal(net.coef, artifact.init.coef) and net.intercept == artifact.init.intercept
    assert np.array_equal(loaded.predict(X), net.predict(X) + trees.predict(X))


def test_hybrid_early_stopping_uses_the_linear_init_on_eval_rows(frames, monkeypatch):
    made = _datasets_capturing(monkeypatch)
    f = _linear_signal(frames)
    stop = EarlyStop(_periods_of(f, JUNE), ("AAA", "BBB"), 50, 5)
    result, (X_eval, y_eval, _, cross_section) = _early_stopping_capturing(
        monkeypatch, f, stop, init_model="linear", loss="mse"
    )
    artifact = result.artifact
    assert 1 <= result.best_iteration <= 50
    # the evaluation Dataset starts from the linear part's prediction on its own rows, as the
    # training one does on the fit rows: read over every pool row of the evaluation periods, as
    # the fold's prediction reads them, then taken at the evaluation rows
    _, evaluated = made
    assert evaluated["reference"] is not None
    assert np.array_equal(evaluated["label"], y_eval)
    assert cross_section.index.equals(_keyed(f, JUNE).index)
    whole = artifact.init.predict(cross_section)
    assert np.array_equal(
        evaluated["init_score"], whole[cross_section.index.get_indexer(X_eval.index)]
    )
    # so the error it stopped on is the hybrid's, the linear part plus the kept rounds (LightGBM
    # holds the labels in float32, hence the tolerance) ...
    hybrid = np.mean((y_eval - artifact.predict(X_eval)) ** 2)
    assert result.eval_mse == pytest.approx(hybrid, rel=1e-6)
    # ... not the trees' alone, which leave the linear signal out
    trees = artifact.booster.predict(
        X_eval[artifact.booster.feature_name()], num_iteration=result.best_iteration, num_threads=8
    )
    assert np.mean((y_eval - trees) ** 2) > 2 * result.eval_mse
    # and it is the error of the fold's own predictions on those rows
    panel = f.panel
    rows = panel[panel["period"].isin(set(stop.periods)) & panel["symbol"].isin(stop.symbols)]
    both = rows.merge(result.predictions, on=["period", "symbol"])
    assert len(both) == len(y_eval) > 0
    assert result.eval_mse == pytest.approx(
        np.mean((both["y"] - both["prediction"]) ** 2), rel=1e-6
    )


@pytest.mark.parametrize("params", [{}, {"risk_adjust": "vol", "smooth_halflife": 3}])
def test_hybrid_early_stopping_error_is_the_folds_own(frames, monkeypatch, params):
    # values missing in the evaluation periods: the net fills each with the median of its
    # period's rows, which the fold's own prediction takes over the period's whole cross-section,
    # not over the evaluation rows (here two of the three names, and only the labelled ones)
    f = _linear_signal(frames)
    june = f.panel.index.isin(rows_in(f.panel, JUNE).index)
    holes = june & (np.random.default_rng(1).random(len(f.panel)) < 0.3)
    X = f.X.copy()
    X.loc[holes, "Volatility_5d"] = np.nan
    f = Frames(f.panel, X)
    stop = EarlyStop(_periods_of(f, JUNE), ("AAA", "BBB"), 50, 5)
    result, (X_eval, y_eval, _, _) = _early_stopping_capturing(
        monkeypatch, f, stop, init_model="linear", loss="mse", **params
    )
    assert X_eval["Volatility_5d"].isna().any() and result.best_iteration is not None
    # the fold's own output on the evaluation rows, before any scaling or smoothing: the units
    # the evaluation labels are in
    own = result.fit_predictions if params else result.predictions["prediction"]
    keys = pd.MultiIndex.from_frame(result.predictions[["period", "symbol"]])
    at = pd.Series(own.to_numpy(), index=keys).reindex(X_eval.index).to_numpy()
    assert len(at) > 0 and np.isfinite(at).all()
    # LightGBM holds the labels in float32: against those, the error it stopped on is the fold's
    # own to 1e-9, and against the float64 labels to float32's precision
    held = y_eval.astype(np.float32).astype(float)
    assert result.eval_mse == pytest.approx(np.mean((held - at) ** 2), rel=0, abs=1e-9)
    assert result.eval_mse == pytest.approx(np.mean((y_eval - at) ** 2), rel=1e-6)


def test_a_hybrid_without_early_stopping_trains_the_booster_from_the_net(frames, monkeypatch):
    # a hybrid fit without early stopping is the net's fit, then one booster trained from its
    # in-sample prediction for n_estimators rounds: nothing of the early-stopping set reaches it
    import lightgbm as lgb

    from ml4trading.models.learned.linear import fit_elastic_net

    real_train, calls = lgb.train, []

    def spy_train(params, train_set, *args, **kwargs):
        calls.append((dict(params), args, kwargs))
        return real_train(params, train_set, *args, **kwargs)

    monkeypatch.setattr(lgb, "train", spy_train)
    made = _datasets_capturing(monkeypatch)
    params = {
        "init_model": "linear",
        "loss": "mse",
        "risk_adjust": "vol",
        "stock_id": "drop",
        "extra_trees": True,
        "feature_fraction": 0.5,
        "bagging_fraction": 0.5,
        "bagging_freq": 1,
        "n_estimators": 30,
    }
    result, X_fit, y_fit = _fit_capturing(monkeypatch, frames["nonlinear"], **params)
    ((given, args, kwargs),) = calls
    assert "metric" not in given and args == () and kwargs == {"num_boost_round": 30}
    (fitted,) = made
    assert set(fitted) == {"label", "categorical_feature", "init_score"}
    # the same booster, trained here by hand from the same pieces
    columns = [c for c in X_fit.columns if c != "stock_id"]
    max_iter = config.METHOD_DEFAULTS["linear"]["model"]["max_iter"]
    net = fit_elastic_net(X_fit[columns], y_fit, 0.01, 0.5, max_iter)
    data = lgb.Dataset(
        X_fit[columns], label=y_fit, categorical_feature=[], init_score=net.predict(X_fit)
    )
    by_hand = real_train(given, data, num_boost_round=30)
    assert result.artifact.booster.model_to_string() == by_hand.model_to_string()


def test_a_smoothed_risk_adjusted_horizon_hybrid_fits(frames):
    # the hybrid with a multi-session label, the volatility scaling and the smoothing at once
    result = _hybrid(frames["nonlinear"], smooth_halflife=3, horizon=5, risk_adjust="vol")
    assert len(result.predictions) > 0 and np.isfinite(result.predictions["prediction"]).all()
    assert np.isfinite(result.fit_predictions).all()
    assert result.horizon == 5 and result.risk_adjust == "vol"


def test_init_model_none_ignores_the_init_parameters(frames, monkeypatch, tmp_path):
    _feature_sets(monkeypatch, tmp_path, small="Return_1d\nWeekday\n")
    f = frames["nonlinear"]
    default = _model("nonlinear").fit_predict_fold(f, FIT, JUNE)
    other = _model(
        "nonlinear",
        init_model="none",
        init_alpha_ratio=0.3,
        init_l1_ratio=0.9,
        init_feature_set="small",
    ).fit_predict_fold(f, FIT, JUNE)
    assert type(other.artifact) is LGBMArtifact
    assert other.artifact.booster.model_to_string() == default.artifact.booster.model_to_string()
    pd.testing.assert_frame_equal(default.predictions, other.predictions, check_exact=True)


def test_the_two_parts_read_their_own_columns(frames, monkeypatch, tmp_path):
    f = frames["nonlinear"]
    _feature_sets(
        monkeypatch,
        tmp_path,
        trees="stock_id\nReturn_1d\nWeekday\n",
        net="Volatility_5d\nstock_id\nReturn_1d\n",
        unknown="Return_1d\nNo_Such_Feature\n",
    )
    params = {"feature_set": "trees", "init_feature_set": "net"}
    result = _hybrid(f, **params)
    artifact = result.artifact
    assert artifact.booster.feature_name() == ["stock_id", "Return_1d", "Weekday"]
    assert artifact.init.feature_names() == ["Volatility_5d", "Return_1d"]  # less the code
    # the artifact's columns: the booster's, then those only the linear part reads
    assert artifact.feature_names() == ["stock_id", "Return_1d", "Weekday", "Volatility_5d"]
    X = _keyed(f, JUNE)
    assert np.array_equal(artifact.predict(X), artifact.predict(X[artifact.feature_names()]))
    # a column neither part reads has no say, in the fit or in the prediction
    other = f.X.copy()
    other["Returns_1h_std"] = other["Returns_1h_std"].to_numpy()[::-1]
    got = _hybrid(Frames(f.panel, other), **params)
    pd.testing.assert_frame_equal(got.predictions, result.predictions, check_exact=True)
    # a column the features lack is refused, by name
    with pytest.raises(ValueError, match="init_feature_set.*No_Such_Feature"):
        _hybrid(f, init_feature_set="unknown")


def test_hybrid_native_importance_covers_the_trees_only(frames, monkeypatch, tmp_path):
    from ml4trading.importance import importance_table

    f = frames["nonlinear"]
    _feature_sets(monkeypatch, tmp_path, trees="stock_id\nReturn_1d\nWeekday\n")
    artifact = _hybrid(f, feature_set="trees").artifact
    X = _keyed(f, JUNE)
    native = artifact.native_importance(X)
    assert list(native.index) == artifact.feature_names()
    # the booster's gain, split count and TreeSHAP, as the booster alone reports them; nothing for
    # a column only the linear part reads
    trees = LGBMArtifact(artifact.booster).native_importance(X)
    pd.testing.assert_frame_equal(native.loc[trees.index], trees, check_dtype=False)
    only_linear = native.index.difference(trees.index)
    assert len(only_linear) > 0 and native.loc[only_linear].isna().all().all()
    # the permutation measures cover every column the hybrid reads
    y = rows_in(f.panel, JUNE)["y"].to_numpy(float)
    table = importance_table(artifact, X, y, y)
    assert list(table.index) == artifact.feature_names()
    assert np.isfinite(table["perm_mse"]).all()


def test_risk_scaling_applies_to_the_sum(frames):
    f = frames["nonlinear"]
    result = _hybrid(f, risk_adjust="vol")
    X = _keyed(f, JUNE)
    output = pd.Series(result.artifact.predict(X), index=X.index)
    keys = pd.MultiIndex.from_frame(result.predictions[["period", "symbol"]])
    # the model's own output, before the scaling to returns, is the linear part plus the trees
    assert len(keys) > 0
    assert np.array_equal(result.fit_predictions.to_numpy(), output.loc[keys].to_numpy())


@pytest.mark.parametrize("early_stopping", [False, True])
def test_linear_trees_can_boost_from_the_linear_score(frames, tmp_path, early_stopping):
    f = frames["nonlinear"]
    stop = EarlyStop(_periods_of(f, JUNE), POOL, 50, 5) if early_stopping else None
    result = _hybrid(f, early_stop=stop, **LINEAR_TREES)
    artifact = result.artifact
    assert "[linear_tree: 1]" in artifact.booster.model_to_string()
    assert len(result.predictions) > 0 and np.isfinite(result.predictions["prediction"]).all()
    artifact.save(tmp_path / "model")
    X = _keyed(f, JUNE)
    assert np.array_equal(type(artifact).load(tmp_path / "model").predict(X), artifact.predict(X))


def test_monotone_with_init_model_is_refused(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    # a constraint binds the trees, not the linear initial score they are added to
    constrained = {"loss": "mse", "monotone_constraints": {"Return_1d": 1}}
    with pytest.raises(ValueError, match="monotone_constraints.*init_model"):
        _model("nonlinear", init_model="linear", **constrained)
    with pytest.raises(ValueError, match="monotone_constraints.*init_model"):
        train(
            "nonlinear",
            tmp_path / "run",
            learned_data,
            grid={"model.init_model": ["none", "linear"]},  # one candidate is enough
            overrides={f"model.{k}": v for k, v in constrained.items()},
        )
    # either alone is fine, and so is a mapping that sets no sign
    _model("nonlinear", **constrained)
    _model("nonlinear", init_model="linear", loss="mse")
    _model("nonlinear", init_model="linear", monotone_constraints={"Return_1d": 0})


def test_bad_init_parameters_are_refused_before_loading(learned_data, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the model was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    _feature_sets(monkeypatch, tmp_path, only_code="stock_id\n")
    for bad, message in (
        ({"init_model": "ridge"}, "init_model.*'ridge'"),
        ({"init_model": None}, "init_model.*None"),
        ({"init_alpha_ratio": 0.0}, "init_alpha_ratio"),
        ({"init_alpha_ratio": -0.1}, "init_alpha_ratio"),
        ({"init_l1_ratio": 0.0}, "init_l1_ratio"),
        ({"init_l1_ratio": 1.5}, "init_l1_ratio"),
        ({"init_feature_set": "nope"}, "nope.*known"),
        ({"init_feature_set": "only_code"}, "init_feature_set.*only_code.*stock_id"),
    ):
        with pytest.raises(ValueError, match=message):  # each refused before any data loads
            _model("nonlinear", **{"init_model": "linear", **bad})
    with pytest.raises(ValueError, match="init_model.*'ridge'"):
        train("nonlinear", tmp_path / "run", learned_data, overrides={"model.init_model": "ridge"})
    with pytest.raises(ValueError, match="init_alpha_ratio"):
        train(
            "nonlinear",
            tmp_path / "run",
            learned_data,
            grid={"model.init_alpha_ratio": [0.01, 0.0]},
        )
    # the parameters are nonlinear's: linear has no initial score
    for name in ("init_model", "init_alpha_ratio", "init_l1_ratio", "init_feature_set"):
        with pytest.raises(ValueError, match=f"unknown parameter.*{name}"):
            _model("linear", **{name: "none"})
    _model("nonlinear", init_model="linear", init_alpha_ratio=1.0, init_l1_ratio=1.0)


def test_a_hybrid_run_saves_both_parts(learned_data, tmp_path, monkeypatch):
    from ml4trading.diagnose import diagnose

    # the trees read three columns, the linear part every one but the code
    _feature_sets(monkeypatch, tmp_path, trees="stock_id\nReturn_1d\nWeekday\n")
    run = train(
        "nonlinear",
        tmp_path / "run",
        learned_data,
        overrides={**SHORT_FOLDS, "model.init_model": "linear", "model.feature_set": "trees"},
        importance=True,
    )
    meta = json.loads((run / "run.json").read_text())
    assert meta["config"]["model"]["init_model"] == "linear"
    folds = sorted((run / "folds").iterdir())
    assert len(folds) > 0
    for fdir in folds:
        model = fdir / "model"
        files = sorted(str(p.relative_to(model)) for p in model.rglob("*") if p.is_file())
        assert files == ["init/model.npz", "model.txt"]
        table = pd.read_parquet(fdir / "val" / "importance.parquet").set_index("feature")
        assert table.index.is_unique and np.isfinite(table["perm_mse"]).all()
        assert table["gain"].notna().sum() == 3 and len(table) > 3  # the trees' columns only
    # the run's diagnosis skips the measures the trees do not have
    importance = pd.read_csv(diagnose(run) / "importance.csv").set_index("feature")
    assert importance.loc[["Return_1d", "Weekday"], "gain_mean"].notna().all()
    assert importance.drop(index=["stock_id", "Return_1d", "Weekday"])["gain_mean"].isna().all()
    assert backtest(run, learned_data)["n_periods"] > 0


# ---- horizon labels (ml4trading/models/learned/horizon.py) ----

SHARED_DEFAULTS = {
    "horizon": 1,
    "horizon_scale": "sum",
    "smooth_halflife": 0,
    "extra_features": False,
    "risk_adjust": "none",
    "vol_window": 63,
}


def _forward_sum(panel, h):
    """Each row's sum of its symbol's next ``h`` labels by pandas' own reversed rolling window:
    NaN unless all ``h`` exist and are finite. An independent reference for ``horizon_labels``."""
    return (
        panel.groupby("symbol", sort=False)["y"]
        .transform(lambda y: y.iloc[::-1].rolling(h).sum().iloc[::-1])
        .to_numpy()
    )


def _next_open_rule(panel, end):
    """The fit-row rule from before horizons existed (a label exits at its symbol's next open),
    kept here as the reference ``horizon: 1`` must still equal."""
    later = panel.loc[panel["session"] > end, "session"]
    if later.empty:
        return np.ones(len(panel), dtype=bool)
    exits = panel.groupby("symbol", observed=True, sort=False)["session"].shift(-1)
    return (exits <= later.min()).to_numpy()


@pytest.mark.parametrize("method", METHODS)
def test_the_shared_parameters_default_to_todays_behaviour(method):
    model = config.METHOD_DEFAULTS[method]["model"]
    assert {k: model[k] for k in SHARED_DEFAULTS} == SHARED_DEFAULTS
    for key in SHARED_DEFAULTS:  # all are required of a learned model
        short = {k: v for k, v in config.resolve(method).model.items() if k != key}
        with pytest.raises(ValueError, match=f"missing parameter.*{key}"):
            get_model(method)(**short)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "params, match",
    [
        ({"horizon": 0}, "horizon must be"),
        ({"horizon": -3}, "horizon must be"),
        ({"horizon": 2.5}, "horizon must be"),
        ({"horizon": True}, "horizon must be"),
        ({"horizon": "5"}, "horizon must be"),
        ({"horizon_scale": "median"}, "horizon_scale must be"),
        ({"smooth_halflife": -1}, "smooth_halflife must be"),
        ({"smooth_halflife": float("nan")}, "smooth_halflife must be"),
        ({"smooth_halflife": float("inf")}, "smooth_halflife must be"),
        ({"smooth_halflife": True}, "smooth_halflife must be"),
        ({"smooth_halflife": "3"}, "smooth_halflife must be"),
        ({"extra_features": 1}, "extra_features must be"),
        ({"extra_features": "yes"}, "extra_features must be"),
    ],
)
def test_a_bad_shared_parameter_is_refused(method, params, match):
    try:
        _model(method, **params)
    except ValueError as err:
        assert match in str(err)
    else:
        pytest.fail(f"{params} was accepted")


@pytest.mark.parametrize("method", METHODS)
def test_good_shared_parameters_are_accepted(method):
    for params in (
        {"horizon": 5, "horizon_scale": "mean"},
        {"horizon": np.int64(3)},
        {"smooth_halflife": 2.5},
        {"smooth_halflife": 3, "extra_features": True},
    ):
        _model(method, **params)


@pytest.mark.parametrize("method", METHODS)
def test_defaults_are_bit_identical(frames, method):
    f = frames[method]
    default = _model(method).fit_predict_fold(f, FIT, JUNE)
    spelled = _model(method, **SHARED_DEFAULTS).fit_predict_fold(f, FIT, JUNE)
    pd.testing.assert_frame_equal(default.predictions, spelled.predictions, check_exact=True)
    assert (default.n_fit_rows, default.baseline_prediction, default.fit_label_mean) == (
        spelled.n_fit_rows,
        spelled.baseline_prediction,
        spelled.fit_label_mean,
    )
    # at horizon 1 the fit's rows are the labelled rows of the window, as they always were
    window = rows_in(f.panel, FIT)
    assert default.n_fit_rows == int(np.isfinite(window["y"].to_numpy(dtype=float)).sum())
    assert (default.horizon, default.horizon_scale) == (1, "sum")
    # the mean of one label is the label
    mean1 = _model(method, horizon_scale="mean").fit_predict_fold(f, FIT, JUNE)
    pd.testing.assert_frame_equal(default.predictions, mean1.predictions, check_exact=True)


def test_horizon_one_purges_exactly_what_the_next_open_rule_purged(frames):
    panel = frames["linear"].panel
    in_gap = (panel["symbol"] == "BBB") & panel["session"].between("2021-05-24", "2021-06-04")
    last = panel["session"].max()
    ends = [
        FIT[1],
        pd.Timestamp("2021-05-22"),  # a weekend
        pd.Timestamp("2021-06-03"),  # inside BBB's gap
        last - pd.Timedelta(days=1),
        last,
        last + pd.Timedelta(days=10),
        pd.Timestamp("2020-12-31"),
    ]
    for rows in (panel, panel[~in_gap].reset_index(drop=True)):
        _, exits = horizon_labels(rows, 1, "sum")
        for end in ends:
            assert np.array_equal(realized_by(rows, exits, end), _next_open_rule(rows, end))


@pytest.mark.parametrize("scale", ["sum", "mean"])
def test_the_fit_trains_on_the_horizon_label(frames, monkeypatch, scale):
    f = frames["nonlinear"]
    result, _, y_fit = _fit_capturing(monkeypatch, f, horizon=3, horizon_scale=scale)
    panel = f.panel
    y_h = _forward_sum(panel, 3) / (3 if scale == "mean" else 1)
    exits = panel.groupby("symbol", sort=False)["session"].shift(-3)
    after = panel.loc[panel["session"] > FIT[1], "session"].min()
    rows = (
        panel["session"].between(*FIT).to_numpy() & np.isfinite(y_h) & (exits <= after).to_numpy()
    )
    assert result.n_fit_rows == int(rows.sum()) == len(y_fit)
    assert np.allclose(y_fit, y_h[rows], rtol=0, atol=1e-12)
    assert result.fit_label_mean == float(np.mean(y_fit))
    assert (result.horizon, result.horizon_scale) == (3, scale)


@pytest.mark.parametrize("method", METHODS)
def test_the_fit_purges_unrealized_horizon_labels(frames, method):
    # a fit row's 5-label window reaches into the sessions after the fit window for its last
    # four rows; those rows are purged, so what happens after the window changes no prediction
    f = frames[method]
    model = lambda: _model(method, horizon=5)  # noqa: E731
    want = model().fit_predict_fold(f, FIT, JUNE)
    panel = f.panel.copy()
    panel.loc[panel["session"] > FIT[1], "y"] += 0.5
    got = model().fit_predict_fold(Frames(panel, f.X), FIT, JUNE)
    pd.testing.assert_frame_equal(got.predictions, want.predictions)
    assert got.n_fit_rows == want.n_fit_rows
    # the purge removed the last four rows of each of the three symbols
    one = _model(method).fit_predict_fold(f, FIT, JUNE)
    assert want.n_fit_rows == one.n_fit_rows - 4 * 3
    # not vacuous: a label inside the window, realized in time, does move the fit
    inside = f.panel.copy()
    inside.loc[inside["session"].between("2021-05-03", "2021-05-07"), "y"] += 0.5
    moved = model().fit_predict_fold(Frames(inside, f.X), FIT, JUNE)
    assert not moved.predictions.equals(want.predictions)


@pytest.mark.parametrize("method", METHODS)
def test_a_gap_in_a_horizon_window_purges_the_row(learned_data, method):
    # BBB stops trading for 10 sessions right after the fit window. With h = 2 the labels of its
    # last two fit rows both contain the one that exits at the open after the gap, inside the
    # period the fit is scored on; AAA's and CCC's last row exits a session past the window's
    model = _model(method, horizon=2)
    pool = model.symbol_pool(all_symbols(learned_data))
    loaded = model.loaded_symbols(pool)
    bars, divs = load_bars(learned_data, loaded), load_dividends(learned_data, loaded)
    sessions = bars["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    gap = sorted(sessions[sessions > FIT[1]].unique())[:10]
    frames = model.build_frames(
        bars[~((bars["symbol"] == "BBB") & sessions.isin(gap))], divs, 390, pool
    )
    panel = frames.panel
    _, exits = horizon_labels(panel, 2, "sum")
    in_fit = panel.index.isin(rows_in(panel, FIT).index)
    purged = in_fit & ~realized_by(panel, exits, FIT[1])
    assert panel.loc[purged, "symbol"].value_counts().to_dict() == {"BBB": 2, "AAA": 1, "CCC": 1}

    want = model.fit_predict_fold(frames, FIT, JUNE)
    last = rows_in(panel[panel["symbol"] == "BBB"], FIT).index[-1]
    assert panel.loc[last, "session"] == FIT[1]
    after_gap = panel.copy()
    after_gap.loc[last, "y"] += 0.5  # the label that exits at the open after the gap
    got = model.fit_predict_fold(Frames(after_gap, frames.X), FIT, JUNE)
    pd.testing.assert_frame_equal(got.predictions, want.predictions)
    assert got.n_fit_rows == want.n_fit_rows
    # not vacuous: the label before it is realized in time, and is fitted
    before = panel.copy()
    before.loc[last - 1, "y"] += 0.5
    assert before.loc[last - 1, "symbol"] == "BBB"
    moved = model.fit_predict_fold(Frames(before, frames.X), FIT, JUNE)
    assert not moved.predictions.equals(want.predictions)


@pytest.mark.parametrize("method", METHODS)
def test_validation_labels_respect_the_window(learned_data, tmp_path, method):
    # Review Focus 2: with h = 3 the last two scored periods' labels exit after the window
    overrides = {**SHORT_FOLDS, "model.horizon": 3}
    run = train(method, tmp_path / "h3", learned_data, overrides=overrides)
    one = train(method, tmp_path / "h1", learned_data, overrides=SHORT_FOLDS)
    panel = load_frames(config.resolve(method), learned_data)[1].panel
    ahead = panel.assign(ahead=_forward_sum(panel, 3))[["period", "symbol", "ahead"]]
    folds = sorted((run / "folds").iterdir())
    assert len(folds) == 3
    for fdir in folds:
        fold = json.loads((fdir / "fold.json").read_text())
        val = pd.read_parquet(fdir / "val" / "predictions.parquet")
        periods = sorted(val["period"].unique())
        late = val["period"].isin(periods[-2:]).to_numpy()
        assert len(periods) > 10 and late.sum() > 0
        assert np.isnan(val.loc[late, "label"]).all()
        assert np.isfinite(val.loc[~late, "label"]).all()
        assert np.isfinite(val["y"]).all()  # the book and the IC keep the one-session return
        expected = val.merge(ahead, on=["period", "symbol"])["ahead"].to_numpy()
        assert np.allclose(val.loc[~late, "label"], expected[~late], rtol=0, atol=1e-12)
        assert fold["model_params"]["horizon"] == 3
        assert all(r is not None and np.isfinite(r) for r in fold["candidate_val_r2"])
        assert np.isfinite(fold["val_diagnostics"]["r2_mse"])
        # the one-session run scores the same rows with the same y
        base = pd.read_parquet(one / "folds" / fdir.name / "val" / "predictions.parquet")
        assert base["label"].notna().all()
        assert base[["period", "symbol", "y"]].equals(val[["period", "symbol", "y"]])


def test_a_grid_can_search_the_horizon(learned_data, tmp_path):
    grid = {"model.horizon": [1, 3]}
    run = train("linear", tmp_path / "run", learned_data, grid=grid, overrides=SHORT_FOLDS)
    for fdir in sorted((run / "folds").iterdir()):
        fold = json.loads((fdir / "fold.json").read_text())
        r2 = fold["candidate_val_r2"]
        assert len(r2) == 2 and all(r is not None and np.isfinite(r) for r in r2)
        assert r2[0] != r2[1]  # each candidate is scored on labels of its own horizon
        assert fold["model_params"]["horizon"] in (1, 3)


def test_scored_val_leaves_unrealized_labels_out_of_the_cross_section():
    days = pd.bdate_range("2021-01-04", periods=6)
    nan = np.nan
    panel = pd.DataFrame(
        {
            "period": [*days, *days],
            "session": [*days, *days],
            "symbol": ["A"] * 6 + ["B"] * 6,
            "y": [0.01, 0.02, 0.03, 0.04, 0.05, nan, 0.10, 0.00, 0.20, 0.00, 0.10, nan],
            "open_px": 10.0,
        }
    )
    preds = pd.DataFrame({"period": days[:4], "symbol": "A", "prediction": 0.001})
    fit = FoldResult(
        empty_predictions(), None, 1, "mse", 0.0, 0.0, LabelTransform("demean"), None, None, 3
    )
    scored = _scored_val(fit, preds, panel, list(days[:4]))
    # A's 3-label sums are .06 .09 .12; B's are .30 .20 .30: rows 0 and 1 exit by days[4] (the
    # first session after the window), rows 2 and 3 after it
    assert scored["y"].tolist() == [0.01, 0.02, 0.03, 0.04]  # the one-session return, all kept
    assert np.allclose(
        scored["label"], [-0.12, -0.055, nan, nan], rtol=0, atol=1e-15, equal_nan=True
    )
    assert _val_r2(
        FoldResult(
            empty_predictions(), None, 1, "mse", 0.0, 0.0, fit.label_transform, None, None, 3
        ),
        scored,
    ) == pytest.approx(
        1
        - np.mean((np.array([-0.12, -0.055]) - 0.001) ** 2)
        / np.mean(np.array([-0.12, -0.055]) ** 2)
    )


def test_the_validation_r2_scores_only_finite_labels():
    fit = FoldResult(empty_predictions(), None, 1, "mse", 0.0, 0.0, LabelTransform("raw"))
    scored = pd.DataFrame({"label": [0.01, np.nan, 0.03], "prediction": [0.01, 0.5, 0.02]})
    # the model's squared error 1e-4 / 2 against the baseline's (1e-4 + 9e-4) / 2
    assert _val_r2(fit, scored) == pytest.approx(1 - (0.5 * 1e-4) / (0.5 * 1e-3))
    assert _val_r2(fit, scored.assign(label=np.nan)) is None


@pytest.mark.parametrize("method", METHODS)
def test_a_fold_records_its_horizon(frames, method):
    f = frames[method]
    result = _model(method, horizon=4, horizon_scale="mean").fit_predict_fold(f, FIT, JUNE)
    assert (result.horizon, result.horizon_scale) == (4, "mean")
    empty = _model(method, horizon=4).fit_predict_fold(
        f, (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")), JUNE
    )
    assert empty.artifact is None and (empty.horizon, empty.horizon_scale) == (4, "sum")


def test_early_stopping_labels_respect_the_window(frames, monkeypatch):
    f = frames["nonlinear"]
    june = _periods_of(f, JUNE)
    stop = EarlyStop(june, POOL, 50, 5)
    result, eval_set = _early_stopping_capturing(monkeypatch, f, stop, horizon=3)
    X_eval, y_eval, _, _ = eval_set
    # the last two sessions of June exit after July's first open
    assert set(X_eval.index.get_level_values("period")) == set(june[:-2])
    assert np.isfinite(y_eval).all() and np.isfinite(result.eval_mse)
    ahead = f.panel.assign(ahead=_forward_sum(f.panel, 3)).set_index(["period", "symbol"])["ahead"]
    assert np.allclose(y_eval, ahead.reindex(X_eval.index).to_numpy(), rtol=0, atol=1e-12)
    # at one session the whole month is evaluated, as always
    _, one = _early_stopping_capturing(monkeypatch, f, stop)
    assert set(one[0].index.get_level_values("period")) == set(june)


def test_an_intraday_fit_keeps_its_rows_at_horizon_one(learned_data):
    # without hold_overnight a label exits at its own close: nothing is purged at horizon 1, and
    # a longer horizon is purged by the (safe) bound of the open of the row after its last label
    cfg = config.resolve("linear", {"model.hold_overnight": False})
    f = load_frames(cfg, learned_data)[1]
    window = rows_in(f.panel, FIT)
    one = get_model("linear")(**cfg.model).fit_predict_fold(f, FIT, JUNE)
    assert one.n_fit_rows == len(window) == int(np.isfinite(window["y"]).sum())
    cfg5 = config.resolve("linear", {"model.hold_overnight": False, "model.horizon": 5})
    five = get_model("linear")(**cfg5.model).fit_predict_fold(f, FIT, JUNE)
    assert five.n_fit_rows == one.n_fit_rows - 4 * 3
