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
from ml4trading.models import FoldResult, Frames, get_model
from ml4trading.models.base import empty_predictions, rows_in
from ml4trading.models.learned.base import EarlyStop
from ml4trading.models.learned.labels import LabelTransform
from ml4trading.models.learned.linear import impute_by_period
from ml4trading.models.learned.losses import baseline_prediction, loss_value, oos_r2
from ml4trading.models.learned.nonlinear import NonlinearModel
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import _scored_val, train

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
    assert NonlinearModel.FRAME_PARAMS == ("hold_overnight", "feature_transform")
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


def test_default_categorical_parameters_change_nothing(frames, monkeypatch):
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

    before = {  # the parameters of a default fit before the categorical ones existed
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
    assert seen["params"] == {**before, **categorical}
    assert seen["dataset_kwargs"]["categorical_feature"] == ["stock_id"]
    assert seen["train_kwargs"] == {"num_boost_round": 100}

    # the same fit without those four keys (LightGBM fills in the same values) predicts the same
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
    X_eval, y_eval, _ = eval_set
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
    X_eval, _, given = eval_set
    assert given is stop
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
