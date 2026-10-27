import json
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
from ml4trading.models import Frames, get_model
from ml4trading.models.base import rows_in
from ml4trading.models.learned.linear import impute_by_period
from ml4trading.models.learned.losses import baseline_prediction, loss_value, oos_r2
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import train

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


@pytest.mark.parametrize("method", METHODS)
def test_causal(learned_data, method):
    model = _model(method)
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
