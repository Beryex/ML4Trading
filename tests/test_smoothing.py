"""Prediction smoothing (``model.smooth_halflife``): an exponential average of each symbol's
traded predictions over its own recent rows, read from a lookback before the predict window."""

import json
import math

import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset

from ml4trading import config
from ml4trading.data import all_symbols, load_bars, load_dividends
from ml4trading.models import Frames, get_model
from ml4trading.models.base import rows_in
from ml4trading.models.learned import base as learned_base
from ml4trading.models.learned.base import EarlyStop, LearnedModel
from ml4trading.models.learned.losses import oos_r2
from ml4trading.models.learned.risk_adjust import prediction_multiplier, trailing_volatility
from ml4trading.models.learned.smoothing import smooth_predictions
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import train

METHODS = ["linear", "nonlinear"]
FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))
ONE_FOLD = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-06-30",
}


@pytest.fixture(scope="module")
def learned_data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("smoothing") / "data")


@pytest.fixture(scope="module")
def frames(learned_data):
    return {m: load_frames(config.resolve(m), learned_data)[1] for m in METHODS}


def _model(method, **params):
    cfg = config.resolve(method, {f"model.{k}": v for k, v in params.items()})
    return get_model(method)(**cfg.model)


def _keys(rows):
    return pd.MultiIndex.from_arrays([rows["period"], rows["symbol"]], names=["period", "symbol"])


def _raw_output(f, artifact, rows):
    """The fitted artifact's output on the panel ``rows``, as the fit's units."""
    at = f.panel.index.get_indexer(rows.index)
    return artifact.predict(f.X.iloc[at].set_axis(_keys(rows)))


def _rows_from(panel, window, sessions_before):
    """The panel rows of ``window`` and of the ``sessions_before`` sessions before it (all of
    them when the data starts sooner)."""
    sessions = np.sort(panel["session"].unique())
    first = rows_in(panel, window)["session"].min().to_datetime64()
    start = sessions[max(0, int(np.searchsorted(sessions, first)) - sessions_before)]
    return rows_in(panel, (pd.Timestamp(start), window[1]))


def _hand_ema(rows, values, halflife):
    """Each row's exponential average of its symbol's values up to it: the recursion
    ``s = (1 - alpha) s + alpha x`` seeded at the symbol's first value, in a plain loop."""
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    state: dict[str, float] = {}
    out = np.empty(len(rows))
    for i, (symbol, x) in enumerate(zip(rows["symbol"], values, strict=True)):
        state[symbol] = x if symbol not in state else (1.0 - alpha) * state[symbol] + alpha * x
        out[i] = state[symbol]
    return out


def _by_key(predictions: pd.DataFrame) -> pd.Series:
    return predictions.set_index(["period", "symbol"])["prediction"].sort_index()


def _hand_series(rows, values, window) -> pd.Series:
    """``values`` of the rows inside ``window``, indexed like ``_by_key``."""
    inside = rows["session"].between(*window).to_numpy()
    return pd.Series(values[inside], index=_keys(rows[inside])).sort_index()


def _assert_same(got: pd.Series, want: pd.Series, atol=1e-12):
    assert got.index.equals(want.index)
    assert np.allclose(got.to_numpy(), want.to_numpy(), rtol=0, atol=atol)


# ---- the average ----


@pytest.mark.parametrize("method, halflife", [("linear", 1), ("linear", 2.5), ("nonlinear", 3)])
def test_smoothing_is_an_ema_of_the_raw_predictions(frames, method, halflife):
    f = frames[method]
    raw = _model(method).fit_predict_fold(f, FIT, JUNE)
    smooth = _model(method, smooth_halflife=halflife).fit_predict_fold(f, FIT, JUNE)
    # the same fit: smoothing only changes what is predicted
    assert smooth.n_fit_rows == raw.n_fit_rows
    assert smooth.baseline_prediction == raw.baseline_prediction
    # 3 * ceil(half_life) sessions of lookback, then the window
    rows = _rows_from(f.panel, JUNE, 3 * math.ceil(halflife))
    want = _hand_series(rows, _hand_ema(rows, _raw_output(f, raw.artifact, rows), halflife), JUNE)
    _assert_same(_by_key(smooth.predictions), want)
    # only the predict window comes back, and the raw predictions are not the smoothed ones
    assert smooth.predictions["period"].between(*JUNE).all()
    assert len(smooth.predictions) == len(raw.predictions)
    assert not np.allclose(smooth.predictions["prediction"], raw.predictions["prediction"])


def test_smoothing_uses_the_lookback(frames):
    f = frames["linear"]
    raw = _model("linear").fit_predict_fold(f, FIT, JUNE)
    smooth = _model("linear", smooth_halflife=3).fit_predict_fold(f, FIT, JUNE)
    first = f.panel.loc[f.panel["session"].between(*JUNE), "session"].min()

    def on_first_day(result):
        return result.predictions[result.predictions["period"] == first].set_index("symbol")

    # the first window row is not its raw value: the rows before the window enter the average
    moved = on_first_day(smooth)["prediction"] - on_first_day(raw)["prediction"]
    assert len(moved) == 3 and moved.abs().min() > 1e-9
    # 3 * ceil(3) = 9 sessions of lookback, no more and no fewer
    for before, same in [(9, True), (0, False), (8, False), (10, False), (12, False)]:
        rows = _rows_from(f.panel, JUNE, before)
        values = _hand_ema(rows, _raw_output(f, raw.artifact, rows), 3)
        got = _hand_series(rows, values, JUNE).to_numpy()
        close = np.allclose(_by_key(smooth.predictions).to_numpy(), got, rtol=0, atol=1e-12)
        assert close == same, f"a lookback of {before} sessions"


def test_the_lookback_is_clipped_to_the_data_start(frames):
    f = frames["linear"]
    early = (pd.Timestamp("2021-01-06"), pd.Timestamp("2021-01-29"))  # only two sessions before
    raw = _model("linear").fit_predict_fold(f, FIT, early)
    smooth = _model("linear", smooth_halflife=3).fit_predict_fold(f, FIT, early)
    rows = rows_in(f.panel, (f.panel["session"].min(), early[1]))
    assert rows["session"].min() < early[0]
    want = _hand_series(rows, _hand_ema(rows, _raw_output(f, raw.artifact, rows), 3), early)
    _assert_same(_by_key(smooth.predictions), want)


# ---- a symbol's own rows (Review Focus 3) ----


def _without_rows(f, drop):
    kept = ~np.asarray(drop)
    return Frames(f.panel[kept].reset_index(drop=True), f.X[kept].reset_index(drop=True))


@pytest.mark.parametrize("method", METHODS)
def test_smoothing_follows_each_symbols_rows(frames, method):
    f = frames[method]
    panel = f.panel
    gap = pd.to_datetime(["2021-06-08", "2021-06-09", "2021-06-10"])
    started = pd.Timestamp("2021-06-10")
    # BBB did not trade three sessions of the window; CCC's first row is inside it
    skipped = (panel["symbol"] == "BBB") & panel["session"].isin(gap)
    not_yet = (panel["symbol"] == "CCC") & (panel["session"] < started)
    g = _without_rows(f, skipped | not_yet)
    halflife = 3
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    raw = _model(method).fit_predict_fold(g, FIT, JUNE).predictions
    smooth = _model(method, smooth_halflife=halflife).fit_predict_fold(g, FIT, JUNE).predictions
    assert smooth[["period", "symbol"]].equals(raw[["period", "symbol"]])

    def step(symbol):
        """The symbol's raw and smoothed values, and whether each row steps from the previous."""
        mine = smooth["symbol"] == symbol
        return (
            smooth.loc[mine, "period"],
            raw.loc[mine, "prediction"],
            smooth.loc[mine, "prediction"],
        )

    period, x, s = step("CCC")
    assert period.min() == started  # no row before it
    x, s = x.to_numpy(), s.to_numpy()
    assert s[0] == pytest.approx(x[0], abs=1e-15)  # seeded at its first row
    assert np.allclose(s[1:], (1 - alpha) * s[:-1] + alpha * x[1:], rtol=0, atol=1e-12)

    period, x, s = step("BBB")
    assert not period.isin(gap).any()
    period, x, s = period.to_numpy(), x.to_numpy(), s.to_numpy()
    # every step, the one across the gap too, is one row of the symbol's own
    assert np.allclose(s[1:], (1 - alpha) * s[:-1] + alpha * x[1:], rtol=0, atol=1e-12)
    after = int(np.flatnonzero(period > gap[-1].to_datetime64())[0])
    assert period[after - 1] == pd.Timestamp("2021-06-07").to_datetime64()
    # in calendar sessions the gap would have cost four steps of decay, a different number
    decayed = (1 - alpha) ** 4 * s[after - 1] + (1 - (1 - alpha) ** 4) * x[after]
    assert abs(s[after] - decayed) > 1e-9


def test_a_window_without_sessions_has_nothing_to_smooth(frames):
    # (nonlinear cannot predict zero rows at all, smoothed or not)
    after_the_data = (pd.Timestamp("2022-03-01"), pd.Timestamp("2022-03-31"))
    result = _model("linear", smooth_halflife=3).fit_predict_fold(
        frames["linear"], FIT, after_the_data
    )
    assert result.predictions.empty and result.artifact is not None


# ---- a missing prediction ----


def test_a_missing_prediction_stays_missing():
    # a row with no prediction (or an infinite one) is not traded that period: the average must
    # not hand it the last value it carried, which pandas' ewm would
    days = pd.bdate_range("2021-06-01", periods=7)
    rows = pd.concat(
        [pd.DataFrame({"period": days, "symbol": symbol}) for symbol in ("A", "B")],
        ignore_index=True,
    )
    nan, inf = np.nan, np.inf
    a = np.array([nan, 1.0, 2.0, nan, 4.0, inf, 6.0])  # leading and interior NaN, an infinity
    b = np.array([0.5, 1.5, -1.0, 2.0, 0.0, 1.0, 3.0])
    halflife = 3
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    got = smooth_predictions(np.concatenate([a, b]), rows, halflife)
    got_a, got_b = got[:7], got[7:]

    assert np.isnan(got_a[[0, 3, 5]]).all()  # missing in, missing out
    assert np.isfinite(got_a[[1, 2, 4, 6]]).all()
    # seeded at the first finite value, not at the leading NaN
    assert got_a[1] == pytest.approx(1.0, abs=1e-15)
    assert got_a[2] == pytest.approx((1 - alpha) * 1.0 + alpha * 2.0, abs=1e-15)
    # a missing row is a session that passed: the older average weighs (1 - alpha) once more
    old = (1 - alpha) ** 2
    after_nan = (old * got_a[2] + alpha * 4.0) / (old + alpha)
    assert got_a[4] == pytest.approx(after_nan, abs=1e-12)
    assert abs(got_a[4] - ((1 - alpha) * got_a[2] + alpha * 4.0)) > 1e-6  # not a plain step
    after_inf = (old * got_a[4] + alpha * 6.0) / (old + alpha)  # an infinity is missing too
    assert got_a[6] == pytest.approx(after_inf, abs=1e-12)
    # the other symbol is a plain average, untouched by A's missing rows
    assert np.allclose(got_b, _hand_ema(rows.iloc[7:], b, halflife), rtol=0, atol=1e-12)


# ---- off ----


@pytest.mark.parametrize("method", METHODS)
def test_smoothing_off_is_bit_identical(frames, method, monkeypatch):
    f = frames[method]
    default = _model(method).fit_predict_fold(f, FIT, JUNE)

    def forbidden(*args, **kwargs):
        raise AssertionError("a prediction was smoothed under smooth_halflife=0")

    monkeypatch.setattr(learned_base, "smooth_predictions", forbidden, raising=False)
    for off in (0, 0.0):
        again = _model(method, smooth_halflife=off).fit_predict_fold(f, FIT, JUNE)
        pd.testing.assert_frame_equal(default.predictions, again.predictions, check_exact=True)
        assert again.fit_predictions is None
    assert default.fit_predictions is None
    # and they are the artifact's own output on the window's rows, nothing more
    rows = rows_in(f.panel, JUNE)
    assert np.array_equal(default.predictions["prediction"], _raw_output(f, default.artifact, rows))


# ---- the fit-unit predictions ----


@pytest.mark.parametrize(
    "method, risk_adjust, halflife",
    [("linear", "none", 3), ("linear", "vol", 0), ("nonlinear", "vol_return", 3)],
)
def test_fit_predictions_are_the_models_own_output(frames, method, risk_adjust, halflife):
    f = frames[method]
    result = _model(method, risk_adjust=risk_adjust, smooth_halflife=halflife).fit_predict_fold(
        f, FIT, JUNE
    )
    preds, fit_units = result.predictions, result.fit_predictions
    assert len(preds) > 0 and fit_units.index.equals(preds.index)  # row for row
    # the artifact's output on the window's rows: before the risk scaling and the smoothing
    rows = rows_in(f.panel, JUNE)
    z = pd.Series(_raw_output(f, result.artifact, rows), index=_keys(rows))
    keys = pd.MultiIndex.from_frame(preds[["period", "symbol"]])
    assert np.allclose(fit_units.to_numpy(), z.loc[keys].to_numpy(), rtol=1e-10, atol=0)
    # what the trader reads is something else
    assert not np.allclose(preds["prediction"], fit_units)
    if risk_adjust == "vol" and halflife == 0:  # only the period's median volatility away
        sigma = trailing_volatility(f.panel, result.vol_window)
        multiplier = prediction_multiplier(sigma, f.panel["period"].to_numpy(), "vol")
        m = pd.Series(multiplier, index=_keys(f.panel)).loc[keys].to_numpy()
        assert np.allclose(preds["prediction"], fit_units * m, rtol=1e-12, atol=0)


# ---- with the risk scaling ----


@pytest.mark.parametrize("method, scale", [("linear", "vol"), ("nonlinear", "vol_return")])
def test_the_lookback_is_scaled_like_the_window(frames, method, scale):
    f = frames[method]
    panel = f.panel
    result = _model(method, risk_adjust=scale, smooth_halflife=3).fit_predict_fold(f, FIT, JUNE)
    rows = _rows_from(panel, JUNE, 9)
    z = _raw_output(f, result.artifact, rows)
    # the same causal sigma as the window's; a period's median is over the rows predicted in it,
    # which for a lookback period is that period's whole pool, as for a window period
    sigma = trailing_volatility(panel, result.vol_window)[panel.index.get_indexer(rows.index)]
    scaled = z * prediction_multiplier(sigma, rows["period"].to_numpy(), scale)
    assert np.isfinite(scaled).all()
    want = _hand_series(rows, _hand_ema(rows, scaled, 3), JUNE)
    _assert_same(_by_key(result.predictions), want)
    assert set(result.predictions["period"]) == set(rows_in(panel, JUNE)["period"])


# ---- early stopping ----


def test_early_stopping_stays_on_the_raw_output(frames, monkeypatch):
    f = frames["nonlinear"]
    seen = []
    cls = get_model("nonlinear")
    real = cls._fit

    def spy(self, X_fit, y_fit, eval_set=None):
        seen.append(eval_set)
        return real(self, X_fit, y_fit, eval_set)

    monkeypatch.setattr(cls, "_fit", spy)
    stop = EarlyStop(
        tuple(sorted(rows_in(f.panel, JUNE)["period"].unique())), ("AAA", "BBB"), 50, 5
    )
    raw = _model("nonlinear").fit_predict_fold(f, FIT, JUNE, early_stop=stop)
    smooth = _model("nonlinear", smooth_halflife=3).fit_predict_fold(f, FIT, JUNE, early_stop=stop)
    # the same booster stopped at the same round on the same raw squared error ...
    (X_raw, y_raw, _, _), (X_smooth, y_smooth, _, _) = seen
    assert X_raw.index.equals(X_smooth.index) and np.array_equal(y_raw, y_smooth)
    assert raw.best_iteration is not None
    assert (smooth.best_iteration, smooth.eval_mse) == (raw.best_iteration, raw.eval_mse)
    # ... and its window predictions come back smoothed, like any other
    rows = _rows_from(f.panel, JUNE, 9)
    want = _hand_series(rows, _hand_ema(rows, _raw_output(f, raw.artifact, rows), 3), JUNE)
    _assert_same(_by_key(smooth.predictions), want)


# ---- the validation R² reads the model's output ----


@pytest.mark.parametrize("method", METHODS)
def test_the_recorded_r2_is_of_the_raw_output(learned_data, tmp_path, monkeypatch, method):
    fits = []
    real = LearnedModel.fit_predict_fold

    def spy(self, *args, **kwargs):
        fits.append(real(self, *args, **kwargs))
        return fits[-1]

    monkeypatch.setattr(LearnedModel, "fit_predict_fold", spy)
    overrides = {**ONE_FOLD, "model.risk_adjust": "vol", "model.smooth_halflife": 3}
    run = train(method, tmp_path / "run", learned_data, overrides=overrides)
    (fdir,) = sorted((run / "folds").iterdir())
    fold = json.loads((fdir / "fold.json").read_text())
    val = pd.read_parquet(fdir / "val" / "predictions.parquet")
    result = fits[0]  # the validation fit; the deploy fit is the second
    assert fold["model_params"]["smooth_halflife"] == 3
    assert result.fit_predictions is not None

    keys = pd.MultiIndex.from_frame(val[["period", "symbol"]])
    made = pd.MultiIndex.from_frame(result.predictions[["period", "symbol"]])
    z = pd.Series(result.fit_predictions.to_numpy(), index=made).loc[keys].to_numpy()
    known = np.isfinite(val["label"].to_numpy())
    assert known.sum() > 10
    want = oos_r2(val["label"].to_numpy()[known], z[known], result.fit_label_mean, "mse")
    assert fold["val_diagnostics"]["r2_mse"] == pytest.approx(want, rel=1e-9)
    if method == "linear":  # the squared-error fit: its two R² are the same quantity
        assert fold["candidate_val_r2"][0] == pytest.approx(want, rel=1e-9)
    # a rebuild from the traded predictions would divide the median volatility out of an
    # average, which is not the model's output
    panel = load_frames(config.resolve(method), learned_data)[1].panel
    multiplier = prediction_multiplier(
        trailing_volatility(panel, result.vol_window), panel["period"].to_numpy(), "vol"
    )
    m = pd.Series(multiplier, index=_keys(panel)).loc[keys].to_numpy()
    assert not np.allclose(val["prediction"].to_numpy() / m, z, rtol=1e-6)


# ---- causality ----


@pytest.mark.parametrize("method", METHODS)
def test_causal(learned_data, method):
    model = _model(method, smooth_halflife=3)
    pool = model.symbol_pool(all_symbols(learned_data))
    loaded = model.loaded_symbols(pool)
    bars, divs = load_bars(learned_data, loaded), load_dividends(learned_data, loaded)
    assert_causal(model, bars, divs, 390, pool, FIT, JUNE, n_checks=3)
