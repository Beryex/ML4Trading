"""The volatility-scaled target (``ml4trading.models.learned.risk_adjust``): sigma from the
symbol's own past labels, the fit's units ``y_h / sigma``, and the prediction scaled back."""

import json

import numpy as np
import pandas as pd
import pytest
from conftest import write_dataset

from ml4trading import config
from ml4trading.data import all_symbols, load_bars, load_dividends
from ml4trading.diagnostics import dispersion
from ml4trading.models import get_model
from ml4trading.models.base import rows_in
from ml4trading.models.learned import risk_adjust as risk_adjust_mod
from ml4trading.models.learned.base import EarlyStop
from ml4trading.models.learned.horizon import window_labels
from ml4trading.models.learned.labels import LabelTransform
from ml4trading.models.learned.losses import oos_r2
from ml4trading.models.learned.risk_adjust import (
    MIN_VOL_OBSERVATIONS,
    RISK_ADJUSTS,
    scale_predictions,
    trailing_volatility,
)
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import _fit_units, _scored_features, _scored_val, _val_r2, train

METHODS = ["linear", "nonlinear"]
SCALES = ["vol", "vol_return"]
FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))
FROM_THE_START = (pd.Timestamp("2021-01-04"), pd.Timestamp("2021-05-21"))
JANUARY_TO_FEBRUARY = (pd.Timestamp("2021-01-04"), pd.Timestamp("2021-02-26"))
SHORT_FOLDS = {
    "folds.train_months": 3,
    "folds.val_months": 1,
    "folds.path_start": "2021-06-01",
    "folds.path_end": "2021-08-31",
}
POOL = ("AAA", "BBB", "CCC")
#: A sigma window ends two rows before its row (the label of the previous row exits at the row's
#: own open, which a prediction may not read), so the first row with 21 labels behind it is the
#: 23rd: index 22.
FIRST_SIGMA_ROW = MIN_VOL_OBSERVATIONS + 1


@pytest.fixture(scope="module")
def learned_data(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("risk_adjust") / "data")


@pytest.fixture(scope="module")
def frames(learned_data):
    return {m: load_frames(config.resolve(m), learned_data)[1] for m in METHODS}


def _model(method, **params):
    cfg = config.resolve(method, {f"model.{k}": v for k, v in params.items()})
    return get_model(method)(**cfg.model)


def _sigma_reference(panel, window):
    """Each row's sigma by explicit loops: the std (ddof 1) of the symbol's finite labels in its
    rows i-window-1 .. i-2, NaN with fewer than 21 of them or no spread."""
    out = np.full(len(panel), np.nan)
    for _, rows in panel.groupby("symbol", sort=False):
        at = panel.index.get_indexer(rows.index)
        y = rows["y"].to_numpy(dtype=float)
        for i in range(len(y)):
            past = y[max(0, i - window - 1) : max(0, i - 1)]
            past = past[np.isfinite(past)]
            if len(past) >= MIN_VOL_OBSERVATIONS and np.std(past, ddof=1) > 0:
                out[at[i]] = np.std(past, ddof=1)
    return out


def _forward_sum(panel, h):
    """Each row's sum of its symbol's next ``h`` labels, NaN unless all exist and are finite."""
    return (
        panel.groupby("symbol", sort=False)["y"]
        .transform(lambda y: y.iloc[::-1].rolling(h).sum().iloc[::-1])
        .to_numpy()
    )


def _scaled_fit_rows(panel, window, h, vol_window):
    """The fit rows of ``window`` at horizon ``h`` and their ``y_h / sigma``, by definition."""
    scaled = _forward_sum(panel, h) / _sigma_reference(panel, vol_window)
    exits = panel.groupby("symbol", sort=False)["session"].shift(-h)
    after = panel.loc[panel["session"] > window[1], "session"].min()
    rows = (
        panel["session"].between(*window).to_numpy()
        & np.isfinite(scaled)
        & (exits <= after).to_numpy()
    )
    return rows, scaled


def _random_panel(seed=0, n=120):
    """Two symbols of ``n`` consecutive sessions with random labels, a few of them missing."""
    parts = []
    for k, symbol in enumerate(("A", "B")):
        gen = np.random.Generator(np.random.PCG64(seed + k))
        y = gen.normal(0.0, 0.02, n)
        y[gen.choice(n, 6, replace=False)] = np.nan
        days = pd.bdate_range("2021-01-04", periods=n)
        parts.append(pd.DataFrame({"period": days, "session": days, "symbol": symbol, "y": y}))
    return pd.concat(parts).reset_index(drop=True)


# ---- sigma ----


@pytest.mark.parametrize("window", [21, 30, 63])
def test_sigma_is_the_std_of_the_window_rows(window):
    panel = _random_panel()
    got = trailing_volatility(panel, window)
    assert np.allclose(got, _sigma_reference(panel, window), rtol=1e-12, atol=0, equal_nan=True)
    assert np.isfinite(got).sum() > 30  # not vacuous


def test_sigma_uses_only_past_labels():
    # Review Focus: sigma_t never includes y[t] or later -- nor y[t-1], whose exit is t's own open
    panel = _random_panel()
    sigma = trailing_volatility(panel, 63)
    a = np.flatnonzero(panel["symbol"] == "A")
    b = np.flatnonzero(panel["symbol"] == "B")
    for t in (25, 40, 70, 110):
        for later_too in (False, True):
            bumped = panel.copy()
            rows = a[t:] if later_too else a[[t]]
            bumped.loc[rows, "y"] = bumped.loc[rows, "y"].fillna(0.0) + 0.5
            got = trailing_volatility(bumped, 63)
            # y[t] moves no sigma up to row t+1; the first row that reads it is t+2
            assert np.array_equal(got[a[: t + 2]], sigma[a[: t + 2]], equal_nan=True)
            assert not np.isclose(got[a[t + 2]], sigma[a[t + 2]])
            assert np.array_equal(got[b], sigma[b], equal_nan=True)  # another symbol's: untouched
    # the previous row's label does not enter either
    for t in (25, 70):
        bumped = panel.copy()
        bumped.loc[a[t - 1], "y"] += 0.5
        assert trailing_volatility(bumped, 63)[a[t]] == sigma[a[t]]


def test_sigma_window_edges():
    panel = _random_panel()
    panel["y"] = panel["y"].fillna(0.01)  # every label known
    sigma = trailing_volatility(panel, 30)
    a = np.flatnonzero(panel["symbol"] == "A")
    t = 80  # its window is rows 49 .. 78
    for moved, changes in (
        (t - 2, True),
        (t - 31, True),
        (t - 32, False),
        (t - 1, False),
        (t, False),
    ):
        bumped = panel.copy()
        bumped.loc[a[moved], "y"] += 0.3
        got = trailing_volatility(bumped, 30)
        # (a rolling std updates as the window slides, so a value that left it can leave a
        # rounding error behind: the comparison is to rounding, not bitwise)
        assert (not np.isclose(got[a[t]], sigma[a[t]], rtol=1e-9)) == changes, moved


def test_sigma_needs_21_finite_labels():
    panel = _random_panel()
    panel["y"] = panel["y"].fillna(0.01)
    panel["y"] += np.random.default_rng(3).normal(0, 0.01, len(panel))  # some spread
    a = np.flatnonzero(panel["symbol"] == "A")
    b = np.flatnonzero(panel["symbol"] == "B")
    sigma = trailing_volatility(panel, 63)
    # rows 0 .. 21 have fewer than 21 labels in rows ..t-2; row 22 has exactly 21
    assert np.isnan(sigma[a[:FIRST_SIGMA_ROW]]).all() and np.isfinite(sigma[a[FIRST_SIGMA_ROW]])
    # a missing label among the first 21 takes the count to 20
    gappy = panel.copy()
    gappy.loc[a[5], "y"] = np.nan
    got = trailing_volatility(gappy, 63)
    assert np.isnan(got[a[FIRST_SIGMA_ROW]]) and np.isfinite(got[a[FIRST_SIGMA_ROW + 1]])
    # the second symbol starts afresh: it does not borrow the first's rows
    assert np.isnan(sigma[b[:FIRST_SIGMA_ROW]]).all() and np.isfinite(sigma[b[FIRST_SIGMA_ROW]])


def test_sigma_of_a_label_without_spread_is_nan():
    # a constant label has no volatility to scale by: its label would be a division by zero
    panel = _random_panel()
    panel["y"] = 0.01
    assert np.isnan(trailing_volatility(panel, 63)).all()


def test_a_bad_window_is_refused():
    panel = _random_panel()
    for bad in (0, 20, -1, 21.5, True, "63", None):
        with pytest.raises(ValueError, match="vol_window"):
            trailing_volatility(panel, bad)
    trailing_volatility(panel, np.int64(63))
    with pytest.raises(ValueError, match="order"):
        trailing_volatility(panel.iloc[::-1].reset_index(drop=True), 63)


# ---- the label and the prediction units ----


def test_window_labels_divide_by_sigma_before_the_transform():
    days = list(pd.bdate_range("2021-03-01", periods=4))
    panel = pd.DataFrame(
        {
            "period": days * 2,
            "session": days * 2,
            "symbol": ["A"] * 4 + ["B"] * 4,
            "y": [0.02, 0.04, 0.06, 0.08, 0.03, 0.01, 0.05, 0.02],
        }
    )
    nan = np.nan
    sigma = np.array([0.01, 0.02, 0.02, nan, 0.01, nan, 0.025, 0.02])
    periods = days[:3]
    raw = window_labels(panel, periods, 1, "sum", None, sigma)
    assert np.allclose(raw, [2, 2, 3, nan, 3, nan, 2, nan], equal_nan=True, rtol=0, atol=1e-12)
    # the demean sees the scaled labels, and not the rows without a sigma
    demeaned = window_labels(panel, periods, 1, "sum", LabelTransform("demean"), sigma)
    # (the second day's mean is A's label alone: B has no sigma that day)
    assert np.allclose(demeaned, [-0.5, 0, 0.5, nan, 0.5, nan, -0.5, nan], equal_nan=True)
    # a longer horizon: the horizon label over sigma, and only where it exits by the first
    # session after the window
    two = window_labels(panel, periods[:2], 2, "sum", None, sigma)
    assert np.allclose(two, [6, nan, nan, nan, 4, nan, nan, nan], equal_nan=True)
    # no sigma: as before
    assert np.array_equal(
        window_labels(panel, periods, 1, "sum", None, None),
        window_labels(panel, periods, 1, "sum", None),
        equal_nan=True,
    )


def test_scale_predictions():
    p1, p2 = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")
    raw = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sigma = np.array([0.01, 0.03, np.nan, 0.02, 0.06])
    period = np.array([p1, p1, p1, p2, p2])
    by_return = scale_predictions(raw, sigma, period, "vol_return")
    assert np.allclose(by_return, [0.01, 0.06, np.nan, 0.08, 0.30], equal_nan=True)
    # the period's median over the rows with a sigma: 0.02 on p1, 0.04 on p2; a row without its
    # own sigma stays NaN
    by_median = scale_predictions(raw, sigma, period, "vol")
    assert np.allclose(by_median, [0.02, 0.04, np.nan, 0.16, 0.20], equal_nan=True)
    assert scale_predictions(raw, sigma, period, "none") is raw
    # a period without any sigma has no multiplier
    nothing = scale_predictions(raw[:2], np.full(2, np.nan), period[:2], "vol")
    assert np.isnan(nothing).all()
    with pytest.raises(ValueError, match="risk_adjust"):
        scale_predictions(raw, sigma, period, "median")


# ---- the parameters ----


@pytest.mark.parametrize("method", METHODS)
def test_the_defaults_leave_the_target_alone(method):
    model = config.METHOD_DEFAULTS[method]["model"]
    assert (model["risk_adjust"], model["vol_window"]) == ("none", 63)
    assert RISK_ADJUSTS == ("none", "vol", "vol_return")
    built = _model(method)
    assert (built.risk_adjust, built.vol_window) == ("none", 63)
    for key in ("risk_adjust", "vol_window"):
        short = {k: v for k, v in config.resolve(method).model.items() if k != key}
        with pytest.raises(ValueError, match=f"missing parameter.*{key}"):
            get_model(method)(**short)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "params, match",
    [
        ({"risk_adjust": "volatility"}, "risk_adjust must be"),
        ({"risk_adjust": None}, "risk_adjust must be"),
        ({"risk_adjust": True}, "risk_adjust must be"),
        ({"vol_window": 20}, "vol_window must be"),
        ({"vol_window": 0}, "vol_window must be"),
        ({"vol_window": 63.5}, "vol_window must be"),
        ({"vol_window": True}, "vol_window must be"),
        ({"vol_window": "63"}, "vol_window must be"),
    ],
)
def test_a_bad_target_parameter_is_refused(method, params, match):
    try:
        _model(method, **params)
    except ValueError as err:
        assert match in str(err)
    else:
        pytest.fail(f"{params} was accepted")


@pytest.mark.parametrize("method", METHODS)
def test_good_target_parameters_are_accepted(method):
    for scale in RISK_ADJUSTS:
        _model(method, risk_adjust=scale)
    _model(method, risk_adjust="vol", vol_window=21)
    _model(method, risk_adjust="vol_return", vol_window=np.int64(126))


# ---- the fit ----


def _fit_capturing(monkeypatch, method, f, fit_window=FIT, predict_window=JUNE, **params):
    """A fold's result with the (X, y) its ``_fit`` was handed."""
    seen = {}
    cls = get_model(method)
    real = cls._fit

    def spy(self, X_fit, y_fit, eval_set=None):
        seen["X"], seen["y"] = X_fit, np.array(y_fit)
        return real(self, X_fit, y_fit, eval_set)

    monkeypatch.setattr(cls, "_fit", spy)
    result = _model(method, **params).fit_predict_fold(f, fit_window, predict_window)
    return result, seen["X"], seen["y"]


def _keyed(f, window):
    """The features of ``window``'s rows indexed by (period, symbol), as a model hands them."""
    panel = f.panel
    keys = pd.MultiIndex.from_arrays([panel["period"], panel["symbol"]], names=["period", "symbol"])
    return f.X.set_axis(keys).iloc[panel.index.get_indexer(rows_in(panel, window).index)]


def _sigma_by_key(panel, window=63):
    return pd.Series(
        _sigma_reference(panel, window), index=pd.MultiIndex.from_frame(panel[["period", "symbol"]])
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", SCALES)
@pytest.mark.parametrize(
    "horizon, label, vol_window",
    [(1, "raw", 63), (1, "raw", 30), (3, "raw", 63), (1, "demean", 63), (1, "winsor", 63)],
)
def test_vol_target_trains_on_scaled_labels(
    frames, monkeypatch, method, scale, horizon, label, vol_window
):
    if method == "linear" and label != "raw":
        pytest.skip("linear has no label parameter")
    params = {"horizon": horizon, "vol_window": vol_window, "risk_adjust": scale}
    if method == "nonlinear":
        params["label"] = label
    f = frames[method]
    result, X_fit, y_fit = _fit_capturing(monkeypatch, method, f, **params)
    rows, scaled = _scaled_fit_rows(f.panel, FIT, horizon, vol_window)
    want = scaled[rows]
    if label.startswith("demean"):
        period = f.panel["period"].to_numpy()[rows]
        want = want - pd.Series(want).groupby(period).transform("mean").to_numpy()
    if label.endswith("winsor"):
        lo, hi = np.quantile(want, [0.005, 0.995])
        assert (want < lo).any() or (want > hi).any()  # the clip matters
        want = np.clip(want, lo, hi)
    assert result.n_fit_rows == int(rows.sum()) == len(y_fit)
    assert np.allclose(y_fit, want, rtol=1e-12, atol=1e-12)
    assert result.fit_label_mean == float(np.mean(y_fit))
    assert (result.risk_adjust, result.vol_window) == (scale, vol_window)
    if label == "raw":  # per unit of volatility: a move of several sigma is several, not 0.04
        assert np.abs(y_fit).max() > 1.0 > np.abs(f.panel["y"]).max()


@pytest.mark.parametrize("method", METHODS)
def test_rows_without_a_sigma_are_not_fitted(frames, method):
    f = frames[method]
    plain = _model(method).fit_predict_fold(f, FROM_THE_START, JUNE)
    scaled = _model(method, risk_adjust="vol").fit_predict_fold(f, FROM_THE_START, JUNE)
    # each symbol's first rows have fewer than 21 labels in their window
    assert scaled.n_fit_rows == plain.n_fit_rows - FIRST_SIGMA_ROW * len(POOL)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", SCALES)
def test_short_history_predicts_nan(frames, method, scale):
    f = frames[method]
    window = rows_in(f.panel, JANUARY_TO_FEBRUARY)
    plain = _model(method).fit_predict_fold(f, FROM_THE_START, JANUARY_TO_FEBRUARY).predictions
    got = _model(method, risk_adjust=scale).fit_predict_fold(f, FROM_THE_START, JANUARY_TO_FEBRUARY)
    # without the scale every pool row is predicted; with it a symbol is predicted from the row
    # that has 21 labels behind it on
    assert len(plain) == len(window)
    first = sorted(f.panel["period"].unique())[FIRST_SIGMA_ROW]
    expected = window[window["period"] >= first]
    assert got.predictions["period"].min() == first
    assert len(got.predictions) == len(expected) > 0
    assert set(got.predictions["symbol"]) == set(POOL)
    assert np.isfinite(got.predictions["prediction"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_vol_prediction_ranks_by_z(frames, method):
    f = frames[method]
    result = _model(method, risk_adjust="vol").fit_predict_fold(f, FIT, JUNE)
    X = _keyed(f, JUNE)
    z = pd.Series(result.artifact.predict(X), index=X.index)
    sigma = _sigma_by_key(f.panel)
    got = result.predictions.set_index(["period", "symbol"])["prediction"]
    assert len(got) == len(z) > 30
    multipliers = []
    for period, preds in got.groupby("period"):
        raw = z.loc[preds.index]
        median = sigma.loc[period].median()
        # every name of the period shares the multiplier: the order is z's, the units a return's
        assert np.allclose(preds.to_numpy(), raw.to_numpy() * median, rtol=1e-12, atol=0)
        assert (preds.rank().to_numpy() == raw.rank().to_numpy()).all()
        multipliers.append(median)
    assert len(set(multipliers)) > 5  # and the multiplier does move from day to day


@pytest.mark.parametrize("method", METHODS)
def test_vol_return_prediction_is_z_times_sigma(frames, method):
    f = frames[method]
    result = _model(method, risk_adjust="vol_return").fit_predict_fold(f, FIT, JUNE)
    X = _keyed(f, JUNE)
    z = pd.Series(result.artifact.predict(X), index=X.index)
    got = result.predictions.set_index(["period", "symbol"])["prediction"]
    assert len(got) == len(z) > 30
    want = z.loc[got.index] * _sigma_by_key(f.panel).loc[got.index]
    assert np.allclose(got.to_numpy(), want.to_numpy(), rtol=1e-12, atol=0)
    # the same fit as "vol": only the multiplier differs
    vol = _model(method, risk_adjust="vol").fit_predict_fold(f, FIT, JUNE)
    assert np.array_equal(vol.artifact.predict(X), result.artifact.predict(X))


@pytest.mark.parametrize("method", METHODS)
def test_none_is_bit_identical(frames, method, monkeypatch):
    f = frames[method]
    default = _model(method).fit_predict_fold(f, FIT, JUNE)

    def forbidden(*args, **kwargs):
        raise AssertionError("a sigma was computed under risk_adjust='none'")

    # "none" computes no sigma, whatever the window: it is the code that was there before
    monkeypatch.setattr(risk_adjust_mod, "trailing_volatility", forbidden)
    for params in ({"risk_adjust": "none"}, {"risk_adjust": "none", "vol_window": 21}):
        again = _model(method, **params).fit_predict_fold(f, FIT, JUNE)
        pd.testing.assert_frame_equal(default.predictions, again.predictions, check_exact=True)
        assert (again.n_fit_rows, again.baseline_prediction, again.fit_label_mean) == (
            default.n_fit_rows,
            default.baseline_prediction,
            default.fit_label_mean,
        )
    # nor does the scoring look for one
    periods = sorted(default.predictions["period"].unique())
    scored = _scored_val(default, default.predictions, f.panel, periods)
    pd.testing.assert_series_equal(scored["label"], scored["y"], check_names=False)
    assert _fit_units(default, scored) is None


@pytest.mark.parametrize("method", METHODS)
def test_a_fold_records_its_risk_adjust(frames, method):
    f = frames[method]
    result = _model(method, risk_adjust="vol", vol_window=40).fit_predict_fold(f, FIT, JUNE)
    assert (result.risk_adjust, result.vol_window) == ("vol", 40)
    empty = _model(method, risk_adjust="vol_return", vol_window=40).fit_predict_fold(
        f, (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-06-30")), JUNE
    )
    assert empty.artifact is None
    assert (empty.risk_adjust, empty.vol_window) == ("vol_return", 40)
    plain = _model(method).fit_predict_fold(f, FIT, JUNE)
    assert (plain.risk_adjust, plain.vol_window) == ("none", 63)


# ---- early stopping and the validation scoring ----


def _eval_set_of(monkeypatch, f, periods, **params):
    """A nonlinear fold's result under early stopping on ``periods`` and the evaluation set its
    ``_fit`` was handed."""
    seen = {}
    cls = get_model("nonlinear")
    real = cls._fit

    def spy(self, X_fit, y_fit, eval_set=None):
        seen["eval"] = eval_set
        return real(self, X_fit, y_fit, eval_set)

    monkeypatch.setattr(cls, "_fit", spy)
    stop = EarlyStop(tuple(periods), ("AAA", "BBB"), 50, 5)
    result = _model("nonlinear", **params).fit_predict_fold(f, FIT, JUNE, early_stop=stop)
    return result, seen["eval"]


def test_early_stopping_set_is_in_the_fits_units(frames, monkeypatch):
    f = frames["nonlinear"]
    panel = f.panel
    june = sorted(rows_in(panel, JUNE)["period"].unique())
    result, (X_eval, y_eval, _) = _eval_set_of(
        monkeypatch, f, june, risk_adjust="vol_return", label="demean"
    )
    # y / sigma, demeaned over the pool's three names, then taken at the two evaluation names
    scaled = pd.Series(panel["y"].to_numpy() / _sigma_reference(panel, 63), index=panel.index)
    pool = scaled[panel["period"].isin(set(june)) & scaled.notna()]
    centred = pool - pool.groupby(panel.loc[pool.index, "period"]).transform("mean")
    want = centred[panel.loc[centred.index, "symbol"].isin(["AAA", "BBB"])]
    assert len(y_eval) == len(want) > 0
    assert np.allclose(y_eval, want.to_numpy(), rtol=1e-12, atol=1e-12)
    assert result.best_iteration is not None and np.isfinite(result.eval_mse)
    # rows without a sigma are no evaluation rows: early periods start at the 23rd session
    early = sorted(rows_in(panel, JANUARY_TO_FEBRUARY)["period"].unique())
    _, (X_early, y_early, _) = _eval_set_of(monkeypatch, f, early, risk_adjust="vol")
    assert np.isfinite(y_early).all()
    assert (
        X_early.index.get_level_values("period").min()
        == sorted(panel["period"].unique())[FIRST_SIGMA_ROW]
    )


def test_scored_val_labels_are_y_over_sigma(frames):
    f = frames["linear"]
    panel = f.panel
    result = _model("linear", risk_adjust="vol").fit_predict_fold(f, FROM_THE_START, JUNE)
    periods = sorted(rows_in(panel, JANUARY_TO_FEBRUARY)["period"].unique())
    preds = panel[panel["period"].isin(periods)][["period", "symbol"]].assign(prediction=0.001)
    scored = _scored_val(result, preds, panel, periods)
    sigma = _sigma_by_key(panel)
    keys = pd.MultiIndex.from_frame(scored[["period", "symbol"]])
    assert np.allclose(
        scored["label"], scored["y"].to_numpy() / sigma.loc[keys].to_numpy(), rtol=1e-12, atol=0,
        equal_nan=True,
    )  # fmt: skip
    # the first sessions of every symbol have no sigma, so no label; their y stays for the book
    assert scored["label"].isna().sum() == FIRST_SIGMA_ROW * len(POOL)
    assert np.isfinite(scored["y"]).all()
    # the validation label follows the fit: one without the scale scores the return itself
    plain = _model("linear").fit_predict_fold(f, FROM_THE_START, JUNE)
    raw = _scored_val(plain, preds, panel, periods)
    pd.testing.assert_series_equal(raw["label"], raw["y"], check_names=False)


def test_scored_val_demeans_the_scaled_labels_over_the_pool(frames):
    f = frames["nonlinear"]
    panel = f.panel
    result = _model("nonlinear", risk_adjust="vol", label="demean").fit_predict_fold(f, FIT, JUNE)
    april = sorted(
        rows_in(panel, (pd.Timestamp("2021-04-01"), pd.Timestamp("2021-04-30")))["period"].unique()
    )
    traded = panel["period"].isin(april) & panel["symbol"].isin(["AAA", "BBB"])
    preds = panel[traded][["period", "symbol"]].assign(prediction=0.001)
    scored = _scored_val(result, preds, panel, april)
    scaled = pd.Series(panel["y"].to_numpy() / _sigma_reference(panel, 63), index=panel.index)
    pool = scaled[panel["period"].isin(april) & scaled.notna()]
    centred = (pool - pool.groupby(panel.loc[pool.index, "period"]).transform("mean")).rename(
        "want"
    )
    want = panel[["period", "symbol"]].join(centred, how="inner")
    both = scored.merge(want, on=["period", "symbol"])
    assert len(both) == len(scored) == 2 * len(april)
    assert np.allclose(both["label"], both["want"], rtol=1e-12, atol=1e-12)
    assert abs(scored["label"].mean()) > 1e-6  # the pool's mean, not the two names'


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", SCALES)
def test_the_validation_r2_is_in_the_fits_units(frames, method, scale):
    # the trader reads returns, the label and the baseline are per unit of volatility: the R²
    # compares the fit's own output with them, not the returns
    f = frames[method]
    result = _model(method, risk_adjust=scale).fit_predict_fold(f, FIT, JUNE)
    periods = sorted(result.predictions["period"].unique())
    scored = _scored_val(result, result.predictions, f.panel, periods)
    in_fit_units = _fit_units(result, scored)
    z = result.artifact.predict(_scored_features(f, scored))
    assert np.allclose(in_fit_units, z, rtol=1e-9, atol=1e-12)
    r2 = _val_r2(result, scored, in_fit_units)
    assert r2 == pytest.approx(
        oos_r2(scored["label"].to_numpy(), z, result.baseline_prediction, result.loss)
    )
    assert _val_r2(result, scored) != pytest.approx(r2)  # the returns would be a unit mismatch
    plain = _model(method).fit_predict_fold(f, FIT, JUNE)
    assert _fit_units(plain, _scored_val(plain, plain.predictions, f.panel, periods)) is None


def test_dispersion_reads_the_fits_units_when_given():
    d1, d2 = pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")
    scored = pd.DataFrame(
        {
            "period": [d1, d1, d2, d2],
            "symbol": ["A", "B"] * 2,
            "prediction": [0.01, 0.02, 0.03, 0.04],
            "label": [1.0, 2.0, 3.0, 5.0],
        }
    )
    assert dispersion(scored)["pred_xs_std"] == pytest.approx(0.005)
    got = dispersion(scored, np.array([1.0, 2.0, 3.0, 4.0]))
    assert got["pred_xs_std"] == pytest.approx(0.5)
    assert got["label_xs_std"] == pytest.approx(0.75)  # the label is the same either way


@pytest.mark.parametrize("method", METHODS)
def test_a_run_scores_and_deploys_the_scaled_target(learned_data, tmp_path, method):
    overrides = {**SHORT_FOLDS, "model.risk_adjust": "vol_return"}
    run = train(method, tmp_path / "run", learned_data, overrides=overrides)
    panel = load_frames(config.resolve(method), learned_data)[1].panel
    sigma = _sigma_by_key(panel)
    folds = sorted((run / "folds").iterdir())
    assert len(folds) == 3
    for fdir in folds:
        fold = json.loads((fdir / "fold.json").read_text())
        assert fold["model_params"]["risk_adjust"] == "vol_return"
        assert fold["model_params"]["vol_window"] == 63
        val = pd.read_parquet(fdir / "val" / "predictions.parquet")
        s = sigma.loc[pd.MultiIndex.from_frame(val[["period", "symbol"]])].to_numpy()
        assert len(val) > 10
        assert np.allclose(val["label"], val["y"].to_numpy() / s, rtol=1e-12, atol=0)
        assert all(r is not None and np.isfinite(r) for r in fold["candidate_val_r2"])
        # the diagnostics: the dispersion and the R² read the fit's units, the rest the returns
        diag = fold["val_diagnostics"]
        in_fit_units = pd.Series(val["prediction"].to_numpy() / s, index=val.index)
        want = in_fit_units.groupby(val["period"]).std(ddof=0).median()
        assert diag["pred_xs_std"] == pytest.approx(want, rel=1e-9)
        assert np.isfinite(diag["r2_mse"])
        if method == "linear":  # the squared-error fit: its two R² are the same quantity
            assert diag["r2_mse"] == pytest.approx(fold["candidate_val_r2"][0], rel=1e-9)
        test = pd.read_parquet(fdir / "predictions.parquet")
        assert len(test) > 0 and np.isfinite(test["prediction"]).all()


def test_a_grid_can_search_the_risk_adjust(learned_data, tmp_path):
    grid = {"model.risk_adjust": ["none", "vol", "vol_return"], "model.vol_window": [30, 63]}
    run = train("linear", tmp_path / "run", learned_data, grid=grid, overrides=SHORT_FOLDS)
    for fdir in sorted((run / "folds").iterdir()):
        fold = json.loads((fdir / "fold.json").read_text())
        r2 = fold["candidate_val_r2"]
        assert len(r2) == 6 and all(r is not None and np.isfinite(r) for r in r2)
        assert len(set(r2)) > 1  # each candidate is scored on labels in its own units
        assert fold["model_params"]["risk_adjust"] in RISK_ADJUSTS


# ---- causality ----


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("scale", SCALES)
def test_causal(learned_data, method, scale):
    model = _model(method, risk_adjust=scale)
    pool = model.symbol_pool(all_symbols(learned_data))
    loaded = model.loaded_symbols(pool)
    bars, divs = load_bars(learned_data, loaded), load_dividends(learned_data, loaded)
    assert_causal(model, bars, divs, 390, pool, FIT, JUNE, n_checks=3)
