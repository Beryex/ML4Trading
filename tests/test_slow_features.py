"""Slow cross-sectional features (``ml4trading.features.slow``) and the ``extra_features`` frame
option that adds them to the learned models' features."""

import tracemalloc

import numpy as np
import pandas as pd
import pytest
from conftest import make_bars, write_dataset
from scipy import stats

from ml4trading import config
from ml4trading.config import TIMEZONE
from ml4trading.data import all_symbols, load_bars, load_dividends
from ml4trading.features import (
    SLOW_FEATURES,
    build_features,
    slow_features,
)
from ml4trading.models import get_model
from ml4trading.pipeline import load_frames
from ml4trading.testing import assert_causal
from ml4trading.train import train

METHODS = ["linear", "nonlinear"]
PRICES = ["open", "high", "low", "close"]
FIT = (pd.Timestamp("2021-03-01"), pd.Timestamp("2021-05-21"))
JUNE = (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30"))

#: The session index (0-based, in the symbol's own sessions) at which each feature first has a
#: full window: the unshifted value exists from here on, the shifted row one session later.
FIRST_VALID = {
    "mom_12_1": 252,  # close(d-252)
    "mom_6_1": 126,
    "mom_1m": 21,
    "high_52w": 251,  # 252 closes
    "max_21": 21,  # 21 returns, the first of which is session 1's
    "beta_252": 252,  # 252 returns
    "ivol_63": 63,
    "resid_mom_12_1": 253,  # the returns of sessions d-252..d-21, the first needs close(d-253)
    "amihud_21": 21,
    "rskew_21": 20,  # 21 sessions of 30-minute returns
    "overnight_21": 21,  # the first session has no previous close
    "intraday_21": 20,
    "volatility_252": 252,
}


def _linked_bars(symbol, voo, beta, seed, price0):
    """Bars of ``symbol`` whose 30-minute log steps are ``beta`` times VOO's plus noise, so the
    market-relative features have something to measure; the timestamps are VOO's."""
    noise = np.random.default_rng(seed)
    market_step = np.diff(np.log(voo["close"].to_numpy()), prepend=np.log(price0))
    close = price0 * np.exp(np.cumsum(beta * market_step + noise.normal(0.0, 0.002, len(voo))))
    open_ = np.concatenate(([price0], close[:-1])) * (1 + noise.normal(0, 0.0005, len(voo)))
    out = voo[["ts"]].copy()
    out["open"], out["close"] = open_, close
    out["high"] = np.maximum(open_, close) * 1.001
    out["low"] = np.minimum(open_, close) * 0.999
    out["volume"] = noise.integers(50_000, 150_000, size=len(voo)).astype(float)
    return out.assign(symbol=symbol)


def _market_bars(start, end):
    """AAA (beta 1.3), BBB (beta 0.6) and VOO over ``start``..``end``."""
    voo = make_bars(4, start, end, 350.0)
    return pd.concat(
        [
            _linked_bars("AAA", voo, 1.3, 11, 50.0),
            _linked_bars("BBB", voo, 0.6, 12, 120.0),
            voo.assign(symbol="VOO"),
        ],
        ignore_index=True,
    )


@pytest.fixture(scope="module")
def bars():
    """About 330 sessions: enough for every window of a feature, once."""
    return _market_bars("2020-01-01", "2021-03-31")


@pytest.fixture(scope="module")
def long_bars():
    """About 700 sessions: room for a window that sees neither side of a split date."""
    return _market_bars("2018-01-01", "2020-09-30")


@pytest.fixture(scope="module")
def long_data(tmp_path_factory):
    """The synthetic dataset of the other tests, from early 2020: the slow features have their
    252 sessions well before the June 2021 window."""
    return write_dataset(
        tmp_path_factory.mktemp("slow") / "data", start="2020-01-01", end="2021-07-30"
    )


def _model(method, **params):
    cfg = config.resolve(method, {f"model.{k}": v for k, v in params.items()})
    return get_model(method)(**cfg.model)


# ---- the hand computation: plain numpy and scipy, on the sessions of one symbol ----


def _sessions(bars, symbol):
    """(dates, open, close, volume, the 30-minute log returns of each session) of a symbol, by
    New York date. A session's first return runs from its open to its first bar's close."""
    rows = bars[bars["symbol"] == symbol].sort_values("ts")
    day = rows["ts"].dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d").to_numpy()
    dates, o, c, v, r = [], [], [], [], []
    for d, g in rows.groupby(day, sort=True):
        dates.append(pd.Timestamp(d))
        o.append(g["open"].iloc[0])
        c.append(g["close"].iloc[-1])
        v.append(g["volume"].sum())
        r.append(np.diff(np.log(np.concatenate(([g["open"].iloc[0]], g["close"].to_numpy())))))
    return pd.DatetimeIndex(dates), np.array(o), np.array(c), np.array(v), r


def _unshifted(bars, symbol, i, market="VOO"):
    """Every slow feature of ``symbol`` as of the close of its session ``i`` (0-based)."""
    _, o, c, v, r = _sessions(bars, symbol)
    _, _, cm, _, _ = _sessions(bars, market)
    ret = np.concatenate(([np.nan], c[1:] / c[:-1] - 1))
    mkt = np.concatenate(([np.nan], cm[1:] / cm[:-1] - 1))
    out = {
        "mom_12_1": c[i - 21] / c[i - 252] - 1,
        "mom_6_1": c[i - 21] / c[i - 126] - 1,
        "mom_1m": c[i] / c[i - 21] - 1,
        "high_52w": c[i] / c[i - 251 : i + 1].max(),
        "max_21": ret[i - 20 : i + 1].max(),
        "volatility_252": np.std(ret[i - 251 : i + 1], ddof=1),
        "amihud_21": np.mean(np.abs(ret[i - 20 : i + 1]) / (c[i - 20 : i + 1] * v[i - 20 : i + 1])),
        "rskew_21": stats.skew(np.concatenate(r[i - 20 : i + 1])),  # Fisher-Pearson, biased
        "overnight_21": np.sum(np.log(o[i - 20 : i + 1] / c[i - 21 : i])),
        "intraday_21": np.sum(np.log(c[i - 20 : i + 1] / o[i - 20 : i + 1])),
    }
    y, x = ret[i - 251 : i + 1], mkt[i - 251 : i + 1]
    out["beta_252"] = np.cov(y, x)[0, 1] / np.var(x, ddof=1)
    y, x = ret[i - 62 : i + 1], mkt[i - 62 : i + 1]
    slope, intercept = np.polyfit(x, y, 1)  # an OLS line with an intercept
    out["ivol_63"] = np.std(y - (intercept + slope * x), ddof=1)
    y, x = ret[i - 252 : i - 20], mkt[i - 252 : i - 20]  # sessions d-252 .. d-21
    assert len(y) == 232
    slope, _ = np.polyfit(x, y, 1)
    resid = y - slope * x  # the slope from the same window, no intercept: the sum is not zero
    out["resid_mom_12_1"] = resid.sum() / np.std(resid, ddof=1)
    return out


def test_the_interface_lists_the_thirteen_features():
    assert SLOW_FEATURES == (
        "mom_12_1",
        "mom_6_1",
        "mom_1m",
        "high_52w",
        "max_21",
        "beta_252",
        "ivol_63",
        "resid_mom_12_1",
        "amihud_21",
        "rskew_21",
        "overnight_21",
        "intraday_21",
        "volatility_252",
    )
    assert set(FIRST_VALID) == set(SLOW_FEATURES)


def test_each_slow_feature_matches_a_hand_computation(bars):
    got = slow_features(bars, ["AAA", "BBB"])
    assert list(got.columns) == list(SLOW_FEATURES) and (got.dtypes == np.float32).all()
    assert got.index.names == ["session", "symbol"]
    for symbol in ("AAA", "BBB"):
        dates, *_ = _sessions(bars, symbol)
        for i in (len(dates) - 2, len(dates) - 30):
            want = _unshifted(bars, symbol, i)
            # row d holds the values as of the close of the previous session
            row = got.loc[(dates[i + 1], symbol)]
            assert set(want) == set(SLOW_FEATURES)
            for name, value in want.items():
                assert np.isclose(row[name], value, rtol=1e-5, atol=0), (symbol, i, name)


def test_the_market_alone_and_other_symbols_are_not_rows(bars):
    got = slow_features(bars, ["AAA", "BBB"])
    assert set(got.index.get_level_values("symbol")) == {"AAA", "BBB"}
    dates, *_ = _sessions(bars, "AAA")
    assert len(got.xs("AAA", level="symbol")) == len(dates)


def test_a_benchmark_without_bars_is_refused(bars):
    with pytest.raises(ValueError, match="VOO"):
        slow_features(bars[bars["symbol"] != "VOO"], ["AAA"])


def test_the_result_does_not_depend_on_row_or_pool_order(bars):
    want = slow_features(bars, ["AAA", "BBB"])
    shuffled = bars.sample(frac=1.0, random_state=1)
    got = slow_features(shuffled, ["BBB", "AAA", "BBB"])
    pd.testing.assert_frame_equal(got.sort_index(), want.sort_index(), check_exact=True)


# ---- the shift ----


def test_slow_features_are_shifted_one_session(bars):
    got = slow_features(bars, ["AAA", "BBB"])
    for symbol in ("AAA", "BBB"):
        mine = got.xs(symbol, level="symbol")
        dates, *_ = _sessions(bars, symbol)
        assert mine.index.equals(dates.rename("session"))
        assert mine.iloc[0].isna().all()  # the first session knows nothing
        for i in (len(dates) - 3, len(dates) - 10):  # row d is the unshifted value of d-1
            want = _unshifted(bars, symbol, i)
            for name in SLOW_FEATURES:
                assert np.isclose(mine[name].iloc[i + 1], want[name], rtol=1e-5, atol=0)


def test_rows_are_known_before_their_session(bars):
    t = pd.Timestamp("2021-02-10")
    clean = slow_features(bars, ["AAA", "BBB"])
    noisy = bars.copy()
    future = (noisy["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None) >= t).to_numpy()
    rng = np.random.default_rng(0)
    for col in [*PRICES, "volume"]:
        noisy.loc[future, col] = rng.lognormal(size=int(future.sum()))
    got = slow_features(noisy, ["AAA", "BBB"])
    sessions = clean.index.get_level_values("session")
    upto = sessions <= t  # t's own row holds the day before's values: unchanged too
    pd.testing.assert_frame_equal(got[upto], clean[upto], check_exact=True)
    assert not np.allclose(
        got[sessions > t].to_numpy(), clean[sessions > t].to_numpy(), equal_nan=True
    )


# ---- history ----


def test_short_history_is_nan(bars):
    got = slow_features(bars, ["AAA", "BBB"])
    for symbol in ("AAA", "BBB"):
        mine = got.xs(symbol, level="symbol")
        for name, first in FIRST_VALID.items():
            # the unshifted value exists from session `first`, the shifted row one later
            assert mine[name].iloc[: first + 1].isna().all(), (name, "before its window is full")
            assert mine[name].iloc[first + 1 :].notna().all(), (name, "after")


def test_a_late_listing_gets_no_history_from_the_market(bars):
    """A symbol's windows are its own sessions: VOO's older history does not fill them."""
    dates, *_ = _sessions(bars, "AAA")
    late = bars[(bars["symbol"] != "AAA") | (bars["ts"] >= pd.Timestamp(dates[100], tz=TIMEZONE))]
    got = slow_features(late, ["AAA", "BBB"]).xs("AAA", level="symbol")
    assert len(got) == len(dates) - 100
    assert got["beta_252"].isna().all()  # 230 sessions left: no full year
    assert got["resid_mom_12_1"].isna().all()
    assert got["ivol_63"].iloc[:64].isna().all() and got["ivol_63"].iloc[64:].notna().all()
    assert got["mom_1m"].iloc[:22].isna().all() and got["mom_1m"].iloc[22:].notna().all()


# ---- sessions the symbol and the market do not share ----


def _local_date(frame):
    return frame["ts"].dt.tz_convert(TIMEZONE).dt.strftime("%Y-%m-%d")


def _without_session(bars, symbol, date):
    return bars[~((bars["symbol"] == symbol) & (_local_date(bars) == date))]


def _closes(bars, symbol):
    rows = bars[bars["symbol"] == symbol]
    return rows.groupby(_local_date(rows), sort=True)["close"].last()


@pytest.mark.parametrize("missing", ["AAA", "VOO"])
def test_market_relative_features_align_on_the_sessions_both_have(bars, missing):
    """With a session missing on either side, the market-relative features are those of the
    common sessions (returns between consecutive common sessions); the others use the symbol's
    own sessions."""
    gone = "2020-12-15"
    holed = _without_session(bars, missing, gone)
    got = slow_features(holed, ["AAA"]).xs("AAA", level="symbol")
    own = _sessions(holed, "AAA")[0]
    own_close, market_close = _closes(holed, "AAA"), _closes(holed, "VOO")
    shared = [d for d in own_close.index if d in market_close.index]
    assert len(shared) == len(own) - (missing == "VOO")
    y = own_close.loc[shared].pct_change().to_numpy()
    x = market_close.loc[shared].pct_change().to_numpy()
    for j in (len(shared) - 3, len(shared) - 40):  # a position among the common sessions
        window = slice(j - 251, j + 1)
        want = np.cov(y[window], x[window])[0, 1] / np.var(x[window], ddof=1)
        # the shifted row of the NEXT common session holds the value as of session j
        row = pd.Timestamp(shared[j + 1])
        assert np.isclose(got.loc[row, "beta_252"], want, rtol=1e-5, atol=0), (missing, j)
    clean = slow_features(bars, ["AAA"]).xs("AAA", level="symbol")
    if missing == "VOO":
        # a feature of the symbol alone does not care that the market lacks a day
        pd.testing.assert_series_equal(got["mom_12_1"], clean["mom_12_1"], check_exact=True)
        # but the day the market lacks has no market-relative value (shown the next session)
        after = own[list(own).index(pd.Timestamp(gone)) + 1]
        assert np.isnan(got.loc[after, ["beta_252", "ivol_63", "resid_mom_12_1"]]).all()
        assert got.loc[after, "max_21"] == clean.loc[after, "max_21"]


# ---- split invariance ----


def _scaled(bars, symbol, k, before=None):
    """``symbol``'s prices times ``k`` and its volume divided by ``k``: over its whole history,
    or only before the session date ``before`` (a split date, as a back-adjustment has it)."""
    out = bars.copy()
    mine = out["symbol"] == symbol
    if before is not None:
        mine &= (out["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None) < before).to_numpy()
    out.loc[mine, PRICES] = out.loc[mine, PRICES] * k
    out.loc[mine, "volume"] = out.loc[mine, "volume"] / k
    return out


@pytest.mark.parametrize("market", [False, True])
def test_slow_features_are_split_invariant(bars, market):
    base = slow_features(bars, ["AAA", "BBB"])
    assert base.xs("AAA", level="symbol").iloc[-5:].notna().all(axis=None)  # not all NaN
    changed = _scaled(bars, "VOO" if market else "AAA", 4.0)
    got = slow_features(changed, ["AAA", "BBB"])
    pd.testing.assert_frame_equal(got, base, check_exact=False, rtol=1e-6, atol=0)


@pytest.mark.parametrize("k", [4.0, 0.25])
def test_the_features_of_a_window_off_a_split_do_not_see_its_scale(long_bars, k):
    """Prices times ``k`` and volume divided by it before a split date: every row whose windows
    all lie on one side of the date is unchanged -- a ratio, a return or a dollar volume does
    not move, and no window reaches past its length. A row whose window holds the date's own
    return, which the scale makes a jump, moves."""
    sessions, *_ = _sessions(long_bars, "AAA")
    split = 300
    base = slow_features(long_bars, ["AAA"]).xs("AAA", level="symbol")
    got = slow_features(_scaled(long_bars, "AAA", k, before=sessions[split]), ["AAA"]).xs(
        "AAA", level="symbol"
    )
    rows = np.arange(len(sessions))
    # row r holds session r-1's values: session u = r-1 needs data back to u-253
    before = (rows - 1 < split) & (rows - 1 >= 253)
    after = rows - 1 >= split + 253
    assert before.sum() > 20 and after.sum() > 100
    for name in SLOW_FEATURES:
        for part in (before, after):
            assert np.isfinite(base[name].to_numpy()[part]).all()
            np.testing.assert_allclose(
                got[name].to_numpy()[part], base[name].to_numpy()[part], rtol=1e-6, atol=0
            )
    between = split + 5  # the window of this row holds the jump
    for name in ("volatility_252", "mom_1m"):
        assert not np.isclose(got[name].iloc[between], base[name].iloc[between], rtol=1e-2)


# ---- the frames ----


def _inputs(data_dir, model):
    pool = model.symbol_pool(all_symbols(data_dir))
    loaded = model.loaded_symbols(pool)
    return load_bars(data_dir, loaded), load_dividends(data_dir, loaded), pool


@pytest.mark.parametrize("method", METHODS)
def test_extra_features_off_is_bit_identical(long_data, method):
    model = _model(method)
    bars, divs, pool = _inputs(long_data, model)
    off = model.build_frames(bars, divs, 390, pool)
    keys = pd.MultiIndex.from_arrays(
        [off.panel["session"], off.panel["symbol"]], names=["session", "symbol"]
    )
    # today's frame: the ML4Investment features alone, reindexed to the panel
    today = build_features(bars, pool).reindex(keys).set_axis(off.panel.index)
    pd.testing.assert_frame_equal(off.X, today, check_exact=True)
    assert not set(SLOW_FEATURES) & set(off.X.columns)
    # on: the same columns first, the slow ones after, row for row
    on = _model(method, extra_features=True).build_frames(bars, divs, 390, pool)
    assert list(on.X.columns) == [*off.X.columns, *SLOW_FEATURES]
    pd.testing.assert_frame_equal(on.X[list(off.X.columns)], off.X, check_exact=True)
    slow = slow_features(bars, pool).reindex(keys).set_axis(off.panel.index)
    pd.testing.assert_frame_equal(on.X[list(SLOW_FEATURES)], slow, check_exact=True)
    pd.testing.assert_frame_equal(on.panel, off.panel)
    assert (on.X.dtypes == np.float32).all()
    in_june = ((on.panel["session"] >= JUNE[0]) & (on.panel["session"] <= JUNE[1])).to_numpy()
    assert on.X.loc[in_june, list(SLOW_FEATURES)].notna().all(axis=None)


def test_cs_rank_ranks_the_slow_columns_too(long_data):
    """The ranks join the transform of the frames: each slow column becomes, period by period,
    its average-tie rank scaled to [-1, 1] (Gu, Kelly & Xiu 2020)."""
    raw_model = _model("nonlinear", extra_features=True)
    bars, divs, pool = _inputs(long_data, raw_model)
    raw = raw_model.build_frames(bars, divs, 390, pool)
    ranked = _model("nonlinear", feature_transform="cs_rank", extra_features=True).build_frames(
        bars, divs, 390, pool
    )
    june = raw.panel["session"].between(*JUNE).to_numpy()
    period = raw.panel.loc[june, "period"]
    for name in SLOW_FEATURES:
        values = raw.X.loc[june, name]
        rank = values.groupby(period).rank(method="average")
        n = values.groupby(period).transform("count")
        want = 2.0 * (rank - 1.0) / (n - 1.0) - 1.0
        got = ranked.X.loc[june, name]
        assert (n > 1).all() and got.notna().all()
        np.testing.assert_allclose(got, want, atol=1e-6, err_msg=name)
        assert not np.allclose(got, values)  # the column itself was replaced


def test_the_join_holds_no_second_copy_of_the_features(long_data, monkeypatch):
    """The full-universe frame is about a gigabyte: joining the 13 slow columns on must not copy
    it. The two builds are stubbed with a wide random frame on the panel's own keys."""
    model = _model("nonlinear", extra_features=True)
    bars, divs, pool = _inputs(long_data, model)
    panel = _model("nonlinear").build_frames(bars, divs, 390, pool).panel
    keys = pd.MultiIndex.from_arrays(
        [panel["session"], panel["symbol"]], names=["session", "symbol"]
    )
    rng = np.random.default_rng(0)

    def frame(columns):
        values = rng.normal(size=(len(keys), columns)).astype(np.float32)
        return pd.DataFrame(values, index=keys, columns=[f"c{i}" for i in range(columns)])

    monkeypatch.setattr("ml4trading.models.learned.base.build_features", lambda b, p: frame(2000))
    monkeypatch.setattr("ml4trading.models.learned.base.slow_features", lambda b, p: frame(13))
    size = len(keys) * 2000 * 4
    peaks = {}
    for extra in (False, True):
        tracemalloc.start()
        try:
            _model("nonlinear", extra_features=extra).build_frames(bars, divs, 390, pool)
            peaks[extra] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert size > 5_000_000  # big enough to see
    assert peaks[True] < peaks[False] + size / 4  # a copy of X would add 1.0 x


@pytest.mark.parametrize("method", METHODS)
def test_extra_features_is_a_frame_parameter(method, data_dir, tmp_path, monkeypatch):
    def no_loading(*args, **kwargs):
        raise AssertionError("the data was loaded before the grid was checked")

    monkeypatch.setattr("ml4trading.train.load_frames", no_loading)
    assert "extra_features" in get_model(method).FRAME_PARAMS
    assert "hold_overnight" in get_model(method).FRAME_PARAMS
    with pytest.raises(ValueError, match="extra_features.*cannot be searched"):
        train(method, tmp_path / "run", data_dir, grid={"model.extra_features": [False, True]})


@pytest.mark.parametrize("method", METHODS)
def test_a_fit_uses_the_slow_columns(long_data, method):
    frames = load_frames(config.resolve(method, {"model.extra_features": True}), long_data)[1]
    result = _model(method, extra_features=True).fit_predict_fold(frames, FIT, JUNE)
    names = result.artifact.feature_names()
    assert set(SLOW_FEATURES) <= set(names)
    assert len(result.predictions) > 0 and np.isfinite(result.predictions["prediction"]).all()


@pytest.mark.parametrize("method", METHODS)
def test_causal(long_data, method):
    model = _model(method, horizon=5, smooth_halflife=3, extra_features=True)
    bars, divs, pool = _inputs(long_data, model)
    frames = model.build_frames(bars, divs, 390, pool)
    june = frames.panel["session"].between(*JUNE).to_numpy()
    assert frames.X.loc[june, list(SLOW_FEATURES)].notna().all(axis=None)  # not a vacuous check
    assert_causal(model, bars, divs, 390, pool, FIT, JUNE, n_checks=2)
