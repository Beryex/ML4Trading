import numpy as np
import pandas as pd
import pytest

from ml4trading.config import TIMEZONE
from ml4trading.data import load_bars
from ml4trading.features import (
    FEATURE_ETFS,
    SCALE_DEPENDENT,
    build_features,
    symbol_daily_features,
)

STOCKS = ["AAA", "BBB", "CCC"]
PRICES = ["open", "high", "low", "close"]


@pytest.fixture
def bars(data_dir):
    return load_bars(data_dir, STOCKS + list(FEATURE_ETFS))


def _sessions(frame):
    return frame.index.get_level_values("session")


def _of(bars, symbol):
    return bars[bars["symbol"] == symbol]


def _scaled(bars, symbol, k=4.0):
    """``symbol``'s prices times ``k`` and its volume divided by ``k`` (a k-for-1 reverse split
    applied to the whole history)."""
    out = bars.copy()
    mine = out["symbol"] == symbol
    out.loc[mine, PRICES] = out.loc[mine, PRICES] * k
    out.loc[mine, "volume"] = out.loc[mine, "volume"] / k
    return out


def _same(a, b) -> bool:
    """Equal up to the EPSILON (1e-9) the indicators add to denominators: on a quiet series it
    moves a ratio by ~1e-6 under scaling, where a level moves by a factor of 4."""
    return np.allclose(
        np.asarray(a, dtype=float), np.asarray(b, dtype=float), rtol=1e-4, atol=1e-8, equal_nan=True
    )


def test_rows_are_known_before_their_session(bars):
    t = pd.Timestamp("2021-06-15")
    clean = build_features(bars, STOCKS, workers=1)
    noisy = bars.copy()
    future = (noisy["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None) >= t).to_numpy()
    rng = np.random.default_rng(0)
    for col in [*PRICES, "volume"]:
        noisy.loc[future, col] = rng.lognormal(size=int(future.sum()))
    got = build_features(noisy, STOCKS, workers=1)
    pd.testing.assert_frame_equal(got[_sessions(got) <= t], clean[_sessions(clean) <= t])
    later = _sessions(clean) > t
    assert not _same(got[later].drop(columns="stock_id"), clean[later].drop(columns="stock_id"))


def test_a_row_holds_the_previous_sessions_values(bars):
    feats = build_features(bars, STOCKS, workers=1)
    daily = symbol_daily_features(_of(bars, "AAA"))
    got = feats.xs("AAA", level="symbol")["Return_1d"]
    want = daily["Return_1d"].shift(1).astype("float32")
    assert got.equals(want)  # values, dtype and index; NaN equal


def test_scale_filter_is_the_rule(bars):
    base = build_features(bars, STOCKS, workers=1)
    scaled = build_features(_scaled(bars, "AAA"), STOCKS, workers=1)
    a, b = base.xs("AAA", level="symbol"), scaled.xs("AAA", level="symbol")
    moved = [c for c in base.columns if not _same(a[c], b[c])]
    assert moved == []

    raw = symbol_daily_features(_of(bars, "AAA"))
    raw_scaled = symbol_daily_features(_of(_scaled(bars, "AAA"), "AAA"))
    per_stock = [c for c in SCALE_DEPENDENT if not c.startswith(FEATURE_ETFS)]
    assert [c for c in per_stock if _same(raw[c], raw_scaled[c])] == []
    # every per-stock column the filter keeps reaches the matrix
    assert set(raw.columns) - set(SCALE_DEPENDENT) <= set(base.columns)

    etf_scaled = build_features(_scaled(bars, "VOO"), STOCKS, workers=1)
    voo = [c for c in base.columns if c.startswith("VOO_")]
    assert [c for c in voo if not _same(base[c], etf_scaled[c])] == []
    raw_voo = symbol_daily_features(_of(bars, "VOO"))
    raw_voo_scaled = symbol_daily_features(_of(_scaled(bars, "VOO"), "VOO"))
    assert not _same(raw_voo["MACD_hist_daily"], raw_voo_scaled["MACD_hist_daily"])
    assert {"VOO_MACD_hist_daily", "QQQ_MACD_hist_daily"} <= set(SCALE_DEPENDENT)


def test_etf_columns(bars):
    feats = build_features(bars, STOCKS, workers=1)
    assert set(feats.index.get_level_values("symbol")) == set(STOCKS)
    etf_cols = [c for c in feats.columns if c.startswith(("VOO_", "QQQ_"))]
    assert len(etf_cols) == 14
    assert {"VOO_Return_1d", "QQQ_CMF_20d"} <= set(etf_cols)
    assert not any(c.endswith("MACD_hist_daily") for c in etf_cols)
    per_session = feats[etf_cols].groupby(level="session").nunique(dropna=False)
    assert (per_session <= 1).all().all()


def test_first_row_and_gaps(bars):
    sessions = bars["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    late = (bars["symbol"] == "AAA") & (sessions < pd.Timestamp("2021-03-01"))
    gap = (bars["symbol"] == "BBB") & sessions.between(
        pd.Timestamp("2021-05-10"), pd.Timestamp("2021-05-12")
    )
    cut = bars[~(late | gap)]
    feats = build_features(cut, STOCKS, workers=1)

    first = feats.xs("AAA", level="symbol").iloc[0]
    assert first.name == pd.Timestamp("2021-03-01")
    assert first.drop("stock_id").isna().all() and first["stock_id"] == 0

    bbb = feats.xs("BBB", level="symbol")
    daily = symbol_daily_features(_of(cut, "BBB"))
    assert bbb.loc["2021-05-13", "Return_1d"] == np.float32(daily.loc["2021-05-07", "Return_1d"])


def test_float32_no_inf(bars):
    sessions = bars["ts"].dt.tz_convert(TIMEZONE).dt.tz_localize(None).dt.normalize()
    zero = (bars["symbol"] == "AAA") & (sessions == pd.Timestamp("2021-04-14"))
    quiet = bars.copy()
    quiet.loc[zero, "volume"] = 0.0
    feats = build_features(quiet, STOCKS, workers=1)
    assert (feats.dtypes == np.float32).all()
    assert not np.isinf(feats.to_numpy()).any()


def test_order_and_workers_do_not_matter(bars):
    want = build_features(bars, STOCKS, workers=1)
    shuffled = bars.sample(frac=1.0, random_state=0)
    pd.testing.assert_frame_equal(build_features(shuffled, ["CCC", "AAA", "BBB"], workers=1), want)
    pd.testing.assert_frame_equal(build_features(bars, STOCKS, workers=2), want)
    ids = want["stock_id"].groupby(level="symbol").unique()
    assert {s: list(v) for s, v in ids.items()} == {"AAA": [0], "BBB": [1], "CCC": [2]}
    assert list(want.columns[:1]) == ["stock_id"]
