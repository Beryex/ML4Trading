import numpy as np
import pandas as pd
import pytest

from ml4trading.periods import build_panel, dividend_returns, period_bars

TZ = "America/New_York"


def _bars(rows):
    df = pd.DataFrame(rows, columns=["ts", "symbol", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(TZ)
    return df


def two_sessions():
    return _bars(
        [
            ("2021-01-04 09:30", "X", 10.0, 11.0, 9.5, 10.5, 100),
            ("2021-01-04 10:00", "X", 10.5, 12.0, 10.0, 11.0, 200),
            ("2021-01-04 15:30", "X", 11.0, 11.5, 10.8, 11.2, 300),
            ("2021-01-05 09:30", "X", 11.4, 11.6, 11.0, 11.1, 100),
            ("2021-01-05 10:00", "X", 11.1, 11.3, 10.9, 11.0, 100),
        ]
    )


def test_daily_aggregation_first_max_min_last_sum():
    pb = period_bars(two_sessions(), 390)
    day1 = pb.iloc[0]
    assert day1["period"] == pd.Timestamp("2021-01-04")
    assert (day1["open"], day1["high"], day1["low"], day1["close"], day1["volume"]) == (
        10.0,
        12.0,
        9.5,
        11.2,
        600,
    )


def test_intraday_keys_floor_to_k_minutes():
    pb = period_bars(two_sessions(), 60)
    keys = list(pb["period"].dt.strftime("%m-%d %H:%M"))
    # 09:30 floors into the 09:00 hour bin, 10:00 starts its own
    assert keys == ["01-04 09:00", "01-04 10:00", "01-04 15:00", "01-05 09:00", "01-05 10:00"]


def test_overnight_label_is_next_open_to_open():
    panel = build_panel(two_sessions(), 390, hold_overnight=True)
    first = panel.iloc[0]
    assert first["exit_px"] == 11.4
    assert first["y"] == pytest.approx(11.4 / 10.0 - 1)
    assert np.isnan(panel.iloc[-1]["y"])  # the last period has no next open


def test_intraday_label_is_open_to_close():
    panel = build_panel(two_sessions(), 390, hold_overnight=False)
    assert panel.iloc[0]["y"] == pytest.approx(11.2 / 10.0 - 1)
    assert panel.iloc[1]["y"] == pytest.approx(11.0 / 11.4 - 1)


def test_dividend_paid_to_the_holder_through_the_ex_date_open():
    divs = pd.DataFrame(
        {"symbol": ["X"], "ex_date": pd.to_datetime(["2021-01-05"]), "div_ret": [0.02]}
    )
    panel = build_panel(two_sessions(), 390, hold_overnight=True, dividends=divs)
    # held from the 01-04 open to the 01-05 open: the ex-date is in (entry, exit]
    assert panel.iloc[0]["div_ret"] == 0.02
    assert panel.iloc[0]["y"] == pytest.approx(11.4 / 10.0 - 1 + 0.02)
    # bought at the ex-date's open: nothing
    assert panel.iloc[1]["div_ret"] == 0.0


def test_dividend_interval_sum():
    d = pd.to_datetime(["2021-01-05", "2021-01-07"]).to_numpy()
    entry = pd.to_datetime(["2021-01-04", "2021-01-05", "2021-01-06"]).to_numpy()
    exit_ = pd.to_datetime(["2021-01-08", "2021-01-06", "2021-01-07"]).to_numpy()
    assert list(dividend_returns(entry, exit_, d, [1.0, 2.0])) == [3.0, 0.0, 2.0]


def test_k_must_be_a_multiple_of_the_bar():
    with pytest.raises(ValueError):
        period_bars(two_sessions(), 45)
