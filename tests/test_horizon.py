import warnings

import numpy as np
import pandas as pd
import pytest

from ml4trading.models.learned.horizon import (
    HORIZON_SCALES,
    horizon_labels,
    realized_by,
    window_labels,
)
from ml4trading.models.learned.labels import LabelTransform


def _panel(series: dict[str, list[float]], days: dict[str, list[str]] | None = None):
    """A panel in (symbol, period) order: per symbol its labels ``y`` on consecutive business
    days from 2021-03-01, or on the given ``days`` (a symbol's missing session has no row)."""
    frames = []
    for symbol, y in series.items():
        sessions = (
            pd.bdate_range("2021-03-01", periods=len(y))
            if days is None or symbol not in days
            else pd.DatetimeIndex(days[symbol])
        )
        frames.append(
            pd.DataFrame({"period": sessions, "session": sessions, "symbol": symbol, "y": y})
        )
    return pd.concat(frames).reset_index(drop=True)


def _close(got, want):
    """Equal to rounding, NaN where NaN."""
    return np.allclose(got, want, rtol=0, atol=1e-15, equal_nan=True)


def test_horizon_one_is_the_panel_label():
    y = [0.011, -0.0, 0.5, np.nan, 1e-300, np.nan]
    panel = _panel({"A": y, "B": [0.2, 0.1, 0.3, 0.4, 0.5, np.nan]})
    for scale in HORIZON_SCALES:
        y_h, exits = horizon_labels(panel, 1, scale)
        assert y_h.tobytes() == panel["y"].to_numpy().tobytes()  # NaNs, -0.0 and all
        want = panel.groupby("symbol", sort=False)["session"].shift(-1)
        pd.testing.assert_series_equal(pd.Series(exits), want, check_names=False)
    y_h[0] = 99.0  # a copy: the panel is not edited through it
    assert panel.loc[0, "y"] == 0.011


def test_no_function_raises_a_warning():
    # a NaT without a time unit is deprecated in NumPy; the repo's runs hide warnings, so the
    # test turns them into errors itself
    days = {"B": ["2021-03-01", "2021-03-02", "2021-03-04", "2021-03-05"]}
    panel = _panel({"A": [0.01, 0.02, np.nan, 0.03], "B": [0.01, 0.02, 0.03, 0.04]}, days)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        for scale in HORIZON_SCALES:
            for h in (1, 2, 9):
                _, exits = horizon_labels(panel, h, scale)
                realized_by(panel, exits, pd.Timestamp("2021-03-02"))
                window_labels(panel, list(panel["period"].iloc[:2]), h, scale, None)


def test_sum_and_mean():
    panel = _panel({"A": [0.01, 0.02, -0.01, 0.03]})
    total, exits = horizon_labels(panel, 2, "sum")
    assert _close(total, [0.03, 0.01, 0.02, np.nan])
    mean, _ = horizon_labels(panel, 2, "mean")
    assert _close(mean, [0.015, 0.005, 0.01, np.nan])
    # the exit is the open of the row h after: the exit of the h-th label
    sessions = panel["session"].to_numpy()
    assert exits[0] == sessions[2] and exits[1] == sessions[3]
    assert np.isnat(exits[2]) and np.isnat(exits[3])
    # a longer horizon leaves fewer rows a label
    three, _ = horizon_labels(panel, 3, "sum")
    assert _close(three, [0.02, 0.04, np.nan, np.nan])


def test_a_gap_or_nan_gives_nan():
    # Review Focus 1: no partial sum, ever
    panel = _panel({"A": [0.01, np.nan, 0.02, 0.03, 0.04]})
    y_h, exits = horizon_labels(panel, 3, "sum")
    # a NaN inside the window: NaN; the last h-1 rows have too few following rows
    assert _close(y_h, [np.nan, np.nan, 0.09, np.nan, np.nan])
    assert np.isnat(exits[2:]).all() and not np.isnat(exits[:2]).any()  # the data end: NaT
    mean, _ = horizon_labels(panel, 3, "mean")
    assert np.isnan(mean[[0, 1, 3, 4]]).all() and np.isfinite(mean[2])
    # an infinite label is not finite
    inf = _panel({"A": [0.01, np.inf, 0.02, 0.03]})
    assert _close(horizon_labels(inf, 2, "sum")[0], [np.nan, np.nan, 0.05, np.nan])


def test_a_window_never_crosses_into_the_next_symbol():
    panel = _panel({"A": [0.01, 0.02, 0.03], "B": [0.1, 0.2, 0.3]})
    y_h, exits = horizon_labels(panel, 2, "sum")
    assert _close(y_h, [0.03, 0.05, np.nan, 0.3, 0.5, np.nan])
    assert np.isnat(exits[[1, 2, 4, 5]]).all()  # A's second-to-last row has no row t+2 of its own
    assert exits[0] == panel.loc[2, "session"] and exits[3] == panel.loc[5, "session"]


def test_a_missing_session_moves_the_exit_not_the_label():
    # B did not trade 2021-03-03: its rows span the gap (each y runs to the symbol's next open)
    days = {"B": ["2021-03-01", "2021-03-02", "2021-03-04", "2021-03-05"]}
    panel = _panel({"B": [0.01, 0.02, 0.03, 0.04]}, days)
    y_h, exits = horizon_labels(panel, 2, "sum")
    assert _close(y_h, [0.03, 0.05, 0.07, np.nan])
    assert exits[0] == pd.Timestamp("2021-03-04")  # the row after next, not 2021-03-03
    assert exits[1] == pd.Timestamp("2021-03-05")


def test_a_misordered_panel_is_refused():
    panel = _panel({"A": [0.01, 0.02, 0.03], "B": [0.1, 0.2, 0.3]})
    with pytest.raises(ValueError, match="order"):
        horizon_labels(panel.iloc[[0, 3, 1, 4, 2, 5]].reset_index(drop=True), 2, "sum")
    with pytest.raises(ValueError, match="order"):
        horizon_labels(panel.iloc[[2, 1, 0, 3, 4, 5]].reset_index(drop=True), 2, "sum")


def test_a_bad_horizon_or_scale_is_refused():
    panel = _panel({"A": [0.01, 0.02]})
    for bad in (0, -1, 1.5, True, "2", None):
        with pytest.raises(ValueError, match="horizon"):
            horizon_labels(panel, bad, "sum")
    with pytest.raises(ValueError, match="scale"):
        horizon_labels(panel, 2, "median")


def test_a_horizon_longer_than_the_panel_has_no_label():
    panel = _panel({"A": [0.01, 0.02, 0.03]})
    y_h, exits = horizon_labels(panel, 7, "sum")
    assert np.isnan(y_h).all() and np.isnat(exits).all()
    assert horizon_labels(panel.iloc[:0], 2, "sum")[0].size == 0


def test_realized_by_is_the_exit_on_or_before_the_first_session_after_the_end():
    days = {"B": ["2021-03-01", "2021-03-02", "2021-03-04", "2021-03-05", "2021-03-08"]}
    panel = _panel({"A": [0.01] * 5, "B": [0.01] * 5}, days)
    _, exits = horizon_labels(panel, 1, "sum")
    # the first session after 2021-03-02 is 2021-03-03 (A trades); B's row of 03-02 exits at its
    # next open, 2021-03-04, which lies past it, and each symbol's last row has no exit
    got = realized_by(panel, exits, pd.Timestamp("2021-03-02"))
    assert got.tolist() == [True, True, False, False, False, True, False, False, False, False]
    # no session follows the end: every row qualifies, those without an exit too
    assert realized_by(panel, exits, pd.Timestamp("2021-03-08")).all()


def test_window_labels_at_horizon_one_are_the_transform_of_the_window_labels():
    panel = _panel({"A": [0.01, 0.03, np.nan], "B": [0.05, 0.01, 0.02]})
    periods = list(panel["period"].iloc[:2])
    raw = window_labels(panel, periods, 1, "sum", None)
    assert _close(raw, [0.01, 0.03, np.nan, 0.05, 0.01, np.nan])
    demeaned = window_labels(panel, periods, 1, "sum", LabelTransform("demean"))
    assert _close(demeaned, [-0.02, 0.01, np.nan, 0.02, -0.01, np.nan])
    assert np.isnan(window_labels(panel, [], 1, "sum", None)).all()


def test_window_labels_leave_out_the_rows_realized_after_the_window():
    # Review Focus 2: with h = 3 the labels of the last two periods exit after the window
    panel = _panel({"A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], "B": [0.1] * 6})
    periods = list(panel["period"].iloc[:4])  # sessions 0..3; the first session after is 4
    got = window_labels(panel, periods, 3, "sum", None)
    # row t exits at row t+3: t = 0 -> 3, t = 1 -> 4 (on the first session after); 2 -> 5, 3 -> NaT
    assert _close(got[:6], [0.06, 0.09, *[np.nan] * 4])
    assert _close(got[6:], [0.3, 0.3, *[np.nan] * 4])


def test_a_row_realized_late_does_not_take_part_in_the_cross_section_mean():
    # B skips 2021-03-03, so with h = 2 its labels exit a session later than A's
    days = {
        "B": ["2021-03-01", "2021-03-02", "2021-03-04", "2021-03-05", "2021-03-08", "2021-03-09"]
    }
    panel = _panel({"A": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06], "B": [0.1] * 6}, days)
    periods = [pd.Timestamp("2021-03-01"), pd.Timestamp("2021-03-02")]
    got = window_labels(panel, periods, 2, "sum", LabelTransform("demean"))
    a, b = got[:6], got[6:]
    # 2021-03-01: both exit at row t+2 (03-03 and 03-04); the first session after the window is
    # 03-03, so A (exit 03-03) is realized and B (exit 03-04) is not -- the mean is A's alone
    assert a[0] == 0.0 and np.isnan(b[0])
    assert np.isnan(a[1]) and np.isnan(b[1])
    assert np.isnan(a[2:]).all() and np.isnan(b[2:]).all()
