import numpy as np
import pandas as pd
import pytest

from ml4trading import config
from ml4trading.book import POSITION_COLUMNS, daily_values, run_book
from ml4trading.periods import nyse_sessions


def _block_book(anchor, equity_before, gross, costs):
    return pd.DataFrame(
        {
            "period": [anchor],
            "session": [anchor],
            "equity_before": [equity_before],
            "gross": [gross],
            "spread_cost": [costs],
            "commission": [0.0],
            "borrow": [0.0],
            "net": [gross - costs],
            "ret": [(gross - costs) / equity_before],
            "equity": [equity_before + gross - costs],
            "n_positions": [1],
            "traded_notional": [0.0],
        }
    )


def test_a_weekly_block_is_marked_every_day_and_lands_on_the_book():
    sessions = nyse_sessions()
    start = sessions.get_loc(pd.Timestamp("2021-03-01"))
    anchor = sessions[start - start % 5]  # a 5-session block anchor
    pos = sessions.get_loc(anchor)
    days, exit_day = sessions[pos : pos + 5], sessions[pos + 5]
    opens = [100.0, 102.0, 101.0, 105.0, 104.0]
    exit_px = 106.0
    daily_panel = pd.DataFrame({"symbol": "A", "session": days, "open_px": opens})
    divs = pd.DataFrame({"symbol": ["A"], "ex_date": [days[2]], "div_ret": [0.01]})
    shares, costs = 10.0, 1.0
    gross = shares * (exit_px - 100.0) + shares * 100.0 * 0.01
    book = _block_book(anchor, 10_000.0, gross, costs)
    row = dict.fromkeys(POSITION_COLUMNS, 0.0)
    row.update(
        period=anchor,
        symbol="A",
        prev_shares=0.0,
        shares=shares,
        price=100.0,
        exit_price=exit_px,
        session_end=False,
    )
    daily = daily_values(book, pd.DataFrame([row]), 1950, daily_panel, divs)
    assert list(daily["session"]) == list(days)
    # end-of-day values: marked at the next open; the dividend's ex-date is day 2, so it is
    # earned over day 1's open-to-open span; costs fall on day 0
    want = 10_000.0 - costs + shares * (np.array([102, 101, 105, 104, exit_px]) - 100.0)
    want[1:] += shares * 100.0 * 0.01
    assert daily["equity"].to_numpy() == pytest.approx(want)
    assert daily["equity"].iloc[-1] == pytest.approx(book["equity"].iloc[0])
    assert daily["residual"].abs().max() == pytest.approx(0.0, abs=1e-9)
    assert daily["gross_ret"].iloc[0] == pytest.approx(shares * 2.0 / 10_000.0)
    assert exit_day not in set(daily["session"])


def test_a_daily_book_is_its_own_daily_value(data_dir):
    from ml4trading.pipeline import load_panel

    cfg = config.resolve("investment", {"capital": 2000.0})
    _, panel = load_panel(cfg, data_dir)
    periods = sorted(panel.dropna(subset=["exit_px"])["period"].unique())[:30]
    preds = panel[["period", "symbol"]].assign(prediction=0.01)
    book, positions = run_book(preds, panel, periods, cfg)
    daily = daily_values(book, positions, 390)
    assert daily["ret"].to_numpy() == pytest.approx(book["ret"].to_numpy())
    assert daily["equity"].to_numpy() == pytest.approx(book["equity"].to_numpy())


def test_a_weekly_run_is_valued_daily_end_to_end(data_dir, tmp_path):
    from ml4trading.backtest import backtest
    from ml4trading.train import train

    short = {
        "folds.train_months": 3,
        "folds.val_months": 1,
        "folds.path_start": "2021-06-01",
        "folds.path_end": "2021-11-30",
    }
    run = train("momentum", tmp_path / "run", data_dir, overrides=short)
    metrics = backtest(run, data_dir)
    book = pd.read_csv(run / "backtest" / "book.csv", parse_dates=["period"])
    daily = pd.read_csv(run / "backtest" / "daily.csv", parse_dates=["session"])
    assert metrics["n_days"] == len(daily) > 4 * metrics["n_periods"]
    assert metrics["max_daily_residual"] == pytest.approx(0.0, abs=1e-6)
    # the last day of every block lands on the book's equity
    ends = daily.groupby(daily["session"].isin(book["period"]).cumsum())["equity"].last()
    assert ends.to_numpy() == pytest.approx(book["equity"].to_numpy())
