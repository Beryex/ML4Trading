import numpy as np
import pandas as pd
import pytest

from ml4trading import config
from ml4trading.book import run_book
from ml4trading.metrics import max_drawdown, sharpe, summarize


def _panel(opens, divs=None, symbol="VOO", closes=None):
    days = pd.bdate_range("2021-01-04", periods=len(opens))
    opens = np.asarray(opens, dtype=float)
    exit_px = np.append(opens[1:], np.nan)
    div = np.zeros(len(opens)) if divs is None else np.asarray(divs, dtype=float)
    return pd.DataFrame(
        {
            "period": days,
            "session": days,
            "symbol": symbol,
            "open": opens,
            "high": opens,
            "low": opens,
            "close": opens if closes is None else closes,
            "volume": 1e6,
            "open_px": opens,
            "exit_px": exit_px,
            "div_ret": div,
            "y": exit_px / opens - 1 + div,
        }
    )


def _hold(panel, prediction=0.01):
    return panel[["period", "symbol"]].assign(prediction=prediction)


def test_buy_and_hold_telescopes_and_pays_one_entry():
    cfg = config.resolve("investment", {"costs.spread_bps": 4.0})
    panel = _panel([150.0, 151.0, 149.0, 153.0, 152.0])
    book, positions = run_book(_hold(panel), panel, panel["period"][:-1], cfg)
    assert set(positions["shares"]) == {13.0}  # floor(2000 / 150), then held
    commission = 0.35  # one entry order; entries pay no spread
    assert book["commission"].sum() == pytest.approx(commission)
    assert book["spread_cost"].sum() == 0.0
    assert book["equity"].iloc[-1] == pytest.approx(2000 + 13 * (152 - 150) - commission)


def test_dividends_are_credited_to_the_holder():
    cfg = config.resolve("investment")
    panel = _panel([150.0, 150.0, 150.0], divs=[0.0, 0.01, 0.0])
    book, _ = run_book(_hold(panel), panel, panel["period"][:-1], cfg)
    assert book["gross"].sum() == pytest.approx(13 * 150 * 0.01)


def test_a_symbol_without_a_mark_is_carried_not_sold():
    cfg = config.resolve("investment")
    panel = _panel([150.0, 151.0, 152.0, 153.0])
    gap = panel.drop(index=1).copy()
    gap.loc[0, "exit_px"] = 152.0  # the next available open spans the gap
    book, positions = run_book(_hold(gap), gap, panel["period"][:-1], cfg)
    assert list(positions["shares"]) == [13.0, 13.0, 13.0]
    assert book["commission"].sum() == pytest.approx(0.35)
    assert book["equity"].iloc[-1] == pytest.approx(2000 + 13 * 3 - 0.35)


def test_intraday_flattens_every_session():
    cfg = config.resolve("investment", {"model.hold_overnight": False})
    panel = _panel([100.0, 100.0, 100.0], closes=[101.0, 99.0, 100.0])
    panel["exit_px"] = panel["close"]
    book, _ = run_book(_hold(panel), panel, panel["period"], cfg)
    # every session: one entry and one exit order, the exit paying half the spread
    assert book["commission"].sum() == pytest.approx(3 * 2 * 0.35)
    assert book["gross"].tolist() == pytest.approx([20.0, -20.0, 0.0])
    assert book["spread_cost"].iloc[0] == pytest.approx(20 * 101 * 4e-4 * 0.5)


def test_a_position_worth_more_than_the_budget_is_trimmed():
    # 20 x 101 = 2020$ exceeds the 2000$ basis (the per-name cap is vs the budget)
    cfg = config.resolve("investment")
    panel = _panel([100.0, 101.0, 101.0])
    _, positions = run_book(_hold(panel), panel, panel["period"][:-1], cfg)
    assert list(positions["shares"]) == [20.0, 19.0]


def test_the_budget_is_the_capital_basis_even_when_equity_grows():
    cfg = config.resolve("investment")
    panel = _panel([100.0, 200.0, 200.0, 200.0])
    _, positions = run_book(_hold(panel), panel, panel["period"][:-1], cfg)
    assert list(positions["shares"]) == [20.0, 10.0, 10.0]


def test_metrics():
    assert sharpe([0.01, -0.01, 0.01, -0.01]) == 0.0
    assert sharpe([0.0, 0.0]) == 0.0
    assert sharpe([0.02, 0.0]) == pytest.approx(np.sqrt(252))
    assert max_drawdown([1.0, 1.2, 0.9, 1.3]) == pytest.approx(0.9 / 1.2 - 1)
    assert max_drawdown([1.0, -0.1]) == -1.0


def test_summary_reports_net_of_costs():
    cfg = config.resolve("investment")
    panel = _panel([150.0, 151.0, 152.0])
    book, _ = run_book(_hold(panel), panel, panel["period"][:-1], cfg)
    s = summarize(book)
    assert s["n_periods"] == 2
    assert s["cumulative_return"] == pytest.approx((2000 + 26 - 0.35) / 2000 - 1)


def test_a_period_nothing_can_be_traded_in_is_not_a_book_period():
    from ml4trading.pipeline import window_periods

    panel = _panel([150.0, 151.0, 152.0])  # the last session has no next open
    periods = window_periods(panel, (panel["session"].min(), panel["session"].max()))
    assert periods == list(panel["period"][:-1])
    assert window_periods(panel, (panel["session"].min(), panel["session"].max()), 1) == [
        panel["period"].iloc[1]
    ]


def test_weekly_sharpe_is_annualized_by_blocks_per_year():
    from ml4trading.metrics import series_per_year

    assert series_per_year(390) == 252
    assert series_per_year(1950) == pytest.approx(50.4)
    assert sharpe([0.02, 0.0], series_per_year(1950)) == pytest.approx(np.sqrt(50.4))
