import numpy as np
import pandas as pd
import pytest

from ml4trading import config
from ml4trading.book import POSITION_COLUMNS, run_book
from ml4trading.metrics import alpha_beta, trade_stats


def _rows(symbol, seq):
    """seq: (prev_shares, shares, gross, entry_cost, exit_cost[, session_end, close_cost])."""
    days = pd.bdate_range("2021-01-04", periods=len(seq))
    out = []
    for d, item in zip(days, seq, strict=True):
        before, after, gross, ec, xc, *rest = item
        end, cc = (rest + [False, 0.0])[:2]
        row = dict.fromkeys(POSITION_COLUMNS, 0.0)
        row.update(
            period=d,
            symbol=symbol,
            prev_shares=before,
            shares=after,
            price=10.0,
            gross=gross,
            entry_commission=ec,
            exit_commission=xc,
            close_commission=cc,
            session_end=end,
        )
        out.append(row)
    return out


def test_a_trade_runs_from_flat_to_flat_and_carries_its_costs():
    rows = _rows(
        "A",
        [
            (0, 10, 5.0, 1.0, 0.0),
            (10, 15, 2.0, 1.0, 0.0),
            (15, 5, -1.0, 0.0, 1.0),
            (5, 0, 0.0, 0.0, 1.0),
        ],
    )
    s = trade_stats(pd.DataFrame(rows))
    assert s["n_trades"] == 1
    assert s["mean_trade_pnl"] == pytest.approx(5 + 2 - 1 - 1 - 1 - 1 - 1)
    assert s["trade_hit_rate"] == 1.0 and s["open_trades_at_end"] == 0
    assert s["median_trade_periods"] == 3


def test_a_flip_closes_one_trade_and_opens_the_next():
    rows = _rows("A", [(0, 10, -3.0, 1.0, 0.0), (10, -10, 8.0, 1.0, 1.0), (-10, 0, 0.0, 0.0, 1.0)])
    s = trade_stats(pd.DataFrame(rows))
    assert s["n_trades"] == 2
    assert s["trade_hit_rate"] == 0.5  # the long lost 3 + 1 + 1, the short made 8 - 1 - 1
    assert s["mean_trade_pnl"] == pytest.approx(((-3 - 1 - 1) + (8 - 1 - 1)) / 2)


def test_open_positions_at_the_end_are_not_completed_trades():
    s = trade_stats(pd.DataFrame(_rows("VOO", [(0, 10, 1.0, 1.0, 0.0), (10, 10, 1.0, 0.0, 0.0)])))
    assert s["n_trades"] == 0 and s["trade_hit_rate"] is None and s["open_trades_at_end"] == 1


def test_intraday_trades_end_at_the_session_close():
    rows = _rows("A", [(0, 10, 2.0, 1.0, 0.0, True, 0.5), (0, 10, -1.0, 1.0, 0.0, True, 0.5)])
    s = trade_stats(pd.DataFrame(rows))
    assert s["n_trades"] == 2 and s["trade_hit_rate"] == 0.5


def test_alpha_and_beta_recover_a_linear_relation():
    rng = np.random.default_rng(0)
    b = pd.Series(rng.normal(0, 0.01, 500))
    r = 0.0004 + 1.5 * b
    out = alpha_beta(r, b, per_year=252)
    assert out["beta"] == pytest.approx(1.5)
    assert out["alpha"] == pytest.approx(0.0004 * 252)


def test_the_book_records_every_symbol_it_touches():
    days = pd.bdate_range("2021-01-04", periods=4)
    opens = [100.0, 110.0, 105.0, 120.0]
    panel = pd.DataFrame(
        {
            "period": days,
            "session": days,
            "symbol": "A",
            "open": opens,
            "high": opens,
            "low": opens,
            "close": opens,
            "volume": 1e6,
            "open_px": opens,
            "exit_px": opens[1:] + [np.nan],
            "div_ret": 0.0,
        }
    )
    panel["y"] = panel["exit_px"] / panel["open_px"] - 1
    preds = pd.DataFrame({"period": days[:3], "symbol": "A", "prediction": [0.05, 0.0, 0.0]})
    cfg = config.resolve("investment", {"capital": 2000.0})
    book, positions = run_book(preds, panel, days[:3], cfg)
    assert list(positions["shares"]) == [20.0, 0.0]  # bought, then sold when the signal is 0
    s = trade_stats(positions)
    assert s["n_trades"] == 1
    assert s["mean_trade_pnl"] == pytest.approx(book["net"].sum())  # one trade = the whole book
