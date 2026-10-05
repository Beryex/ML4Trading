import numpy as np
import pandas as pd
import pytest

from ml4trading.config import CostConfig, SelectorConfig, TraderConfig
from ml4trading.costs import commission, leg_amounts
from ml4trading.selector import select_universe
from ml4trading.sizing import dollar_targets, size_shares
from ml4trading.trader import apply_caps, target_weights

COSTS = CostConfig()


# -- costs ---------------------------------------------------------------------------------


def test_commission_has_a_per_order_minimum():
    assert commission(10, COSTS) == pytest.approx(0.35)
    assert commission(1000, COSTS) == pytest.approx(3.5)
    assert commission(0, COSTS) == 0.0


def test_legs_split_a_sign_flip_into_exit_then_entry():
    assert leg_amounts(5, 8) == (3, 0.0)
    assert leg_amounts(8, 5) == (0.0, 3)
    assert leg_amounts(5, -2) == (2, 5)


# -- trader --------------------------------------------------------------------------------


def test_caps_water_fill():
    assert apply_caps(np.array([1.0]), 0.5, 1.0) == pytest.approx([0.5])
    assert apply_caps(np.array([0.6, 0.6]), 1.0, 0.8).sum() == pytest.approx(0.8)
    assert apply_caps(np.full(4, 0.25), 0.1, 1.0) == pytest.approx([0.1] * 4)
    w = apply_caps(np.array([0.7, 0.2, 0.1]), 0.5, 1.0)
    assert w == pytest.approx([0.5, 0.2 + 0.2 * 2 / 3, 0.1 + 0.2 / 3])


def _scores(preds, px=100.0):
    return pd.DataFrame({"symbol": list(preds), "prediction": list(preds.values()), "open_px": px})


def test_top_n_by_abs_prediction_proportional_and_signed():
    trader = TraderConfig(number_of_symbols_to_buy=2, max_symbol_weight=1.0)
    w = target_weights(
        _scores({"A": 0.03, "B": -0.01, "C": 0.005}),
        trader=trader,
        costs=COSTS,
        capital=1e6,
        hold_overnight=True,
    )
    assert w == pytest.approx({"A": 0.75, "B": -0.25})


def test_buy_long_ignores_negative_predictions():
    trader = TraderConfig(number_of_symbols_to_buy=2, symbol_selection_strategy="BUY_LONG")
    w = target_weights(
        _scores({"A": 0.03, "B": -0.05, "C": 0.01}),
        trader=trader,
        costs=COSTS,
        capital=1e6,
        hold_overnight=True,
    )
    assert set(w) == {"A", "C"}


def test_cost_gate_drops_a_prediction_below_its_round_trip_cost():
    trader = TraderConfig(number_of_symbols_to_buy=2, max_symbol_weight=1.0)
    # 0.5 x 4 bp spread = 2 bp, plus two $0.35 commissions on ~$1000 = 7 bp
    w = target_weights(
        _scores({"A": 0.02, "B": 0.0005}),
        trader=trader,
        costs=COSTS,
        capital=2000.0,
        hold_overnight=True,
    )
    assert w == pytest.approx({"A": 1.0})


def test_intraday_metric_is_net_of_the_spread():
    trader = TraderConfig(number_of_symbols_to_buy=2, max_symbol_weight=1.0)
    w = target_weights(
        _scores({"A": 0.0001, "B": 0.01}),
        trader=trader,
        costs=COSTS,
        capital=1e6,
        hold_overnight=False,
    )
    assert w == pytest.approx({"B": 1.0})


# -- selector ------------------------------------------------------------------------------


def test_liquidity_selector_coverage_floor_and_cap():
    sessions = pd.bdate_range("2021-01-04", periods=40)
    rows = []
    for s in sessions:
        rows.append(("LIQ", s, 100.0, 1e5, 0.01))  # $10M/day
        rows.append(("MID", s, 50.0, 1e5, 0.01))  # $5M/day
        rows.append(("THIN", s, 10.0, 1e4, 0.01))  # $100k/day: below the floor
    for s in sessions[:10]:
        rows.append(("NEW", s, 100.0, 1e6, 0.01))  # liquid but only 10 labelled sessions
    panel = pd.DataFrame(rows, columns=["symbol", "session", "close", "volume", "y"])
    window = (sessions[0], sessions[-1])
    assert select_universe(panel, window, SelectorConfig()) == ["LIQ", "MID"]
    assert select_universe(panel, window, SelectorConfig(selection_cap=1)) == ["LIQ"]


# -- sizing --------------------------------------------------------------------------------


def test_floor_then_greedy_top_up_within_budget():
    trader = TraderConfig(max_symbol_weight=1.0, sizing_friction_theta=0.0)
    shares = size_shares(
        {"A": 0.5, "B": 0.5}, {"A": 30.0, "B": 45.0}, 1000.0, {}, {}, trader=trader, costs=COSTS
    )
    # floors 16 x 30 = 480 and 11 x 45 = 495: A's 20$ deficit is worth a share but 30$ more
    # would overspend the budget; B's 5$ deficit is under half a share
    assert shares == {"A": 16.0, "B": 11.0}
    shares = size_shares(
        {"A": 0.5, "B": 0.5}, {"A": 30.0, "B": 45.0}, 1010.0, {}, {}, trader=trader, costs=COSTS
    )
    assert shares == {"A": 17.0, "B": 11.0}  # 505$ target, 25$ deficit, room for one more


def test_an_unaffordable_name_is_dropped_and_the_rest_renormalized():
    trader = TraderConfig(max_symbol_weight=1.0, sizing_friction_theta=0.0)
    t = dollar_targets({"A": 0.9, "B": 0.1}, {"A": 10.0, "B": 500.0}, 1000.0, 0.0, trader)
    assert t == pytest.approx({"A": 1000.0})


def test_carried_names_keep_their_shares_and_reserve_their_value():
    trader = TraderConfig(max_symbol_weight=1.0, sizing_friction_theta=0.0)
    shares = size_shares(
        {"A": 1.0}, {"A": 10.0}, 1000.0, {"Z": 4.0}, {"Z": 100.0}, trader=trader, costs=COSTS
    )
    assert shares["Z"] == 4.0
    assert shares["A"] * 10.0 <= 1000.0 - 400.0


def _size(weights, prices, prev, theta):
    trader = TraderConfig(max_symbol_weight=1.0, sizing_friction_theta=theta)
    return size_shares(weights, prices, 1000.0, prev, {}, trader=trader, costs=COSTS)


def test_friction_keeps_a_held_share_whose_sale_is_not_worth_its_cost():
    weights, prices = {"A": 0.49485, "B": 0.4}, {"A": 10.0, "B": 10.0}
    # target 494.85$: floor 49. Selling the 50th share cuts the mismatch from 5.15$ to 4.85$,
    # 0.30$ -- less than theta x its cost (2 x (0.35 + 0.002)): keep it.
    assert _size(weights, prices, {"A": 50.0}, theta=2.0)["A"] == 50.0
    assert _size(weights, prices, {"A": 50.0}, theta=0.0)["A"] == 49.0


def test_friction_sells_when_the_mismatch_it_removes_pays_for_the_sale():
    # target 500$ at 49$: floor 10; holding 11 is 39$ over, selling one leaves 10$ under
    assert _size({"A": 0.5, "B": 0.5}, {"A": 49.0, "B": 10.0}, {"A": 11.0}, theta=2.0)["A"] == 10.0


def test_a_kept_position_never_breaches_the_name_cap():
    # target 1000$ at 101$: held 10 would be 1010$ > the 1000$ cap -> trimmed to 9
    assert _size({"A": 1.0}, {"A": 101.0}, {"A": 10.0}, theta=2.0) == {"A": 9.0}
