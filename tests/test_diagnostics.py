import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from ml4trading.book import POSITION_COLUMNS
from ml4trading.config import CostConfig, RunConfig, TraderConfig
from ml4trading.diagnostics import (
    dispersion,
    exposure_stats,
    gate_stats,
    ic_summary,
    leg_pnl,
    period_ic,
    validation_diagnostics,
)
from ml4trading.trader import apply_caps, cost_gate, decision_metric, select, target_weights

COSTS = CostConfig()
DAYS = pd.bdate_range("2021-01-04", periods=4)


def _scored(rows):
    """rows: (period index, symbol, prediction, y[, label]) -> a scored frame."""
    out = []
    for period, symbol, prediction, y, *rest in rows:
        out.append(
            {
                "period": DAYS[period],
                "symbol": symbol,
                "prediction": prediction,
                "y": y,
                "label": rest[0] if rest else y,
                "open_px": 100.0,
            }
        )
    return pd.DataFrame(out, columns=["period", "symbol", "prediction", "y", "label", "open_px"])


def _positions(rows):
    """rows: dicts over POSITION_COLUMNS; the omitted columns are 0."""
    out = []
    for r in rows:
        row = dict.fromkeys(POSITION_COLUMNS, 0.0)
        row.update(r)
        out.append(row)
    return pd.DataFrame(out, columns=POSITION_COLUMNS)


def _book(equity_before):
    return pd.DataFrame(
        {"period": DAYS[: len(equity_before)], "equity_before": [float(e) for e in equity_before]}
    )


def _cfg():
    trader = TraderConfig(number_of_symbols_to_buy=3, max_symbol_weight=1.0)
    return RunConfig(
        method="linear",
        K=390,
        model={"hold_overnight": True},
        trader=trader,
        capital=2000.0,
        reinvest=False,
    )


# Three names priced at 100 with $2000: the weights are |p| / sum|p| = 0.5, 0.2667, 0.2333, so the
# $0.35 minimum commission dominates and each name's gate is 2 bp of spread + 0.7 / its notional:
# A 9 bp, B 15.125 bp, C 17 bp. B (16 bp) sits just above its gate, C (14 bp) just below.
PREDS = {"A": 0.0030, "B": 0.0016, "C": 0.0014}
GATES = [0.0009, 0.0015125, 0.0017]


def _gate_scores(preds=PREDS):
    return pd.DataFrame(
        {"symbol": list(preds), "prediction": list(preds.values()), "open_px": 100.0}
    )


# -- the cost gate -------------------------------------------------------------------------


def test_cost_gate_matches_the_trader():
    trader = TraderConfig(number_of_symbols_to_buy=3, max_symbol_weight=1.0)
    scores = _gate_scores()
    pred = scores["prediction"].to_numpy()
    metric = decision_metric(pred, True, trader, COSTS)
    pos = select(metric, scores["symbol"].to_numpy(), 3, "BOTH")
    absm = np.abs(metric[pos])
    w = apply_caps(absm / absm.sum(), 1.0, 1.0)
    px = scores["open_px"].to_numpy()[pos]

    gate = cost_gate(w, px, trader=trader, costs=COSTS, capital=2000.0)
    assert gate == pytest.approx(GATES)
    symbols, pred = scores["symbol"].to_numpy()[pos], pred[pos]
    kept = {s for s, p, g in zip(symbols, pred, gate, strict=True) if p > g}
    assert kept == {"A", "B"}
    weights = target_weights(
        scores, trader=trader, costs=COSTS, capital=2000.0, hold_overnight=True
    )
    assert set(weights) == kept

    # the safety factor scales the whole estimate
    doubled = TraderConfig(cost_safety_factor=2.0)
    both = cost_gate(w, px, trader=doubled, costs=COSTS, capital=2000.0)
    assert both == pytest.approx(2 * gate)
    # a zero notional can never be afforded
    flat = cost_gate(np.array([0.0]), np.array([100.0]), trader=trader, costs=COSTS, capital=1.0)
    assert flat[0] == np.inf


# -- the information coefficient -----------------------------------------------------------


def test_period_ic():
    scored = _scored(
        [
            (0, "A", 1.0, 0.1),
            (0, "B", 2.0, 0.3),
            (0, "C", 3.0, 0.2),
            (1, "A", 1.0, 0.1),  # one row
            (2, "A", 5.0, 0.1),  # constant predictions
            (2, "B", 5.0, 0.3),
            (2, "C", 5.0, 0.2),
        ]
    )
    ic = period_ic(scored)
    assert list(ic.index) == list(DAYS[:3])
    assert ic.iloc[0] == pytest.approx(0.5)
    assert ic.iloc[1:].isna().all()
    s = ic_summary(ic)
    assert s["ic_periods"] == 1
    assert s["ic_mean"] == pytest.approx(0.5)
    assert s["ic_std"] is None and s["ic_t"] is None
    assert s["ic_positive_share"] == 1.0


def test_ic_summary_statistics():
    s = ic_summary(pd.Series([0.5, 1.0, np.nan]))
    assert s["ic_periods"] == 2
    assert s["ic_mean"] == pytest.approx(0.75)
    assert s["ic_std"] == pytest.approx(np.sqrt(0.125))  # ddof 1
    assert s["ic_t"] == pytest.approx(3.0)  # 0.75 / (std / sqrt(2))
    assert ic_summary(pd.Series([0.5, -0.5]))["ic_positive_share"] == 0.5
    none = ic_summary(pd.Series([np.nan, np.nan]))
    assert none == {
        "ic_mean": None,
        "ic_std": None,
        "ic_t": None,
        "ic_positive_share": None,
        "ic_periods": 0,
    }
    assert ic_summary(pd.Series([0.3, 0.3]))["ic_t"] is None  # no spread, no t


def test_period_ic_ignores_rows_without_a_realized_return():
    scored = _scored(
        [
            (0, "A", 1.0, 0.1),
            (0, "B", 2.0, 0.3),
            (0, "C", 3.0, 0.2),
            (0, "D", 9.0, np.nan),  # a symbol's last period has no return
            (1, "A", 1.0, np.nan),
            (1, "B", 2.0, np.nan),
        ]
    )
    ic = period_ic(scored)
    assert ic.iloc[0] == pytest.approx(0.5)  # the NaN row would otherwise change the ranks
    assert np.isnan(ic.iloc[1])
    assert ic_summary(ic)["ic_periods"] == 1
    assert period_ic(scored.iloc[:0]).empty


# -- the long and short legs ---------------------------------------------------------------


def test_leg_pnl_splits_by_held_side():
    positions = _positions(
        [
            {"symbol": "L", "shares": 5.0, "gross": 10.0, "entry_spread": 1.0},
            {"symbol": "S", "shares": -2.0, "gross": -4.0, "borrow": 0.5},
            # exited this period: the short it closes carries the exit cost
            {"symbol": "X", "prev_shares": -5.0, "shares": 0.0, "exit_commission": 0.35},
        ]
    )
    leg = leg_pnl(positions)
    assert leg["long_gross"] == 10.0
    assert leg["long_net"] == 9.0
    assert leg["short_gross"] == -4.0
    assert leg["short_net"] == pytest.approx(-4.85)


def test_leg_pnl_charges_every_cost_column():
    costs = [c for c in POSITION_COLUMNS if "spread" in c or "commission" in c] + ["borrow"]
    assert len(costs) == 7
    positions = _positions(
        [{"symbol": "L", "shares": 1.0, "gross": 10.0, **dict.fromkeys(costs, 1)}]
    )
    assert leg_pnl(positions)["long_net"] == 3.0


# -- exposure and same-sign periods --------------------------------------------------------


def test_exposure_stats():
    positions = _positions(
        [
            {"period": DAYS[0], "symbol": "A", "shares": 2.0, "price": 50.0},
            {"period": DAYS[0], "symbol": "B", "shares": -1.0, "price": 100.0},
            {"period": DAYS[1], "symbol": "A", "shares": 3.0, "price": 50.0},
            # carried at no price: it holds, but adds nothing to the exposure
            {"period": DAYS[1], "symbol": "C", "shares": 4.0, "price": np.nan},
            # period 2 holds nothing: only a closed row
            {"period": DAYS[2], "symbol": "A", "prev_shares": 3.0, "shares": 0.0, "price": 50.0},
        ]
    )
    stats = exposure_stats(positions, _book([1000, 1500, 1500]))
    # period 0: net (100 - 100) / 1000 = 0, gross 200 / 1000; period 1: 150 / 1500 both ways
    assert stats["same_sign_share"] == 0.5
    assert stats["mean_net_exposure"] == pytest.approx(0.05)
    assert stats["mean_gross_exposure"] == pytest.approx(0.15)
    assert stats["holding_share"] == pytest.approx(2 / 3)


def test_an_empty_book_gives_none():
    for positions in (
        _positions([{"period": DAYS[0], "symbol": "A"}, {"period": DAYS[1], "symbol": "A"}]),
        _positions([]),
    ):
        stats = exposure_stats(positions, _book([1000, 1000]))
        assert stats["same_sign_share"] is None
        assert stats["mean_net_exposure"] is None and stats["mean_gross_exposure"] is None
        assert stats["holding_share"] == 0.0  # 0 of 2 periods hold: defined, just zero
        leg = leg_pnl(positions)
        assert leg == {"long_gross": 0.0, "long_net": 0.0, "short_gross": 0.0, "short_net": 0.0}
    no_rows = gate_stats(_scored([]), _book([1000]), _cfg())
    assert no_rows == {
        "gate_selected_mean": None,
        "gate_pass_mean": None,
        "gate_ratio_q10": None,
        "gate_ratio_q50": None,
        "gate_ratio_q90": None,
    }


# -- the cost-gate statistics --------------------------------------------------------------


def _gate_rows():
    rows = [(0, s, p, 0.0) for s, p in PREDS.items()]
    return rows + [(1, "A", 0.5, 0.0)]  # period 1: one name, gate 5.5 bp, ratio 909.09


def test_gate_stats():
    ratios = [0.0030 / GATES[0], 0.0016 / GATES[1], 0.0014 / GATES[2], 0.5 / 0.00055]
    stats = gate_stats(_scored(_gate_rows()), _book([2000, 2000]), _cfg())
    assert stats["gate_selected_mean"] == pytest.approx((3 + 1) / 2)
    assert stats["gate_pass_mean"] == pytest.approx((2 + 1) / 2)
    q10, q50, q90 = np.quantile(ratios, [0.1, 0.5, 0.9])
    assert stats["gate_ratio_q10"] == pytest.approx(q10)
    assert stats["gate_ratio_q50"] == pytest.approx(q50)
    assert stats["gate_ratio_q90"] == pytest.approx(q90)


def test_gate_stats_sizes_against_the_equity_when_reinvesting():
    scored = _scored(_gate_rows()[:3])
    book = _book([2000])
    # reinvest: the capital is the period's equity, whatever cfg.capital says
    rich = replace(_cfg(), capital=1e9, reinvest=True)
    assert gate_stats(scored, book, rich) == gate_stats(scored, book, _cfg())
    # fixed basis: the 1e9 capital makes the commission negligible, so every name passes
    fixed = replace(_cfg(), capital=1e9)
    assert gate_stats(scored, book, fixed)["gate_pass_mean"] == 3.0


def test_gate_stats_skips_unpriced_rows_and_a_ruined_period():
    scored = _scored(_gate_rows()[:3])
    scored.loc[scored["symbol"] == "A", "open_px"] = np.nan  # not tradeable: never reaches the gate
    stats = gate_stats(scored, _book([2000]), _cfg())
    assert stats["gate_selected_mean"] == 2.0
    # equity <= 0 holds nothing, so the period does not count
    assert gate_stats(_scored(_gate_rows()[:3]), _book([0]), _cfg())["gate_pass_mean"] is None


# -- dispersion ----------------------------------------------------------------------------


def test_dispersion_is_the_median_cross_sectional_std():
    scored = _scored(
        [
            (0, "A", 0.0, 0.0, 0.0),
            (0, "B", 2.0, 0.0, 1.0),
            (1, "A", 0.0, 0.0, 0.0),
            (1, "B", 4.0, 0.0, 2.0),
            (2, "A", 0.0, 0.0, 0.0),
            (2, "B", 20.0, 0.0, 6.0),
        ]
    )
    d = dispersion(scored)  # per-period stds (ddof 0): 1, 2, 10 and 0.5, 1, 3
    assert d == {"pred_xs_std": pytest.approx(2.0), "label_xs_std": pytest.approx(1.0)}
    assert dispersion(scored.iloc[:0]) == {"pred_xs_std": None, "label_xs_std": None}


# -- the merged block ----------------------------------------------------------------------

KEYS = {
    "net_sharpe",
    "gross_sharpe",
    "alpha",
    "beta",
    "alpha_t",
    "ic_mean",
    "ic_std",
    "ic_t",
    "ic_positive_share",
    "ic_periods",
    "long_gross",
    "long_net",
    "short_gross",
    "short_net",
    "same_sign_share",
    "mean_net_exposure",
    "mean_gross_exposure",
    "holding_share",
    "gate_selected_mean",
    "gate_pass_mean",
    "gate_ratio_q10",
    "gate_ratio_q50",
    "gate_ratio_q90",
    "pred_xs_std",
    "label_xs_std",
    "r2_mse",
}


def test_validation_diagnostics_is_json_safe():
    scored = _scored(
        [
            (0, "A", 0.0030, 0.1, 1.0),
            (0, "B", 0.0016, 0.3, 3.0),
            (0, "C", 0.0014, 0.2, 2.0),
            (1, "A", 0.5, 0.1, 5.0),  # one row: this period's IC is NaN
        ]
    )
    positions = _positions([{"period": DAYS[0], "symbol": "A", "shares": 2.0, "price": 50.0}])
    metrics = {"net_sharpe": 1.5, "gross_sharpe": float("nan"), "n_days": 3, "alpha": None}
    out = validation_diagnostics(scored, _book([2000, 2000]), positions, metrics, _cfg(), 2.0)
    assert set(out) == KEYS
    json.dumps(out, allow_nan=False)  # no NaN, no numpy scalar
    assert out["net_sharpe"] == 1.5
    assert out["gross_sharpe"] is None and out["alpha"] is None and out["beta"] is None
    assert out["ic_periods"] == 1
    assert out["holding_share"] == 0.5 and out["same_sign_share"] == 1.0
    # labels 1, 3, 2, 5 against the fit mean 2: MSE (1 + 1 + 0 + 9) / 4 = 11/4
    expected = 1 - np.mean(np.square(np.array([1, 3, 2, 5]) - scored["prediction"])) / (11 / 4)
    assert out["r2_mse"] == pytest.approx(expected)


def test_validation_diagnostics_without_a_fit_mean_or_a_held_name():
    # tiny predictions: the gate rejects every name, so nothing is ever held
    scored = _scored([(p, s, 1e-6, 0.1 * (i + 1)) for p in (0, 1) for i, s in enumerate("ABC")])
    out = validation_diagnostics(
        scored, _book([2000, 2000]), _positions([]), {}, _cfg(), fit_label_mean=None
    )
    json.dumps(out, allow_nan=False)
    assert out["r2_mse"] is None and out["net_sharpe"] is None
    assert out["long_net"] == 0.0 and out["short_net"] == 0.0
    assert out["same_sign_share"] is None and out["mean_net_exposure"] is None
    assert out["holding_share"] == 0.0
    assert out["gate_selected_mean"] == 3.0 and out["gate_pass_mean"] == 0.0
    assert 0 < out["gate_ratio_q90"] < 1
    assert out["ic_periods"] == 0  # constant predictions: every IC is undefined
