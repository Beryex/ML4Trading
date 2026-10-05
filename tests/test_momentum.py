import pandas as pd
import pytest

from ml4trading import config
from ml4trading.models import get_model
from ml4trading.pipeline import load_panel
from ml4trading.testing import assert_causal


def _panel():
    days = pd.bdate_range("2021-01-04", periods=4)
    rows = []
    for sym, closes in {"A": [10.0, 11.0, 9.9, 9.9], "B": [20.0, 20.0, 21.0, 22.05]}.items():
        for d, c in zip(days, closes, strict=True):
            rows.append({"period": d, "session": d, "symbol": sym, "close": c})
    return pd.DataFrame(rows)


@pytest.mark.parametrize("reverse, sign", [(False, 1.0), (True, -1.0)])
def test_prediction_is_the_previous_periods_change(reverse, sign):
    panel = _panel()
    model = get_model("momentum")(hold_overnight=True, reverse=reverse)
    out = model.predict(panel, (panel["period"].min(), panel["period"].max()))
    got = {(r.symbol, r.period.day): r.prediction for r in out.itertuples()}
    # day 6 is predicted from day 5's change over day 4, day 7 from day 6's over day 5
    assert got == pytest.approx(
        {
            ("A", 6): sign * 0.1,
            ("A", 7): sign * -0.1,
            ("B", 6): 0.0,
            ("B", 7): sign * 0.05,
        }
    )


def test_defaults():
    cfg = config.resolve("momentum")
    assert cfg.K == 1950 and cfg.hold_overnight and cfg.model["reverse"] is False
    assert cfg.trader.symbol_selection_strategy == "BOTH"


def test_momentum_is_causal(data_dir):
    cfg = config.resolve("momentum")
    _, panel = load_panel(cfg, data_dir)
    for reverse in (False, True):
        model = get_model("momentum")(hold_overnight=True, reverse=reverse)
        assert_causal(model, panel, (pd.Timestamp("2021-06-01"), pd.Timestamp("2021-06-30")))


def test_weekly_momentum_trains_and_replays(data_dir, tmp_path):
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
    gaps = book["period"].diff().dropna().dt.days
    assert gaps.min() >= 5  # one decision per five-session block
    assert 20 <= metrics["n_periods"] <= 27


def test_weekly_k_needs_an_overnight_hold():
    with pytest.raises(ValueError):
        config.resolve("momentum", {"model.hold_overnight": False})
