import pandas as pd
import pytest

from ml4trading import config
from ml4trading.config import FoldConfig
from ml4trading.folds import embargoed_end, fold_at, fold_boundaries


def test_fold_windows():
    f = fold_at("2020-01-01", FoldConfig(train_months=36, val_months=12))
    d = f.to_dict()
    assert d["train"] == ["2016-01-01", "2018-12-31"]
    assert d["val"] == ["2019-01-01", "2019-12-31"]
    assert d["deploy"] == ["2017-01-01", "2019-12-31"]
    assert d["test"] == ["2020-01-01", "2020-01-31"]
    assert f.name == "F2020-01"


def test_test_months_tile_the_path():
    cfg = FoldConfig(path_start="2020-01-01", path_end="2020-06-30")
    bs = fold_boundaries(cfg)
    assert [str(b.date()) for b in bs] == [f"2020-0{m}-01" for m in range(1, 7)]
    for a, b in zip(bs, bs[1:], strict=False):
        assert fold_at(a, cfg).test[1] + pd.Timedelta(days=1) == b


def test_fold_boundary_must_be_a_month_start():
    with pytest.raises(ValueError):
        fold_at("2020-01-15", FoldConfig())


def test_embargoed_end_drops_the_last_sessions():
    sessions = list(pd.bdate_range("2021-01-04", "2021-01-15"))
    assert embargoed_end(pd.Timestamp("2021-01-15"), sessions, 2) == pd.Timestamp("2021-01-13")
    assert embargoed_end(pd.Timestamp("2021-01-05"), sessions, 2) is None


def test_resolve_layers_method_defaults_over_class_defaults():
    cfg = config.resolve("investment")
    assert cfg.K == 390 and cfg.hold_overnight
    assert cfg.trader.number_of_symbols_to_buy == 1
    assert cfg.trader.max_symbol_weight == 1.0
    assert cfg.selector.name == "liquidity"


def test_unknown_keys_are_refused():
    with pytest.raises(ValueError):
        config.resolve("investment", {"trader.width": 3})
    with pytest.raises(ValueError):
        config.resolve("nope")


def test_grid_expands_every_combination():
    base = config.resolve("investment")
    cands = config.expand_grid(base, {"model.a": [1, 2], "model.b": ["x", "y", "z"]})
    assert len(cands) == 6
    assert cands[0].model["a"] == 1 and cands[-1].model["b"] == "z"


def test_round_trip_through_dict():
    cfg = config.resolve("investment", {"capital": 10000.0, "costs.spread_bps": 2.0})
    assert config.RunConfig.from_dict(cfg.to_dict()) == cfg
