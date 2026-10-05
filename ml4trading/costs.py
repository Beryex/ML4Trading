"""Transaction-cost formulas (the policy and its defaults: ``ml4trading.config.CostConfig``).

A change of position is split into LEGS: the part that grows |position| is an entry, the part
that shrinks it is an exit, and a sign flip is a full exit followed by a full entry. Each leg
pays its spread fraction times the quoted spread on its notional, plus one commission order.
"""

from __future__ import annotations

import numpy as np

from ml4trading.config import CostConfig


def commission(shares, cfg: CostConfig):
    """Commission of ONE order of ``shares`` (scalar or array): max(per_share * |shares|,
    min_per_order), and 0 when no share trades."""
    sh = np.abs(np.asarray(shares, dtype=float))
    out = np.where(
        sh > 0.0, np.maximum(cfg.commission_per_share * sh, cfg.commission_min_per_order), 0.0
    )
    return out if out.ndim else float(out)


def leg_amounts(prev: float, cur: float) -> tuple[float, float]:
    """(entry, exit) amounts of prev -> cur; entry + exit == |cur - prev|."""
    if prev * cur < 0:
        return abs(cur), abs(prev)
    d = abs(cur) - abs(prev)
    return (d, 0.0) if d > 0 else (0.0, -d)


def leg_cost(entry_shares: float, exit_shares: float, price: float, cfg: CostConfig):
    """(spread $, commission $) of one symbol's entry and exit legs at ``price``."""
    spread = (
        price
        * cfg.spread
        * (cfg.entry_spread_fraction * entry_shares + cfg.exit_spread_fraction * exit_shares)
    )
    return spread, commission(entry_shares, cfg) + commission(exit_shares, cfg)


def round_trip_spread_fraction(cfg: CostConfig) -> float:
    """A round trip's spread cost in units of the quoted spread."""
    return cfg.entry_spread_fraction + cfg.exit_spread_fraction
