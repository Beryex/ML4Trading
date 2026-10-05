"""Run configuration: every knob a train or backtest run reads, with its default.

A run is resolved from three layers, later ones winning: the class defaults below, the method's
block in ``METHOD_DEFAULTS``, and explicit overrides (a hyperparameter grid, or a test). Unknown
keys are refused by name, so a typo can never silently fall back to a default.
"""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field, fields, replace
from typing import Any

SESSION_MINUTES = 390  # one regular US equity session, 09:30-16:00 ET
BAR_MINUTES = 30  # the dataset's bar width
TRADING_DAYS_PER_YEAR = 252
TIMEZONE = "America/New_York"
BENCHMARK = "VOO"  # the market every book's alpha and beta are measured against


@dataclass(frozen=True)
class FoldConfig:
    """The walk-forward calendar (see ``ml4trading.folds``)."""

    train_months: int = 36
    val_months: int = 12
    step_months: int = 1
    path_start: str = "2020-01-01"  # first test month
    path_end: str = "2026-08-31"  # last test month must end on or before this day
    embargo_days: int = 5  # sessions dropped between a fit window and what it is scored on


@dataclass(frozen=True)
class CostConfig:
    """Transaction costs. The spread is ONE fixed quoted spread for every symbol and period; a
    leg that grows |position| (entry) pays ``entry_spread_fraction`` of it and a leg that
    shrinks |position| (exit) pays ``exit_spread_fraction`` of it -- entries are modelled as
    passive limit orders, exits as market orders. Commission is a tiered per-share schedule
    with a per-order minimum."""

    spread_bps: float = 4.0
    entry_spread_fraction: float = 0.0
    exit_spread_fraction: float = 0.5
    commission_per_share: float = 0.0035
    commission_min_per_order: float = 0.35
    borrow_rate_annual: float = 0.01  # charged on short notional, per held session

    @property
    def spread(self) -> float:
        return self.spread_bps * 1e-4


@dataclass(frozen=True)
class SelectorConfig:
    """``liquidity``: over the selection window, keep a symbol iff it has realized labels on
    enough sessions and its median per-period dollar volume clears the floor; keep the
    ``selection_cap`` most liquid."""

    name: str = "liquidity"
    selection_cap: int = 100
    min_dollar_volume: float = 1_000_000.0
    min_coverage_sessions: int = 20


@dataclass(frozen=True)
class TraderConfig:
    """``proportional``: per period, the top ``number_of_symbols_to_buy`` names by
    |decision metric| under the selection strategy, weighted proportionally to it, capped,
    then cost-gated (see ``ml4trading.trader``)."""

    name: str = "proportional"
    number_of_symbols_to_buy: int = 10
    max_symbol_weight: float = 0.7
    symbol_selection_strategy: str = "BOTH"  # BOTH | BUY_LONG | SELL_SHORT
    sizing_friction_theta: float = 2.0
    gross_exposure_cap: float = 1.0
    cost_safety_factor: float = 1.0


@dataclass(frozen=True)
class RunConfig:
    method: str
    K: int  # period minutes: a multiple of BAR_MINUTES; 390 = a session; 390 x N = N sessions
    model: dict[str, Any] = field(default_factory=dict)
    selector: SelectorConfig = field(default_factory=SelectorConfig)
    trader: TraderConfig = field(default_factory=TraderConfig)
    folds: FoldConfig = field(default_factory=FoldConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    capital: float = 20000.0  # the starting cash
    # True: every period sizes against the whole equity, so gains are reinvested. False: against
    # min(capital, equity) -- a fixed capital basis whose gains above it stay in cash.
    reinvest: bool = True

    @property
    def hold_overnight(self) -> bool:
        return bool(self.model["hold_overnight"])

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunConfig:
        return cls(
            method=d["method"],
            K=int(d["K"]),
            model=dict(d["model"]),
            selector=_build(SelectorConfig, d["selector"]),
            trader=_build(TraderConfig, d["trader"]),
            folds=_build(FoldConfig, d["folds"]),
            costs=_build(CostConfig, d["costs"]),
            capital=float(d["capital"]),
            reinvest=bool(d["reinvest"]),
        )


#: A single-ETF buy-and-hold: one name, the whole (1.0-capped) book, held overnight.
_SINGLE_ETF_HOLD: dict[str, Any] = {
    "K": 390,
    "model": {"hold_overnight": True},
    "selector": {"name": "liquidity"},
    "trader": {"name": "proportional", "number_of_symbols_to_buy": 1, "max_symbol_weight": 1.0},
}

#: Each method's own defaults over the class defaults above.
METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    # 100 % VOO (S&P 500): the buy-and-hold baseline every learned model is compared to.
    "investment": _SINGLE_ETF_HOLD,
    # 100 % QQQ (Nasdaq-100): the technology-heavy buy-and-hold baseline.
    "investment_tech": _SINGLE_ETF_HOLD,
    # Last week's change as the forecast (reverse: its negative), rebalanced once per 5-session
    # block, the default trader on top. ``--set K=390`` is the daily version.
    "momentum": {"K": 1950, "model": {"hold_overnight": True, "reverse": False}},
}


def _build(cls, values: dict):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(values) - known)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} key(s): {unknown}")
    return cls(**values)


def resolve(method: str, overrides: dict[str, Any] | None = None) -> RunConfig:
    """The method's RunConfig: class defaults < ``METHOD_DEFAULTS[method]`` < ``overrides``.

    ``overrides`` uses dotted keys (``"trader.number_of_symbols_to_buy"``, ``"model.alpha"``,
    ``"K"``); a ``model.*`` key may be new (models declare their own parameters), every other
    key must already exist."""
    if method not in METHOD_DEFAULTS:
        raise ValueError(f"unknown method {method!r}; known: {sorted(METHOD_DEFAULTS)}")
    base = METHOD_DEFAULTS[method]
    cfg = RunConfig(
        method=method,
        K=int(base["K"]),
        model=dict(base.get("model", {})),
        selector=_build(SelectorConfig, base.get("selector", {})),
        trader=_build(TraderConfig, base.get("trader", {})),
    )
    return apply_overrides(cfg, overrides or {})


def apply_overrides(cfg: RunConfig, overrides: dict[str, Any]) -> RunConfig:
    for key, value in overrides.items():
        section, _, name = key.partition(".")
        if not name:
            if section not in ("K", "capital", "reinvest"):
                raise ValueError(f"unknown top-level key {key!r}")
            cfg = replace(cfg, **{section: value})
        elif section == "model":
            cfg = replace(cfg, model={**cfg.model, name: value})
        elif section in ("selector", "trader", "folds", "costs"):
            sub = getattr(cfg, section)
            if name not in {f.name for f in fields(sub)}:
                raise ValueError(f"unknown key {key!r}")
            cfg = replace(cfg, **{section: replace(sub, **{name: value})})
        else:
            raise ValueError(f"unknown section in {key!r}")
    validate(cfg)
    return cfg


def expand_grid(cfg: RunConfig, grid: dict[str, list] | None) -> list[RunConfig]:
    """Every combination of ``grid`` (dotted key -> list of values) applied to ``cfg``, in
    the grid's own order; no grid -> ``[cfg]``. A one-element list is a pin."""
    if not grid:
        return [cfg]
    keys = list(grid)
    for k in keys:
        if not isinstance(grid[k], list) or not grid[k]:
            raise ValueError(f"grid key {k!r} needs a non-empty list")
    return [
        apply_overrides(cfg, dict(zip(keys, combo, strict=True)))
        for combo in itertools.product(*(grid[k] for k in keys))
    ]


def validate(cfg: RunConfig) -> None:
    from ml4trading.periods import check_k

    check_k(cfg.K)
    if "hold_overnight" not in cfg.model:
        raise ValueError("model.hold_overnight is required")
    if cfg.K > SESSION_MINUTES and not cfg.model["hold_overnight"]:
        raise ValueError(f"K={cfg.K} spans several sessions: it needs model.hold_overnight")
    if cfg.trader.symbol_selection_strategy not in ("BOTH", "BUY_LONG", "SELL_SHORT"):
        raise ValueError(f"unknown strategy {cfg.trader.symbol_selection_strategy!r}")
    if cfg.folds.step_months != 1:
        raise ValueError("folds.step_months must be 1: every fold is one monthly refit")
