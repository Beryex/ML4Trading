"""Target weights -> whole-share counts (budget-filling, with switching friction).

Inputs per period: signed target weights of the TRADEABLE names (each has a usable entry price),
their prices, the budget ``capital`` (= min(capital basis, equity)), the previous share book,
and the CARRIED names -- held names with no price this period, which cannot trade: they keep
their shares, and their value at their last observed price is RESERVED out of the budget.

1. Dollar targets (affordable set). With nothing reserved, target = |w| x capital. Otherwise,
   and after any drop, the survivors split the free budget gross_exposure_cap x capital -
   reserved in proportion to |w|, each clamped to max_symbol_weight x capital. While some
   survivor's price exceeds twice its target (it rounds to zero shares), the single worst one
   (largest price / target, ties by symbol) is dropped and the rest renormalized.
2. Floors: shares = floor(target / price).
3. Greedy top-up: repeatedly add one share to the name with the largest dollar deficit
   (target - held $, ties by symbol) while (a) total spend + reserved + price <= capital,
   (b) (shares + 1) x price <= max_symbol_weight x capital, and (d) the deficit is at least
   half a share -- plus, with friction, theta x the marginal trading cost of that share.
Switching friction (theta > 0) applies to DISCRETIONARY names -- already held, same sign, still
targeted: a held name above its floor keeps its shares unless selling down improves the
dollar mismatch |shares x price - target| by more than theta x the cost of the sale; then any
kept name breaching the per-name cap or the budget is trimmed share by share (largest dollar
overweight first) back towards its floor. New entries, exits and sign flips always execute.
"""

from __future__ import annotations

import math

from ml4trading.config import CostConfig, TraderConfig
from ml4trading.costs import leg_amounts, leg_cost


def _usable(px) -> bool:
    return px is not None and px == px and 0 < px < math.inf


def dollar_targets(
    weights: dict, prices: dict, capital: float, reserved: float, trader: TraderConfig
) -> dict:
    alive = {
        s: abs(w) for s, w in weights.items() if w == w and w != 0.0 and _usable(prices.get(s))
    }
    free = trader.gross_exposure_cap * capital - reserved
    if free <= 0:
        return {}
    identity = reserved == 0.0
    while alive:
        if identity:
            targets = {s: aw * capital for s, aw in alive.items()}
        else:
            total = sum(alive.values())
            targets = {
                s: min(aw / total * free, trader.max_symbol_weight * capital)
                for s, aw in alive.items()
            }
        worst_ratio, worst = 2.0, None
        for s in sorted(alive):
            t = targets[s]
            ratio = prices[s] / t if t > 0 else math.inf
            if ratio > worst_ratio:
                worst_ratio, worst = ratio, s
        if worst is None:
            return targets
        del alive[worst]
        identity = False
    return {}


def _floor(target: float, px: float) -> int:
    return math.floor(target / px + 1e-9)


def _friction(entry_sh: float, exit_sh: float, px: float, costs: CostConfig) -> float:
    spread, comm = leg_cost(entry_sh, exit_sh, px, costs)
    return spread + comm


def size_shares(
    weights: dict,
    prices: dict,
    capital: float,
    prev: dict,
    carried: dict,
    *,
    trader: TraderConfig,
    costs: CostConfig,
) -> dict:
    """{symbol: signed shares} for the period (module docstring). ``carried`` maps each held
    name with no price this period to its last observed price; those names come back with
    their previous shares unchanged."""
    out: dict = {s: prev[s] for s in carried if prev.get(s)}
    reserved = sum(abs(prev[s]) * px for s, px in carried.items() if prev.get(s) and _usable(px))
    targets = dollar_targets(weights, prices, capital, reserved, trader)
    theta = float(trader.sizing_friction_theta)
    cap_notional = capital * trader.max_symbol_weight
    tradeable = [s for s, w in weights.items() if w == w and w != 0.0 and _usable(prices.get(s))]

    counts: dict = {}
    base: dict = {}
    prev_abs: dict = {}  # discretionary names only
    spent = 0.0
    for s in tradeable:
        px, t, w = float(prices[s]), targets.get(s, 0.0), weights[s]
        b = _floor(t, px)
        n = float(b)
        p = prev.get(s, 0.0)
        if theta > 0 and abs(p) > 0 and (p > 0) == (w > 0) and t > 0:
            prev_abs[s] = abs(p)
            if abs(p) < b:
                n = abs(p)  # the greedy owns the buy-up, at its friction
            elif abs(p) > b:
                improvement = abs(abs(p) * px - t) - abs(b * px - t)
                if improvement < theta * _friction(0.0, abs(p) - b, px, costs):
                    n = abs(p)  # selling down is not worth its cost: keep
        base[s] = float(b)
        counts[s] = n
        spent += n * px

    # a kept count may breach the per-name cap or (after a loss) the budget: trim it back
    for s in sorted(counts):
        px = float(prices[s])
        while counts[s] * px > cap_notional + 1e-9 and counts[s] > 0:
            step = min(1.0, counts[s])
            counts[s] -= step
            spent -= step * px
    while spent + reserved > capital:
        worst, worst_over = None, -math.inf
        for s in sorted(counts):
            if counts[s] <= base[s]:
                continue
            over = counts[s] * float(prices[s]) - targets.get(s, 0.0)
            if over > worst_over:
                worst, worst_over = s, over
        if worst is None:
            break
        step = min(1.0, counts[worst] - base[worst])
        counts[worst] -= step
        spent -= step * float(prices[worst])

    while True:
        best, best_deficit = None, 0.0
        for s in sorted(counts):
            px = float(prices[s])
            if spent + reserved + px > capital:  # (a)
                continue
            n = counts[s]
            if (n + 1.0) * px > cap_notional + 1e-9:  # (b)
                continue
            deficit = targets.get(s, 0.0) - n * px
            mtc = 0.0
            if s in prev_abs:
                e0, x0 = leg_amounts(prev_abs[s], n)
                e1, x1 = leg_amounts(prev_abs[s], n + 1.0)
                mtc = _friction(e1, x1, px, costs) - _friction(e0, x0, px, costs)
            if deficit < px / 2.0 - 1e-9 + theta * mtc:  # (d)
                continue
            if best is None or deficit > best_deficit:
                best, best_deficit = s, deficit
        if best is None:
            break
        counts[best] += 1.0
        spent += float(prices[best])

    for s in tradeable:
        if counts.get(s, 0.0) > 0:
            out[s] = counts[s] if weights[s] > 0 else -counts[s]
    return out
