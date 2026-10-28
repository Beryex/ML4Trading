"""Is a Sharpe ratio really higher than another, and did a search just pick the luckiest one?

Two small statistics for the final report, both on Sharpe ratios measured PER PERIOD (mean over
population std, as ``ml4trading.metrics.sharpe`` does before its sqrt(per_year)).

* ``sharpe_difference_test``: a paired test of equal Sharpe ratios on the dates two return
  series share. The statistic is Jobson & Korkie (1981), "Performance Hypothesis Testing with
  the Sharpe and Treynor Measures", Journal of Finance 36(4), with the algebra corrected by
  Memmel (2003), "Performance Hypothesis Testing with the Sharpe Ratio", Finance Letters 1:
  with correlation rho and T paired periods, the asymptotic variance of SR1 - SR2 is

      (2 - 2 rho + (SR1^2 + SR2^2 - 2 SR1 SR2 rho^2) / 2) / T.

  It assumes iid returns, which daily returns are not quite (fat tails, volatility clustering,
  autocorrelation). Ledoit & Wolf (2008), "Robust Performance Hypothesis Testing with the
  Sharpe Ratio", Journal of Empirical Finance 15(5), is the robust alternative (a studentized
  bootstrap of the difference); it is not implemented here.
* ``deflated_sharpe``: Bailey & Lopez de Prado (2014), "The Deflated Sharpe Ratio: Correcting
  for Selection Bias, Backtest Overfitting and Non-Normality", Journal of Portfolio Management
  40(5). Picking the best of N tried strategies inflates the winner's Sharpe even when none has
  an edge; the deflated ratio is the probability that the winner's true Sharpe beats the best
  one would expect from N trials of no skill, given the track record's length, skewness and
  kurtosis.

The standard normal distribution comes from the standard library, so these functions add no
dependency.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

from ml4trading.config import TRADING_DAYS_PER_YEAR

_NORMAL = NormalDist()
EULER_GAMMA = 0.5772156649015329  # Euler-Mascheroni constant, as in Bailey & Lopez de Prado (2014)


def sharpe_difference_test(
    r1: pd.Series, r2: pd.Series, per_year: int = TRADING_DAYS_PER_YEAR
) -> dict:
    """Paired test that two return series have the same Sharpe ratio (module docstring).

    Both series are indexed by period (the session); only the dates present and finite in both
    are used, and ``n`` counts them. ``sharpe_1``, ``sharpe_2``, ``diff`` (``sharpe_1`` minus
    ``sharpe_2``) and ``se`` are annualized by sqrt(``per_year``); ``z`` and ``p_two_sided``
    (against the standard normal) do not depend on the annualization. ``corr`` is the
    correlation of the two series. A pair with no spread in its Sharpe estimate (``se`` of
    zero, e.g. one series twice) has ``z`` 0 when the Sharpes agree and infinite otherwise."""
    joined = pd.concat([r1.rename("r1"), r2.rename("r2")], axis=1, join="inner").dropna()
    n = len(joined)
    if n < 3:
        raise ValueError(f"a paired Sharpe test needs at least 3 common dates, got {n}")
    x, y = joined["r1"].to_numpy(dtype=float), joined["r2"].to_numpy(dtype=float)
    sd1, sd2 = float(x.std()), float(y.std())
    if sd1 == 0 or sd2 == 0:
        raise ValueError("a constant return series has no Sharpe ratio")
    sr1, sr2 = float(x.mean()) / sd1, float(y.mean()) / sd2
    rho = float(((x - x.mean()) * (y - y.mean())).mean() / (sd1 * sd2))
    rho = min(max(rho, -1.0), 1.0)  # rounding can push a perfect correlation just past 1
    var = (2.0 - 2.0 * rho + 0.5 * (sr1**2 + sr2**2 - 2.0 * sr1 * sr2 * rho**2)) / n
    diff, se = sr1 - sr2, math.sqrt(max(var, 0.0))
    if se > 0:
        z = diff / se
    else:
        z = 0.0 if diff == 0 else math.copysign(math.inf, diff)
    root = math.sqrt(per_year)
    return {
        "sharpe_1": sr1 * root,
        "sharpe_2": sr2 * root,
        "diff": diff * root,
        "se": se * root,
        "z": z,
        "p_two_sided": 2.0 * _NORMAL.cdf(-abs(z)),
        "n": n,
        "corr": rho,
    }


def deflated_sharpe(
    sr: float, n_trials: int, sr_trials_std: float, n_obs: int, skew: float, kurt: float
) -> dict:
    """Bailey & Lopez de Prado's deflated Sharpe ratio (module docstring).

    ``sr`` is the selected strategy's Sharpe ratio per period, measured over ``n_obs`` periods
    with return skewness ``skew`` and kurtosis ``kurt`` (not excess: 3 for normal returns);
    ``n_trials`` strategies were tried and ``sr_trials_std`` is the standard deviation of their
    per-period Sharpe ratios. Returns ``expected_max_sr``, the best per-period Sharpe N trials
    without any skill would show on average (0 for a single trial: nothing was selected), and
    ``dsr``, the probability that the strategy's true Sharpe exceeds it."""
    if n_trials < 1:
        raise ValueError(f"n_trials must be at least 1, got {n_trials}")
    if n_obs < 2:
        raise ValueError(f"n_obs must be at least 2, got {n_obs}")
    if sr_trials_std < 0:
        raise ValueError(f"sr_trials_std must not be negative, got {sr_trials_std}")
    if n_trials == 1:
        sr0 = 0.0
    else:
        sr0 = sr_trials_std * (
            (1.0 - EULER_GAMMA) * _NORMAL.inv_cdf(1.0 - 1.0 / n_trials)
            + EULER_GAMMA * _NORMAL.inv_cdf(1.0 - 1.0 / (n_trials * math.e))
        )
    # variance of the Sharpe estimate under skewed, fat-tailed returns (times n_obs - 1)
    var_term = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr**2
    if var_term <= 0:
        raise ValueError(f"the Sharpe estimate's variance term is {var_term:.4g}, not positive")
    stat = (sr - sr0) * math.sqrt(n_obs - 1) / math.sqrt(var_term)
    return {"expected_max_sr": sr0, "dsr": _NORMAL.cdf(stat)}


def return_moments(returns: pd.Series | np.ndarray) -> tuple[float, float, float]:
    """(per-period Sharpe, skewness, kurtosis) of a return series, all with the population std
    (so the Sharpe matches ``ml4trading.metrics.sharpe``); the kurtosis is not excess."""
    r = np.asarray(returns, dtype=float)
    sd = float(r.std())
    if r.size < 2 or sd == 0:
        raise ValueError("a constant return series has no Sharpe ratio")
    z = (r - r.mean()) / sd
    return float(r.mean()) / sd, float((z**3).mean()), float((z**4).mean())
