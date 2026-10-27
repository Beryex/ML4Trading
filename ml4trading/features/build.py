"""The feature matrix of a pool: every symbol's daily features, ML4Investment's cross-sectional
ranks and ETF columns, the scale filter, the symbol code, and the one-session shift."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from ml4trading.features.daily import symbol_daily_features

#: The ETFs whose features every stock row carries (ML4Investment embeds eight; the dataset has
#: these two). They are loaded beside the pool and never become rows of their own.
FEATURE_ETFS = ("VOO", "QQQ")

#: Every column that a stock's price/volume scale moves (price and volume levels, indicators in
#: price units, and ratios with a level in the denominator). Dropped: the bars are split-adjusted
#: back from the end of the dataset, so such a level in the past tells which companies split
#: later -- the ones that grew. ``tests/test_features.py`` holds this tuple equal to the rule.
SCALE_DEPENDENT = (
    "Open_first",
    "High_max",
    "Low_min",
    "Close_last",
    "Close_std",
    "Close_mean",
    "Volume_sum",
    "Volume_std",
    "Volume_mean",
    "ATR_91_mean",
    "OBV_last",
    "ATR_7_mean",
    "ATR_21_mean",
    "MA_7_last",
    "MA_21_last",
    "MA_Diff_7_21_last",
    "Signed_Volume_MA7_last",
    "Signed_Volume_MA7_mean",
    "MACD_line_last",
    "MACD_signal_last",
    "MACD_hist_last",
    "MA10",
    "MA50",
    "Close_Momentum_Slope",
    "MACD_line_daily",
    "MACD_signal_daily",
    "MACD_hist_daily",
    "ATR_14d",
    "BB_Upper_daily",
    "BB_Lower_daily",
    "BB_Middle_daily",
    "Volume_sum_lag1",
    "Volume_sum_lag2",
    "Volume_sum_lag3",
    "Volume_sum_lag5",
    "ATR_91_mean_lag1",
    "ATR_91_mean_lag2",
    "ATR_91_mean_lag3",
    "ATR_91_mean_lag5",
    "MACD_hist_last_lag1",
    "MACD_hist_last_lag2",
    "MACD_hist_last_lag3",
    "MACD_hist_daily_lag1",
    "MACD_hist_daily_lag2",
    "MACD_hist_daily_lag3",
    "Vol_Price_Synergy",
    "Risk_Adj_Momentum_5d",
    "Risk_Adj_Momentum_1d",
    "MACD_Diff",
    "OBV_last_diff",
    "OBV_Price_Momentum",
    "Daily_Range",
    "Daily_Range_for_pct_change",
    "Range_MA5",
    "Range_Std_5d",
    "Range_to_Volume",
    "Volume_Accel",
    "Volume_Trend_5d",
    "VOO_MACD_hist_daily",
    "QQQ_MACD_hist_daily",
)


def _add_cross_stock_rank_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-stock ranking features after daily aggregation (ML4Investment's, verbatim).

    Args:
        daily_df: Combined daily DataFrame for all stocks.

    Returns:
        DataFrame with ranking features added.
    """
    daily_df["Relative_Return_Rank"] = daily_df.groupby(level=0)["Return_1d"].rank(pct=True)

    features_to_rank = [
        "Volatility_5d",
        "Volume_MA_ratio",
        "MA5_Deviation",
        "OC_Momentum",
        "Overnight_Gap_Daily",
        "CMF_20d",
    ]
    for feature in features_to_rank:
        if feature in daily_df.columns:
            daily_df[f"Rank_{feature}"] = daily_df.groupby(level=0)[feature].rank(pct=True)
    return daily_df


def _embed_etf_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """ML4Investment's ETF embedding with ``FEATURE_ETFS`` and the ``symbol`` column: each
    ETF's eight features become ``<ETF>_<feature>`` columns of every stock row of the same
    session, and the ETF rows are dropped."""
    etf_list = [s for s in daily_df["symbol"].unique() if s in FEATURE_ETFS]

    features_to_embed_from_etfs = [
        "Return_1d",
        "Return_5d",
        "Volatility_5d",
        "RSI_14d",
        "MACD_hist_daily",
        "Volume_MA_ratio",
        "ADX_14d",
        "CMF_20d",
    ]

    etf_data = daily_df[daily_df["symbol"].isin(etf_list)]
    etf_data_indexed = etf_data.set_index("symbol", append=True)
    etf_features_subset = etf_data_indexed[features_to_embed_from_etfs]
    etf_features_wide = etf_features_subset.unstack("symbol")
    etf_features_wide.columns = [
        f"{stock}_{feature}" for feature, stock in etf_features_wide.columns
    ]

    stocks_only_df = daily_df[~daily_df["symbol"].isin(etf_list)].copy()

    return pd.merge(
        stocks_only_df, etf_features_wide, left_index=True, right_index=True, how="left"
    )


def _daily_frames(groups: list[pd.DataFrame], workers: int) -> list[pd.DataFrame]:
    if workers <= 1:
        return [symbol_daily_features(g) for g in groups]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        return list(pool.map(symbol_daily_features, groups, chunksize=4))


def build_features(bars: pd.DataFrame, pool: list[str], workers: int | None = None) -> pd.DataFrame:
    """The pool's features: one float32 row per (session, symbol) the symbol has bars on,
    holding what was known at that session's open -- the daily features of the symbol's
    PREVIOUS session (the first session of a symbol is all NaN), with ``stock_id`` (the
    symbol's position in the sorted pool) first. ``bars`` holds the pool and ``FEATURE_ETFS``;
    the ranks are taken over all of them, as ML4Investment does. ``workers`` processes compute
    the per-symbol features (None: ``min(8, cpus, symbols // 8)``); the result does not depend on
    it, nor on the row order of ``bars`` or ``pool``."""
    stocks = sorted(set(pool))
    absent = [e for e in FEATURE_ETFS if e not in set(bars["symbol"])]
    if absent:
        raise ValueError(f"build_features: no bars for the feature ETF(s) {absent}")
    symbols = sorted(set(stocks) | set(FEATURE_ETFS))
    by_symbol = dict(tuple(bars[bars["symbol"].isin(symbols)].groupby("symbol", sort=True)))
    loaded = [s for s in symbols if s in by_symbol]
    if workers is None:
        workers = min(8, os.cpu_count() or 1, max(1, len(loaded) // 8))
    frames = _daily_frames([by_symbol[s] for s in loaded], workers)
    daily_df = pd.concat([f.assign(symbol=s) for s, f in zip(loaded, frames, strict=True)])
    daily_df = _add_cross_stock_rank_features(daily_df).copy()  # one block again
    daily_df = _embed_etf_features(daily_df)
    daily_df = daily_df.drop(columns=[c for c in SCALE_DEPENDENT if c in daily_df.columns])

    daily_df = daily_df.set_index("symbol", append=True).sort_index(level=["symbol", "session"])
    features = daily_df.groupby(level="symbol", sort=False).shift(1)
    features = features.replace([np.inf, -np.inf], np.nan).astype(np.float32)
    codes = {s: i for i, s in enumerate(stocks)}
    stock_id = features.index.get_level_values("symbol").map(codes).astype(np.float32)
    features.insert(0, "stock_id", np.asarray(stock_id))
    return features
