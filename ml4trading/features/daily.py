"""One symbol's daily features from its 30-minute bars: ML4Investment's per-stock pipeline.

The ``_add_*`` stages and the aggregation rules are ML4Investment's own, verbatim
(``ml4trading.features`` names the source); ``_aggregate_to_daily`` and
``symbol_daily_features`` are this package's adaptation (one row per New York session the
symbol has bars on).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ml4trading.config import BAR_MINUTES, SESSION_MINUTES, TIMEZONE
from ml4trading.features.indicators import (
    EPSILON,
    calculate_adx,
    calculate_atr,
    calculate_bbands,
    calculate_cci,
    calculate_macd,
    calculate_mfi,
    calculate_obv,
    calculate_rsi,
    calculate_stoch,
)

#: ML4Investment's ``DATA_PER_DAY``: the 30-minute bars of a full session.
BARS_PER_SESSION = SESSION_MINUTES // BAR_MINUTES


def _add_basic_calculations(df: pd.DataFrame) -> pd.DataFrame:
    """Add basic intraday return and range features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with basic return and range features.
    """
    df["Returns_1h"] = df["Close"].pct_change(fill_method=None)
    df["Return_abs"] = df["Returns_1h"].abs()
    df["Intraday_Range"] = (df["High"] - df["Low"]) / df["Open"]
    range_high_low = df["High"] - df["Low"]
    df["Price_Efficiency"] = (df["Close"] - df["Open"]) / (range_high_low + EPSILON)
    df["High_Low_Ratio"] = df["High"] / (df["Low"] + EPSILON)
    df["Close_Open_Ratio"] = df["Close"] / (df["Open"] + EPSILON)
    df["Body_Range_Ratio"] = (df["Close"] - df["Open"]).abs() / (range_high_low + EPSILON)
    df["Upper_Shadow"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / (
        range_high_low + EPSILON
    )
    df["Lower_Shadow"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / (
        range_high_low + EPSILON
    )
    df["Hourly_Volatility"] = df["Returns_1h"].rolling(BARS_PER_SESSION).std()
    df["Hourly_Open_Gap"] = df["Open"].pct_change(fill_method=None)
    return df


def _add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add intraday technical indicator features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with technical indicators.
    """
    df["RSI_91"] = calculate_rsi(df["Close"], BARS_PER_SESSION * 14)
    df["ATR_91"] = calculate_atr(df["High"], df["Low"], df["Close"], BARS_PER_SESSION * 14)
    df["Bollinger_Width"] = (
        df["Close"].rolling(BARS_PER_SESSION * 3).std()
        / df["Close"].rolling(BARS_PER_SESSION * 3).mean()
    ) * 100
    df["OBV"] = calculate_obv(df)

    df["RSI_7"] = calculate_rsi(df["Close"], BARS_PER_SESSION)
    df["RSI_21"] = calculate_rsi(df["Close"], BARS_PER_SESSION * 3)
    df["ATR_7"] = calculate_atr(df["High"], df["Low"], df["Close"], BARS_PER_SESSION)
    df["ATR_21"] = calculate_atr(df["High"], df["Low"], df["Close"], BARS_PER_SESSION * 3)
    df["MA_7"] = df["Close"].rolling(BARS_PER_SESSION).mean()
    df["MA_21"] = df["Close"].rolling(BARS_PER_SESSION * 3).mean()
    df["MA_Diff_7_21"] = df["MA_7"] - df["MA_21"]
    df["Close_vs_MA7"] = df["Close"] / (df["MA_7"] + EPSILON) - 1
    df["Close_vs_MA21"] = df["Close"] / (df["MA_21"] + EPSILON) - 1

    df["ADX_14h"] = calculate_adx(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["CCI_14h"] = calculate_cci(df["High"], df["Low"], df["Close"], timeperiod=14)
    df["MFI_14h"] = calculate_mfi(df["High"], df["Low"], df["Close"], df["Volume"], timeperiod=14)
    upper, middle, lower = calculate_bbands(df["Close"], timeperiod=20, nbdev=2)
    df["BB_Upper"] = upper
    df["BB_Lower"] = lower
    df["BB_Middle"] = middle
    df["Close_vs_BB"] = (df["Close"] - df["BB_Middle"]) / (
        df["BB_Upper"] - df["BB_Lower"] + EPSILON
    )
    return df


def _add_price_volume_interaction(df: pd.DataFrame) -> pd.DataFrame:
    """Add price/volume interaction features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with price/volume interaction features.
    """
    df["Volume_Spike"] = df["Volume"] / (
        df["Volume"].rolling(BARS_PER_SESSION * 3).mean() + EPSILON
    )
    df["Price_Volume_Correlation"] = df["Close"].rolling(BARS_PER_SESSION).corr(df["Volume"])
    df["Price_Volume_Correlation"] = df["Price_Volume_Correlation"].replace(
        [np.inf, -np.inf], np.nan
    )
    df["Price_Volume_Correlation"] = df["Price_Volume_Correlation"].fillna(0)
    df["Volume_Change_Ratio"] = df["Volume"].pct_change(fill_method=None)
    df["Volume_Change_Ratio"] = df["Volume_Change_Ratio"].replace([np.inf, -np.inf], np.nan)
    df["Signed_Volume"] = np.sign(df["Returns_1h"]) * df["Volume"]
    df["Signed_Volume_MA7"] = df["Signed_Volume"].rolling(BARS_PER_SESSION).mean()
    return df


def _add_flow_direction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add flow intensity and direction consistency features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with flow and direction features.
    """
    df["Flow_Intensity"] = (np.sign(df["Close"].diff()) * df["Volume"]) / (
        df["Volume"].rolling(21).mean()
    )
    df["Flow_Intensity"] = df["Flow_Intensity"].rolling(3).mean()
    df["Direction_Consistency"] = (
        (np.sign(df["Close"].diff()) == np.sign(df["Close"].diff().shift(1))).rolling(3).sum()
    )
    return df


def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum and volatility ratio features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with momentum indicators.
    """
    df["Momentum_Accel"] = df["Close"].pct_change(3, fill_method=None) - df["Close"].pct_change(
        BARS_PER_SESSION, fill_method=None
    )
    df["Vol_Ratio"] = df["Close"].rolling(BARS_PER_SESSION).std() / (
        df["Close"].rolling(BARS_PER_SESSION * 5).std() + EPSILON
    )
    df["ROC_1h"] = df["Close"].pct_change(1, fill_method=None)
    df["ROC_3h"] = df["Close"].pct_change(3, fill_method=None)
    df["ROC_7h"] = df["Close"].pct_change(BARS_PER_SESSION, fill_method=None)
    return df


def _add_date_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a plain date column derived from the DatetimeIndex.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with a Date column.
    """
    idx = pd.DatetimeIndex(df.index)
    df["Date"] = idx.date
    return df


def _add_vwap_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add daily and cumulative VWAP features.

    Args:
        df: Intraday OHLCV DataFrame with Date column.

    Returns:
        DataFrame with VWAP features.
    """
    price_vol_product = df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3
    volume_series = df["Volume"]
    cum_price_vol = price_vol_product.groupby(df["Date"]).cumsum()
    cum_vol = volume_series.groupby(df["Date"]).cumsum()
    df["VWAP_Daily"] = cum_price_vol / (cum_vol + EPSILON)
    df["VWAP_Cumulative"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / (
        df["Volume"].cumsum() + EPSILON
    )
    df["VWAP_Deviation_Daily"] = (df["Close"] - df["VWAP_Daily"]) / (df["VWAP_Daily"] + EPSILON)
    df["VWAP_Deviation_Cumulative"] = (df["Close"] - df["VWAP_Cumulative"]) / (
        df["VWAP_Cumulative"] + EPSILON
    )
    return df


def _add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add MACD features for intraday close prices.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with MACD features.
    """
    close_prices = df["Close"]
    macd_line, macd_signal, macd_hist = calculate_macd(close_prices)
    df["MACD_line"] = macd_line
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    return df


def _add_stochastic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add stochastic oscillator features.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with stochastic %K and %D.
    """
    high_rolling = df["High"].rolling(BARS_PER_SESSION * 2).max()
    low_rolling = df["Low"].rolling(BARS_PER_SESSION * 2).min()
    k = (df["Close"] - low_rolling) / (high_rolling - low_rolling + EPSILON) * 100
    df["Stochastic_K"] = k
    df["Stochastic_D"] = k.rolling(3).mean()
    return df


def _add_relative_price_position(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative price position within recent range.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with Relative_Price_Position feature.
    """
    rolling_max = df["Close"].rolling(BARS_PER_SESSION * 2).max()
    rolling_min = df["Close"].rolling(BARS_PER_SESSION * 2).min()
    df["Relative_Price_Position"] = (df["Close"] - rolling_min) / (
        rolling_max - rolling_min + EPSILON
    )
    return df


def _add_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-of-day intraday features.

    Args:
        df: Intraday OHLCV DataFrame with Date column.

    Returns:
        DataFrame with intraday time and session features.
    """
    idx = pd.DatetimeIndex(df.index)
    df["Hour_Index"] = idx.hour * 100 + idx.minute
    df["Time_Portion"] = df["Hour_Index"].apply(lambda x: 0 if x < 1130 else (1 if x < 1430 else 2))

    morning_mask = df["Hour_Index"] < 1230
    df["Morning_Volume_Ratio"] = df["Volume"].where(morning_mask).rolling(3).sum() / (
        df["Volume"].rolling(BARS_PER_SESSION).sum() + EPSILON
    )

    afternoon_mask = df["Hour_Index"] >= 1330
    df["Afternoon_Return"] = df["Returns_1h"].where(afternoon_mask).rolling(3).mean()

    late_afternoon_mask = df["Hour_Index"] >= 1430
    df["Late_Afternoon_Volume_Ratio"] = df["Volume"].where(late_afternoon_mask).rolling(2).sum() / (
        df["Volume"].rolling(BARS_PER_SESSION).sum() + EPSILON
    )
    df["Late_Momentum"] = df["Returns_1h"].where(late_afternoon_mask).rolling(2).mean()

    end_mask = (df["Hour_Index"] >= 1430) & (df["Hour_Index"] <= 1530)
    end_volume_sum_daily = df["Volume"].where(end_mask, 0).groupby(df["Date"]).transform("sum")
    total_volume_daily = df["Volume"].groupby(df["Date"]).transform("sum")
    df["End_Volume_Ratio"] = end_volume_sum_daily / (total_volume_daily + EPSILON)

    df["First_Hour_Return"] = df.groupby("Date")["Returns_1h"].transform("first")
    df["Last_Hour_Return"] = df.groupby("Date")["Returns_1h"].transform("last")
    df["First_Hour_Volume_Ratio"] = df.groupby("Date")["Volume"].transform("first") / (
        df.groupby("Date")["Volume"].transform("sum") + EPSILON
    )
    df["Last_Hour_Volume_Ratio"] = df.groupby("Date")["Volume"].transform("last") / (
        df.groupby("Date")["Volume"].transform("sum") + EPSILON
    )
    return df


def _add_divergence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add price/volume divergence flags.

    Args:
        df: Intraday OHLCV DataFrame.

    Returns:
        DataFrame with divergence features.
    """
    price_up = df["Close"].pct_change(fill_method=None) > 0
    volume_down = df["Volume"].pct_change(fill_method=None) < 0
    df["Price_Up_Vol_Down_Div"] = (price_up & volume_down).astype(int)

    price_down = df["Close"].pct_change(fill_method=None) < 0
    volume_up = df["Volume"].pct_change(fill_method=None) > 0
    df["Price_Down_Vol_Up_Div"] = (price_down & volume_up).astype(int)
    return df


def _build_daily_aggregation_rules() -> dict[str, Any]:
    """Build aggregation rules for intraday-to-daily rollups.

    Returns:
        Mapping from column names to aggregation functions.
    """
    return {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": ["last", "std", "mean"],
        "Volume": ["sum", "std", "mean"],
        "Returns_1h": ["mean", "std", "median", "skew"],
        "Hourly_Open_Gap": ["mean", "std"],
        "Return_abs": ["sum", "mean"],
        "Intraday_Range": ["last", "mean"],
        "Price_Efficiency": ["mean", "last", "std"],
        "High_Low_Ratio": ["mean", "last"],
        "Close_Open_Ratio": ["mean", "last"],
        "Body_Range_Ratio": ["mean", "last"],
        "Upper_Shadow": ["mean", "last"],
        "Lower_Shadow": ["mean", "last"],
        "Hourly_Volatility": ["mean", "last"],
        "RSI_91": "last",
        "ATR_91": "mean",
        "Bollinger_Width": "last",
        "OBV": "last",
        "RSI_7": "last",
        "RSI_21": "last",
        "ATR_7": "mean",
        "ATR_21": "mean",
        "MA_7": "last",
        "MA_21": "last",
        "MA_Diff_7_21": "last",
        "Close_vs_MA7": "last",
        "Close_vs_MA21": "last",
        "ADX_14h": "last",
        "CCI_14h": "last",
        "MFI_14h": "last",
        "Close_vs_BB": "last",
        "Volume_Spike": ["max", "mean"],
        "Price_Volume_Correlation": ["mean", "last"],
        "Volume_Change_Ratio": ["mean", "std"],
        "Signed_Volume_MA7": ["last", "mean"],
        "Flow_Intensity": ["last", "mean"],
        "Direction_Consistency": ["mean", "last"],
        "Momentum_Accel": ["last", "mean"],
        "Vol_Ratio": ["last", "mean"],
        "ROC_3h": ["last", "mean"],
        "ROC_7h": ["last", "mean"],
        "VWAP_Deviation_Daily": ["mean", "last", "std"],
        "VWAP_Deviation_Cumulative": ["mean", "last"],
        "MACD_line": "last",
        "MACD_signal": "last",
        "MACD_hist": "last",
        "Stochastic_K": "last",
        "Stochastic_D": "last",
        "Relative_Price_Position": ["last", "mean"],
        "Hour_Index": ["first", "last"],
        "Time_Portion": "last",
        "Morning_Volume_Ratio": ["last", "mean"],
        "Afternoon_Return": ["last", "mean"],
        "Late_Afternoon_Volume_Ratio": ["last", "mean"],
        "Late_Momentum": ["last", "mean"],
        "End_Volume_Ratio": ["last"],
        "First_Hour_Return": "first",
        "Last_Hour_Return": "last",
        "First_Hour_Volume_Ratio": "first",
        "Last_Hour_Volume_Ratio": "last",
        "Price_Up_Vol_Down_Div": ["sum", "mean"],
        "Price_Down_Vol_Up_Div": ["sum", "mean"],
    }


def _add_daily_return_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily gap and return features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with daily return features.
    """
    prev_close_daily = daily_df["Close_last"].shift(1)
    daily_df["Overnight_Gap_Daily"] = (daily_df["Open_first"] - prev_close_daily) / (
        prev_close_daily + EPSILON
    )

    daily_df["Return_1d"] = daily_df["Close_last"].pct_change(1, fill_method=None)
    daily_df["Return_3d"] = daily_df["Close_last"].pct_change(3, fill_method=None)
    daily_df["Return_5d"] = daily_df["Close_last"].pct_change(5, fill_method=None)
    daily_df["Return_10d"] = daily_df["Close_last"].pct_change(10, fill_method=None)
    daily_df["Return_20d"] = daily_df["Close_last"].pct_change(20, fill_method=None)
    return daily_df


def _add_daily_volatility_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily volatility and deviation features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with volatility-related features.
    """
    daily_df["Volatility_5d"] = daily_df["Return_1d"].rolling(5).std()
    daily_df["Volatility_Change"] = daily_df["Volatility_5d"].diff()
    daily_df["Volatility_10d"] = daily_df["Return_1d"].rolling(10).std()
    daily_df["Volatility_20d"] = daily_df["Return_1d"].rolling(20).std()
    daily_df["Volatility_Skew"] = daily_df["Volatility_5d"] / (daily_df["Volatility_10d"] + EPSILON)
    daily_df["Momentum_to_Volatility"] = daily_df["Return_1d"] / (
        daily_df["Volatility_5d"] + EPSILON
    )
    daily_df["Volume_MA_ratio"] = daily_df["Volume_sum"] / (
        daily_df["Volume_sum"].rolling(5).mean() + EPSILON
    )
    daily_df["MA5_Deviation"] = (
        daily_df["Close_last"] / (daily_df["Close_last"].rolling(5).mean() + EPSILON) - 1
    )
    daily_df["MA20_Deviation"] = (
        daily_df["Close_last"] / (daily_df["Close_last"].rolling(20).mean() + EPSILON) - 1
    )
    return daily_df


def _add_daily_moving_average_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily moving average and ratio features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with moving average features.
    """
    daily_df["MA10"] = daily_df["Close_last"].rolling(10).mean()
    daily_df["MA50"] = daily_df["Close_last"].rolling(50).mean()
    daily_df["MA10_Deviation"] = daily_df["Close_last"] / (daily_df["MA10"] + EPSILON) - 1
    daily_df["MA50_Deviation"] = daily_df["Close_last"] / (daily_df["MA50"] + EPSILON) - 1
    daily_df["MA5_vs_MA20"] = (
        daily_df["Close_last"].rolling(5).mean()
        / (daily_df["Close_last"].rolling(20).mean() + EPSILON)
        - 1
    )
    return daily_df


def _add_daily_momentum_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily momentum and weekday position features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with daily momentum and weekday features.
    """
    assert isinstance(daily_df.index, pd.DatetimeIndex)
    daily_df["Weekday"] = daily_df.index.day_of_week
    daily_df["Relative_Rank_Change"] = daily_df["Relative_Price_Position_last"].diff(3)
    daily_df["OC_Momentum"] = (daily_df["Close_last"] - daily_df["Open_first"]) / (
        daily_df["Open_first"] + EPSILON
    )
    daily_df["Close_Momentum_Slope"] = (
        daily_df["Close_last"]
        .rolling(5)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )
    return daily_df


def _add_daily_technical_indicators(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily technical indicator features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with daily technical indicators.
    """
    daily_df["RSI_14d"] = calculate_rsi(daily_df["Close_last"], 14)
    close_prices_daily = daily_df["Close_last"]
    macd_line_daily, macd_signal_daily, macd_hist_daily = calculate_macd(close_prices_daily)
    daily_df["MACD_line_daily"] = macd_line_daily
    daily_df["MACD_signal_daily"] = macd_signal_daily
    daily_df["MACD_hist_daily"] = macd_hist_daily
    slowk, slowd = calculate_stoch(
        daily_df["High_max"],
        daily_df["Low_min"],
        daily_df["Close_last"],
        fastk_period=14,
        slowk_period=3,
        slowd_period=3,
    )
    daily_df["Stochastic_K_daily"] = slowk
    daily_df["Stochastic_D_daily"] = slowd
    daily_df["ATR_14d"] = calculate_atr(
        daily_df["High_max"], daily_df["Low_min"], daily_df["Close_last"], 14
    )
    daily_df["ADX_14d"] = calculate_adx(
        daily_df["High_max"], daily_df["Low_min"], daily_df["Close_last"], timeperiod=14
    )
    daily_df["CCI_14d"] = calculate_cci(
        daily_df["High_max"], daily_df["Low_min"], daily_df["Close_last"], timeperiod=14
    )
    daily_df["MFI_14d"] = calculate_mfi(
        daily_df["High_max"],
        daily_df["Low_min"],
        daily_df["Close_last"],
        daily_df["Volume_sum"],
        timeperiod=14,
    )
    upper_d, middle_d, lower_d = calculate_bbands(daily_df["Close_last"], timeperiod=20, nbdev=2)
    daily_df["BB_Upper_daily"] = upper_d
    daily_df["BB_Lower_daily"] = lower_d
    daily_df["BB_Middle_daily"] = middle_d
    daily_df["Close_vs_BB_daily"] = (daily_df["Close_last"] - middle_d) / (
        upper_d - lower_d + EPSILON
    )
    return daily_df


def _add_lag_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features for selected daily metrics.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with lagged features.
    """
    lag_features_config = {
        "Return_1d": [1, 2, 3, 5, 10],
        "Volatility_5d": [1, 2, 3, 5],
        "Volume_sum": [1, 2, 3, 5],
        "Volume_MA_ratio": [1, 2, 3],
        "OC_Momentum": [1, 2, 3],
        "Overnight_Gap_Daily": [1, 2, 3],
        "ATR_91_mean": [1, 2, 3, 5],
        "RSI_91_last": [1, 2, 3, 5],
        "MACD_hist_last": [1, 2, 3],
        "Stochastic_K_last": [1, 2],
        "Relative_Price_Position_last": [1, 2, 3],
        "VWAP_Deviation_Daily_last": [1, 2],
        "RSI_14d": [1, 2, 3],
        "MACD_hist_daily": [1, 2, 3],
        "Stochastic_K_daily": [1, 2],
    }

    for base, lags in lag_features_config.items():
        for lag in lags:
            daily_df[f"{base}_lag{lag}"] = daily_df[base].shift(lag)

    return daily_df


def _add_synergy_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add feature combinations for risk-adjusted momentum and synergy.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with synergy features.
    """
    daily_df["Vol_Price_Synergy"] = daily_df["Volume_sum"] * daily_df["Returns_1h_mean"]
    daily_df["Risk_Adj_Momentum_5d"] = daily_df["Return_5d"] / (daily_df["ATR_14d"] + EPSILON)
    daily_df["Risk_Adj_Momentum_1d"] = daily_df["Return_1d"] / (daily_df["ATR_14d"] + EPSILON)
    return daily_df


def _add_trend_signals(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily trend signal features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with trend signal features.
    """
    daily_df["MACD_Diff"] = daily_df["MACD_line_last"] - daily_df["MACD_signal_last"]
    daily_df["MACD_Direction"] = (daily_df["MACD_Diff"].diff().astype(float) > 0).astype(int)
    daily_df["Trend_Continuation_3d"] = (
        (daily_df["Return_1d"] > 0)
        & (daily_df["Return_1d"].shift(1) > 0)
        & (daily_df["Return_1d"].shift(2) > 0)
    ).astype(int)
    daily_df["Trend_Reversal"] = (
        (daily_df["Return_1d"] * daily_df["Return_1d"].shift(1)) < 0
    ).astype(int)
    return daily_df


def _add_price_volume_divergence_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add extended price/volume divergence features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with divergence features.
    """
    daily_df["Volume_Change"] = daily_df["Volume_sum"].pct_change(fill_method=None)
    daily_df["Price_vs_Volume_Diff"] = daily_df["Return_1d"] - daily_df["Volume_Change"]
    daily_df["OBV_last_diff"] = daily_df["OBV_last"].diff()
    daily_df["OBV_Price_Momentum"] = daily_df["OBV_last_diff"] * daily_df["Return_1d"]
    daily_df["OBV_Price_Divergence"] = (
        (daily_df["OBV_last_diff"] > 0) & (daily_df["Return_1d"] < 0)
    ) | ((daily_df["OBV_last_diff"] < 0) & (daily_df["Return_1d"] > 0)).astype(int)
    return daily_df


def _add_time_cycle_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-based time and cycle position features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with time cycle features.
    """
    assert isinstance(daily_df.index, pd.DatetimeIndex)
    daily_df["Month"] = daily_df.index.month
    daily_df["Day_of_Month"] = daily_df.index.day
    daily_df["Day_of_Year"] = daily_df.index.day_of_year
    daily_df["Week_of_Year"] = daily_df.index.isocalendar().week.astype(int)
    daily_df["Week_of_Quarter"] = (daily_df["Week_of_Year"] - 1) % 13 + 1
    return daily_df


def _add_opening_deviation_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add opening deviation and momentum features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with opening deviation features.
    """
    daily_df["Open_vs_MA5"] = (
        daily_df["Open_first"] / (daily_df["Close_last"].rolling(5).mean().shift(1) + EPSILON) - 1
    )
    daily_df["Open_vs_MA20"] = (
        daily_df["Open_first"] / (daily_df["Close_last"].rolling(20).mean().shift(1) + EPSILON) - 1
    )
    daily_df["Open_Volatility_3d"] = (
        daily_df["Open_first"].pct_change(fill_method=None).rolling(3).std()
    )
    daily_df["Open_Momentum_3d"] = daily_df["Open_first"].pct_change(3, fill_method=None)
    return daily_df


def _add_range_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add daily range compression and expansion features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with range-related features.
    """
    daily_df["Daily_Range"] = daily_df["High_max"] - daily_df["Low_min"]
    daily_df["Daily_Range_for_pct_change"] = daily_df["Daily_Range"].replace(0, np.nan)
    daily_df["Daily_Range_for_pct_change"] = daily_df["Daily_Range_for_pct_change"].ffill()
    daily_df["Daily_Range_for_pct_change"] = daily_df["Daily_Range_for_pct_change"].fillna(EPSILON)
    daily_df["Range_Change"] = daily_df["Daily_Range_for_pct_change"].pct_change(fill_method=None)
    daily_df["Range_MA5"] = daily_df["Daily_Range"].rolling(5).mean()
    daily_df["Range_vs_MA5"] = daily_df["Daily_Range"] / (daily_df["Range_MA5"] + EPSILON)
    daily_df["Range_Std_5d"] = daily_df["Daily_Range"].rolling(5).std()
    daily_df["Range_to_Volume"] = daily_df["Daily_Range"] / (daily_df["Volume_sum"] + EPSILON)
    return daily_df


def _add_volume_acceleration_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add volume acceleration and trend features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with volume acceleration features.
    """
    daily_df["Volume_Accel"] = daily_df["Volume_sum"].diff(1) - daily_df["Volume_sum"].diff(2)
    daily_df["Volume_Shock_5d"] = daily_df["Volume_sum"].rolling(5).std() / (
        daily_df["Volume_sum"].rolling(5).mean() + EPSILON
    )
    daily_df["Volume_Trend_5d"] = (
        daily_df["Volume_sum"]
        .rolling(5)
        .apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0], raw=True)
    )
    return daily_df


def _add_flow_strength_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add flow strength and money flow features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with flow strength features.
    """
    daily_df["Flow_to_Volume"] = daily_df["OBV_last_diff"] / (daily_df["Volume_sum"] + EPSILON)
    mfm = (
        (daily_df["Close_last"] - daily_df["Low_min"])
        - (daily_df["High_max"] - daily_df["Close_last"])
    ) / (daily_df["High_max"] - daily_df["Low_min"] + EPSILON)
    mfv = mfm * daily_df["Volume_sum"]
    daily_df["CMF_20d"] = mfv.rolling(20).sum() / (
        daily_df["Volume_sum"].rolling(20).sum() + EPSILON
    )
    return daily_df


def _add_trend_normalization_features(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Add trend normalization features.

    Args:
        daily_df: Daily aggregated DataFrame.

    Returns:
        DataFrame with normalized trend features.
    """
    daily_df["Normalized_Trend_Strength"] = daily_df["Return_5d"] / (
        daily_df["Volatility_5d"] + EPSILON
    )
    daily_df["Normalized_Trend_Strength_20d"] = daily_df["Return_20d"] / (
        daily_df["Volatility_20d"] + EPSILON
    )
    daily_df["Normalized_OC_Momentum"] = daily_df["OC_Momentum"] / (
        daily_df["Volatility_5d"] + EPSILON
    )
    return daily_df


def _aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """The intraday frame aggregated by ``_build_daily_aggregation_rules`` into one row per New
    York session date (named ``session``, tz-naive), columns flattened ``<column>_<function>``
    and forward-filled as ML4Investment does."""
    session = pd.DatetimeIndex(df.index).tz_localize(None).normalize().rename("session")
    daily_df = df.groupby(session).agg(_build_daily_aggregation_rules())  # type: ignore
    daily_df.columns = ["_".join(col).strip() for col in daily_df.columns]
    return daily_df.ffill()


_INTRADAY_STAGES = (
    _add_basic_calculations,
    _add_technical_indicators,
    _add_price_volume_interaction,
    _add_flow_direction_features,
    _add_momentum_indicators,
    _add_date_column,
    _add_vwap_features,
    _add_macd_features,
    _add_stochastic_features,
    _add_relative_price_position,
    _add_intraday_features,
    _add_divergence_features,
)

_DAILY_STAGES = (
    _add_daily_return_features,
    _add_daily_volatility_features,
    _add_daily_moving_average_features,
    _add_daily_momentum_features,
    _add_daily_technical_indicators,
    _add_lag_features,
    _add_synergy_features,
    _add_trend_signals,
    _add_price_volume_divergence_features,
    _add_time_cycle_features,
    _add_opening_deviation_features,
    _add_range_features,
    _add_volume_acceleration_features,
    _add_flow_strength_features,
    _add_trend_normalization_features,
)


def symbol_daily_features(bars: pd.DataFrame) -> pd.DataFrame:
    """One symbol's daily features: ``bars`` (``ts, open, high, low, close, volume``, any row
    order) -> one row per session it has bars on, indexed by the tz-naive session date; each row
    computed from that session's bars and earlier ones."""
    df = bars.sort_values("ts").set_index("ts")[["open", "high", "low", "close", "volume"]]
    df = df.rename(columns=str.capitalize).astype(float)
    df.index = pd.DatetimeIndex(df.index).tz_convert(TIMEZONE)
    for stage in _INTRADAY_STAGES:
        df = stage(df)
    daily_df = _aggregate_to_daily(df)
    for stage in _DAILY_STAGES:
        daily_df = stage(daily_df).copy()
    return daily_df
