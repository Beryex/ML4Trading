"""Technical indicators on one symbol's bars: ML4Investment's own implementations, verbatim
(``ml4trading.features`` names the source)."""

from __future__ import annotations

import numpy as np
import pandas as pd

EPSILON = 1e-9


def calculate_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Args:
        series: Close price series.

    Returns:
        Tuple of (macd_line, macd_signal, macd_hist) as Series.
    """
    fast_ema = series.ewm(span=12, adjust=False).mean()
    slow_ema = series.ewm(span=26, adjust=False).mean()

    macd_line = fast_ema - slow_ema
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - macd_signal

    return macd_line, macd_signal, macd_hist


def calculate_rsi(series: pd.Series, window: int) -> pd.Series:
    """Compute Relative Strength Index (RSI).

    Args:
        series: Price series.
        window: RSI window length.

    Returns:
        RSI values as a Series.
    """
    delta = series.diff().astype(float)
    gain = delta.where(delta > 0, 0).fillna(0)
    loss = -delta.where(delta < 0, 0).fillna(0)

    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    rs = avg_gain / (avg_loss + EPSILON)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int) -> pd.Series:
    """Calculate Average True Range (ATR).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: ATR window length.

    Returns:
        ATR values as a Series.
    """
    hl = high - low
    hc = (high - close.shift()).abs()
    lc = (low - close.shift()).abs()

    true_range = pd.DataFrame({"hl": hl, "hc": hc, "lc": lc}).max(axis=1)
    atr = true_range.ewm(com=window - 1, min_periods=window).mean()
    return atr


def calculate_obv(df: pd.DataFrame) -> pd.Series:
    """Compute On-Balance Volume (OBV).

    Args:
        df: DataFrame containing Close and Volume columns.

    Returns:
        OBV values as a Series.
    """
    price_change = df["Close"].diff().fillna(0)
    signed_volume = np.sign(price_change) * df["Volume"]
    return signed_volume.cumsum()  # type: ignore


def calculate_stoch(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    fastk_period: int = 14,
    slowk_period: int = 3,
    slowd_period: int = 3,
) -> tuple[pd.Series, pd.Series]:
    """Calculate Slow Stochastic Oscillator (%K and %D).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        fastk_period: Fast %K window.
        slowk_period: Slow %K smoothing window.
        slowd_period: Slow %D smoothing window.

    Returns:
        Tuple of (slowk, slowd) Series.
    """
    min_required_len = fastk_period + slowk_period + slowd_period - 2
    if (
        high.isna().all()
        or low.isna().all()
        or close.isna().all()
        or high.count() < min_required_len
    ):
        nan_series = pd.Series(np.nan, index=high.index)
        return nan_series, nan_series

    lowest_low = low.rolling(window=fastk_period, min_periods=fastk_period).min()
    highest_high = high.rolling(window=fastk_period, min_periods=fastk_period).max()

    range_high_low = highest_high - lowest_low
    fastk = 100.0 * (close - lowest_low) / (range_high_low + EPSILON)

    slowk = fastk.rolling(window=slowk_period, min_periods=slowk_period).mean()

    slowd = slowk.rolling(window=slowd_period, min_periods=slowd_period).mean()

    return slowk, slowd


def calculate_bbands(close: pd.Series, timeperiod: int = 20, nbdev: float = 2.0):
    """Calculate Bollinger Bands.

    Args:
        close: Close price series.
        timeperiod: Rolling window length.
        nbdev: Standard deviation multiplier.

    Returns:
        Tuple of (upper, middle, lower) band Series.
    """
    if close.isna().all() or close.count() < timeperiod:
        nan_series = pd.Series(np.nan, index=close.index)
        return nan_series, nan_series, nan_series

    middle = close.rolling(window=timeperiod, min_periods=timeperiod).mean()
    std_dev = close.rolling(window=timeperiod, min_periods=timeperiod).std()
    upper = middle + (nbdev * std_dev)
    lower = middle - (nbdev * std_dev)
    return upper, middle, lower


def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14):
    """Calculate Commodity Channel Index (CCI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        timeperiod: Rolling window length.

    Returns:
        CCI values as a Series.
    """
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod:
        return pd.Series(np.nan, index=high.index)

    tp = (high + low + close) / 3
    tp_sma = tp.rolling(window=timeperiod, min_periods=timeperiod).mean()

    mean_dev = tp.rolling(window=timeperiod, min_periods=timeperiod).apply(
        lambda x: np.mean(np.abs(x - np.nanmean(x))) if not np.isnan(x).all() else np.nan,
        raw=True,
    )

    cci = (tp - tp_sma) / (0.015 * mean_dev + EPSILON)
    return cci


def calculate_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    timeperiod: int = 14,
) -> pd.Series:
    """Calculate Money Flow Index (MFI).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        volume: Volume series.
        timeperiod: Rolling window length.

    Returns:
        MFI values as a Series.
    """
    if (
        high.isna().all()
        or low.isna().all()
        or close.isna().all()
        or volume.isna().all()
        or high.count() < timeperiod
    ):
        return pd.Series(np.nan, index=high.index)

    tp = (high + low + close) / 3
    rmf = tp * volume
    tp_diff = tp.diff(1).astype(float)

    pmf = rmf.where(tp_diff > 0, 0.0).rolling(window=timeperiod, min_periods=timeperiod).sum()
    nmf = rmf.where(tp_diff < 0, 0.0).rolling(window=timeperiod, min_periods=timeperiod).sum()

    mfr = pmf / (nmf + EPSILON)
    mfi = 100.0 - (100.0 / (1.0 + mfr))

    mfi[nmf == 0] = 100.0

    return mfi


def calculate_atr_wilder(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int):
    """Calculate ATR with Wilder smoothing (alpha=1/N).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        timeperiod: Rolling window length.

    Returns:
        ATR values as a Series.
    """
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod + 1:
        return pd.Series(np.nan, index=high.index)

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)

    atr = tr.ewm(alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod).mean()
    return atr


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, timeperiod: int = 14):
    """Calculate Average Directional Movement Index (ADX).

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        timeperiod: Rolling window length.

    Returns:
        ADX values as a Series.
    """
    if high.isna().all() or low.isna().all() or close.isna().all() or high.count() < timeperiod * 2:
        return pd.Series(np.nan, index=high.index)

    move_up = high.diff(1).astype(float)
    move_down = -low.diff(1).astype(float)

    plus_dm = pd.Series(
        np.where((move_up > move_down) & (move_up > 0), move_up, 0.0), index=high.index
    ).fillna(0)
    minus_dm = pd.Series(
        np.where((move_down > move_up) & (move_down > 0), move_down, 0.0),
        index=high.index,
    ).fillna(0)

    atr = calculate_atr_wilder(high, low, close, timeperiod)

    plus_dm_smooth = plus_dm.ewm(
        alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod
    ).mean()
    minus_dm_smooth = minus_dm.ewm(
        alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod
    ).mean()

    plus_di = 100.0 * (plus_dm_smooth / (atr + EPSILON))
    minus_di = 100.0 * (minus_dm_smooth / (atr + EPSILON))

    di_diff = abs(plus_di - minus_di)
    di_sum = plus_di + minus_di

    dx = 100.0 * (di_diff / (di_sum + EPSILON))
    dx[di_sum < EPSILON] = 0.0

    adx = dx.ewm(alpha=1.0 / timeperiod, adjust=False, min_periods=timeperiod).mean()

    return adx
