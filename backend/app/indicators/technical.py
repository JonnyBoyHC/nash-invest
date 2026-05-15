"""Technical indicators: RSI, MACD, Bollinger Bands, ATR, OBV.

All functions accept ordered lists/arrays of close prices (and optionally high/low/volume).
Returns dicts suitable for JSON serialization.
"""

import math
import numpy as np
from typing import Optional


def _clean(arr):
    """Replace NaN with None for JSON serialization."""
    if isinstance(arr, np.ndarray):
        return [None if isinstance(v, float) and math.isnan(v) else v for v in arr.tolist()]
    return [None if isinstance(v, float) and math.isnan(v) else v for v in arr]


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index (Wilder's smoothing)."""
    if len(closes) < period + 1:
        return np.full(len(closes), np.nan)

    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Wilder's smoothing: initial SMA, then EMA
    avg_gain = np.full(len(closes), np.nan)
    avg_loss = np.full(len(closes), np.nan)
    avg_gain[period] = gains[:period].mean()
    avg_loss[period] = losses[:period].mean()

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
        avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss != 0)
    rsi_vals = 100.0 - (100.0 / (1.0 + rs))
    return _clean(np.round(rsi_vals, 2))


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    result = np.full(len(arr), np.nan)
    if len(arr) < period:
        return result
    result[period - 1] = arr[:period].mean()
    alpha = 2.0 / (period + 1.0)
    for i in range(period, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def sma(arr: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    result = np.full(len(arr), np.nan)
    if len(arr) < period:
        return result
    cumsum = np.cumsum(np.insert(arr, 0, 0.0))
    result[period - 1:] = (cumsum[period:] - cumsum[:-period]) / period
    return result


def macd(
    closes: np.ndarray,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict:
    """MACD line, signal line, histogram."""
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line[slow - 1:], signal)  # start where macd_line is defined
    # Pad signal_line to same length as closes
    signal_padded = np.full(len(closes), np.nan)
    start = slow - 1
    signal_padded[start:] = np.pad(
        signal_line,
        (0, len(closes) - start - len(signal_line)),
        constant_values=np.nan,
    )[: len(closes) - start]
    histogram = macd_line - signal_padded
    return {
        "macd_line": _clean(np.round(macd_line, 4)),
        "signal_line": _clean(np.round(signal_padded, 4)),
        "histogram": _clean(np.round(histogram, 4)),
    }


def bollinger_bands(
    closes: np.ndarray,
    period: int = 20,
    num_std: float = 2.0,
) -> dict:
    """Bollinger Bands: middle (SMA), upper, lower, %B, bandwidth."""
    middle = sma(closes, period)
    # Rolling std
    rolling_std = np.full(len(closes), np.nan)
    for i in range(period - 1, len(closes)):
        rolling_std[i] = np.std(closes[i - period + 1 : i + 1], ddof=1)

    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std

    # %B = (price - lower) / (upper - lower)
    band_range = upper - lower
    pct_b = np.divide(
        closes - lower, band_range,
        out=np.full_like(closes, np.nan), where=band_range != 0,
    )

    # Bandwidth = (upper - lower) / middle
    bandwidth = np.divide(
        band_range, middle,
        out=np.full_like(closes, np.nan), where=middle != 0,
    )

    return {
        "upper": _clean(np.round(upper, 2)),
        "middle": _clean(np.round(middle, 2)),
        "lower": _clean(np.round(lower, 2)),
        "pct_b": _clean(np.round(pct_b, 4)),
        "bandwidth": _clean(np.round(bandwidth, 4)),
    }


def atr(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    period: int = 14,
) -> np.ndarray:
    """Average True Range (Wilder's smoothing)."""
    n = len(closes)
    if n < 2:
        return np.full(n, np.nan)

    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]

    tr = np.maximum(
        highs - lows,
        np.maximum(
            np.abs(highs - prev_close),
            np.abs(lows - prev_close),
        ),
    )

    atr_vals = np.full(n, np.nan)
    atr_vals[period] = tr[1:period + 1].mean()  # first TR at index 1

    for i in range(period + 1, n):
        atr_vals[i] = (atr_vals[i - 1] * (period - 1) + tr[i]) / period

    return np.round(atr_vals, 2)


def obv(closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """On-Balance Volume."""
    n = len(closes)
    if n < 2:
        return np.cumsum(volumes) if n > 0 else np.array([])

    obv_vals = np.zeros(n)
    obv_vals[0] = volumes[0]

    for i in range(1, n):
        if closes[i] > closes[i - 1]:
            obv_vals[i] = obv_vals[i - 1] + volumes[i]
        elif closes[i] < closes[i - 1]:
            obv_vals[i] = obv_vals[i - 1] - volumes[i]
        else:
            obv_vals[i] = obv_vals[i - 1]

    return obv_vals
