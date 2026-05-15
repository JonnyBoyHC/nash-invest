"""Technical indicators API: RSI, MACD, Bollinger Bands."""
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
import numpy as np

from app.database import get_db
from app.models.market import Asset, PriceData
from app.indicators.technical import rsi, macd, bollinger_bands
import math

router = APIRouter(prefix="/api/indicators", tags=["indicators"])


def _get_price_arrays(db: Session, ticker: str, days: int):
    """Fetch price data and return numpy arrays + dates."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not tracked")

    cutoff = date.today() - timedelta(days=days + 5)
    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id, PriceData.date >= cutoff)
        .order_by(PriceData.date.asc())
        .all()
    )

    if len(prices) < 30:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 30 price records, got {len(prices)}",
        )

    dates = [p.date.isoformat() for p in prices]
    closes = np.array([p.close for p in prices], dtype=float)
    volumes = np.array([p.volume or 0 for p in prices], dtype=float)

    return dates, closes, volumes


@router.get("/{ticker}/rsi")
def get_rsi(
    ticker: str,
    period: int = Query(default=14, ge=2, le=100),
    days: int = Query(default=365, ge=30, le=1825),
    db: Session = Depends(get_db),
):
    """RSI (Relative Strength Index) with overbought/oversold levels."""
    dates, closes, _ = _get_price_arrays(db, ticker, days)
    values = rsi(closes, period)

    # Latest value and signal
    valid = [v for v in values if v is not None]
    latest = float(valid[-1]) if len(valid) > 0 else None
    signal = "oversold" if latest is not None and latest < 30 else \
             "overbought" if latest is not None and latest > 70 else "neutral"

    return {
        "ticker": ticker.upper(),
        "period": period,
        "dates": dates,
        "values": values,
        "latest": latest,
        "signal": signal,
        "overbought": 70,
        "oversold": 30,
    }


@router.get("/{ticker}/macd")
def get_macd(
    ticker: str,
    fast: int = Query(default=12, ge=2, le=100),
    slow: int = Query(default=26, ge=3, le=200),
    signal_period: int = Query(default=9, ge=2, le=100),
    days: int = Query(default=365, ge=60, le=1825),
    db: Session = Depends(get_db),
):
    """MACD line, signal line, and histogram."""
    if slow <= fast:
        raise HTTPException(status_code=400, detail="slow period must be > fast period")

    dates, closes, _ = _get_price_arrays(db, ticker, days)
    result = macd(closes, fast, slow, signal_period)

    # Latest values
    def _last_valid(arr):
        valid = [v for v in arr if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return valid[-1] if valid else None

    latest_macd = _last_valid(result["macd_line"])
    latest_signal = _last_valid(result["signal_line"])
    latest_hist = _last_valid(result["histogram"])
    direction = "bullish" if latest_hist is not None and latest_hist > 0 else \
                "bearish" if latest_hist is not None and latest_hist < 0 else "neutral"
    crossover = None
    if latest_macd is not None and latest_signal is not None:
        # Check last 2 bars for crossover
        macd_vals = [v for v in result["macd_line"] if v is not None and not np.isnan(v)]
        sig_vals = [v for v in result["signal_line"] if v is not None and not np.isnan(v)]
        if len(macd_vals) >= 2 and len(sig_vals) >= 2:
            if macd_vals[-2] <= sig_vals[-2] and macd_vals[-1] > sig_vals[-1]:
                crossover = "bullish_cross"
            elif macd_vals[-2] >= sig_vals[-2] and macd_vals[-1] < sig_vals[-1]:
                crossover = "bearish_cross"

    return {
        "ticker": ticker.upper(),
        "params": {"fast": fast, "slow": slow, "signal": signal_period},
        "dates": dates,
        **result,
        "latest": {
            "macd_line": latest_macd,
            "signal_line": latest_signal,
            "histogram": latest_hist,
            "direction": direction,
            "crossover": crossover,
        },
    }


@router.get("/{ticker}/bollinger")
def get_bollinger(
    ticker: str,
    period: int = Query(default=20, ge=5, le=200),
    num_std: float = Query(default=2.0, ge=1.0, le=5.0),
    days: int = Query(default=365, ge=30, le=1825),
    db: Session = Depends(get_db),
):
    """Bollinger Bands, %B, and bandwidth."""
    dates, closes, _ = _get_price_arrays(db, ticker, days)
    result = bollinger_bands(closes, period, num_std)

    def _last_valid(arr):
        valid = [v for v in arr if v is not None and not (isinstance(v, float) and np.isnan(v))]
        return valid[-1] if valid else None

    latest_close = float(closes[-1]) if len(closes) > 0 else None
    latest_upper = _last_valid(result["upper"])
    latest_middle = _last_valid(result["middle"])
    latest_lower = _last_valid(result["lower"])
    latest_pct_b = _last_valid(result["pct_b"])
    latest_bandwidth = _last_valid(result["bandwidth"])

    # Position signal
    squeeze = latest_bandwidth is not None and latest_bandwidth < 0.05
    position = None
    if latest_pct_b is not None:
        if latest_pct_b > 1.0:
            position = "above_upper"
        elif latest_pct_b < 0.0:
            position = "below_lower"
        elif latest_pct_b > 0.8:
            position = "near_upper"
        elif latest_pct_b < 0.2:
            position = "near_lower"
        else:
            position = "mid_range"

    return {
        "ticker": ticker.upper(),
        "params": {"period": period, "num_std": num_std},
        "dates": dates,
        **result,
        "latest": {
            "close": latest_close,
            "upper": latest_upper,
            "middle": latest_middle,
            "lower": latest_lower,
            "pct_b": latest_pct_b,
            "bandwidth": latest_bandwidth,
            "squeeze": squeeze,
            "position": position,
        },
    }


@router.get("/{ticker}/all")
def get_all_indicators(
    ticker: str,
    days: int = Query(default=365, ge=60, le=1825),
    db: Session = Depends(get_db),
):
    """All indicators in one call for dashboard efficiency."""
    dates, closes, _ = _get_price_arrays(db, ticker, days)

    rsi_vals = rsi(closes)
    macd_result = macd(closes)
    bb_result = bollinger_bands(closes)

    def _last(arr):
        valid_vals = [v for v in arr if v is not None and not (isinstance(v, float) and math.isnan(v))]
        return valid_vals[-1] if valid_vals else None

    return {
        "ticker": ticker.upper(),
        "dates": dates,
        "close": closes.tolist(),
        "rsi": {
            "values": rsi_vals,
            "latest": float(_last(rsi_vals)) if _last(rsi_vals) is not None else None,
            "signal": "oversold" if (_last(rsi_vals) is not None and _last(rsi_vals) < 30) else
                      "overbought" if (_last(rsi_vals) is not None and _last(rsi_vals) > 70) else "neutral",
        },
        "macd": {
            **macd_result,
            "latest_direction": "bullish" if (_last(macd_result["histogram"]) is not None and _last(macd_result["histogram"]) > 0) else
                               "bearish" if (_last(macd_result["histogram"]) is not None and _last(macd_result["histogram"]) < 0) else "neutral",
        },
        "bollinger": {
            **bb_result,
            "latest_position": "above_upper" if (_last(bb_result["pct_b"]) is not None and _last(bb_result["pct_b"]) > 1.0) else
                               "below_lower" if (_last(bb_result["pct_b"]) is not None and _last(bb_result["pct_b"]) < 0.0) else
                               "near_upper" if (_last(bb_result["pct_b"]) is not None and _last(bb_result["pct_b"]) > 0.8) else
                               "near_lower" if (_last(bb_result["pct_b"]) is not None and _last(bb_result["pct_b"]) < 0.2) else "mid_range",
        },
    }
