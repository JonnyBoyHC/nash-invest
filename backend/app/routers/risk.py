"""Risk metrics API: VaR, CVaR, Sharpe, Max Drawdown, Beta."""
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd

from app.database import get_db
from app.models.market import Asset, PriceData

router = APIRouter(prefix="/api/risk", tags=["risk"])

# Risk-free rate (approximate 1-year Treasury yield, updated periodically)
RISK_FREE_RATE = 0.045  # 4.5%


def _daily_returns(prices: list) -> np.ndarray:
    """Compute daily log returns from ordered PriceData list."""
    closes = np.array([p.close for p in prices], dtype=float)
    return np.diff(np.log(closes))


def _get_spy_returns(db: Session, start_date: date) -> Optional[np.ndarray]:
    """Get SPY daily returns for beta calculation, or None."""
    spy = db.query(Asset).filter(Asset.ticker == "SPY").first()
    if not spy:
        return None
    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == spy.id, PriceData.date >= start_date)
        .order_by(PriceData.date.asc())
        .all()
    )
    if len(prices) < 21:
        return None
    return _daily_returns(prices)


def compute_risk_metrics(db: Session, ticker: str, lookback_days: int = 252) -> dict:
    """Compute all risk metrics for a ticker."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not tracked")

    cutoff = date.today() - timedelta(days=lookback_days + 5)
    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id, PriceData.date >= cutoff)
        .order_by(PriceData.date.asc())
        .all()
    )

    if len(prices) < 21:
        raise HTTPException(status_code=400, detail=f"Need at least 21 price records, got {len(prices)}")

    returns = _daily_returns(prices)
    n = len(returns)
    current_price = float(prices[-1].close)

    # ── Basic stats ────────────────────────────────────────────────
    daily_mean = float(np.mean(returns))
    daily_vol = float(np.std(returns, ddof=1))
    annual_return = daily_mean * 252
    annual_vol = daily_vol * np.sqrt(252)

    # ── Sharpe Ratio ───────────────────────────────────────────────
    excess_daily = daily_mean - (RISK_FREE_RATE / 252)
    sharpe = (excess_daily / daily_vol) * np.sqrt(252) if daily_vol > 0 else 0

    # ── VaR 95% (parametric, 1-day) ────────────────────────────────
    # Assume Normal; for fat tails we use historical percentile
    from scipy import stats as sp_stats
    var_95_parametric = float(sp_stats.norm.ppf(0.05, daily_mean, daily_vol))
    var_95_historical = float(np.percentile(returns, 5))
    var_95_dollar = current_price * (1 - np.exp(var_95_historical))

    # ── CVaR 95% (expected shortfall) ──────────────────────────────
    tail_losses = returns[returns <= var_95_historical]
    cvar_95 = float(np.mean(tail_losses)) if len(tail_losses) > 0 else var_95_historical
    cvar_95_dollar = current_price * (1 - np.exp(cvar_95))

    # ── Max Drawdown ───────────────────────────────────────────────
    closes = np.array([p.close for p in prices], dtype=float)
    peak = np.maximum.accumulate(closes)
    drawdowns = (closes - peak) / peak
    max_dd = float(np.min(drawdowns))
    max_dd_idx = int(np.argmin(drawdowns))
    peak_idx = int(np.argmax(closes[:max_dd_idx + 1])) if max_dd_idx > 0 else 0
    max_dd_start = prices[peak_idx].date.isoformat() if peak_idx < len(prices) else None
    max_dd_end = prices[max_dd_idx].date.isoformat() if max_dd_idx < len(prices) else None

    # Current drawdown from ATH
    ath = float(np.max(closes))
    current_dd = float((current_price - ath) / ath)

    # ── Beta vs SPY ────────────────────────────────────────────────
    spy_rets = _get_spy_returns(db, cutoff)
    beta = None
    if spy_rets is not None and len(spy_rets) > 0:
        min_len = min(len(returns), len(spy_rets))
        r = returns[-min_len:]
        s = spy_rets[-min_len:]
        cov = np.cov(r, s)[0, 1]
        var_bench = np.var(s, ddof=1)
        beta = round(float(cov / var_bench), 4) if var_bench > 0 else None

    # ── Rolling volatility ─────────────────────────────────────────
    window = min(21, n)
    rolling_vol = pd.Series(returns).rolling(window).std().dropna().values
    vol_current = float(rolling_vol[-1]) * np.sqrt(252) if len(rolling_vol) > 0 else annual_vol
    vol_min = float(np.min(rolling_vol)) * np.sqrt(252) if len(rolling_vol) > 0 else annual_vol
    vol_max = float(np.max(rolling_vol)) * np.sqrt(252) if len(rolling_vol) > 0 else annual_vol

    # ── Return distribution stats ──────────────────────────────────
    skew = float(sp_stats.skew(returns))
    kurtosis = float(sp_stats.kurtosis(returns))  # excess kurtosis

    return {
        "ticker": asset.ticker,
        "current_price": round(current_price, 2),
        "lookback_days": lookback_days,
        "observations": n,
        "returns": {
            "daily_mean_pct": round(daily_mean * 100, 4),
            "daily_vol_pct": round(daily_vol * 100, 4),
            "annual_return_pct": round(annual_return * 100, 2),
            "annual_vol_pct": round(annual_vol * 100, 2),
            "skewness": round(skew, 4),
            "excess_kurtosis": round(kurtosis, 4),
        },
        "sharpe_ratio": round(sharpe, 4),
        "var_95_1day": {
            "parametric_pct": round(var_95_parametric * 100, 4),
            "historical_pct": round(var_95_historical * 100, 4),
            "dollar": round(var_95_dollar, 2),
            "dollar_formatted": f"${var_95_dollar:,.2f}",
        },
        "cvar_95_1day": {
            "historical_pct": round(cvar_95 * 100, 4),
            "dollar": round(cvar_95_dollar, 2),
            "dollar_formatted": f"${cvar_95_dollar:,.2f}",
        },
        "max_drawdown": {
            "pct": round(max_dd * 100, 2),
            "start_date": max_dd_start,
            "end_date": max_dd_end,
        },
        "current_drawdown_pct": round(current_dd * 100, 2),
        "beta_vs_spy": beta,
        "volatility_cone": {
            "current_annual_pct": round(vol_current * 100, 2),
            "min_annual_pct": round(vol_min * 100, 2),
            "max_annual_pct": round(vol_max * 100, 2),
            "window_days": window,
        },
    }


@router.get("/{ticker}")
def get_risk_metrics(
    ticker: str,
    lookback_days: int = Query(default=252, ge=21, le=1260),
    db: Session = Depends(get_db),
):
    """Full risk report: VaR, CVaR, Sharpe, max drawdown, beta, volatility."""
    return compute_risk_metrics(db, ticker, lookback_days)


@router.get("/summary/all")
def get_risk_summary(db: Session = Depends(get_db)):
    """Quick risk summary for all tracked assets."""
    assets = db.query(Asset).all()
    results = []
    for asset in assets:
        try:
            m = compute_risk_metrics(db, asset.ticker)
            results.append({
                "ticker": m["ticker"],
                "sharpe": m["sharpe_ratio"],
                "var_95_dollar": m["var_95_1day"]["dollar"],
                "max_dd_pct": m["max_drawdown"]["pct"],
                "annual_vol_pct": m["returns"]["annual_vol_pct"],
                "annual_return_pct": m["returns"]["annual_return_pct"],
                "beta": m["beta_vs_spy"],
                "current_drawdown_pct": m["current_drawdown_pct"],
            })
        except Exception as e:
            results.append({"ticker": asset.ticker, "error": str(e)})
    return results
