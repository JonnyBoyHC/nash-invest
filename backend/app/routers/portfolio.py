"""Portfolio API: correlation matrix, efficient frontier, optimal weights."""
from datetime import date, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
import numpy as np

from app.database import get_db
from app.models.market import Asset, PriceData
from app.portfolio.optimizer import covariance_matrix, efficient_frontier
from app.config import settings

router = APIRouter(prefix="/api/portfolio", tags=["portfolio"])


def _get_all_price_data(db: Session, days: int = 252) -> dict:
    """Fetch aligned price arrays for all tracked assets."""
    cutoff = date.today() - timedelta(days=days + 5)
    assets = db.query(Asset).all()

    price_data = {}
    for asset in assets:
        prices = (
            db.query(PriceData)
            .filter(PriceData.asset_id == asset.id, PriceData.date >= cutoff)
            .order_by(PriceData.date.asc())
            .all()
        )
        if prices:
            close_array = np.array([p.close for p in prices], dtype=float)
            price_data[asset.ticker] = close_array

    return price_data


@router.get("/correlation")
def get_correlation(
    days: int = Query(default=252, ge=30, le=1260),
    db: Session = Depends(get_db),
):
    """Correlation and covariance matrices for all tracked assets."""
    price_data = _get_all_price_data(db, days)

    if len(price_data) < 2:
        return {"error": "Need at least 2 assets with price data"}

    returns = {}
    for ticker, prices in price_data.items():
        if len(prices) >= 21:
            returns[ticker] = np.diff(np.log(prices))

    if len(returns) < 2:
        return {"error": "Need at least 2 assets with sufficient returns"}

    result = covariance_matrix(returns)
    result["lookback_days"] = days
    return result


@router.get("/efficient-frontier")
def get_efficient_frontier(
    days: int = Query(default=252, ge=30, le=1260),
    n_portfolios: int = Query(default=5000, ge=100, le=50000),
    db: Session = Depends(get_db),
):
    """Efficient frontier and optimal portfolios (max Sharpe, min vol, equal weight)."""
    price_data = _get_all_price_data(db, days)

    if len(price_data) < 2:
        return {"error": "Need at least 2 assets with price data"}

    returns = {}
    for ticker, prices in price_data.items():
        if len(prices) >= 21:
            returns[ticker] = np.diff(np.log(prices))

    if len(returns) < 2:
        return {"error": "Need at least 2 assets with sufficient returns"}

    result = efficient_frontier(returns, n_portfolios=n_portfolios)
    result["lookback_days"] = days
    return result


@router.get("/summary")
def get_portfolio_summary(
    days: int = Query(default=252, ge=30, le=1260),
    db: Session = Depends(get_db),
):
    """Combined correlation + efficient frontier summary."""
    price_data = _get_all_price_data(db, days)

    if len(price_data) < 2:
        return {"error": "Need at least 2 assets with price data"}

    returns = {}
    for ticker, prices in price_data.items():
        if len(prices) >= 21:
            returns[ticker] = np.diff(np.log(prices))

    if len(returns) < 2:
        return {"error": "Need at least 2 assets with sufficient returns"}

    corr_result = covariance_matrix(returns)
    ef_result = efficient_frontier(returns)

    return {
        "lookback_days": days,
        "correlation": corr_result["correlation"],
        "stats": corr_result["stats"],
        "efficient_frontier": ef_result["efficient_frontier"],
        "optimal_portfolios": ef_result["optimal_portfolios"],
    }
