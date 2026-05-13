"""Market data API endpoints."""
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd

from app.database import get_db
from app.models.market import Asset, PriceData
from app.pipelines.data_fetcher import sync_prices, sync_watchlist, ensure_asset
from app.config import settings

router = APIRouter(prefix="/api/market", tags=["market"])


@router.get("/assets")
def list_assets(db: Session = Depends(get_db)):
    """List all tracked assets."""
    assets = db.query(Asset).all()
    return [
        {
            "id": a.id,
            "ticker": a.ticker,
            "name": a.name,
            "asset_type": a.asset_type,
            "currency": a.currency,
            "added_at": a.added_at.isoformat() if a.added_at else None,
        }
        for a in assets
    ]


@router.post("/assets/{ticker}")
def add_asset(ticker: str, db: Session = Depends(get_db)):
    """Add a ticker to tracking. Syncs 1 year of history."""
    try:
        asset = ensure_asset(db, ticker.upper())
        db.commit()
        sync_prices(ticker.upper(), db=db)
        return {"status": "ok", "asset": {"id": asset.id, "ticker": asset.ticker, "name": asset.name}}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/prices/{ticker}")
def get_prices(
    ticker: str,
    days: int = Query(default=90, ge=1, le=3650),
    db: Session = Depends(get_db),
):
    """Get price history for a ticker."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not tracked")

    cutoff = date.today() - timedelta(days=days)
    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id, PriceData.date >= cutoff)
        .order_by(PriceData.date.asc())
        .all()
    )

    return {
        "ticker": asset.ticker,
        "name": asset.name,
        "prices": [
            {
                "date": p.date.isoformat(),
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume,
            }
            for p in prices
        ],
    }


@router.post("/sync")
def trigger_sync(
    tickers: Optional[str] = Query(default=None, description="Comma-separated, default: watchlist"),
    db: Session = Depends(get_db),
):
    """Sync prices for tickers. Defaults to configured watchlist."""
    tlist = [t.strip().upper() for t in (tickers or "").split(",") if t.strip()]
    if not tlist:
        tlist = settings.watchlist
    results = sync_watchlist(tlist)
    return {"synced": results}


@router.get("/watchlist")
def get_watchlist(db: Session = Depends(get_db)):
    """Get latest price + daily change for every watched asset."""
    assets = db.query(Asset).all()
    results = []
    for asset in assets:
        latest = (
            db.query(PriceData)
            .filter(PriceData.asset_id == asset.id)
            .order_by(PriceData.date.desc())
            .limit(2)
            .all()
        )
        if not latest:
            continue

        current = latest[0]
        change_pct = None
        if len(latest) > 1 and latest[1].close and latest[1].close > 0:
            change_pct = round(
                ((current.close - latest[1].close) / latest[1].close) * 100, 2
            )

        results.append({
            "ticker": asset.ticker,
            "name": asset.name,
            "price": current.close,
            "date": current.date.isoformat(),
            "change_pct": change_pct,
            "volume": current.volume,
        })

    return results
