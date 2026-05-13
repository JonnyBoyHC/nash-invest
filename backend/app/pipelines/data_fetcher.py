"""Fetch market data via yfinance and store locally."""
import logging
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.database import SessionLocal
from app.models.market import Asset, PriceData

logger = logging.getLogger(__name__)


def ensure_asset(db: Session, ticker: str) -> Asset:
    """Get or create an Asset row."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        info = yf.Ticker(ticker).info
        asset = Asset(
            ticker=ticker.upper(),
            name=info.get("shortName") or info.get("longName") or ticker,
            asset_type="equity",
            currency=info.get("currency", "USD"),
        )
        db.add(asset)
        db.flush()
    return asset


def fetch_price_history(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    period: str = "1y",
) -> pd.DataFrame:
    """Download price history from Yahoo Finance."""
    if end is None:
        end = date.today()
    if start is None:
        start = end - timedelta(days=365)

    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    # Normalize MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def sync_prices(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    db: Optional[Session] = None,
) -> int:
    """Fetch prices for `ticker` and upsert into the database. Returns rows inserted."""
    close_db = db is None
    if db is None:
        db = SessionLocal()

    try:
        asset = ensure_asset(db, ticker)
        df = fetch_price_history(ticker, start=start, end=end)

        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return 0

        rows = 0
        for _, row in df.iterrows():
            trade_date = row.get("date")
            if isinstance(trade_date, pd.Timestamp):
                trade_date = trade_date.date()

            stmt = sqlite_insert(PriceData).values(
                asset_id=asset.id,
                date=trade_date,
                open=float(row.get("open", 0)) if pd.notna(row.get("open")) else None,
                high=float(row.get("high", 0)) if pd.notna(row.get("high")) else None,
                low=float(row.get("low", 0)) if pd.notna(row.get("low")) else None,
                close=float(row["close"]),
                volume=float(row.get("volume", 0)) if pd.notna(row.get("volume")) else None,
                adjusted_close=float(row["close"]),
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=["asset_id", "date"],
                set_={
                    "open": stmt.excluded.open,
                    "high": stmt.excluded.high,
                    "low": stmt.excluded.low,
                    "close": stmt.excluded.close,
                    "volume": stmt.excluded.volume,
                    "adjusted_close": stmt.excluded.adjusted_close,
                },
            )
            db.execute(stmt)
            rows += 1

        db.commit()
        logger.info(f"Synced {rows} price records for {ticker}")
        return rows
    finally:
        if close_db:
            db.close()


def sync_watchlist(
    tickers: list[str],
    start: Optional[date] = None,
) -> dict[str, int]:
    """Sync prices for every ticker in the watchlist. Returns {ticker: rows}."""
    db = SessionLocal()
    try:
        results = {}
        for ticker in tickers:
            try:
                n = sync_prices(ticker.strip().upper(), start=start, db=db)
                results[ticker] = n
            except Exception as e:
                logger.error(f"Failed to sync {ticker}: {e}")
                results[ticker] = -1
        return results
    finally:
        db.close()
