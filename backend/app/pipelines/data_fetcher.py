"""Fetch market data via yfinance and store locally."""
import logging
import time
from datetime import date, datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from app.database import SessionLocal
from app.models.market import Asset, PriceData

logger = logging.getLogger(__name__)

# Seconds to pause between API calls to avoid Yahoo 429 rate limits
RATE_LIMIT_DELAY = 2.0


def ensure_asset(db: Session, ticker: str) -> Asset:
    """Get or create an Asset row. Falls back to ticker name if info fetch fails."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        name = ticker
        asset_type = "equity"
        currency = "USD"
        try:
            info = yf.Ticker(ticker).info
            name = info.get("shortName") or info.get("longName") or ticker
            currency = info.get("currency", "USD")
        except Exception:
            logger.warning(f"Could not fetch info for {ticker}, using ticker as name")
        asset = Asset(
            ticker=ticker.upper(),
            name=name,
            asset_type=asset_type,
            currency=currency,
        )
        db.add(asset)
        db.flush()
    return asset


def fetch_price_history(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    period: str = "1y",
    retries: int = 3,
) -> pd.DataFrame:
    """Download price history from Yahoo Finance with retry on rate limit."""
    if end is None:
        end = date.today()
    if start is None:
        start = end - timedelta(days=365)

    for attempt in range(retries):
        try:
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

        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = (attempt + 1) * RATE_LIMIT_DELAY
                logger.warning(f"Rate limited on {ticker}, retrying in {wait}s (attempt {attempt+1}/{retries})")
                time.sleep(wait)
            else:
                raise

    return pd.DataFrame()


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
    """Sync prices for every ticker in the watchlist. Returns {ticker: rows}.

    Adds a pause between tickers to avoid Yahoo Finance rate limiting (429)."""
    db = SessionLocal()
    try:
        results = {}
        for i, ticker in enumerate(tickers):
            t = ticker.strip().upper()
            try:
                n = sync_prices(t, start=start, db=db)
                results[ticker] = n
            except Exception as e:
                logger.error(f"Failed to sync {t}: {e}")
                results[ticker] = -1
            # Pause between tickers to avoid 429 — skip after the last one
            if i < len(tickers) - 1:
                time.sleep(RATE_LIMIT_DELAY)
        return results
    finally:
        db.close()
