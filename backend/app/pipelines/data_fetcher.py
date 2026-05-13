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

MAX_RETRIES = 4
BACKOFF_BASE = 3.0  # seconds, doubles each retry


def ensure_asset(db: Session, ticker: str) -> Asset:
    """Get or create an Asset row. Does NOT hit Yahoo — uses ticker as name.
    For full company info, call refresh_asset_info separately."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        asset = Asset(
            ticker=ticker.upper(),
            name=ticker,
            asset_type="equity",
            currency="USD",
        )
        db.add(asset)
        db.flush()
    return asset


def refresh_asset_info(db: Session, ticker: str) -> None:
    """Backfill company name/currency from Yahoo (rate-limited, call sparingly)."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        return
    for attempt in range(MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).info
            asset.name = info.get("shortName") or info.get("longName") or ticker
            asset.currency = info.get("currency", "USD")
            db.commit()
            return
        except Exception as e:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Info refresh for {ticker} failed (attempt {attempt+1}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)


def fetch_bulk_prices(
    tickers: list[str],
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> dict[str, pd.DataFrame]:
    """Download price history for multiple tickers in a SINGLE Yahoo call.

    Returns {ticker: DataFrame}. One request regardless of ticker count."""
    if end is None:
        end = date.today()
    if start is None:
        start = end - timedelta(days=365)

    for attempt in range(MAX_RETRIES):
        try:
            df = yf.download(
                tickers,
                start=start.isoformat(),
                end=(end + timedelta(days=1)).isoformat(),
                progress=False,
                auto_adjust=True,
            )

            if df.empty:
                if attempt < MAX_RETRIES - 1:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    logger.warning(f"Empty bulk response, retrying in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                return {}

            # Parse multi-ticker result into per-ticker DataFrames
            result = {}
            for t in tickers:
                t_upper = t.upper()
                try:
                    if len(tickers) == 1:
                        ticker_df = df.copy()
                    elif "Close" in df.columns.levels[0] if isinstance(df.columns, pd.MultiIndex) else False:
                        ticker_df = df.xs(t_upper, axis=1, level=0)
                    else:
                        # Single ticker result
                        ticker_df = df.copy()
                except KeyError:
                    logger.warning(f"No price data in bulk result for {t_upper}")
                    continue

                ticker_df = ticker_df.reset_index()
                ticker_df.columns = [c.lower().replace(" ", "_") for c in ticker_df.columns]
                if isinstance(ticker_df.columns, pd.MultiIndex):
                    ticker_df.columns = ticker_df.columns.get_level_values(0)
                result[t_upper] = ticker_df

            return result

        except Exception as e:
            wait = BACKOFF_BASE * (2 ** attempt)
            logger.warning(f"Bulk download failed (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(wait)
            else:
                raise

    return {}


def fetch_price_history(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """Download price history for a single ticker with retry. Prefer fetch_bulk_prices
    for multiple tickers to avoid rate limits."""
    result = fetch_bulk_prices([ticker], start=start, end=end)
    return result.get(ticker.upper(), pd.DataFrame())


def upsert_price_data(db: Session, asset_id: int, df: pd.DataFrame) -> int:
    """Insert/update PriceData rows from a DataFrame. Returns count."""
    rows = 0
    for _, row in df.iterrows():
        trade_date = row.get("date")
        if isinstance(trade_date, pd.Timestamp):
            trade_date = trade_date.date()

        close_val = row.get("close")
        if close_val is None or (hasattr(close_val, '__iter__') and pd.isna(close_val)):
            continue

        stmt = sqlite_insert(PriceData).values(
            asset_id=asset_id,
            date=trade_date,
            open=float(row.get("open", 0)) if pd.notna(row.get("open")) else None,
            high=float(row.get("high", 0)) if pd.notna(row.get("high")) else None,
            low=float(row.get("low", 0)) if pd.notna(row.get("low")) else None,
            close=float(close_val),
            volume=float(row.get("volume", 0)) if pd.notna(row.get("volume")) else None,
            adjusted_close=float(close_val),
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
    return rows


def sync_prices(
    ticker: str,
    start: Optional[date] = None,
    end: Optional[date] = None,
    db: Optional[Session] = None,
) -> int:
    """Fetch and store prices for a single ticker. Returns rows inserted."""
    close_db = db is None
    if db is None:
        db = SessionLocal()
    try:
        asset = ensure_asset(db, ticker)
        df = fetch_price_history(ticker, start=start, end=end)
        if df.empty:
            logger.warning(f"No data returned for {ticker}")
            return 0
        rows = upsert_price_data(db, asset.id, df)
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
    """Sync all watchlist tickers using a SINGLE bulk Yahoo download.

    Avoids per-ticker rate limiting — one HTTP request for all tickers."""
    db = SessionLocal()
    try:
        # Ensure all assets exist (no Yahoo calls)
        assets = {}
        for t in tickers:
            t_upper = t.strip().upper()
            assets[t_upper] = ensure_asset(db, t_upper)
        db.commit()

        # Bulk download all tickers in ONE request
        logger.info(f"Bulk downloading prices for {list(assets.keys())}")
        all_data = fetch_bulk_prices(list(assets.keys()), start=start)

        results = {}
        for ticker_upper, asset in assets.items():
            df = all_data.get(ticker_upper, pd.DataFrame())
            if df.empty:
                logger.warning(f"No price data in bulk result for {ticker_upper}")
                results[ticker_upper] = 0
            else:
                rows = upsert_price_data(db, asset.id, df)
                results[ticker_upper] = rows
                logger.info(f"Synced {rows} price records for {ticker_upper}")

        db.commit()
        return results
    except Exception as e:
        logger.error(f"Bulk watchlist sync failed: {e}")
        return {t: -1 for t in tickers}
    finally:
        db.close()
