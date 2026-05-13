"""Market data SQLAlchemy models."""
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, UniqueConstraint, Index
)
from sqlalchemy.orm import relationship
from app.database import Base


class Asset(Base):
    """A tracked security (stock, ETF, index, crypto)."""
    __tablename__ = "assets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(200))
    asset_type = Column(String(20), default="equity")  # equity, etf, index, crypto
    currency = Column(String(10), default="USD")
    added_at = Column(DateTime, default=datetime.utcnow)

    prices = relationship("PriceData", back_populates="asset", cascade="all, delete-orphan")


class PriceData(Base):
    """Daily OHLCV for an asset."""
    __tablename__ = "price_data"
    __table_args__ = (
        UniqueConstraint("asset_id", "date", name="uq_asset_date"),
        Index("idx_price_asset_date", "asset_id", "date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float, nullable=False)
    volume = Column(Float)
    adjusted_close = Column(Float)

    asset = relationship("Asset", back_populates="prices")
