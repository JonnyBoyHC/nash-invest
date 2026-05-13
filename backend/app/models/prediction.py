"""Prediction SQLAlchemy models."""
from datetime import datetime, date
from sqlalchemy import (
    Column, Integer, String, Float, Date, DateTime, Text, JSON, ForeignKey, Index
)
from sqlalchemy.orm import relationship
from app.database import Base


class ModelVersion(Base):
    """Tracks each deployed model version for audit trail."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(20), nullable=False)
    description = Column(Text)
    params = Column(JSON)  # hyperparameters snapshot
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="model_version")


class Prediction(Base):
    """A single forecast: predicted distribution parameters and the actual outcome."""
    __tablename__ = "predictions"
    __table_args__ = (
        Index("idx_pred_asset_date", "asset_id", "target_date"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    asset_id = Column(Integer, nullable=False)
    model_version_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    target_date = Column(Date, nullable=False)
    made_at = Column(DateTime, default=datetime.utcnow)

    # Predicted distribution (Normal for starters; extend to T/others)
    pred_mean = Column(Float, nullable=False)
    pred_std = Column(Float, nullable=False)
    pred_5th = Column(Float)  # lower credible interval
    pred_95th = Column(Float)  # upper credible interval

    # What actually happened (backfilled after target date passes)
    actual_close = Column(Float, nullable=True)
    error = Column(Float, nullable=True)  # pred_mean - actual_close
    z_score = Column(Float, nullable=True)  # (actual - pred_mean) / pred_std

    model_version = relationship("ModelVersion", back_populates="predictions")
