"""Prediction API endpoints: generate forecasts, score past predictions."""
from datetime import date, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
import numpy as np

from app.database import get_db
from app.models.market import Asset, PriceData
from app.models.prediction import ModelVersion, Prediction
from app.predictors.bayesian import BayesianReturnsPredictor, MODEL_NAME, MODEL_VERSION
from app.predictors.base import ForecastSet
from app.predictors.student_t import predict_student_t, compare_models
from app.predictors.garch import volatility_forecast_ci
from app.config import settings

router = APIRouter(prefix="/api/predictions", tags=["predictions"])


def _get_model_version(db: Session, model_name: str, version: str, params: dict) -> ModelVersion:
    """Get or create a model version record."""
    mv = (
        db.query(ModelVersion)
        .filter(ModelVersion.model_name == model_name, ModelVersion.version == version)
        .first()
    )
    if not mv:
        mv = ModelVersion(
            model_name=model_name,
            version=version,
            description=f"Auto-created {date.today().isoformat()}",
            params=params,
        )
        db.add(mv)
        db.flush()
    return mv


@router.post("/generate/{ticker}")
def generate_forecast(
    ticker: str,
    horizon_days: int = Query(default=None, ge=1, le=365),
    save: bool = Query(default=True),
    db: Session = Depends(get_db),
):
    """Run the Bayesian MCMC model and produce a forecast for `ticker`."""
    if horizon_days is None:
        horizon_days = settings.default_forecast_days

    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"{ticker} not tracked. Add it first via /api/market/assets/{ticker}")

    # Get price history
    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id)
        .order_by(PriceData.date.asc())
        .all()
    )
    if len(prices) < 5:
        raise HTTPException(status_code=400, detail=f"Need at least 5 price records for {ticker}, got {len(prices)}")

    close_prices = np.array([p.close for p in prices], dtype=float)

    # Run MCMC predictor
    predictor = BayesianReturnsPredictor(
        samples=settings.mcmc_samples,
        tune=settings.mcmc_tune,
    )
    forecast_set: ForecastSet = predictor.forecast(ticker.upper(), close_prices, horizon_days)

    # Save predictions to DB
    if save:
        model_version = _get_model_version(
            db,
            forecast_set.model_name,
            forecast_set.model_version,
            {"samples": settings.mcmc_samples, "tune": settings.mcmc_tune},
        )
        for f in forecast_set.forecasts:
            pred = Prediction(
                asset_id=asset.id,
                model_version_id=model_version.id,
                target_date=f.target_date,
                pred_mean=f.pred_mean,
                pred_std=f.pred_std,
                pred_5th=f.pred_5th,
                pred_95th=f.pred_95th,
            )
            db.add(pred)
        db.commit()

    return {
        "ticker": forecast_set.ticker,
        "current_price": forecast_set.current_price,
        "model": forecast_set.model_name,
        "version": forecast_set.model_version,
        "generated_at": forecast_set.generated_at.isoformat(),
        "forecasts": [
            {
                "target_date": f.target_date.isoformat(),
                "pred_mean": round(f.pred_mean, 2),
                "pred_std": round(f.pred_std, 2),
                "ci_lower": round(f.pred_5th, 2),
                "ci_upper": round(f.pred_95th, 2),
            }
            for f in forecast_set.forecasts
        ],
    }


@router.get("/history/{ticker}")
def get_prediction_history(
    ticker: str,
    days: int = Query(default=30, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Get past predictions for a ticker, including actual outcomes and errors."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"{ticker} not tracked")

    cutoff = date.today() - timedelta(days=days)
    preds = (
        db.query(Prediction)
        .filter(Prediction.asset_id == asset.id, Prediction.target_date >= cutoff)
        .order_by(Prediction.target_date.asc())
        .all()
    )

    return {
        "ticker": asset.ticker,
        "predictions": [
            {
                "id": p.id,
                "target_date": p.target_date.isoformat(),
                "pred_mean": p.pred_mean,
                "pred_std": p.pred_std,
                "ci_lower": p.pred_5th,
                "ci_upper": p.pred_95th,
                "actual_close": p.actual_close,
                "error": p.error,
                "z_score": round(p.z_score, 4) if p.z_score is not None else None,
                "made_at": p.made_at.isoformat() if p.made_at else None,
            }
            for p in preds
        ],
    }


@router.post("/backfill")
def backfill_outcomes(
    ticker: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    """Fill in actual_close for predictions whose target date has passed."""
    query = db.query(Prediction).filter(
        Prediction.target_date <= date.today(),
        Prediction.actual_close.is_(None),
    )
    if ticker:
        asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
        if not asset:
            raise HTTPException(status_code=404, detail=f"{ticker} not tracked")
        query = query.filter(Prediction.asset_id == asset.id)

    preds = query.all()
    updated = 0
    for pred in preds:
        actual = (
            db.query(PriceData)
            .filter(
                PriceData.asset_id == pred.asset_id,
                PriceData.date == pred.target_date,
            )
            .first()
        )
        if actual and actual.close:
            pred.actual_close = actual.close
            pred.error = round(pred.pred_mean - actual.close, 4)
            if pred.pred_std and pred.pred_std > 0:
                pred.z_score = round((actual.close - pred.pred_mean) / pred.pred_std, 4)
            updated += 1

    db.commit()
    return {"backfilled": updated, "total_predictions": len(preds)}


@router.get("/model-scores/{ticker}")
def get_model_scores(
    ticker: str,
    days: int = Query(default=90, ge=1, le=365),
    db: Session = Depends(get_db),
):
    """Score predictions: RMSE, MAE, coverage (how often actual fell within CI)."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"{ticker} not tracked")

    cutoff = date.today() - timedelta(days=days)
    preds = (
        db.query(Prediction)
        .filter(
            Prediction.asset_id == asset.id,
            Prediction.target_date >= cutoff,
            Prediction.actual_close.isnot(None),
        )
        .all()
    )

    if not preds:
        return {"ticker": ticker, "scores": None, "message": "No scored predictions yet"}

    errors = [p.error for p in preds if p.error is not None]
    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    mae = float(np.mean(np.abs(errors)))

    in_ci = sum(
        1 for p in preds
        if p.pred_5th is not None and p.pred_95th is not None
        and p.actual_close and p.pred_5th <= p.actual_close <= p.pred_95th
    )
    coverage = round(in_ci / len(preds), 4) if preds else 0

    return {
        "ticker": ticker,
        "n_predictions": len(preds),
        "scores": {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "ci_coverage": coverage,
            "mean_z_score": round(float(np.mean([p.z_score for p in preds if p.z_score is not None])), 4),
        },
    }


@router.post("/generate-student-t/{ticker}")
def generate_student_t_forecast(
    ticker: str,
    horizon_days: int = Query(default=7, ge=1, le=60),
    samples: int = Query(default=500, ge=100, le=5000),
    tune: int = Query(default=500, ge=100, le=5000),
    db: Session = Depends(get_db),
):
    """Run Student-T MCMC forecast (fat tails for TSLA-like assets)."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"{ticker} not tracked")

    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id)
        .order_by(PriceData.date.asc())
        .all()
    )
    if len(prices) < 20:
        raise HTTPException(status_code=400, detail=f"Need at least 20 prices, got {len(prices)}")

    close_prices = np.array([p.close for p in prices], dtype=float)
    result = predict_student_t(close_prices, horizon_days=horizon_days, samples=samples, tune=tune)
    result["ticker"] = ticker.upper()

    return result


@router.post("/compare-models/{ticker}")
def compare_likelihoods(
    ticker: str,
    db: Session = Depends(get_db),
):
    """Compare Normal vs Student-T likelihood via LOO cross-validation."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"{ticker} not tracked")

    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id)
        .order_by(PriceData.date.asc())
        .all()
    )
    if len(prices) < 30:
        raise HTTPException(status_code=400, detail=f"Need at least 30 prices, got {len(prices)}")

    close_prices = np.array([p.close for p in prices], dtype=float)
    result = compare_models(close_prices)
    result["ticker"] = ticker.upper()

    return result


@router.get("/garch/{ticker}")
def get_garch_forecast(
    ticker: str,
    horizon_days: int = Query(default=7, ge=1, le=60),
    db: Session = Depends(get_db),
):
    """GARCH(1,1) volatility forecast with dynamic CIs."""
    asset = db.query(Asset).filter(Asset.ticker == ticker.upper()).first()
    if not asset:
        raise HTTPException(status_code=404, detail=f"{ticker} not tracked")

    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id)
        .order_by(PriceData.date.asc())
        .all()
    )
    if len(prices) < 30:
        raise HTTPException(status_code=400, detail=f"Need at least 30 prices, got {len(prices)}")

    close_prices = np.array([p.close for p in prices], dtype=float)
    returns = np.diff(np.log(close_prices))

    result = volatility_forecast_ci(returns, close_prices, horizon_days=horizon_days)
    result["ticker"] = ticker.upper()
    result["current_price"] = round(float(close_prices[-1]), 2)

    return result
