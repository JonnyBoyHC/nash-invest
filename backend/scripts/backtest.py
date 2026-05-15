"""Walk-forward backtest: generate historical predictions, backfill outcomes, score.

For each ticker, walks backward through the price history in steps,
generating 7-day forecasts at each step. Compares against actual outcomes.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
import numpy as np

from app.database import SessionLocal, engine, Base
from app.models.market import Asset, PriceData
from app.models.prediction import ModelVersion, Prediction
from app.predictors.bayesian import BayesianReturnsPredictor, MODEL_NAME, MODEL_VERSION
from app.predictors.base import ForecastSet

# ── config ──────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
HORIZON = 7
MIN_TRAIN_DAYS = 60       # need at least 60 days of data to train
STEP_DAYS = 30            # generate a forecast every 30 days going backward
MCMC_SAMPLES = 500        # fewer samples for speed (2000 is overkill for quick test)
MCMC_TUNE = 500

# ── helpers ──────────────────────────────────────────────────────────────

def backtest_ticker(db, ticker: str, predictor) -> dict:
    """Walk-forward backtest for a single ticker. Returns score dict."""
    asset = db.query(Asset).filter(Asset.ticker == ticker).first()
    if not asset:
        return {"ticker": ticker, "error": "not found"}

    prices = (
        db.query(PriceData)
        .filter(PriceData.asset_id == asset.id)
        .order_by(PriceData.date.asc())
        .all()
    )
    if len(prices) < MIN_TRAIN_DAYS + HORIZON:
        return {"ticker": ticker, "error": f"need {MIN_TRAIN_DAYS + HORIZON} prices, got {len(prices)}"}

    close_all = np.array([p.close for p in prices], dtype=float)
    dates_all = [p.date for p in prices]

    # Walk backward from the end, every STEP_DAYS
    n_generated = 0
    predictions = []
    model_version_id = None

    for cutoff_idx in range(len(prices) - HORIZON, MIN_TRAIN_DAYS, -STEP_DAYS):
        train_prices = close_all[:cutoff_idx]
        # The next HORIZON days are our test window
        test_window = list(range(cutoff_idx, min(cutoff_idx + HORIZON, len(prices))))

        try:
            fs: ForecastSet = predictor.forecast(ticker, train_prices, HORIZON)
        except Exception as e:
            print(f"  [{ticker}] fit failed at idx {cutoff_idx}: {e}")
            continue

        # Get or create model version once
        if model_version_id is None:
            mv = (
                db.query(ModelVersion)
                .filter(ModelVersion.model_name == MODEL_NAME, ModelVersion.version == MODEL_VERSION)
                .first()
            )
            if not mv:
                mv = ModelVersion(
                    model_name=MODEL_NAME,
                    version=MODEL_VERSION,
                    description=f"Backtest {date.today().isoformat()}",
                    params={"samples": MCMC_SAMPLES, "tune": MCMC_TUNE},
                )
                db.add(mv)
                db.flush()
            model_version_id = mv.id

        for forecast, actual_idx in zip(fs.forecasts, test_window):
            actual_px = close_all[actual_idx]
            actual_dt = dates_all[actual_idx]

            pred = Prediction(
                asset_id=asset.id,
                model_version_id=model_version_id,
                target_date=actual_dt,
                pred_mean=forecast.pred_mean,
                pred_std=forecast.pred_std,
                pred_5th=forecast.pred_5th,
                pred_95th=forecast.pred_95th,
                actual_close=float(actual_px),
                error=round(float(forecast.pred_mean - actual_px), 4),
            )
            if forecast.pred_std > 0:
                pred.z_score = round((float(actual_px) - forecast.pred_mean) / forecast.pred_std, 4)
            predictions.append(pred)
            n_generated += 1

    # Bulk save
    if predictions:
        db.bulk_save_objects(predictions)
        db.commit()

    # Score these predictions
    errors = [p.error for p in predictions if p.error is not None]
    z_scores = [p.z_score for p in predictions if p.z_score is not None]
    in_ci = sum(
        1 for p in predictions
        if p.pred_5th and p.pred_95th and p.actual_close
        and p.pred_5th <= p.actual_close <= p.pred_95th
    )

    rmse = float(np.sqrt(np.mean(np.array(errors) ** 2)))
    mae = float(np.mean(np.abs(errors)))

    return {
        "ticker": ticker,
        "n_predictions": len(predictions),
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mean_abs_error_pct": f"{round(mae / float(np.mean(close_all)) * 100, 3)}%",
        "ci_coverage": round(in_ci / len(predictions), 4) if predictions else 0,
        "mean_z": round(float(np.mean(z_scores)), 4),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main():
    db = SessionLocal()
    predictor = BayesianReturnsPredictor(samples=MCMC_SAMPLES, tune=MCMC_TUNE, chains=1)

    print("=" * 64)
    print("Nash Invest — Walk-Forward Backtest")
    print(f"Model: {MODEL_NAME} v{MODEL_VERSION}")
    print(f"Horizon: {HORIZON}d  |  MCMC: {MCMC_SAMPLES}s/{MCMC_TUNE}t")
    print(f"Step: every {STEP_DAYS}d  |  Min train: {MIN_TRAIN_DAYS}d")
    print("=" * 64)
    print()

    results = []
    for ticker in TICKERS:
        print(f"[{ticker}] Running walk-forward backtest...")
        r = backtest_ticker(db, ticker, predictor)
        results.append(r)

    db.close()

    # ── summary ──────────────────────────────────────────────────────────
    print()
    print(f"{'Ticker':<8} {'Preds':<7} {'RMSE($)':<10} {'MAE($)':<10} {'MAE%':<10} {'CI Cov':<8} {'Mean Z':<8}")
    print("-" * 64)
    for r in results:
        if "error" in r:
            print(f"{r['ticker']:<8} ERROR: {r['error']}")
        else:
            print(
                f"{r['ticker']:<8} {r['n_predictions']:<7} "
                f"{r['rmse']:<10} {r['mae']:<10} "
                f"{r['mean_abs_error_pct']:<10} {r['ci_coverage']:<8} {r['mean_z']:<8}"
            )

    print()
    avg_rmse = np.mean([r["rmse"] for r in results if "rmse" in r])
    avg_coverage = np.mean([r["ci_coverage"] for r in results if "ci_coverage" in r])
    print(f"Average RMSE: ${avg_rmse:.4f}  |  Average CI coverage: {avg_coverage:.2%}")
    print()
    print("Done. Scores available via: GET /api/predictions/model-scores/{ticker}")


if __name__ == "__main__":
    main()
