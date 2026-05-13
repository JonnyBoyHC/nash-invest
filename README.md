# Nash Invest

**MCMC-driven Bayesian investment research platform.**

Built by Nash 📈 — quantitative analysis, probability distributions, market forecasting.

## Architecture

```
backend/           # Python FastAPI server
  app/
    main.py        # Entry point, startup/shutdown
    config.py      # Pydantic settings via .env
    database.py    # SQLAlchemy + SQLite
    models/        # DB models (Asset, PriceData, Prediction)
    pipelines/     # Data ingestion (yfinance)
    predictors/    # Forecasting models (Bayesian MCMC)
    routers/       # REST API endpoints
```

## Quick Start

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API Endpoints

### Market Data
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/market/assets` | List tracked assets |
| POST | `/api/market/assets/AAPL` | Add ticker + sync history |
| GET | `/api/market/prices/AAPL?days=90` | Price history |
| POST | `/api/market/sync` | Sync watchlist prices |
| GET | `/api/market/watchlist` | Latest prices + daily change |

### Predictions
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predictions/generate/AAPL?horizon_days=7` | Run MCMC forecast |
| GET | `/api/predictions/history/AAPL?days=30` | Past predictions + outcomes |
| POST | `/api/predictions/backfill` | Fill actual outcomes |
| GET | `/api/predictions/model-scores/AAPL?days=90` | RMSE, MAE, CI coverage |

### System
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Service health check |

## Prediction Loop

1. **Sync** → fetch latest prices via yfinance
2. **Predict** → MCMC samples a posterior over returns, projects prices forward
3. **Compare** → once the target date passes, backfill actuals
4. **Refine** → view model scores, adjust priors, iterate

## Tech Stack

- **FastAPI** — async Python API
- **PyMC** — probabilistic programming / MCMC
- **yfinance** — free market data
- **SQLite** — zero-config database
- **Plotly** — server-side charts (coming to frontend)

## Roadmap

- [ ] Frontend dashboard (Lovable.dev)
- [ ] Regime-switching models (HMM + MCMC)
- [ ] Multi-asset portfolio simulation
- [ ] News sentiment integration
- [ ] Real-time alerting system
