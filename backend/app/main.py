"""Nash Invest — FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import settings
from app.database import init_db
from app.pipelines.data_fetcher import sync_watchlist
from app.routers import market, predictions, risk, indicators

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("nash-invest")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB, sync watchlist prices. Shutdown: clean up."""
    logger.info("Initializing database...")
    init_db()
    logger.info(f"Syncing watchlist: {settings.watchlist}")
    try:
        results = sync_watchlist(settings.watchlist)
        logger.info(f"Watchlist sync results: {results}")
    except Exception as e:
        logger.warning(f"Initial sync partially failed (this is OK if offline): {e}")
    yield
    logger.info("Shutting down Nash Invest API")


app = FastAPI(
    title="Nash Invest",
    description="Investment research platform with MCMC-driven Bayesian price forecasting.",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — open during development; lock down for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router)
app.include_router(predictions.router)
app.include_router(risk.router)
app.include_router(indicators.router)

# Static dashboard
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/dashboard")
def dashboard():
    """Serve the HTML5 dashboard."""
    return FileResponse("static/dashboard.html")


@app.get("/")
def root():
    """Landing when opening the server in a browser — API has no HTML homepage."""
    return {
        "service": "nash-invest",
        "message": "API is running. Use the paths below or open /docs for Swagger UI.",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/health",
        "dashboard": "/dashboard",
        "market": "/api/market",
        "predictions": "/api/predictions",
        "risk": "/api/risk",
        "indicators": "/api/indicators",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "nash-invest", "version": "0.1.0"}
