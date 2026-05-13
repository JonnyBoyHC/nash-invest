"""Nash Invest — Configuration via environment variables."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    database_url: str = "sqlite:///./nash_invest.db"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    watchlist: List[str] = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    default_forecast_days: int = 7
    mcmc_samples: int = 2000
    mcmc_tune: int = 1000

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
