"""Abstract base class for all predictors."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Optional, Dict, Any

import numpy as np


@dataclass
class Forecast:
    """Output of any predictor: a distribution, not a point."""
    ticker: str
    target_date: date
    pred_mean: float
    pred_std: float
    pred_5th: float
    pred_95th: float
    samples: Optional[np.ndarray] = None  # raw MCMC samples (optional, can be large)


@dataclass
class ForecastSet:
    """Multiple forecasts for one ticker across time horizon."""
    ticker: str
    current_price: float
    forecasts: list[Forecast]
    model_name: str
    model_version: str
    generated_at: date


class BasePredictor(ABC):
    """Every predictor implements `forecast` to produce a distribution."""

    @abstractmethod
    def forecast(
        self,
        ticker: str,
        prices: np.ndarray,
        horizon_days: int = 7,
    ) -> ForecastSet:
        ...
