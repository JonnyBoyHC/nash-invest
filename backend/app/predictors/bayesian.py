"""Bayesian price predictor using PyMC MCMC sampling.

Models daily log-returns as normally distributed with a Student-T prior
on the mean (heavy-tailed prior accounts for black swan events).
Generates a full posterior predictive distribution for each horizon day.
"""
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pymc as pm
import arviz as az

from app.predictors.base import BasePredictor, Forecast, ForecastSet

MODEL_NAME = "bayesian-returns"
MODEL_VERSION = "1.0.0"


class BayesianReturnsPredictor(BasePredictor):
    """MCMC model: log-returns ~ Normal(mu, sigma) with weakly informative priors."""

    def __init__(
        self,
        samples: int = 2000,
        tune: int = 1000,
        chains: int = 2,
        random_seed: int = 42,
    ):
        self.samples = samples
        self.tune = tune
        self.chains = chains
        self.random_seed = random_seed
        self._trace = None
        self._mu = None
        self._sigma = None

    def fit(self, prices: np.ndarray) -> None:
        """Fit the MCMC model on log-returns."""
        if len(prices) < 10:
            # Fallback: simple mean/std from limited data
            rets = np.diff(np.log(prices))
            self._mu = np.mean(rets)
            self._sigma = np.std(rets) if np.std(rets) > 0 else 0.01
            return

        log_returns = np.diff(np.log(prices))

        with pm.Model() as model:
            # Weakly informative priors
            mu = pm.Normal("mu", mu=0, sigma=0.1)
            sigma = pm.HalfStudentT("sigma", nu=3, sigma=0.05)

            likelihood = pm.Normal(
                "returns", mu=mu, sigma=sigma, observed=log_returns
            )

            trace = pm.sample(
                draws=self.samples,
                tune=self.tune,
                chains=self.chains,
                random_seed=self.random_seed,
                progressbar=False,
                idata_kwargs={"log_likelihood": True},
            )

        self._trace = trace
        self._mu = float(trace.posterior["mu"].mean())
        self._sigma = float(trace.posterior["sigma"].mean())

    def forecast(
        self,
        ticker: str,
        prices: np.ndarray,
        horizon_days: int = 7,
    ) -> ForecastSet:
        """Generate forecasts for each day in the horizon."""
        self.fit(prices)
        current_price = float(prices[-1])

        if self._trace is not None:
            # Use posterior samples for predictive distribution
            posterior_mu = self._trace.posterior["mu"].values.flatten()
            posterior_sigma = self._trace.posterior["sigma"].values.flatten()
        else:
            # Fallback: use point estimates
            posterior_mu = np.array([self._mu])
            posterior_sigma = np.array([self._sigma])

        forecasts = []
        rng = np.random.default_rng(self.random_seed)

        for day in range(1, horizon_days + 1):
            # Compound returns over `day` days
            # Each day: r_t ~ Normal(mu, sigma)
            # T-day cumulative: sum of T normals ~ Normal(T*mu, sqrt(T)*sigma)
            sampled_mu = rng.choice(posterior_mu, size=min(1000, len(posterior_mu)))
            sampled_sigma = rng.choice(posterior_sigma, size=min(1000, len(posterior_sigma)))

            cumulative_mean = day * sampled_mu
            cumulative_std = np.sqrt(day) * sampled_sigma

            # Predictive price distribution
            pred_prices = current_price * np.exp(
                rng.normal(cumulative_mean, cumulative_std)
            )

            target_dt = date.today() + timedelta(days=day)
            forecasts.append(Forecast(
                ticker=ticker,
                target_date=target_dt,
                pred_mean=float(np.mean(pred_prices)),
                pred_std=float(np.std(pred_prices)),
                pred_5th=float(np.percentile(pred_prices, 5)),
                pred_95th=float(np.percentile(pred_prices, 95)),
                samples=pred_prices,
            ))

        return ForecastSet(
            ticker=ticker,
            current_price=current_price,
            forecasts=forecasts,
            model_name=MODEL_NAME,
            model_version=MODEL_VERSION,
            generated_at=date.today(),
        )
