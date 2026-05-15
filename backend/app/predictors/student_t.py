"""Student-T likelihood for Bayesian price prediction.

Replaces the Normal likelihood with a Student-T distribution,
which has fatter tails (controlled by degrees of freedom nu).

This is especially important for high-volatility assets like TSLA
where extreme returns occur more often than a Normal predicts.
"""

import numpy as np
import pymc as pm
import arviz as az
from typing import Optional
from dataclasses import dataclass
from datetime import date, timedelta

from app.predictors.base import Forecast


@dataclass
class StudentTForecast:
    """Student-T forecast container (extends base Forecast with nu)."""
    ticker: str
    current_price: float
    target_date: date
    horizon_days: int
    pred_mean: float
    pred_median: float
    ci_lower: float
    ci_upper: float
    nu: Optional[float] = None  # degrees of freedom
    sigma: Optional[float] = None  # scale parameter


def predict_student_t(
    prices: np.ndarray,
    horizon_days: int = 7,
    samples: int = 500,
    tune: int = 500,
    chains: int = 2,
    ci_width: float = 0.90,
    nu_prior_mean: float = 4.0,
    nu_prior_sd: float = 5.0,
    random_seed: Optional[int] = None,
) -> dict:
    """
    Bayesian price forecast using Student-T returns.

    Model:
        r_t ~ StudentT(nu, mu, sigma)
        nu ~ Gamma(2, 0.5)  # weakly informative, mean ~4
        mu ~ Normal(0, 0.01)
        sigma ~ HalfNormal(0.05)

    Args:
        prices: 1-D array of historical close prices (chronological)
        horizon_days: forecast horizon in trading days
        samples: MCMC samples per chain
        tune: tuning steps per chain
        chains: number of MCMC chains
        ci_width: credible interval width (e.g. 0.90 → 90% CI)
        nu_prior_mean: prior mean for degrees of freedom
        nu_prior_sd: prior sd for degrees of freedom
        random_seed: RNG seed for reproducibility

    Returns:
        dict with forecasts, diagnostics, and model info
    """
    log_returns = np.diff(np.log(prices))

    if len(log_returns) < 20:
        raise ValueError(f"Need at least 20 returns, got {len(log_returns)}")

    alpha = (1 - ci_width) / 2  # tail probability on each side

    with pm.Model() as model:
        # Degrees of freedom: Gamma prior, mean ~nu_prior_mean, sd ~nu_prior_sd
        # Shape = mean²/sd², Rate = mean/sd²
        gamma_alpha = (nu_prior_mean ** 2) / (nu_prior_sd ** 2)
        gamma_beta = nu_prior_mean / (nu_prior_sd ** 2)
        nu = pm.Gamma("nu", alpha=gamma_alpha, beta=gamma_beta)

        # Location and scale
        mu = pm.Normal("mu", mu=0, sigma=0.01)
        sigma = pm.HalfNormal("sigma", sigma=0.05)

        # Likelihood
        pm.StudentT("returns", nu=nu, mu=mu, sigma=sigma, observed=log_returns)

        # Sample
        idata = pm.sample(
            draws=samples,
            tune=tune,
            chains=chains,
            random_seed=random_seed,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    # Extract posterior
    posterior = idata.posterior.stack(draws=("chain", "draw"))
    mu_samples = posterior["mu"].values
    sigma_samples = posterior["sigma"].values
    nu_samples = posterior["nu"].values

    # Parameter summaries
    mu_mean = float(np.mean(mu_samples))
    sigma_mean = float(np.mean(sigma_samples))
    nu_mean = float(np.mean(nu_samples))

    # Generate forecasts via simulation from posterior
    last_price = prices[-1]
    n_samples = len(mu_samples)

    # Compound return over horizon: draw Student-T(mean over draws)
    forecasts = []

    for day in range(1, horizon_days + 1):
        # For each posterior draw, simulate one forward path for 'day' steps
        # Use the posterior mean parameters for each draw
        horizon_returns = np.zeros(n_samples)
        for i in range(n_samples):
            # Simulate day steps from Student-T with draw i's parameters
            daily_r = np.random.standard_t(df=nu_samples[i], size=day)
            # Scale and shift: actual return = mu + sigma * standard_t
            # (standard_t has variance = nu/(nu-2) for nu > 2, but we just scale)
            horizon_returns[i] = np.sum(mu_samples[i] + sigma_samples[i] * daily_r)

        pred_prices = last_price * np.exp(horizon_returns)

        mean_pred = float(np.mean(pred_prices))
        median_pred = float(np.median(pred_prices))
        lower = float(np.quantile(pred_prices, alpha))
        upper = float(np.quantile(pred_prices, 1 - alpha))

        forecasts.append({
            "day": day,
            "pred_mean": round(mean_pred, 2),
            "pred_median": round(median_pred, 2),
            "ci_lower": round(lower, 2),
            "ci_upper": round(upper, 2),
        })

    # Compute diagnostics
    r_hat = {}
    try:
        summary = az.rhat(idata)
        for var_name in ["mu", "sigma", "nu"]:
            if var_name in summary:
                r_hat[var_name] = round(float(summary[var_name].values), 4)
    except Exception:
        pass

    # Effective sample size
    ess = {}
    try:
        ess_result = az.ess(idata)
        for var_name in ["mu", "sigma", "nu"]:
            if var_name in ess_result:
                ess[var_name] = int(ess_result[var_name].values)
    except Exception:
        pass

    return {
        "ticker": None,  # filled by caller
        "current_price": round(float(last_price), 2),
        "forecasts": forecasts,
        "parameters": {
            "mu": round(mu_mean, 6),
            "mu_pct": round(mu_mean * 100, 4),
            "sigma": round(sigma_mean, 6),
            "sigma_pct": round(sigma_mean * 100, 4),
            "nu": round(nu_mean, 2),
            "nu_lt_5": float(np.mean(nu_samples < 5)),
            "nu_lt_10": float(np.mean(nu_samples < 10)),
        },
        "diagnostics": {
            "r_hat": r_hat,
            "ess": ess,
        },
        "model": "student_t",
        "ci_width": ci_width,
        "n_observations": len(log_returns),
    }


def compare_models(prices: np.ndarray) -> dict:
    """
    Quick comparison between Normal and Student-T likelihoods.

    Returns LOO-IC comparison for model selection.
    """
    log_returns = np.diff(np.log(prices))

    # Normal model
    with pm.Model() as normal_model:
        mu = pm.Normal("mu", mu=0, sigma=0.01)
        sigma = pm.HalfNormal("sigma", sigma=0.05)
        pm.Normal("returns", mu=mu, sigma=sigma, observed=log_returns)
        normal_idata = pm.sample(
            draws=500, tune=500, chains=2,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    # Student-T model
    with pm.Model() as st_model:
        gamma_alpha = 4.0**2 / 5.0**2
        gamma_beta = 4.0 / 5.0**2
        nu = pm.Gamma("nu", alpha=gamma_alpha, beta=gamma_beta)
        mu = pm.Normal("mu", mu=0, sigma=0.01)
        sigma = pm.HalfNormal("sigma", sigma=0.05)
        pm.StudentT("returns", nu=nu, mu=mu, sigma=sigma, observed=log_returns)
        st_idata = pm.sample(
            draws=500, tune=500, chains=2,
            progressbar=False,
            idata_kwargs={"log_likelihood": True},
        )

    # Compare
    try:
        compare_df = az.compare(
            {"normal": normal_idata, "student_t": st_idata},
            ic="loo",
        )
        comparison = compare_df.to_dict("index")
    except Exception:
        comparison = {"error": "LOO comparison failed (likely too few observations)"}

    # Extract parameter summaries
    norm_post = normal_idata.posterior.stack(draws=("chain", "draw"))
    st_post = st_idata.posterior.stack(draws=("chain", "draw"))

    return {
        "normal": {
            "mu": round(float(norm_post["mu"].mean()), 6),
            "sigma": round(float(norm_post["sigma"].mean()), 6),
        },
        "student_t": {
            "mu": round(float(st_post["mu"].mean()), 6),
            "sigma": round(float(st_post["sigma"].mean()), 6),
            "nu": round(float(st_post["nu"].mean()), 2),
        },
        "loo_comparison": comparison,
    }
