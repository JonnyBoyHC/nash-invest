"""GARCH(1,1) volatility model for conditional heteroskedasticity.

Models volatility clustering: high-vol periods follow high-vol,
low-vol periods follow low-vol. The standard model:

    r_t = μ + ε_t
    ε_t = σ_t * z_t,  z_t ~ N(0,1) or StudentT(ν)
    σ²_t = ω + α * ε²_{t-1} + β * σ²_{t-1}

Provides volatility forecasts that are more responsive to recent
market conditions than simple historical volatility.
"""

import numpy as np
from typing import Optional
from dataclasses import dataclass


@dataclass
class GARCHResult:
    """Fitted GARCH model with forecasts."""
    omega: float
    alpha: float
    beta: float
    persistence: float  # alpha + beta (should be < 1 for stationarity)
    unconditional_vol: float  # sqrt(ω / (1 - α - β)) annualized
    long_run_vol: float  # unconditional daily vol
    fitted_vol: np.ndarray  # in-sample conditional vol
    forecast_vol: np.ndarray  # out-of-sample vol forecast
    convergence: bool
    iterations: int
    log_likelihood: float


def fit_garch(
    returns: np.ndarray,
    max_iter: int = 1000,
    tol: float = 1e-6,
    lr: float = 0.001,
) -> GARCHResult:
    """
    Fit GARCH(1,1) via gradient descent on negative log-likelihood.

    Parameters are constrained: ω>0, α≥0, β≥0, α+β<1.

    Args:
        returns: array of log returns
        max_iter: maximum gradient descent iterations
        tol: convergence tolerance on parameter change
        lr: learning rate for gradient descent

    Returns:
        GARCHResult with fitted parameters and volatility forecasts
    """
    n = len(returns)
    if n < 30:
        raise ValueError(f"Need at least 30 returns, got {n}")

    # De-mean returns
    mu = np.mean(returns)
    eps = returns - mu

    # Initial variance estimate
    init_var = np.var(returns)

    # Initialize parameters (transformed to unconstrained space for optimization)
    # Use softplus/logistic transforms to enforce constraints
    def to_params(raw):
        """Unconstrained → constrained parameters."""
        w = np.exp(raw[0])  # ω > 0
        a = 1 / (1 + np.exp(-raw[1]))  # α ∈ (0, 1)
        # β: we need 0 < β < 1-α. Parameterize as fraction of remaining
        b = (1 - a) / (1 + np.exp(-raw[2]))
        return w, a, b

    def neg_loglik(raw):
        """Negative log-likelihood for GARCH(1,1)."""
        w, a, b = to_params(raw)
        if w <= 0 or a < 0 or b < 0 or a + b >= 1:
            return 1e10

        sigma2 = np.zeros(n)
        sigma2[0] = init_var

        for t in range(1, n):
            sigma2[t] = w + a * eps[t-1]**2 + b * sigma2[t-1]

        if np.any(sigma2 <= 0):
            return 1e10

        # Gaussian log-likelihood
        ll = -0.5 * np.sum(np.log(2 * np.pi * sigma2) + eps**2 / sigma2)
        return -ll

    # Initialize: ω = sample_var * 0.1, α = 0.1, β = 0.8
    raw = np.array([np.log(init_var * 0.1), 0.0, 0.0])  # log(ω), logit(α), logit(β/(1-α))

    prev_loss = neg_loglik(raw)
    best_raw = raw.copy()
    best_loss = prev_loss

    for iteration in range(max_iter):
        # Numerical gradient
        grad = np.zeros(3)
        h = 1e-6
        for i in range(3):
            raw_plus = raw.copy()
            raw_plus[i] += h
            grad[i] = (neg_loglik(raw_plus) - prev_loss) / h

        # Gradient descent with line search
        step = lr
        for _ in range(10):
            new_raw = raw - step * grad
            new_loss = neg_loglik(new_raw)
            if new_loss < prev_loss:
                break
            step *= 0.5
        else:
            break  # can't improve

        # Update
        raw = new_raw
        prev_loss = new_loss

        if new_loss < best_loss:
            best_loss = new_loss
            best_raw = raw.copy()

        # Check convergence
        if np.max(np.abs(grad * lr)) < tol:
            break

    # Final parameters
    omega, alpha, beta = to_params(best_raw)

    # Compute fitted volatility
    sigma2 = np.zeros(n)
    sigma2[0] = init_var
    for t in range(1, n):
        sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]

    fitted_vol = np.sqrt(sigma2)

    # Unconditional (long-run) volatility
    persistence = alpha + beta
    if persistence < 1:
        long_run_var = omega / (1 - persistence)
        long_run_vol = np.sqrt(long_run_var)
    else:
        long_run_vol = float('nan')

    # Forecast: for horizon h, E[σ²_{T+h}] = LR_var + (α+β)^{h-1} * (σ²_T - LR_var)
    horizon = 21  # 1 month forecasting
    forecast_sigma2 = np.zeros(horizon)
    last_sigma2 = sigma2[-1]
    last_eps2 = eps[-1]**2

    if persistence < 1:
        lr_var = long_run_var ** 2
        for h in range(horizon):
            if h == 0:
                forecast_sigma2[h] = omega + alpha * last_eps2 + beta * last_sigma2
            else:
                forecast_sigma2[h] = lr_var + persistence * (forecast_sigma2[h-1] - lr_var)
    else:
        forecast_sigma2[:] = sigma2[-1]

    forecast_vol = np.sqrt(forecast_sigma2)

    return GARCHResult(
        omega=float(omega),
        alpha=float(alpha),
        beta=float(beta),
        persistence=float(persistence),
        unconditional_vol=float(long_run_vol * np.sqrt(252)),
        long_run_vol=float(long_run_vol),
        fitted_vol=fitted_vol,
        forecast_vol=forecast_vol,
        convergence=iteration < max_iter - 1,
        iterations=iteration + 1,
        log_likelihood=float(-best_loss),
    )


def volatility_forecast_ci(
    returns: np.ndarray,
    prices: np.ndarray,
    horizon_days: int = 7,
    ci_width: float = 0.90,
) -> dict:
    """
    Combined GARCH volatility + price forecast with volatility-based CI.

    Uses GARCH to estimate time-varying volatility, then constructs
    forecast intervals that expand when volatility is high.
    """
    garch = fit_garch(returns)

    mu = np.mean(returns)
    last_price = prices[-1]
    current_vol = float(garch.fitted_vol[-1])

    # Forecast volatility path
    forecast_vols = []
    for h in range(min(horizon_days, len(garch.forecast_vol))):
        forecast_vols.append(float(garch.forecast_vol[h]))

    # Pad with long-run vol if horizon > 21
    while len(forecast_vols) < horizon_days:
        forecast_vols.append(float(garch.long_run_vol))

    # Price forecast with GARCH-based CIs
    from scipy import stats as sp_stats
    alpha = (1 - ci_width) / 2
    z_lower = sp_stats.norm.ppf(alpha)
    z_upper = sp_stats.norm.ppf(1 - alpha)

    forecasts = []
    cum_return_mean = 0
    cum_var = 0

    for day in range(horizon_days):
        daily_vol = forecast_vols[day]
        cum_return_mean += mu
        cum_var += daily_vol ** 2

        cum_vol = np.sqrt(cum_var)

        pred_mean = float(last_price * np.exp(cum_return_mean))
        ci_lower = float(last_price * np.exp(cum_return_mean + z_lower * cum_vol))
        ci_upper = float(last_price * np.exp(cum_return_mean + z_upper * cum_vol))

        forecasts.append({
            "day": day + 1,
            "pred_mean": round(pred_mean, 2),
            "ci_lower": round(ci_lower, 2),
            "ci_upper": round(ci_upper, 2),
            "daily_vol_pct": round(daily_vol * 100, 4),
            "cum_vol_pct": round(cum_vol * 100, 4),
        })

    return {
        "garch_params": {
            "omega": garch.omega,
            "alpha": garch.alpha,
            "beta": garch.beta,
            "persistence": garch.persistence,
            "unconditional_vol_annual_pct": round(garch.unconditional_vol * 100, 2),
            "current_daily_vol_pct": round(current_vol * 100, 4),
            "convergence": garch.convergence,
            "iterations": garch.iterations,
        },
        "forecasts": forecasts,
        "model": "garch_normal",
        "ci_width": ci_width,
    }
