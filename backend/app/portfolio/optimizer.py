"""Portfolio analytics: correlation, efficient frontier, optimization.

Mean-variance optimization (Markowitz) with:
- Correlation / covariance matrix
- Efficient frontier curve
- Maximum Sharpe ratio portfolio
- Minimum variance portfolio
- Equal-weight portfolio as baseline
"""

import numpy as np
from typing import List, Dict, Optional


RISK_FREE_RATE = 0.045  # annualized


def compute_returns_matrix(
    price_data: Dict[str, np.ndarray],
    min_periods: int = 20,
) -> Dict[str, np.ndarray]:
    """
    Align daily log returns across tickers.
    
    Args:
        price_data: {ticker: close_prices_array} (chronological)
        min_periods: minimum observations required per ticker
    
    Returns:
        {ticker: returns_array} for the common date range
    """
    returns = {}
    for ticker, prices in price_data.items():
        if len(prices) >= min_periods + 1:
            returns[ticker] = np.diff(np.log(prices))
    return returns


def covariance_matrix(
    returns: Dict[str, np.ndarray],
    annualize: bool = True,
) -> dict:
    """
    Compute covariance and correlation matrices.
    
    Returns dict with matrix data ready for JSON/Plotly.
    """
    tickers = sorted(returns.keys())
    n = len(tickers)
    
    # Align to minimum length
    min_len = min(len(r) for r in returns.values())
    aligned = np.column_stack([returns[t][-min_len:] for t in tickers])
    
    # Covariance matrix (daily)
    cov_daily = np.cov(aligned, rowvar=False)
    
    # Correlation matrix
    std = np.sqrt(np.diag(cov_daily))
    corr = cov_daily / np.outer(std, std)
    
    # Annualize
    cov_annual = cov_daily * 252 if annualize else cov_daily
    
    # Annualized stats per ticker
    annual_return = np.mean(aligned, axis=0) * 252
    annual_vol = np.std(aligned, axis=0, ddof=1) * np.sqrt(252)
    sharpe = (annual_return - RISK_FREE_RATE) / annual_vol
    
    return {
        "tickers": tickers,
        "n_observations": min_len,
        "stats": [
            {
                "ticker": tickers[i],
                "annual_return_pct": round(float(annual_return[i]) * 100, 2),
                "annual_vol_pct": round(float(annual_vol[i]) * 100, 2),
                "sharpe": round(float(sharpe[i]), 4),
            }
            for i in range(n)
        ],
        "correlation": {
            "labels": tickers,
            "matrix": [[round(float(corr[i][j]), 4) for j in range(n)] for i in range(n)],
        },
        "covariance": {
            "labels": tickers,
            "matrix": [[round(float(cov_annual[i][j]), 8) for j in range(n)] for i in range(n)],
        },
    }


def efficient_frontier(
    returns: Dict[str, np.ndarray],
    n_portfolios: int = 100,
) -> dict:
    """
    Generate the efficient frontier via Monte Carlo simulation.
    
    Returns the frontier curve and optimal portfolios.
    """
    tickers = sorted(returns.keys())
    n_assets = len(tickers)
    
    if n_assets < 2:
        return {"error": "Need at least 2 assets for portfolio optimization"}
    
    # Align returns
    min_len = min(len(r) for r in returns.values())
    aligned = np.column_stack([returns[t][-min_len:] for t in tickers])
    
    mean_returns = np.mean(aligned, axis=0)
    cov_matrix = np.cov(aligned, rowvar=False)
    
    # Monte Carlo portfolios
    np.random.seed(42)
    results = np.zeros((3, n_portfolios))
    weights_record = np.zeros((n_portfolios, n_assets))
    
    for i in range(n_portfolios):
        # Random weights
        w = np.random.random(n_assets)
        w /= w.sum()
        weights_record[i] = w
        
        # Portfolio return and volatility
        port_return = np.sum(mean_returns * w) * 252
        port_vol = np.sqrt(w.T @ cov_matrix @ w) * np.sqrt(252)
        port_sharpe = (port_return - RISK_FREE_RATE) / port_vol if port_vol > 0 else 0
        
        results[0, i] = port_return
        results[1, i] = port_vol
        results[2, i] = port_sharpe
    
    # Find optimal portfolios
    max_sharpe_idx = np.argmax(results[2])
    min_vol_idx = np.argmin(results[1])
    
    max_sharpe_weights = weights_record[max_sharpe_idx]
    min_vol_weights = weights_record[min_vol_idx]
    equal_weights = np.ones(n_assets) / n_assets
    
    # Equal weight portfolio stats
    eq_ret = np.sum(mean_returns * equal_weights) * 252
    eq_vol = np.sqrt(equal_weights.T @ cov_matrix @ equal_weights) * np.sqrt(252)
    eq_sharpe = (eq_ret - RISK_FREE_RATE) / eq_vol if eq_vol > 0 else 0
    
    # Generate frontier curve (sorted by volatility)
    sort_idx = np.argsort(results[1])
    frontier_vol = results[1, sort_idx]
    frontier_ret = results[0, sort_idx]
    frontier_sharpe = results[2, sort_idx]
    
    # Select Pareto-optimal points (highest return for each vol level)
    pareto_vol = []
    pareto_ret = []
    max_ret = -np.inf
    for v, r in zip(frontier_vol, frontier_ret):
        if r > max_ret:
            pareto_vol.append(float(v * 100))
            pareto_ret.append(float(r * 100))
            max_ret = r
    
    def _format_weights(w):
        return {tickers[i]: round(float(w[i]) * 100, 1) for i in range(n_assets)}
    
    return {
        "tickers": tickers,
        "n_observations": min_len,
        "efficient_frontier": {
            "volatility_pct": pareto_vol,
            "return_pct": pareto_ret,
        },
        "optimal_portfolios": {
            "max_sharpe": {
                "return_pct": round(float(results[0, max_sharpe_idx]) * 100, 2),
                "volatility_pct": round(float(results[1, max_sharpe_idx]) * 100, 2),
                "sharpe": round(float(results[2, max_sharpe_idx]), 4),
                "weights": _format_weights(max_sharpe_weights),
            },
            "min_volatility": {
                "return_pct": round(float(results[0, min_vol_idx]) * 100, 2),
                "volatility_pct": round(float(results[1, min_vol_idx]) * 100, 2),
                "sharpe": round(float(results[2, min_vol_idx]), 4),
                "weights": _format_weights(min_vol_weights),
            },
            "equal_weight": {
                "return_pct": round(float(eq_ret) * 100, 2),
                "volatility_pct": round(float(eq_vol) * 100, 2),
                "sharpe": round(float(eq_sharpe), 4),
                "weights": _format_weights(equal_weights),
            },
        },
    }
