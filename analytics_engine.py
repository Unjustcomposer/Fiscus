import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta


def get_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.04):
    """
    Downloads 1y daily data for public equity tickers, computes Volatility, Beta (vs SPY), 
    Sharpe Ratio, Sortino Ratio, CVaR, Max Drawdown, and historical Value at Risk (95%).
    Returns (metrics_df, correlation_matrix)
    """
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
        
    public_assets = df[(df["Side"] == "Asset") & (df["Ticker"].str.strip() != "")]
    tickers = [t.strip().upper() for t in public_assets["Ticker"].unique() if t and str(t).lower() != "nan"]
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
        
    end = datetime.today()
    start = end - timedelta(days=365)
    
    all_tickers = tickers + ["SPY"]
    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False)["Close"]
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
        
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    if isinstance(data, pd.Series):
        if len(all_tickers) == 1:
            data = pd.DataFrame({all_tickers[0]: data})
        else:
            return pd.DataFrame(), pd.DataFrame()
            
    returns = data.pct_change().dropna()
    
    metrics = []
    spy_returns = returns["SPY"] if "SPY" in returns.columns else None
    spy_var = spy_returns.var() if spy_returns is not None else 1.0
    
    for t in tickers:
        if t in returns.columns:
            tkr_ret = returns[t]
            mu = tkr_ret.mean() * 252
            vol = tkr_ret.std() * np.sqrt(252)
            sharpe = (mu - risk_free_rate) / vol if vol > 0 else 0
            
            # Beta
            beta = 1.0
            if spy_returns is not None and spy_var > 0:
                cov = np.cov(tkr_ret, spy_returns)[0][1]
                beta = cov / spy_var
                
            # VaR 95% Historical (1-day)
            var_95 = np.percentile(tkr_ret, 5) * 100

            # CVaR (Expected Shortfall) — average loss beyond VaR
            var_threshold = np.percentile(tkr_ret, 5)
            tail_losses = tkr_ret[tkr_ret <= var_threshold]
            cvar_95 = float(tail_losses.mean() * 100) if len(tail_losses) > 0 else var_95

            # Sortino Ratio — penalizes only downside volatility
            downside = tkr_ret[tkr_ret < 0]
            downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else vol
            sortino = (mu - risk_free_rate) / downside_vol if downside_vol > 0 else 0

            # Maximum Drawdown
            cum_returns = (1 + tkr_ret).cumprod()
            running_max = cum_returns.cummax()
            drawdown = (cum_returns - running_max) / running_max
            max_dd = float(drawdown.min() * 100)

            # Composite Risk Score (0-100, higher = riskier)
            risk_score = _compute_risk_score(vol, beta, max_dd, cvar_95, sharpe)
            
            metrics.append({
                "Ticker": t,
                "Expected Return (%)": round(mu * 100, 2),
                "Volatility (%)": round(vol * 100, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Sortino Ratio": round(sortino, 2),
                "Beta": round(beta, 2),
                "1-Day VaR 95%": round(var_95, 2),
                "CVaR 95%": round(cvar_95, 2),
                "Max Drawdown (%)": round(max_dd, 2),
                "Risk Score": risk_score,
            })
            
    metrics_df = pd.DataFrame(metrics)
    corr_matrix = returns[[t for t in tickers if t in returns.columns]].corr().round(2)
    return metrics_df, corr_matrix


def _compute_risk_score(vol: float, beta: float, max_dd: float, cvar: float, sharpe: float) -> int:
    """
    Composite risk score 0-100.  Higher = riskier.
    Blends volatility, beta, drawdown, tail risk, and return-efficiency.
    """
    # Normalize each component to 0-100 range
    vol_score = min(vol * 100 / 60, 1.0) * 25          # 60% ann vol = max
    beta_score = min(abs(beta) / 2.5, 1.0) * 20        # beta 2.5 = max
    dd_score = min(abs(max_dd) / 50, 1.0) * 25         # 50% drawdown = max
    cvar_score = min(abs(cvar) / 5, 1.0) * 20          # 5% daily CVaR = max
    efficiency = max(0, 1 - max(sharpe, 0) / 2) * 10   # good sharpe = lower risk
    
    score = int(vol_score + beta_score + dd_score + cvar_score + efficiency)
    return max(0, min(100, score))


def run_monte_carlo(initial_value: float, mu: float, sigma: float, years: int = 10, sim_count: int = 1000):
    """
    Geometric Brownian Motion simulator.
    Returns a dataframe of pathways (Years x Sims)
    """
    dt = 1
    steps = years
    
    np.random.seed(42)
    Z = np.random.normal(size=(steps, sim_count))
    
    daily_returns = np.exp((mu - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt) * Z)
    
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = initial_value * daily_returns[0]
    
    for t in range(1, steps):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    initial_row = np.full((1, sim_count), initial_value)
    paths = np.vstack([initial_row, price_paths])
    
    df = pd.DataFrame(paths)
    df.index.name = "Year"
    return df


def run_monte_carlo_stock(current_price: float, mu: float, sigma: float,
                          days: int = 30, sim_count: int = 5000) -> dict:
    """
    Monte Carlo simulation for a single stock price.
    Returns percentile forecasts aligned to business days.
    """
    dt = 1 / 252
    np.random.seed(None)  # random seed each run
    Z = np.random.normal(size=(days, sim_count))
    
    log_returns = (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
    price_paths = current_price * np.exp(np.cumsum(log_returns, axis=0))
    
    # Prepend current price
    initial = np.full((1, sim_count), current_price)
    all_paths = np.vstack([initial, price_paths])
    
    pcts = {}
    for p in [5, 25, 50, 75, 95]:
        pcts[p] = np.percentile(all_paths, p, axis=1)
    
    final = all_paths[-1]
    return {
        'percentiles': pcts,
        'final_values': final,
        'median_price': float(np.median(final)),
        'mean_price': float(np.mean(final)),
        'p5': float(np.percentile(final, 5)),
        'p95': float(np.percentile(final, 95)),
        'prob_up': float((final > current_price).mean() * 100),
        'days': days,
    }


def exponential_smoothing_forecast(prices: np.ndarray, forecast_days: int = 30,
                                    alpha: float = 0.3, beta: float = 0.1) -> dict:
    """
    Double Exponential Smoothing (Holt's method) for price forecasting.
    Uses level + trend components. No external dependencies needed.
    
    Returns dict with forecast values and confidence bands.
    """
    if len(prices) < 10:
        return {'success': False, 'error': 'Need at least 10 data points.'}
    
    prices = prices.astype(float)
    n = len(prices)
    
    # Initialize level and trend
    level = prices[0]
    trend = np.mean(np.diff(prices[:10]))
    
    levels = [level]
    trends = [trend]
    fitted = [level + trend]
    
    # Fit on historical data
    for t in range(1, n):
        new_level = alpha * prices[t] + (1 - alpha) * (level + trend)
        new_trend = beta * (new_level - level) + (1 - beta) * trend
        level = new_level
        trend = new_trend
        levels.append(level)
        trends.append(trend)
        fitted.append(level + trend)
    
    # Calculate residual standard deviation for confidence bands
    fitted_arr = np.array(fitted[:n])
    residuals = prices - fitted_arr
    residual_std = float(np.std(residuals))
    
    # Forecast
    forecasts = []
    for h in range(1, forecast_days + 1):
        forecasts.append(level + h * trend)
    
    forecasts = np.array(forecasts)
    
    # Widening confidence bands (uncertainty grows with horizon)
    band_widths = np.array([residual_std * np.sqrt(h) * 1.96 for h in range(1, forecast_days + 1)])
    upper = forecasts + band_widths
    lower = forecasts - band_widths
    
    # Compute fit quality (RMSE on last 20% of data)
    test_start = int(n * 0.8)
    test_actual = prices[test_start:]
    test_fitted = fitted_arr[test_start:]
    rmse = float(np.sqrt(np.mean((test_actual - test_fitted) ** 2)))
    mape = float(np.mean(np.abs((test_actual - test_fitted) / test_actual)) * 100)
    
    return {
        'success': True,
        'forecast': forecasts,
        'upper': upper,
        'lower': lower,
        'fitted': fitted_arr,
        'rmse': round(rmse, 4),
        'mape': round(mape, 2),
        'residual_std': round(residual_std, 4),
        'method': 'Double Exponential Smoothing (Holt)',
        'params': {'alpha': alpha, 'beta': beta},
    }
