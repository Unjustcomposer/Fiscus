import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

def get_risk_metrics(df: pd.DataFrame, risk_free_rate: float = 0.04):
    """
    Downloads 1y daily data for public equity tickers, computes Volatility, Beta (vs SPY), 
    Sharpe Ratio, and historical Value at Risk (95%).
    Returns (metrics_df, correlation_matrix)
    """
    if df.empty or "Ticker" not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
        
    public_assets = df[(df["Side"] == "Asset") & (df["Ticker"].str.strip() != "")]
    tickers = [t.strip().upper() for t in public_assets["Ticker"].unique() if t and str(t).lower() != "nan"]
    if not tickers:
        return pd.DataFrame(), pd.DataFrame()
        
    # Download 1y data
    end = datetime.today()
    start = end - timedelta(days=365)
    
    # Also fetch SPY for market return
    all_tickers = tickers + ["SPY"]
    try:
        data = yf.download(all_tickers, start=start, end=end, progress=False)["Close"]
    except Exception:
        return pd.DataFrame(), pd.DataFrame()
        
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
        
    # Handle single ticker case
    if isinstance(data, pd.Series):
        if len(all_tickers) == 1:
            data = pd.DataFrame({all_tickers[0]: data})
        else:
            return pd.DataFrame(), pd.DataFrame()
            
    # Calculate daily returns
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
            var_95 = np.percentile(tkr_ret, 5) * 100 # percentage format
            
            metrics.append({
                "Ticker": t,
                "Expected Return (%)": round(mu * 100, 2),
                "Volatility (%)": round(vol * 100, 2),
                "Sharpe Ratio": round(sharpe, 2),
                "Beta": round(beta, 2),
                "1-Day VaR 95%": round(var_95, 2)
            })
            
    metrics_df = pd.DataFrame(metrics)
    corr_matrix = returns[tickers].corr().round(2)
    return metrics_df, corr_matrix

def run_monte_carlo(initial_value: float, mu: float, sigma: float, years: int = 10, sim_count: int = 1000):
    """
    Geometric Brownian Motion simulator.
    Returns a dataframe of pathways (Years x Sims)
    """
    # Annual steps
    dt = 1
    steps = years
    
    # Generate random normal matrix
    np.random.seed(42)
    Z = np.random.normal(size=(steps, sim_count))
    
    # Calculate daily drifts and shocks
    daily_returns = np.exp((mu - (sigma ** 2) / 2) * dt + sigma * np.sqrt(dt) * Z)
    
    # Create price paths starting at initial_value
    price_paths = np.zeros_like(daily_returns)
    price_paths[0] = initial_value * daily_returns[0]
    
    for t in range(1, steps):
        price_paths[t] = price_paths[t-1] * daily_returns[t]
        
    # Append initial value row for T=0
    initial_row = np.full((1, sim_count), initial_value)
    paths = np.vstack([initial_row, price_paths])
    
    df = pd.DataFrame(paths)
    df.index.name = "Year"
    return df
