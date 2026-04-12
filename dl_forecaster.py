# dl_forecaster.py — Deep Learning Stock Price Forecaster (LSTM via PyTorch)
# Trains a stacked LSTM on historical price data and forecasts future prices.

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# LSTM MODEL (PyTorch)
# ============================================================================
class LSTMModel(nn.Module):
    """Stacked LSTM for time series forecasting."""
    
    def __init__(self, input_size=1, hidden1=128, hidden2=64, dense_size=32):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size, hidden1, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(hidden2, dense_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(dense_size, 1)

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out[:, -1, :])  # Take last timestep
        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        return out


# ============================================================================
# DATA UTILITIES
# ============================================================================
def _download_data(ticker: str, years: int = 2) -> pd.DataFrame:
    """Download historical close prices for a ticker."""
    end = datetime.today()
    start = end - timedelta(days=years * 365)
    try:
        df = yf.download(ticker, start=start, end=end, progress=False)
        if df.empty:
            return pd.DataFrame()
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df[["Close"]].dropna()
    except Exception:
        return pd.DataFrame()


def _prepare_data(series: np.ndarray, lookback: int = 60):
    """Create sliding window sequences for LSTM training."""
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(series.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled)):
        X.append(scaled[i - lookback:i, 0])
        y.append(scaled[i, 0])

    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y, scaler, scaled


# ============================================================================
# TRAINING & FORECASTING
# ============================================================================
def train_and_forecast(ticker: str, forecast_days: int = 30, lookback: int = 60, epochs: int = 50) -> dict:
    """
    End-to-end: download data, train LSTM, forecast future prices.
    
    Returns dict with:
        - 'ticker': str
        - 'historical': DataFrame (date, close)
        - 'forecast': DataFrame (date, predicted, upper, lower)
        - 'metrics': dict (mse, rmse on test set)
        - 'model_summary': str
        - 'success': bool
        - 'error': str (if failed)
    """
    try:
        # 1. Download data
        df = _download_data(ticker)
        if df.empty or len(df) < lookback + 50:
            return {
                'success': False,
                'error': f"Insufficient data for {ticker}. Need at least {lookback + 50} days, got {len(df)}.",
                'ticker': ticker
            }

        close_prices = df["Close"].values.astype(float)

        # 2. Prepare data
        X, y, scaler, scaled = _prepare_data(close_prices, lookback)

        # Train/test split (80/20)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        # Convert to PyTorch tensors
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        X_train_t = torch.FloatTensor(X_train).to(device)
        y_train_t = torch.FloatTensor(y_train).to(device)
        X_test_t = torch.FloatTensor(X_test).to(device)
        y_test_t = torch.FloatTensor(y_test).to(device)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

        # 3. Build & train model
        model = LSTMModel().to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()

        # 4. Evaluate on test set
        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_t).cpu().numpy().flatten()
        
        y_test_np = y_test.flatten()
        
        # Inverse transform for metrics
        y_test_actual = scaler.inverse_transform(y_test_np.reshape(-1, 1)).flatten()
        y_pred_actual = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()

        mse = float(np.mean((y_test_actual - y_pred_actual) ** 2))
        rmse = float(np.sqrt(mse))
        mape = float(np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100)

        # 5. Forecast future prices using MC Dropout for uncertainty quantification
        # Instead of constant ±1.96×RMSE bands, we run N forward passes with
        # dropout enabled (MC Dropout) to estimate predictive variance.
        # This produces confidence intervals that naturally widen with horizon.
        
        MC_SAMPLES = 50  # Number of Monte Carlo forward passes
        mc_forecasts = []
        
        for mc_run in range(MC_SAMPLES):
            last_sequence = torch.FloatTensor(scaled[-lookback:].reshape(1, lookback, 1)).to(device)
            current_seq = last_sequence.clone()
            run_predictions = []
            
            # Enable dropout for MC sampling (key insight: model.train() activates dropout)
            model.train()
            with torch.no_grad():
                for _ in range(forecast_days):
                    pred = model(current_seq).cpu().numpy()[0, 0]
                    run_predictions.append(pred)
                    current_seq = torch.roll(current_seq, -1, dims=1)
                    current_seq[0, -1, 0] = float(pred)
            
            run_prices = scaler.inverse_transform(
                np.array(run_predictions).reshape(-1, 1)
            ).flatten()
            mc_forecasts.append(run_prices)
        
        mc_array = np.array(mc_forecasts)  # Shape: (MC_SAMPLES, forecast_days)
        
        # MC Dropout estimates
        predictions = mc_array.mean(axis=0)      # Predictive mean
        pred_std = mc_array.std(axis=0)           # Predictive std (widens with horizon)
        upper = predictions + 1.96 * pred_std     # 95% CI upper
        lower = predictions - 1.96 * pred_std     # 95% CI lower
        
        # Switch back to eval for any subsequent use
        model.eval()

        # Build forecast DataFrame
        last_date = df.index[-1]
        forecast_dates = pd.bdate_range(start=last_date + timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({
            "Date": forecast_dates,
            "Predicted": predictions,
            "Upper (95%)": upper,
            "Lower (95%)": lower,
        })

        # Historical DataFrame
        hist_df = pd.DataFrame({
            "Date": df.index,
            "Close": close_prices,
        })

        # Model summary
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        summary_lines = [
            f"Model: LSTMModel (PyTorch)",
            f"Device: {device}",
            f"Architecture:",
            f"  LSTM Layer 1: input=1 → hidden=128 (return sequences)",
            f"  Dropout: 0.2",
            f"  LSTM Layer 2: input=128 → hidden=64",
            f"  Dropout: 0.2",
            f"  Dense: 64 → 32 (ReLU)",
            f"  Output: 32 → 1",
            f"",
            f"Total Parameters: {total_params:,}",
            f"Trainable Parameters: {trainable_params:,}",
            f"Lookback Window: {lookback} days",
        ]

        return {
            'success': True,
            'ticker': ticker,
            'historical': hist_df,
            'forecast': forecast_df,
            'metrics': {
                'mse': round(mse, 4),
                'rmse': round(rmse, 4),
                'mape': round(mape, 2),
                'test_samples': len(y_test),
                'train_samples': len(y_train),
                'epochs': epochs,
                'lookback': lookback,
            },
            'model_summary': "\n".join(summary_lines),
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'ticker': ticker,
        }


def get_portfolio_tickers(portfolio_df: pd.DataFrame) -> list:
    """Extract valid public equity tickers from the portfolio."""
    if portfolio_df.empty or "Ticker" not in portfolio_df.columns:
        return []
    tickers = portfolio_df[
        (portfolio_df["Side"] == "Asset") &
        (portfolio_df["Ticker"].str.strip() != "")
    ]["Ticker"].unique()
    return [t.strip().upper() for t in tickers if t and str(t).lower() not in ("nan", "none", "")]
