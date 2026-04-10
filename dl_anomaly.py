# dl_anomaly.py — Deep Learning Anomaly Detection (Autoencoder via PyTorch)
# Trains an autoencoder on historical return patterns to detect anomalous holdings.

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ============================================================================
# AUTOENCODER MODEL (PyTorch)
# ============================================================================
class Autoencoder(nn.Module):
    """Symmetric autoencoder for anomaly detection."""
    
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# ============================================================================
# ANOMALY DETECTOR
# ============================================================================
class PortfolioAnomalyDetector:
    """
    Autoencoder-based anomaly detection for portfolio holdings.
    
    Learns the 'normal' statistical profile of each asset's returns,
    then flags assets whose current behavior deviates significantly
    from historical norms (high reconstruction error).
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.threshold = None

    def _compute_features(self, returns: pd.Series) -> dict:
        """Compute a feature vector from a return series."""
        if returns.empty or len(returns) < 20:
            return None

        return {
            'mean_return': float(returns.mean()),
            'volatility': float(returns.std()),
            'skewness': float(returns.skew()),
            'kurtosis': float(returns.kurtosis()),
            'max_drawdown': float((returns.cumsum() - returns.cumsum().cummax()).min()),
            'var_95': float(np.percentile(returns, 5)),
            'positive_ratio': float((returns > 0).mean()),
            'autocorrelation': float(returns.autocorr(lag=1)) if len(returns) > 1 else 0.0,
            'max_gain': float(returns.max()),
            'max_loss': float(returns.min()),
        }

    def analyze_portfolio(self, portfolio_df: pd.DataFrame, epochs: int = 100) -> dict:
        """
        Full pipeline: download data, extract features, train autoencoder,
        detect anomalies.
        """
        try:
            from sklearn.preprocessing import StandardScaler

            # Extract tickers
            if portfolio_df.empty or "Ticker" not in portfolio_df.columns:
                return {'success': False, 'error': 'No portfolio data.'}

            assets = portfolio_df[
                (portfolio_df["Side"] == "Asset") &
                (portfolio_df["Ticker"].str.strip() != "")
            ].copy()

            tickers = [t.strip().upper() for t in assets["Ticker"].unique()
                        if t and str(t).lower() not in ("nan", "none", "")]

            if len(tickers) < 3:
                return {
                    'success': False,
                    'error': f"Need at least 3 tickers with valid data. Found: {len(tickers)}."
                }

            # Download 1 year of data
            end = datetime.today()
            start = end - timedelta(days=365)

            try:
                data = yf.download(tickers, start=start, end=end, progress=False)["Close"]
            except Exception as e:
                return {'success': False, 'error': f"Failed to download data: {e}"}

            if data.empty:
                return {'success': False, 'error': 'No price data returned.'}

            if isinstance(data, pd.Series):
                data = pd.DataFrame({tickers[0]: data})

            returns = data.pct_change().dropna()

            # Build feature matrix using rolling windows
            window_size = 30
            valid_tickers = [t for t in tickers if t in returns.columns]
            
            if len(valid_tickers) < 3:
                return {
                    'success': False,
                    'error': f"Only {len(valid_tickers)} tickers had valid data. Need at least 3."
                }

            all_features = []
            all_labels = []

            for t in valid_tickers:
                series = returns[t].dropna()
                for i in range(window_size, len(series)):
                    window = series.iloc[i - window_size:i]
                    feats = self._compute_features(window)
                    if feats:
                        all_features.append(list(feats.values()))
                        all_labels.append(t)

            if len(all_features) < 20:
                return {'success': False, 'error': 'Insufficient data points for training.'}

            feature_names = list(self._compute_features(returns[valid_tickers[0]].dropna()[:30]).keys())
            X = np.array(all_features)

            # Scale features
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Build and train autoencoder
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            input_dim = X_scaled.shape[1]
            
            self.model = Autoencoder(input_dim).to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

            X_tensor = torch.FloatTensor(X_scaled).to(device)
            dataset = TensorDataset(X_tensor, X_tensor)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            self.model.train()
            for epoch in range(epochs):
                for batch_X, batch_target in loader:
                    optimizer.zero_grad()
                    output = self.model(batch_X)
                    loss = criterion(output, batch_target)
                    loss.backward()
                    optimizer.step()

            # Compute reconstruction errors
            self.model.eval()
            with torch.no_grad():
                X_reconstructed = self.model(X_tensor).cpu().numpy()
            
            errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            self.threshold = float(np.mean(errors) + 2 * np.std(errors))

            # Compute current score for each ticker
            results = []
            for t in valid_tickers:
                series = returns[t].dropna()
                if len(series) < window_size:
                    continue

                recent = series.iloc[-window_size:]
                feats = self._compute_features(recent)
                if not feats:
                    continue

                feat_vec = np.array([list(feats.values())])
                feat_scaled = self.scaler.transform(feat_vec)
                feat_scaled = np.nan_to_num(feat_scaled, nan=0.0, posinf=0.0, neginf=0.0)

                feat_tensor = torch.FloatTensor(feat_scaled).to(device)
                with torch.no_grad():
                    reconstructed = self.model(feat_tensor).cpu().numpy()
                
                error = float(np.mean((feat_scaled - reconstructed) ** 2))
                is_anomaly = error > self.threshold

                if error > self.threshold * 2:
                    risk = "🔴 Critical"
                elif error > self.threshold:
                    risk = "🟡 Elevated"
                elif error > self.threshold * 0.5:
                    risk = "🟠 Watch"
                else:
                    risk = "🟢 Normal"

                name_matches = assets[assets["Ticker"].str.strip().str.upper() == t]
                name = name_matches["Name"].values[0] if not name_matches.empty else t

                results.append({
                    'Ticker': t,
                    'Name': name,
                    'Reconstruction Error': round(error, 6),
                    'Threshold': round(self.threshold, 6),
                    'Anomaly': '⚠️ YES' if is_anomaly else '✅ No',
                    'Risk Level': risk,
                    'Volatility (30d)': round(float(feats['volatility']) * 100, 2),
                    'Skewness': round(float(feats['skewness']), 3),
                })

            results_df = pd.DataFrame(results).sort_values(
                'Reconstruction Error', ascending=False
            )

            # Model summary
            total_params = sum(p.numel() for p in self.model.parameters())
            summary_lines = [
                "Model: Autoencoder (PyTorch)",
                f"Device: {device}",
                f"Architecture:",
                f"  Encoder: {input_dim} → 64 (ReLU) → 32 (ReLU) → 16 (ReLU)",
                f"  Decoder: 16 → 32 (ReLU) → 64 (ReLU) → {input_dim}",
                f"  Dropout: 0.2 (encoder & decoder)",
                f"",
                f"Total Parameters: {total_params:,}",
                f"Training Samples: {len(X)}",
                f"Feature Dimensions: {input_dim}",
                f"Anomaly Threshold: {self.threshold:.6f}",
            ]

            return {
                'success': True,
                'results': results_df,
                'threshold': self.threshold,
                'feature_names': feature_names,
                'model_summary': "\n".join(summary_lines),
                'training_samples': len(X),
                'tickers_analyzed': len(valid_tickers),
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
            }
