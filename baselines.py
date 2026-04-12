# baselines.py — Baseline Forecasting Models
# Implements Naive, Random Walk, ARIMA, Buy-and-Hold, and SMA baselines
# for rigorous comparison against ensemble models.

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


# ============================================================================
# ABSTRACT BASE — Unified interface for all models
# ============================================================================

class BaselineModel(ABC):
    """
    Abstract base class for all forecasting models (baselines and advanced).
    Every model must implement fit(), predict(), and name().
    """

    @abstractmethod
    def fit(self, prices: np.ndarray) -> None:
        """Fit the model on historical price data."""
        pass

    @abstractmethod
    def predict(self, forecast_days: int) -> dict:
        """
        Generate forecast.

        Returns:
            {
                'forecast': np.ndarray of predicted prices,
                'end_price': float (final predicted price),
                'method': str,
            }
        """
        pass

    @abstractmethod
    def model_name(self) -> str:
        """Return human-readable model name."""
        pass


# ============================================================================
# NAIVE FORECAST — Last known price repeated
# ============================================================================

class NaiveForecast(BaselineModel):
    """
    Naive baseline: predicts the last known close price for all future days.
    This is the simplest possible baseline — any model that cannot beat this
    has no predictive value.

    ŷ_{t+h} = y_t  ∀ h ∈ [1, H]
    """

    def __init__(self):
        self.last_price = None

    def fit(self, prices: np.ndarray) -> None:
        self.last_price = float(prices[-1])

    def predict(self, forecast_days: int) -> dict:
        forecast = np.full(forecast_days, self.last_price)
        return {
            'forecast': forecast,
            'end_price': self.last_price,
            'method': 'Naive (Last Close)',
        }

    def model_name(self) -> str:
        return 'Naive'


# ============================================================================
# RANDOM WALK — Last close + cumulative noise
# ============================================================================

class RandomWalkForecast(BaselineModel):
    """
    Random Walk baseline: models price as a random walk with drift.
    Uses historical mean and std of log returns.

    ln(S_{t+1}) = ln(S_t) + μΔt + σ√Δt · Z,  Z ~ N(0,1)

    This is the theoretical null hypothesis for market efficiency.
    """

    def __init__(self, n_simulations: int = 1000):
        self.last_price = None
        self.mu = None
        self.sigma = None
        self.n_simulations = n_simulations

    def fit(self, prices: np.ndarray) -> None:
        prices = np.asarray(prices, dtype=float)
        self.last_price = prices[-1]
        log_returns = np.diff(np.log(prices))
        self.mu = float(np.mean(log_returns))
        self.sigma = float(np.std(log_returns))

    def predict(self, forecast_days: int) -> dict:
        dt = 1.0
        Z = np.random.normal(size=(forecast_days, self.n_simulations))
        log_returns = self.mu * dt + self.sigma * np.sqrt(dt) * Z
        log_paths = np.cumsum(log_returns, axis=0)
        price_paths = self.last_price * np.exp(log_paths)

        median_forecast = np.median(price_paths, axis=1)
        return {
            'forecast': median_forecast,
            'end_price': float(median_forecast[-1]),
            'p5': float(np.percentile(price_paths[-1], 5)),
            'p95': float(np.percentile(price_paths[-1], 95)),
            'method': f'Random Walk (μ={self.mu:.6f}, σ={self.sigma:.4f})',
        }

    def model_name(self) -> str:
        return 'Random Walk'


# ============================================================================
# ARIMA FORECAST
# ============================================================================

class ARIMAForecast(BaselineModel):
    """
    ARIMA(p,d,q) baseline using statsmodels.
    Auto-selects order by minimizing AIC over a grid search.

    Model: Φ(B)(1-B)^d y_t = Θ(B) ε_t

    Default grid: p ∈ {1..5}, d ∈ {0,1}, q ∈ {0,1}
    """

    def __init__(self, max_p: int = 5, max_d: int = 1, max_q: int = 1):
        self.max_p = max_p
        self.max_d = max_d
        self.max_q = max_q
        self.best_order = None
        self.fitted_model = None
        self.prices = None

    def fit(self, prices: np.ndarray) -> None:
        from statsmodels.tsa.arima.model import ARIMA
        import warnings

        self.prices = np.asarray(prices, dtype=float)
        best_aic = float('inf')
        best_order = (1, 1, 0)
        best_model = None

        for p in range(1, self.max_p + 1):
            for d in range(self.max_d + 1):
                for q in range(self.max_q + 1):
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            model = ARIMA(self.prices, order=(p, d, q))
                            fitted = model.fit()
                            if fitted.aic < best_aic:
                                best_aic = fitted.aic
                                best_order = (p, d, q)
                                best_model = fitted
                    except Exception:
                        continue

        self.best_order = best_order
        self.fitted_model = best_model

        # If all ARIMA fits failed, fall back to (1,1,0)
        if self.fitted_model is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(self.prices, order=(1, 1, 0))
                    self.fitted_model = model.fit()
                    self.best_order = (1, 1, 0)
            except Exception:
                pass

    def predict(self, forecast_days: int) -> dict:
        if self.fitted_model is None:
            # Fallback to naive if ARIMA completely fails
            last = float(self.prices[-1]) if self.prices is not None else 0.0
            return {
                'forecast': np.full(forecast_days, last),
                'end_price': last,
                'method': 'ARIMA (failed → Naive fallback)',
            }

        try:
            fc = self.fitted_model.forecast(steps=forecast_days)
            forecast = np.asarray(fc, dtype=float)
            return {
                'forecast': forecast,
                'end_price': float(forecast[-1]),
                'method': f'ARIMA{self.best_order}',
                'aic': float(self.fitted_model.aic),
            }
        except Exception:
            last = float(self.prices[-1]) if self.prices is not None else 0.0
            return {
                'forecast': np.full(forecast_days, last),
                'end_price': last,
                'method': f'ARIMA{self.best_order} (forecast failed)',
            }

    def model_name(self) -> str:
        return f'ARIMA{self.best_order}' if self.best_order else 'ARIMA'


# ============================================================================
# BUY AND HOLD — Benchmark return
# ============================================================================

class BuyAndHoldBaseline(BaselineModel):
    """
    Buy-and-Hold baseline for signal evaluation.
    Always predicts "up" — equivalent to buying and holding the asset.

    Used to evaluate whether active signal generation adds value.
    The forecast is a simple linear extrapolation of the historical
    annualized return.
    """

    def __init__(self):
        self.last_price = None
        self.daily_return = None

    def fit(self, prices: np.ndarray) -> None:
        prices = np.asarray(prices, dtype=float)
        self.last_price = prices[-1]
        total_return = prices[-1] / prices[0]
        n_days = len(prices)
        self.daily_return = float(total_return ** (1 / n_days) - 1)

    def predict(self, forecast_days: int) -> dict:
        forecast = np.array([
            self.last_price * (1 + self.daily_return) ** (d + 1)
            for d in range(forecast_days)
        ])
        return {
            'forecast': forecast,
            'end_price': float(forecast[-1]),
            'daily_return': self.daily_return,
            'annual_return': float((1 + self.daily_return) ** 252 - 1),
            'method': 'Buy & Hold',
        }

    def model_name(self) -> str:
        return 'Buy & Hold'


# ============================================================================
# SMA CROSSOVER — Signal-based baseline
# ============================================================================

class SMAForecast(BaselineModel):
    """
    Simple Moving Average Crossover signal generator.
    Uses 50-day and 200-day SMA for golden/death cross signals.

    Signal: +1 (buy) when SMA50 > SMA200, -1 (sell) when SMA50 < SMA200.
    Price forecast: extrapolates the direction of the shorter SMA.
    """

    def __init__(self, short_window: int = 50, long_window: int = 200):
        self.short_window = short_window
        self.long_window = long_window
        self.prices = None
        self.signal = 0
        self.trend_slope = 0.0

    def fit(self, prices: np.ndarray) -> None:
        self.prices = np.asarray(prices, dtype=float)
        n = len(self.prices)

        if n < self.long_window:
            # Not enough data for long SMA, use short only
            if n >= self.short_window:
                sma_short = np.mean(self.prices[-self.short_window:])
                self.signal = 1 if self.prices[-1] > sma_short else -1
            else:
                self.signal = 0
            self.trend_slope = float(np.mean(np.diff(self.prices[-min(20, n):])))
            return

        sma_short = np.mean(self.prices[-self.short_window:])
        sma_long = np.mean(self.prices[-self.long_window:])
        self.signal = 1 if sma_short > sma_long else -1

        # Trend slope from short SMA series
        sma_series = pd.Series(self.prices).rolling(self.short_window).mean().dropna().values
        if len(sma_series) >= 5:
            self.trend_slope = float(np.mean(np.diff(sma_series[-5:])))
        else:
            self.trend_slope = 0.0

    def predict(self, forecast_days: int) -> dict:
        last_price = float(self.prices[-1])
        forecast = np.array([
            last_price + self.trend_slope * (d + 1)
            for d in range(forecast_days)
        ])
        return {
            'forecast': forecast,
            'end_price': float(forecast[-1]),
            'signal': self.signal,
            'method': f'SMA({self.short_window}/{self.long_window}) Crossover',
        }

    def model_name(self) -> str:
        return f'SMA {self.short_window}/{self.long_window}'


# ============================================================================
# MODEL WRAPPER — Wraps existing project models into BaselineModel interface
# ============================================================================

class EnsembleModelWrapper(BaselineModel):
    """
    Wraps the project's EnsembleForecaster into the BaselineModel interface
    for use in the backtesting engine.
    """

    def __init__(self, run_lstm: bool = True, epochs: int = 25,
                 sentiment_score: float = None):
        self.run_lstm = run_lstm
        self.epochs = epochs
        self.sentiment_score = sentiment_score
        self.ticker = None
        self._result = None

    def fit(self, prices: np.ndarray) -> None:
        # Ensemble expects a ticker and downloads its own data.
        # For backtesting, we pass pre-downloaded data instead.
        # This is handled in the backtesting engine directly.
        pass

    def predict(self, forecast_days: int) -> dict:
        # Backtesting engine calls forecast() directly
        return {'forecast': np.array([]), 'end_price': 0.0, 'method': 'Ensemble'}

    def model_name(self) -> str:
        return 'Ensemble'


class LSTMModelWrapper(BaselineModel):
    """Wraps dl_forecaster into BaselineModel interface."""

    def __init__(self, epochs: int = 30):
        self.epochs = epochs

    def fit(self, prices: np.ndarray) -> None:
        pass

    def predict(self, forecast_days: int) -> dict:
        return {'forecast': np.array([]), 'end_price': 0.0, 'method': 'LSTM'}

    def model_name(self) -> str:
        return 'LSTM'


class MonteCarloModelWrapper(BaselineModel):
    """Wraps analytics_engine.run_monte_carlo_stock into BaselineModel interface."""

    def __init__(self):
        self.last_price = None
        self.mu = None
        self.sigma = None

    def fit(self, prices: np.ndarray) -> None:
        prices = np.asarray(prices, dtype=float)
        self.last_price = float(prices[-1])
        returns = pd.Series(prices).pct_change().dropna()
        self.mu = float(returns.mean() * 252)
        self.sigma = float(returns.std() * np.sqrt(252))

    def predict(self, forecast_days: int) -> dict:
        from analytics_engine import run_monte_carlo_stock
        result = run_monte_carlo_stock(self.last_price, self.mu, self.sigma,
                                       forecast_days)
        return {
            'forecast': result['percentiles'][50][1:],  # skip current price
            'end_price': result['median_price'],
            'p5': result['p5'],
            'p95': result['p95'],
            'prob_up': result['prob_up'],
            'method': 'Monte Carlo GBM',
        }

    def model_name(self) -> str:
        return 'Monte Carlo'


class ExpSmoothingModelWrapper(BaselineModel):
    """Wraps analytics_engine.exponential_smoothing_forecast."""

    def __init__(self):
        self.prices = None

    def fit(self, prices: np.ndarray) -> None:
        self.prices = np.asarray(prices, dtype=float)

    def predict(self, forecast_days: int) -> dict:
        from analytics_engine import exponential_smoothing_forecast
        result = exponential_smoothing_forecast(self.prices, forecast_days)
        if not result.get('success'):
            return {
                'forecast': np.full(forecast_days, float(self.prices[-1])),
                'end_price': float(self.prices[-1]),
                'method': 'Exp Smoothing (failed)',
            }
        return {
            'forecast': result['forecast'],
            'end_price': float(result['forecast'][-1]),
            'rmse': result['rmse'],
            'mape': result['mape'],
            'method': 'Double Exponential Smoothing (Holt)',
        }

    def model_name(self) -> str:
        return 'Exp Smoothing'
