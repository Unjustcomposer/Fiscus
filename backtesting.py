# backtesting.py — Walk-Forward Backtesting Engine
# Implements expanding-window evaluation over historical data,
# logging predictions, actuals, signals, and returns per timestamp.
# Uses concurrent.futures for parallel model evaluation within each window.

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from evaluation import (
    rmse, mape, mae, directional_accuracy, hit_rate, profit_factor,
    sharpe_ratio, calmar_ratio, max_drawdown, information_ratio,
    EvaluationReport, paired_ttest, wilcoxon_test, diebold_mariano_test,
)
from baselines import (
    NaiveForecast, RandomWalkForecast, ARIMAForecast,
    BuyAndHoldBaseline, SMAForecast,
    MonteCarloModelWrapper, ExpSmoothingModelWrapper,
)


# ============================================================================
# WALK-FORWARD BACKTESTER
# ============================================================================

class WalkForwardBacktester:
    """
    Walk-Forward Backtesting Engine with expanding-window evaluation.

    Algorithm:
        for t in range(min_train_days, len(data) - forecast_horizon, step_size):
            train = data[:t]
            test  = data[t : t + forecast_horizon]
            for model in models:
                pred = model.fit(train).predict(forecast_horizon)
                log(timestamp=t, model, predicted=pred[-1], actual=test[-1])

    Parameters:
        ticker: Stock ticker symbol
        lookback_years: Years of historical data to download (default 3)
        forecast_horizon: Number of days to forecast ahead (default 21 = 1 month)
        step_size: Days between evaluation windows (default 21 = monthly)
        min_train_days: Minimum training window size (default 252 = 1 year)
    """

    AVAILABLE_MODELS = [
        'naive', 'random_walk', 'arima', 'buy_and_hold', 'sma',
        'monte_carlo', 'exp_smoothing', 'lstm', 'ensemble',
    ]

    def __init__(self, ticker: str, lookback_years: int = 3,
                 forecast_horizon: int = 21, step_size: int = 21,
                 min_train_days: int = 252):
        self.ticker = ticker.strip().upper()
        self.lookback_years = lookback_years
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        self.min_train_days = min_train_days

        self.prices = None
        self.dates = None
        self.results = []       # Per-window results
        self.all_predictions = []  # Flat log of every prediction
        self._downloaded = False

    def _download_data(self) -> bool:
        """Download historical price data."""
        end = datetime.today()
        start = end - timedelta(days=self.lookback_years * 365)
        try:
            df = yf.download(self.ticker, start=start, end=end, progress=False)
            if df.empty:
                return False
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self.prices = df["Close"].dropna().values.astype(float)
            self.dates = df.index.tolist()
            self._downloaded = True
            return True
        except Exception:
            return False

    def _create_model(self, model_name: str):
        """Instantiate a model by name."""
        models = {
            'naive': NaiveForecast,
            'random_walk': RandomWalkForecast,
            'arima': ARIMAForecast,
            'buy_and_hold': BuyAndHoldBaseline,
            'sma': SMAForecast,
            'monte_carlo': MonteCarloModelWrapper,
            'exp_smoothing': ExpSmoothingModelWrapper,
        }
        cls = models.get(model_name)
        if cls:
            return cls()
        return None

    def _run_lstm_window(self, train_prices: np.ndarray,
                         forecast_days: int) -> dict:
        """Train and forecast with LSTM on a single window."""
        try:
            from dl_forecaster import LSTMModel, _prepare_data
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset

            lookback = 60
            if len(train_prices) < lookback + 30:
                return None

            X, y, scaler, scaled = _prepare_data(train_prices, lookback)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            X_t = torch.FloatTensor(X).to(device)
            y_t = torch.FloatTensor(y).to(device)

            dataset = TensorDataset(X_t, y_t)
            loader = DataLoader(dataset, batch_size=32, shuffle=True)

            model = LSTMModel().to(device)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            model.train()
            for epoch in range(25):  # Lighter training for backtesting speed
                for bx, by in loader:
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()

            # Forecast
            last_seq = torch.FloatTensor(
                scaled[-lookback:].reshape(1, lookback, 1)).to(device)
            preds_scaled = []
            current = last_seq.clone()
            model.eval()
            with torch.no_grad():
                for _ in range(forecast_days):
                    p = model(current).cpu().numpy()[0, 0]
                    preds_scaled.append(p)
                    current = torch.roll(current, -1, dims=1)
                    current[0, -1, 0] = float(p)

            forecast = scaler.inverse_transform(
                np.array(preds_scaled).reshape(-1, 1)
            ).flatten()

            return {
                'forecast': forecast,
                'end_price': float(forecast[-1]),
                'method': 'LSTM (Backtest)',
            }
        except Exception:
            return None

    def _run_ensemble_window(self, train_prices: np.ndarray,
                             forecast_days: int,
                             sentiment_score: float = None) -> dict:
        """Run ensemble forecast on a single window using raw price data."""
        try:
            returns = pd.Series(train_prices).pct_change().dropna()
            current_price = float(train_prices[-1])
            ann_mu = float(returns.mean() * 252)
            ann_sigma = float(returns.std() * np.sqrt(252))

            model_forecasts = {}
            model_weights = {}

            # Exponential Smoothing
            try:
                from analytics_engine import exponential_smoothing_forecast
                ets = exponential_smoothing_forecast(train_prices, forecast_days)
                if ets.get('success'):
                    model_forecasts['exp_smoothing'] = float(ets['forecast'][-1])
                    model_weights['exp_smoothing'] = max(
                        0.1, 1.0 / max(ets['mape'], 0.5))
            except Exception:
                pass

            # Monte Carlo
            try:
                from analytics_engine import run_monte_carlo_stock
                mc = run_monte_carlo_stock(
                    current_price, ann_mu, ann_sigma, forecast_days)
                model_forecasts['monte_carlo'] = mc['median_price']
                model_weights['monte_carlo'] = 1.0
            except Exception:
                pass

            if not model_forecasts:
                return None

            # Sentiment gating
            if sentiment_score is not None:
                gates = {
                    'exp_smoothing': 1.0 + 0.5 * sentiment_score,
                    'monte_carlo': 1.0 - 0.4 * sentiment_score,
                }
                model_weights = {
                    k: w * gates.get(k, 1.0) for k, w in model_weights.items()
                }

            total_w = sum(model_weights.values())
            normalized = {k: w / total_w for k, w in model_weights.items()}

            consensus = sum(
                model_forecasts[k] * normalized[k] for k in model_forecasts)

            return {
                'forecast': np.array([consensus]),
                'end_price': float(consensus),
                'method': 'Ensemble (Backtest)',
            }
        except Exception:
            return None

    def run(self, models: list = None,
            sentiment_score: float = None,
            progress_callback=None) -> pd.DataFrame:
        """
        Execute walk-forward backtesting.

        Parameters:
            models: List of model names to evaluate.
                    Default: all baselines + exp_smoothing + monte_carlo
            sentiment_score: Optional sentiment for ensemble gating
            progress_callback: Optional callable(progress_pct, msg)

        Returns:
            DataFrame with columns:
                [window, date, model, predicted_price, actual_price,
                 return_pred, return_actual, signal, correct_direction]
        """
        if not self._downloaded:
            if not self._download_data():
                raise ValueError(f"Failed to download data for {self.ticker}")

        if models is None:
            models = ['naive', 'random_walk', 'arima', 'buy_and_hold',
                       'sma', 'monte_carlo', 'exp_smoothing']

        n = len(self.prices)
        total_required = self.min_train_days + self.forecast_horizon

        if n < total_required:
            raise ValueError(
                f"Insufficient data: {n} days available, "
                f"need at least {total_required} "
                f"(min_train={self.min_train_days} + "
                f"horizon={self.forecast_horizon})")

        # Calculate windows
        windows = list(range(
            self.min_train_days,
            n - self.forecast_horizon + 1,
            self.step_size
        ))
        total_steps = len(windows) * len(models)
        current_step = 0

        self.results = []
        self.all_predictions = []

        # Determine parallelism: use threads for I/O-bound models
        max_workers = min(len(models), 4)

        for w_idx, train_end in enumerate(windows):
            train_prices = self.prices[:train_end]
            test_start = train_end
            test_end = min(train_end + self.forecast_horizon, n)
            test_prices = self.prices[test_start:test_end]
            actual_end_price = float(test_prices[-1])
            train_end_price = float(train_prices[-1])

            actual_return = (actual_end_price - train_end_price) / train_end_price
            window_date = (self.dates[train_end]
                           if train_end < len(self.dates) else None)

            # --- Parallel model evaluation within this window ---
            def _eval_model(model_name):
                """Evaluate a single model; returns (model_name, result, elapsed)."""
                t0 = time.time()
                try:
                    if model_name == 'lstm':
                        res = self._run_lstm_window(
                            train_prices, self.forecast_horizon)
                    elif model_name == 'ensemble':
                        res = self._run_ensemble_window(
                            train_prices, self.forecast_horizon,
                            sentiment_score)
                    else:
                        mdl = self._create_model(model_name)
                        if mdl:
                            mdl.fit(train_prices)
                            res = mdl.predict(self.forecast_horizon)
                        else:
                            res = None
                except Exception:
                    res = None
                return (model_name, res, time.time() - t0)

            # Submit all models for this window concurrently
            futures = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for model_name in models:
                    futures[executor.submit(_eval_model, model_name)] = model_name

                for future in as_completed(futures):
                    current_step += 1
                    m_name = futures[future]
                    if progress_callback:
                        pct = current_step / total_steps
                        progress_callback(
                            pct,
                            f"Window {w_idx+1}/{len(windows)} — {m_name}")

                    try:
                        model_name, pred_result, elapsed = future.result()
                    except Exception:
                        continue

                    if pred_result is None:
                        continue

                    predicted_end = float(pred_result['end_price'])
                    predicted_return = (
                        (predicted_end - train_end_price) / train_end_price
                    )
                    signal = 1 if predicted_return > 0 else -1 if predicted_return < 0 else 0
                    correct_direction = (
                        (signal > 0 and actual_return > 0) or
                        (signal < 0 and actual_return < 0)
                    )

                    record = {
                        'window': w_idx,
                        'date': window_date,
                        'train_end_idx': train_end,
                        'model': model_name,
                        'predicted_price': predicted_end,
                        'actual_price': actual_end_price,
                        'predicted_return': round(predicted_return * 100, 4),
                        'actual_return': round(actual_return * 100, 4),
                        'signal': signal,
                        'correct_direction': correct_direction,
                        'abs_error': abs(predicted_end - actual_end_price),
                        'pct_error': abs(
                            (predicted_end - actual_end_price) / actual_end_price
                        ) * 100,
                        'elapsed_sec': round(elapsed, 3),
                    }
                    self.all_predictions.append(record)

        return pd.DataFrame(self.all_predictions)

    def get_summary(self) -> pd.DataFrame:
        """
        Aggregate metrics per model across all windows.

        Returns DataFrame with one row per model:
            [Model, RMSE, MAPE, MAE, DA, Hit Rate, Profit Factor,
             Avg Elapsed, Windows]
        """
        if not self.all_predictions:
            return pd.DataFrame()

        df = pd.DataFrame(self.all_predictions)
        summaries = []

        for model_name in df['model'].unique():
            mdf = df[df['model'] == model_name]
            actuals = mdf['actual_price'].values
            predicted = mdf['predicted_price'].values

            _signals = mdf['signal'].values
            _returns = mdf['actual_return'].values / 100  # Convert back

            da = float(mdf['correct_direction'].mean() * 100)
            hr = hit_rate(_signals, _returns)
            pf = profit_factor(_signals, _returns)

            summaries.append({
                'Model': model_name,
                'RMSE': round(rmse(actuals, predicted), 4),
                'MAPE (%)': round(mape(actuals, predicted), 2),
                'MAE': round(mae(actuals, predicted), 4),
                'DA (%)': round(da, 2),
                'Hit Rate (%)': round(hr, 2),
                'Profit Factor': round(pf, 4),
                'Avg Time (s)': round(mdf['elapsed_sec'].mean(), 3),
                'Windows': len(mdf),
            })

        summary_df = pd.DataFrame(summaries)
        return summary_df.sort_values('RMSE')

    def get_significance_tests(self) -> list:
        """
        Run pairwise statistical significance tests between all models.

        Returns list of dicts with test results.
        """
        if not self.all_predictions:
            return []

        df = pd.DataFrame(self.all_predictions)
        models = df['model'].unique().tolist()
        tests = []

        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                ma, mb = models[i], models[j]
                dfa = df[df['model'] == ma].sort_values('window')
                dfb = df[df['model'] == mb].sort_values('window')

                # Align on common windows
                common = set(dfa['window']) & set(dfb['window'])
                if len(common) < 10:
                    continue

                common = sorted(common)
                a_errors = dfa[dfa['window'].isin(common)]['abs_error'].values
                b_errors = dfb[dfb['window'].isin(common)]['abs_error'].values

                n = min(len(a_errors), len(b_errors))
                a_errors, b_errors = a_errors[:n], b_errors[:n]

                raw_a = (dfa[dfa['window'].isin(common)]['actual_price'].values[:n] -
                         dfa[dfa['window'].isin(common)]['predicted_price'].values[:n])
                raw_b = (dfb[dfb['window'].isin(common)]['actual_price'].values[:n] -
                         dfb[dfb['window'].isin(common)]['predicted_price'].values[:n])

                test = {
                    'model_a': ma,
                    'model_b': mb,
                    'n_windows': n,
                    'mean_error_a': round(float(np.mean(a_errors)), 4),
                    'mean_error_b': round(float(np.mean(b_errors)), 4),
                    'paired_ttest': paired_ttest(a_errors, b_errors),
                    'wilcoxon': wilcoxon_test(a_errors, b_errors),
                    'diebold_mariano': diebold_mariano_test(raw_a, raw_b),
                }
                tests.append(test)

        return tests

    def get_equity_curve(self) -> pd.DataFrame:
        """
        Compute cumulative return curves for each model based on signals.

        Returns DataFrame with date index and one column per model.
        """
        if not self.all_predictions:
            return pd.DataFrame()

        df = pd.DataFrame(self.all_predictions)
        pivot = df.pivot_table(
            index='window', columns='model', values='actual_return')

        # For each model, the strategy return is:
        # signal * actual_return (only capture return when signal is correct dir)
        signals_pivot = df.pivot_table(
            index='window', columns='model', values='signal')

        strategy_returns = signals_pivot * pivot / 100  # Convert pct to decimal
        cumulative = (1 + strategy_returns).cumprod()
        cumulative.index.name = 'Window'

        return cumulative
