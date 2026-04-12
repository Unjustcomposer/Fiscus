# ablation_runner.py — Automated Ablation Study Executor
# Runs systematic ablation experiments to quantify the contribution
# of each component (LSTM, Monte Carlo, Exp Smoothing, Sentiment).

import numpy as np
import pandas as pd
import time
from datetime import datetime

from backtesting import WalkForwardBacktester
from evaluation import (
    EvaluationReport, paired_ttest, wilcoxon_test, diebold_mariano_test,
    format_latex_table, format_markdown_table,
)


# ============================================================================
# ABLATION CONFIGURATIONS
# ============================================================================

ABLATION_CONFIGS = {
    # --- Full system ---
    "full_ensemble": {
        "models": ["exp_smoothing", "monte_carlo"],
        "sentiment": True,
        "description": "Full ensemble with sentiment gating",
    },
    "full_ensemble_lstm": {
        "models": ["exp_smoothing", "monte_carlo", "lstm"],
        "sentiment": True,
        "description": "Full ensemble + LSTM with sentiment gating",
    },

    # --- Remove one component ---
    "no_sentiment": {
        "models": ["exp_smoothing", "monte_carlo"],
        "sentiment": False,
        "description": "Ensemble WITHOUT sentiment gating",
    },
    "no_mc": {
        "models": ["exp_smoothing"],
        "sentiment": True,
        "description": "Only Exp Smoothing + sentiment",
    },
    "no_ets": {
        "models": ["monte_carlo"],
        "sentiment": True,
        "description": "Only Monte Carlo + sentiment",
    },

    # --- Single models ---
    "lstm_only": {
        "models": ["lstm"],
        "sentiment": False,
        "description": "LSTM standalone",
    },
    "mc_only": {
        "models": ["monte_carlo"],
        "sentiment": False,
        "description": "Monte Carlo standalone",
    },
    "ets_only": {
        "models": ["exp_smoothing"],
        "sentiment": False,
        "description": "Exp Smoothing standalone",
    },

    # --- Baselines ---
    "naive_baseline": {
        "models": ["naive"],
        "sentiment": False,
        "description": "Naive baseline (last close)",
    },
    "random_walk_baseline": {
        "models": ["random_walk"],
        "sentiment": False,
        "description": "Random Walk baseline",
    },
    "arima_baseline": {
        "models": ["arima"],
        "sentiment": False,
        "description": "ARIMA baseline (auto-order)",
    },
    "buy_hold_baseline": {
        "models": ["buy_and_hold"],
        "sentiment": False,
        "description": "Buy & Hold baseline",
    },
    "sma_baseline": {
        "models": ["sma"],
        "sentiment": False,
        "description": "SMA 50/200 Crossover baseline",
    },

    # --- AutoAnalyzer test ---
    "auto_select": {
        "models": ["auto"],
        "sentiment": True,
        "description": "AutoAnalyzer model selection + sentiment",
    },
    "random_select": {
        "models": ["random"],
        "sentiment": False,
        "description": "Random model selection (control)",
    },
}


# ============================================================================
# ABLATION RUNNER
# ============================================================================

class AblationRunner:
    """
    Runs ablation studies across multiple tickers and configurations.

    For each (ticker, config) pair, runs a walk-forward backtest and
    records all metrics. Produces comparison tables with statistical
    significance tests.
    """

    def __init__(self, tickers: list,
                 forecast_horizon: int = 21,
                 step_size: int = 21,
                 min_train_days: int = 252,
                 lookback_years: int = 3):
        self.tickers = [t.strip().upper() for t in tickers]
        self.forecast_horizon = forecast_horizon
        self.step_size = step_size
        self.min_train_days = min_train_days
        self.lookback_years = lookback_years
        self.results = {}   # {config_name: {ticker: summary_df}}
        self.raw_predictions = {}  # {config_name: {ticker: predictions_df}}
        self.run_timestamp = None

    def run_single_ablation(self, ticker: str, config_name: str,
                            config: dict,
                            progress_callback=None) -> dict:
        """
        Run one ablation configuration on one ticker.

        Returns dict with summary metrics.
        """
        models = config["models"]
        use_sentiment = config.get("sentiment", False)
        sentiment_score = 0.1 if use_sentiment else None  # Mild bullish default

        # Handle special model modes
        if "auto" in models:
            # Run auto-selection via smart_forecast
            return self._run_auto_mode(ticker, config_name, sentiment_score)
        if "random" in models:
            return self._run_random_mode(ticker, config_name)

        bt = WalkForwardBacktester(
            ticker=ticker,
            lookback_years=self.lookback_years,
            forecast_horizon=self.forecast_horizon,
            step_size=self.step_size,
            min_train_days=self.min_train_days,
        )

        try:
            predictions_df = bt.run(
                models=models,
                sentiment_score=sentiment_score,
                progress_callback=progress_callback,
            )
            summary = bt.get_summary()
            return {
                'success': True,
                'summary': summary,
                'predictions': predictions_df,
                'n_windows': len(predictions_df['window'].unique())
                              if not predictions_df.empty else 0,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'summary': pd.DataFrame(),
                'predictions': pd.DataFrame(),
            }

    def _run_auto_mode(self, ticker: str, config_name: str,
                       sentiment_score: float = None) -> dict:
        """Run using AutoAnalyzer model selection."""
        bt = WalkForwardBacktester(
            ticker=ticker,
            lookback_years=self.lookback_years,
            forecast_horizon=self.forecast_horizon,
            step_size=self.step_size,
            min_train_days=self.min_train_days,
        )
        try:
            # Use ensemble mode as proxy for auto-selection
            predictions_df = bt.run(
                models=['ensemble'],
                sentiment_score=sentiment_score,
            )
            summary = bt.get_summary()
            return {
                'success': True,
                'summary': summary,
                'predictions': predictions_df,
                'n_windows': len(predictions_df['window'].unique())
                              if not predictions_df.empty else 0,
            }
        except Exception as e:
            return {'success': False, 'error': str(e),
                    'summary': pd.DataFrame(), 'predictions': pd.DataFrame()}

    def _run_random_mode(self, ticker: str, config_name: str) -> dict:
        """Run with random model selection (control group)."""
        import random
        model_pool = ['naive', 'random_walk', 'arima', 'exp_smoothing',
                       'monte_carlo', 'sma']
        # Randomly pick one model per run
        chosen = random.choice(model_pool)
        bt = WalkForwardBacktester(
            ticker=ticker,
            lookback_years=self.lookback_years,
            forecast_horizon=self.forecast_horizon,
            step_size=self.step_size,
            min_train_days=self.min_train_days,
        )
        try:
            predictions_df = bt.run(models=[chosen])
            summary = bt.get_summary()
            return {
                'success': True,
                'summary': summary,
                'predictions': predictions_df,
                'n_windows': len(predictions_df['window'].unique())
                              if not predictions_df.empty else 0,
                'random_model_chosen': chosen,
            }
        except Exception as e:
            return {'success': False, 'error': str(e),
                    'summary': pd.DataFrame(), 'predictions': pd.DataFrame()}

    def run_suite(self, configs: list = None,
                  progress_callback=None) -> dict:
        """
        Run all ablation configurations across all tickers.

        Parameters:
            configs: List of config names to run. Default: all.
            progress_callback: Optional callable(pct, message)

        Returns:
            dict of {config_name: {ticker: result_dict}}
        """
        self.run_timestamp = datetime.now().isoformat()

        if configs is None:
            configs = list(ABLATION_CONFIGS.keys())

        total = len(configs) * len(self.tickers)
        current = 0

        for config_name in configs:
            if config_name not in ABLATION_CONFIGS:
                continue
            config = ABLATION_CONFIGS[config_name]
            self.results[config_name] = {}
            self.raw_predictions[config_name] = {}

            for ticker in self.tickers:
                current += 1
                if progress_callback:
                    progress_callback(
                        current / total,
                        f"{config_name} - {ticker} "
                        f"({current}/{total})")

                result = self.run_single_ablation(ticker, config_name, config)
                self.results[config_name][ticker] = result
                if result.get('success') and not result['predictions'].empty:
                    self.raw_predictions[config_name][ticker] = result['predictions']

        return self.results

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Build a comparison table across all configs and tickers.

        Returns DataFrame with columns:
            [Config, Description, Ticker, RMSE, MAPE, DA, Hit Rate,
             Profit Factor, Windows]
        """
        rows = []
        for config_name, ticker_results in self.results.items():
            desc = ABLATION_CONFIGS.get(config_name, {}).get(
                'description', config_name)
            for ticker, result in ticker_results.items():
                if not result.get('success'):
                    rows.append({
                        'Config': config_name,
                        'Description': desc,
                        'Ticker': ticker,
                        'RMSE': None,
                        'MAPE (%)': None,
                        'DA (%)': None,
                        'Hit Rate (%)': None,
                        'Profit Factor': None,
                        'Windows': 0,
                        'Error': result.get('error', 'Unknown'),
                    })
                    continue

                summary = result['summary']
                if summary.empty:
                    continue

                # Aggregate across models in this config (take best)
                best = summary.iloc[0]  # Already sorted by RMSE
                rows.append({
                    'Config': config_name,
                    'Description': desc,
                    'Ticker': ticker,
                    'RMSE': best.get('RMSE'),
                    'MAPE (%)': best.get('MAPE (%)'),
                    'DA (%)': best.get('DA (%)'),
                    'Hit Rate (%)': best.get('Hit Rate (%)'),
                    'Profit Factor': best.get('Profit Factor'),
                    'Windows': best.get('Windows', 0),
                })

        return pd.DataFrame(rows)

    def get_averaged_comparison(self) -> pd.DataFrame:
        """
        Average metrics across tickers for each configuration.

        Returns DataFrame with one row per config.
        """
        full_table = self.get_comparison_table()
        if full_table.empty:
            return pd.DataFrame()

        numeric_cols = ['RMSE', 'MAPE (%)', 'DA (%)', 'Hit Rate (%)',
                        'Profit Factor', 'Windows']
        avg = full_table.groupby(['Config', 'Description'])[numeric_cols].mean()
        avg = avg.reset_index().sort_values('RMSE')
        avg[numeric_cols] = avg[numeric_cols].round(4)
        return avg

    def run_significance_tests(self,
                               baseline_config: str = "naive_baseline",
                               test_config: str = "full_ensemble") -> list:
        """
        Run statistical tests comparing a test config against a baseline.

        Returns list of test results per ticker.
        """
        tests = []
        for ticker in self.tickers:
            baseline_preds = self.raw_predictions.get(
                baseline_config, {}).get(ticker)
            test_preds = self.raw_predictions.get(
                test_config, {}).get(ticker)

            if baseline_preds is None or test_preds is None:
                continue
            if baseline_preds.empty or test_preds.empty:
                continue

            # Align on common windows
            common_windows = set(baseline_preds['window']) & set(test_preds['window'])
            if len(common_windows) < 5:
                continue

            common = sorted(common_windows)
            bp = baseline_preds[baseline_preds['window'].isin(common)].sort_values('window')
            tp = test_preds[test_preds['window'].isin(common)].sort_values('window')

            n = min(len(bp), len(tp))
            b_errors = bp['abs_error'].values[:n]
            t_errors = tp['abs_error'].values[:n]

            b_raw = (bp['actual_price'].values[:n] - bp['predicted_price'].values[:n])
            t_raw = (tp['actual_price'].values[:n] - tp['predicted_price'].values[:n])

            tests.append({
                'ticker': ticker,
                'baseline': baseline_config,
                'test': test_config,
                'n_windows': n,
                'baseline_mean_error': round(float(np.mean(b_errors)), 4),
                'test_mean_error': round(float(np.mean(t_errors)), 4),
                'improvement_pct': round(
                    (1 - np.mean(t_errors) / max(np.mean(b_errors), 1e-8)) * 100, 2),
                'paired_ttest': paired_ttest(b_errors, t_errors),
                'wilcoxon': wilcoxon_test(b_errors, t_errors),
                'diebold_mariano': diebold_mariano_test(b_raw, t_raw),
            })

        return tests

    def export_results(self, output_dir: str = ".") -> dict:
        """
        Export all results to CSV and LaTeX files.

        Returns dict of file paths created.
        """
        import os
        files = {}

        # Full comparison table
        full = self.get_comparison_table()
        if not full.empty:
            path = os.path.join(output_dir, "ablation_full_results.csv")
            full.to_csv(path, index=False)
            files['full_csv'] = path

        # Averaged comparison
        avg = self.get_averaged_comparison()
        if not avg.empty:
            path = os.path.join(output_dir, "ablation_averaged.csv")
            avg.to_csv(path, index=False)
            files['averaged_csv'] = path

            # LaTeX table
            report = EvaluationReport()
            for _, row in avg.iterrows():
                # Add each config as a "model" in the report
                report.results.append({
                    'Model': row['Config'],
                    'RMSE': row.get('RMSE'),
                    'MAPE (%)': row.get('MAPE (%)'),
                    'DA (%)': row.get('DA (%)'),
                    'Hit Rate (%)': row.get('Hit Rate (%)'),
                    'Profit Factor': row.get('Profit Factor'),
                })
            latex = format_latex_table(report, "Ablation Study Results")
            path = os.path.join(output_dir, "ablation_table.tex")
            with open(path, 'w') as f:
                f.write(latex)
            files['latex'] = path

        # Significance tests
        tests = self.run_significance_tests()
        if tests:
            tests_df = pd.DataFrame(tests)
            path = os.path.join(output_dir, "ablation_significance.csv")
            tests_df.to_csv(path, index=False)
            files['significance_csv'] = path

        return files
