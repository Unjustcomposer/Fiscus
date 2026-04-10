# intelligence_engine.py — Unified Intelligence Engine
# Orchestrates LSTM, Monte Carlo, Exponential Smoothing into ensemble forecasts,
# computes composite risk profiles, and generates actionable insights.

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta


# ============================================================================
# AUTO ANALYZER — Smart Model Selection
# ============================================================================
class AutoAnalyzer:
    """
    Evaluates data characteristics for each ticker and recommends the
    best forecasting model based on:
      - Data length (more data → LSTM viable)
      - Volatility regime (high vol → Monte Carlo better)
      - Trend strength (strong trend → Exponential Smoothing)
      - Return distribution (fat tails → Monte Carlo)
    """

    @staticmethod
    def evaluate_ticker(prices: np.ndarray) -> dict:
        """
        Analyze price data and return characteristics + recommended model.
        """
        n = len(prices)
        returns = np.diff(prices) / prices[:-1]
        
        # Data length score
        data_score = min(n / 500, 1.0)  # 500+ days = full score
        
        # Volatility regime
        ann_vol = float(np.std(returns) * np.sqrt(252))
        vol_regime = "low" if ann_vol < 0.15 else "medium" if ann_vol < 0.35 else "high"
        
        # Trend strength (slope of linear regression on recent 60 days)
        recent = prices[-min(60, n):]
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent, 1)[0]
        trend_strength = abs(slope) / np.mean(recent)  # normalized
        has_strong_trend = trend_strength > 0.001
        
        # Fat tails (excess kurtosis)
        kurt = float(pd.Series(returns).kurtosis())
        has_fat_tails = kurt > 3
        
        # Stationarity proxy: ratio of recent vol to historical vol
        if n > 120:
            recent_vol = np.std(returns[-60:])
            hist_vol = np.std(returns[:-60])
            vol_ratio = recent_vol / hist_vol if hist_vol > 0 else 1.0
            regime_change = abs(vol_ratio - 1.0) > 0.5
        else:
            vol_ratio = 1.0
            regime_change = False
        
        # --- MODEL SELECTION LOGIC ---
        reasons = []
        scores = {"lstm": 0.0, "monte_carlo": 0.0, "exp_smoothing": 0.0}
        
        # LSTM: needs lots of data, works best with stable patterns
        if n >= 200:
            scores["lstm"] += 3.0
            reasons.append("Sufficient data for LSTM training")
        if vol_regime in ("low", "medium") and not regime_change:
            scores["lstm"] += 2.0
            reasons.append("Stable volatility regime favors LSTM")
        
        # Monte Carlo: best for high volatility, fat tails, regime changes
        if vol_regime == "high":
            scores["monte_carlo"] += 3.0
            reasons.append("High volatility favors stochastic simulation")
        if has_fat_tails:
            scores["monte_carlo"] += 2.0
            reasons.append("Fat-tailed returns suit Monte Carlo")
        if regime_change:
            scores["monte_carlo"] += 2.0
            reasons.append("Volatility regime change detected")
        
        # Exponential Smoothing: best for clear trends, limited data
        if has_strong_trend:
            scores["exp_smoothing"] += 3.0
            reasons.append("Strong trend detected — trend model advantages")
        if n < 200:
            scores["exp_smoothing"] += 2.0
            reasons.append("Limited data — statistical model more reliable")
        if vol_regime == "low":
            scores["exp_smoothing"] += 1.5
            reasons.append("Low volatility suits trend extrapolation")
        
        # Always give Monte Carlo a base score (it's always somewhat valid)
        scores["monte_carlo"] += 1.0
        
        # Pick best model
        best_model = max(scores, key=scores.get)
        
        # If scores are close, recommend ensemble
        sorted_scores = sorted(scores.values(), reverse=True)
        use_ensemble = (sorted_scores[0] - sorted_scores[1]) < 1.5
        
        return {
            "data_points": n,
            "ann_volatility": round(ann_vol * 100, 1),
            "vol_regime": vol_regime,
            "trend_strength": round(trend_strength * 10000, 2),
            "has_strong_trend": has_strong_trend,
            "kurtosis": round(kurt, 2),
            "has_fat_tails": has_fat_tails,
            "regime_change": regime_change,
            "model_scores": {k: round(v, 1) for k, v in scores.items()},
            "best_model": best_model,
            "use_ensemble": use_ensemble,
            "reasons": reasons,
        }

    @staticmethod
    def get_model_label(model_name: str) -> str:
        return {
            "lstm": "🧠 LSTM Neural Network",
            "monte_carlo": "🎲 Monte Carlo GBM",
            "exp_smoothing": "📈 Exponential Smoothing (Holt)",
        }.get(model_name, model_name)


# ============================================================================
# ENSEMBLE FORECASTER
# ============================================================================
class EnsembleForecaster:
    """
    Runs multiple forecasting models on a ticker and produces a
    confidence-weighted consensus prediction with auto-voting.
    
    Models:
        1. LSTM Neural Network  (dl_forecaster)
        2. Monte Carlo GBM      (analytics_engine)
        3. Double Exponential Smoothing / Holt's Method (analytics_engine)
    """

    def __init__(self):
        self.models_available = {
            'lstm': False,
            'monte_carlo': True,
            'exp_smoothing': True,
        }
        # Check if LSTM is available
        try:
            import torch
            self.models_available['lstm'] = True
        except ImportError:
            pass

    def forecast(self, ticker: str, forecast_days: int = 30,
                 epochs: int = 30, run_lstm: bool = True) -> dict:
        """
        Run all available models and produce ensemble forecast.
        
        Returns:
            {
                'ticker': str,
                'models': {model_name: result_dict, ...},
                'consensus': {...},
                'agreement_score': float (0-100),
                'signal': str,
                'success': bool,
            }
        """
        # Download shared historical data
        end = datetime.today()
        start = end - timedelta(days=730)  # 2 years
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                return {'success': False, 'error': f'No data for {ticker}', 'ticker': ticker}
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            prices = df["Close"].dropna().values.astype(float)
            if len(prices) < 60:
                return {'success': False, 'error': f'Insufficient data ({len(prices)} days)', 'ticker': ticker}
        except Exception as e:
            return {'success': False, 'error': str(e), 'ticker': ticker}

        current_price = float(prices[-1])
        returns = pd.Series(prices).pct_change().dropna()
        ann_mu = float(returns.mean() * 252)
        ann_sigma = float(returns.std() * np.sqrt(252))

        model_results = {}
        model_forecasts = {}  # end-of-horizon median price per model
        model_weights = {}

        # ── Model 1: Exponential Smoothing ──
        try:
            from analytics_engine import exponential_smoothing_forecast
            ets_result = exponential_smoothing_forecast(prices, forecast_days)
            if ets_result['success']:
                model_results['exp_smoothing'] = {
                    'forecast': ets_result['forecast'],
                    'upper': ets_result['upper'],
                    'lower': ets_result['lower'],
                    'end_price': float(ets_result['forecast'][-1]),
                    'rmse': ets_result['rmse'],
                    'mape': ets_result['mape'],
                    'method': ets_result['method'],
                }
                model_forecasts['exp_smoothing'] = float(ets_result['forecast'][-1])
                # Weight based on inverse MAPE (lower error = higher weight)
                model_weights['exp_smoothing'] = max(0.1, 1.0 / max(ets_result['mape'], 0.5))
        except Exception:
            pass

        # ── Model 2: Monte Carlo GBM ──
        try:
            from analytics_engine import run_monte_carlo_stock
            mc_result = run_monte_carlo_stock(current_price, ann_mu, ann_sigma, forecast_days)
            model_results['monte_carlo'] = {
                'percentiles': {k: v.tolist() for k, v in mc_result['percentiles'].items()},
                'end_price': mc_result['median_price'],
                'p5': mc_result['p5'],
                'p95': mc_result['p95'],
                'prob_up': mc_result['prob_up'],
                'method': 'Monte Carlo GBM (5000 sims)',
            }
            model_forecasts['monte_carlo'] = mc_result['median_price']
            model_weights['monte_carlo'] = 1.0  # baseline weight
        except Exception:
            pass

        # ── Model 3: LSTM Neural Network ──
        if run_lstm and self.models_available['lstm']:
            try:
                from dl_forecaster import train_and_forecast
                lstm_result = train_and_forecast(ticker, forecast_days, epochs=epochs)
                if lstm_result['success']:
                    fc = lstm_result['forecast']
                    model_results['lstm'] = {
                        'forecast': fc['Predicted'].values.tolist(),
                        'upper': fc['Upper (95%)'].values.tolist(),
                        'lower': fc['Lower (95%)'].values.tolist(),
                        'end_price': float(fc['Predicted'].values[-1]),
                        'rmse': lstm_result['metrics']['rmse'],
                        'mape': lstm_result['metrics']['mape'],
                        'dates': fc['Date'].tolist(),
                        'method': 'Stacked LSTM (PyTorch)',
                        'historical': lstm_result['historical'],
                    }
                    model_forecasts['lstm'] = float(fc['Predicted'].values[-1])
                    # LSTM gets higher base weight (deep learning advantage)
                    model_weights['lstm'] = max(0.1, 1.5 / max(lstm_result['metrics']['mape'], 0.5))
            except Exception:
                pass

        if not model_forecasts:
            return {'success': False, 'error': 'All models failed.', 'ticker': ticker}

        # ── Ensemble Consensus ──
        total_weight = sum(model_weights.values())
        normalized = {k: w / total_weight for k, w in model_weights.items()}

        consensus_price = sum(model_forecasts[k] * normalized[k] for k in model_forecasts)
        expected_return = (consensus_price - current_price) / current_price * 100

        # Agreement score: how much models agree (low std = high agreement)
        if len(model_forecasts) > 1:
            forecast_values = list(model_forecasts.values())
            forecast_std = np.std(forecast_values)
            forecast_mean = np.mean(forecast_values)
            cv = forecast_std / abs(forecast_mean) if forecast_mean != 0 else 1.0
            agreement_score = max(0, min(100, int((1 - min(cv, 1.0)) * 100)))
        else:
            agreement_score = 50  # single model = moderate confidence

        # Directional consensus
        directions = {k: ('up' if v > current_price else 'down') for k, v in model_forecasts.items()}
        up_votes = sum(1 for d in directions.values() if d == 'up')
        total_votes = len(directions)
        
        # Signal generation
        signal = _generate_signal(expected_return, agreement_score, up_votes, total_votes)

        return {
            'success': True,
            'ticker': ticker,
            'current_price': current_price,
            'models': model_results,
            'model_forecasts': model_forecasts,
            'model_weights': normalized,
            'consensus': {
                'price': round(consensus_price, 2),
                'return_pct': round(expected_return, 2),
                'agreement_score': agreement_score,
                'up_votes': up_votes,
                'total_votes': total_votes,
                'directions': directions,
            },
            'signal': signal,
            'forecast_days': forecast_days,
        }

    def smart_forecast(self, ticker: str, forecast_days: int = 30, epochs: int = 25) -> dict:
        """
        Auto-select the best model based on data characteristics, run it,
        and return the result with reasoning.
        
        Unlike forecast(), this evaluates data first, then runs only the
        optimal model (or ensemble if scores are close).
        """
        # Download data
        end = datetime.today()
        start = end - timedelta(days=730)
        try:
            df = yf.download(ticker, start=start, end=end, progress=False)
            if df.empty:
                return {'success': False, 'error': f'No data for {ticker}', 'ticker': ticker}
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            prices = df["Close"].dropna().values.astype(float)
            if len(prices) < 30:
                return {'success': False, 'error': f'Insufficient data ({len(prices)} days)', 'ticker': ticker}
        except Exception as e:
            return {'success': False, 'error': str(e), 'ticker': ticker}

        # Evaluate data characteristics
        analysis = AutoAnalyzer.evaluate_ticker(prices)
        best_model = analysis['best_model']
        use_ensemble = analysis['use_ensemble']

        # If ensemble is recommended or LSTM is unavailable, run full ensemble
        if use_ensemble or (best_model == 'lstm' and not self.models_available['lstm']):
            result = self.forecast(ticker, forecast_days, epochs,
                                   run_lstm=(best_model == 'lstm' and self.models_available['lstm']))
            if result.get('success'):
                result['auto_analysis'] = analysis
                result['selection_mode'] = 'ensemble'
                result['selection_reason'] = 'Model scores were close — ensemble provides better coverage'
            return result

        # Run only the best model
        current_price = float(prices[-1])
        returns = pd.Series(prices).pct_change().dropna()
        ann_mu = float(returns.mean() * 252)
        ann_sigma = float(returns.std() * np.sqrt(252))

        model_results = {}
        model_forecasts = {}
        model_weights = {}

        if best_model == 'exp_smoothing':
            try:
                from analytics_engine import exponential_smoothing_forecast
                ets = exponential_smoothing_forecast(prices, forecast_days)
                if ets['success']:
                    model_results['exp_smoothing'] = {
                        'forecast': ets['forecast'], 'upper': ets['upper'], 'lower': ets['lower'],
                        'end_price': float(ets['forecast'][-1]), 'rmse': ets['rmse'],
                        'mape': ets['mape'], 'method': ets['method'],
                    }
                    model_forecasts['exp_smoothing'] = float(ets['forecast'][-1])
                    model_weights['exp_smoothing'] = 1.0
            except Exception:
                pass

        elif best_model == 'monte_carlo':
            try:
                from analytics_engine import run_monte_carlo_stock
                mc = run_monte_carlo_stock(current_price, ann_mu, ann_sigma, forecast_days)
                model_results['monte_carlo'] = {
                    'percentiles': {k: v.tolist() for k, v in mc['percentiles'].items()},
                    'end_price': mc['median_price'], 'p5': mc['p5'], 'p95': mc['p95'],
                    'prob_up': mc['prob_up'], 'method': 'Monte Carlo GBM (5000 sims)',
                }
                model_forecasts['monte_carlo'] = mc['median_price']
                model_weights['monte_carlo'] = 1.0
            except Exception:
                pass

        elif best_model == 'lstm' and self.models_available['lstm']:
            try:
                from dl_forecaster import train_and_forecast
                lstm = train_and_forecast(ticker, forecast_days, epochs=epochs)
                if lstm['success']:
                    fc = lstm['forecast']
                    model_results['lstm'] = {
                        'forecast': fc['Predicted'].values.tolist(),
                        'upper': fc['Upper (95%)'].values.tolist(),
                        'lower': fc['Lower (95%)'].values.tolist(),
                        'end_price': float(fc['Predicted'].values[-1]),
                        'rmse': lstm['metrics']['rmse'], 'mape': lstm['metrics']['mape'],
                        'dates': fc['Date'].tolist(), 'method': 'Stacked LSTM (PyTorch)',
                        'historical': lstm['historical'],
                    }
                    model_forecasts['lstm'] = float(fc['Predicted'].values[-1])
                    model_weights['lstm'] = 1.0
            except Exception:
                pass

        # If best model failed, fall back to ensemble
        if not model_forecasts:
            result = self.forecast(ticker, forecast_days, epochs, run_lstm=self.models_available['lstm'])
            if result.get('success'):
                result['auto_analysis'] = analysis
                result['selection_mode'] = 'fallback_ensemble'
                result['selection_reason'] = f'{AutoAnalyzer.get_model_label(best_model)} failed — fell back to ensemble'
            return result

        # Build result
        consensus_price = list(model_forecasts.values())[0]
        expected_return = (consensus_price - current_price) / current_price * 100
        signal = _generate_signal(expected_return, 60, 1 if expected_return > 0 else 0, 1)

        total_w = sum(model_weights.values())
        normalized = {k: w / total_w for k, w in model_weights.items()}

        return {
            'success': True,
            'ticker': ticker,
            'current_price': current_price,
            'models': model_results,
            'model_forecasts': model_forecasts,
            'model_weights': normalized,
            'consensus': {
                'price': round(consensus_price, 2),
                'return_pct': round(expected_return, 2),
                'agreement_score': 60,
                'up_votes': 1 if expected_return > 0 else 0,
                'total_votes': 1,
                'directions': {best_model: 'up' if expected_return > 0 else 'down'},
            },
            'signal': signal,
            'forecast_days': forecast_days,
            'auto_analysis': analysis,
            'selection_mode': 'auto_best',
            'selection_reason': (
                f"Selected {AutoAnalyzer.get_model_label(best_model)} — "
                + "; ".join(analysis['reasons'][:3])
            ),
        }

    def auto_analyze_portfolio(self, tickers: list, forecast_days: int = 30) -> dict:
        """
        Run smart_forecast on all tickers and return portfolio-level results.
        """
        results = []
        for ticker in tickers:
            result = self.smart_forecast(ticker, forecast_days)
            results.append(result)
        return results


def _generate_signal(expected_return: float, agreement: int, up_votes: int, total: int) -> str:
    """Generate trading signal from ensemble metrics."""
    bullish_ratio = up_votes / total if total > 0 else 0.5

    if expected_return > 10 and agreement > 70 and bullish_ratio >= 0.67:
        return "🟢 STRONG BUY"
    elif expected_return > 3 and agreement > 50 and bullish_ratio >= 0.5:
        return "🟢 BUY"
    elif expected_return < -10 and agreement > 70 and bullish_ratio <= 0.33:
        return "🔴 STRONG SELL"
    elif expected_return < -3 and agreement > 50 and bullish_ratio <= 0.5:
        return "🔴 SELL"
    else:
        return "🟡 HOLD"


# ============================================================================
# INSIGHT GENERATOR
# ============================================================================
class InsightGenerator:
    """
    Takes ensemble forecast results + risk metrics + sentiment data
    and produces actionable natural language insights.
    """

    def generate_ticker_insight(self, ensemble_result: dict,
                                risk_row: dict = None,
                                sentiment_label: str = None) -> dict:
        """Generate insight for a single ticker."""
        if not ensemble_result.get('success'):
            return {'ticker': ensemble_result.get('ticker', '?'), 'insights': [], 'signal': 'N/A'}

        ticker = ensemble_result['ticker']
        consensus = ensemble_result['consensus']
        signal = ensemble_result['signal']
        models = ensemble_result['models']
        current = ensemble_result['current_price']
        target = consensus['price']
        ret = consensus['return_pct']
        agreement = consensus['agreement_score']

        insights = []

        # Price direction insight
        direction = "upside" if ret > 0 else "downside"
        insights.append(
            f"**Ensemble Consensus:** {len(models)} models project "
            f"**{abs(ret):.1f}% {direction}** to ${target:,.2f} "
            f"over {ensemble_result['forecast_days']} days "
            f"(Agreement: {agreement}%)."
        )

        # Model agreement insight
        dirs = consensus['directions']
        if consensus['up_votes'] == consensus['total_votes']:
            insights.append("✅ **Unanimous bullish** — all models agree on upward movement.")
        elif consensus['up_votes'] == 0:
            insights.append("⚠️ **Unanimous bearish** — all models project decline.")
        else:
            bull_models = [k for k, v in dirs.items() if v == 'up']
            bear_models = [k for k, v in dirs.items() if v == 'down']
            insights.append(
                f"📊 **Mixed signals** — Bullish: {', '.join(bull_models)} | "
                f"Bearish: {', '.join(bear_models)}"
            )

        # Model-specific highlights
        if 'lstm' in models:
            m = models['lstm']
            insights.append(f"🧠 LSTM target: ${m['end_price']:,.2f} (MAPE: {m['mape']:.1f}%)")
        if 'monte_carlo' in models:
            m = models['monte_carlo']
            insights.append(
                f"🎲 Monte Carlo: ${m['p5']:,.2f} – ${m['p95']:,.2f} range "
                f"({m['prob_up']:.0f}% probability of gain)"
            )
        if 'exp_smoothing' in models:
            m = models['exp_smoothing']
            insights.append(f"📈 Trend Model: ${m['end_price']:,.2f} (MAPE: {m['mape']:.1f}%)")

        # Risk insight
        if risk_row:
            rs = risk_row.get('Risk Score', 50)
            vol = risk_row.get('Volatility (%)', 0)
            max_dd = risk_row.get('Max Drawdown (%)', 0)
            level = "🟢 Low" if rs < 30 else "🟡 Moderate" if rs < 60 else "🔴 High"
            insights.append(
                f"⚡ Risk Score: **{rs}/100** ({level}) — "
                f"Vol: {vol:.1f}%, Max DD: {max_dd:.1f}%"
            )

        # Sentiment insight
        if sentiment_label:
            insights.append(f"📰 Market Sentiment: **{sentiment_label}**")

        return {
            'ticker': ticker,
            'signal': signal,
            'target_price': target,
            'expected_return': ret,
            'agreement': agreement,
            'insights': insights,
        }

    def generate_portfolio_summary(self, ticker_insights: list) -> str:
        """Generate a portfolio-level intelligence summary."""
        if not ticker_insights:
            return "No intelligence data available."

        strong_buys = [t for t in ticker_insights if 'STRONG BUY' in t['signal']]
        buys = [t for t in ticker_insights if t['signal'] == '🟢 BUY']
        holds = [t for t in ticker_insights if 'HOLD' in t['signal']]
        sells = [t for t in ticker_insights if 'SELL' in t['signal']]
        strong_sells = [t for t in ticker_insights if 'STRONG SELL' in t['signal']]

        avg_return = np.mean([t['expected_return'] for t in ticker_insights])
        avg_agreement = np.mean([t['agreement'] for t in ticker_insights])

        lines = [
            "## 📊 Portfolio Intelligence Summary\n",
            f"**Tickers Analyzed:** {len(ticker_insights)} | "
            f"**Avg Expected Return:** {avg_return:+.1f}% | "
            f"**Avg Model Agreement:** {avg_agreement:.0f}%\n",
            "### Signal Distribution\n",
        ]

        if strong_buys:
            lines.append(f"- 🟢 **Strong Buy:** {', '.join(t['ticker'] for t in strong_buys)}")
        if buys:
            lines.append(f"- 🟢 **Buy:** {', '.join(t['ticker'] for t in buys)}")
        if holds:
            lines.append(f"- 🟡 **Hold:** {', '.join(t['ticker'] for t in holds)}")
        if sells:
            lines.append(f"- 🔴 **Sell:** {', '.join(t['ticker'] for t in sells)}")
        if strong_sells:
            lines.append(f"- 🔴 **Strong Sell:** {', '.join(t['ticker'] for t in strong_sells)}")

        lines.append("\n### Top Opportunities\n")
        sorted_by_return = sorted(ticker_insights, key=lambda x: x['expected_return'], reverse=True)
        for t in sorted_by_return[:3]:
            lines.append(
                f"- **{t['ticker']}**: {t['signal']} — "
                f"Target ${t['target_price']:,.2f} ({t['expected_return']:+.1f}%), "
                f"Agreement {t['agreement']}%"
            )

        if any(t['expected_return'] < -5 for t in ticker_insights):
            lines.append("\n### ⚠️ Risk Alerts\n")
            for t in sorted_by_return:
                if t['expected_return'] < -5:
                    lines.append(
                        f"- **{t['ticker']}**: {t['signal']} — "
                        f"Projected {t['expected_return']:+.1f}% decline"
                    )

        return "\n".join(lines)
