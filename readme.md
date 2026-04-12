# 🏛️ fiscus Dashboard

**An Ensemble Intelligence System for Multi-Asset Portfolio Management**

Integrates LSTM forecasting, transformer-based sentiment analysis (FinBERT), autoencoder anomaly detection, and automatic model selection into a unified decision-support platform for family offices.

---

## 🔬 Research Features

| Feature | Description |
|---------|-------------|
| **Walk-Forward Backtesting** | Expanding-window evaluation with monthly rebalance steps over 2-3 years of data |
| **Baseline Models** | Naive, Random Walk, ARIMA (auto-order), Buy-and-Hold, SMA Crossover |
| **Evaluation Metrics** | RMSE, MAPE, Directional Accuracy, Hit Rate, Profit Factor, Sharpe, Calmar, Information Ratio |
| **Statistical Tests** | Paired t-test, Wilcoxon signed-rank, Diebold-Mariano test |
| **Ablation Studies** | 15 configurations systematically removing components to quantify contribution |
| **Sentiment-Weighted Ensemble** | FinBERT score dynamically re-weights model ensemble via gating function |
| **MC Dropout Uncertainty** | 50 stochastic forward passes produce confidence intervals that widen with horizon |
| **F1-Calibrated Anomaly Detection** | Threshold sweep (P80-P99) with synthetic labels for optimal anomaly sensitivity |
| **Mathematical Formulations** | All equations documented in LaTeX-exportable format |

## 🏗️ Architecture

```
Family_Office_Portfolio_Tracker/
├── app.py                    # Main Streamlit dashboard (entry point)
├── pages/
│   └── page_backtesting.py   # Walk-forward backtesting & ablation UI
│
├── # ── Core Data Layer ──
├── utils.py                  # Portfolio CRUD, calculations, XIRR
├── database.py               # SQLAlchemy ORM (SQLite)
├── market_data.py            # yfinance price fetcher + FX rates
├── target_allocation.py      # IPS engine & rebalancing
│
├── # ── Analytics ──
├── analytics_engine.py       # Risk metrics, Monte Carlo, Holt's smoothing
├── ai_advisor.py             # Data-driven advisory (portfolio-gap scoring)
│
├── # ── Deep Learning ──
├── dl_forecaster.py          # LSTM with MC Dropout uncertainty
├── dl_anomaly.py             # Autoencoder with F1-calibrated threshold
├── dl_sentiment.py           # FinBERT transformer sentiment analysis
├── intelligence_engine.py    # Ensemble orchestration + sentiment gating
│
├── # ── Research Infrastructure ──
├── backtesting.py            # Walk-forward backtesting engine
├── baselines.py              # Baseline models (Naive, RW, ARIMA, B&H, SMA)
├── evaluation.py             # Metrics suite + statistical significance tests
├── ablation_runner.py        # Automated ablation study executor
├── math_formulations.py      # LaTeX-exportable mathematical documentation
│
├── # ── Security ──
├── auth.py                   # streamlit-authenticator login wall
│
├── # ── Paper ──
├── paper/
│   └── paper_skeleton.tex    # LaTeX paper skeleton (IEEE format)
│
└── requirements.txt
```

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

**Default credentials:** `admin` / `admin123`

## 📊 Key Modules

### Ensemble Forecasting with Sentiment Gating
The system runs three models simultaneously — **LSTM**, **Monte Carlo GBM**, and **Holt's Exponential Smoothing** — then fuses their predictions using inverse-MAPE weighting modulated by a **FinBERT sentiment gating function**:

```
w_i' = w_i^(base) × g(s, m_i)

g(s, LSTM)          = 1 + 0.3s
g(s, Monte Carlo)   = 1 - 0.4s  
g(s, Exp Smoothing) = 1 + 0.5s
```

Where `s ∈ [-1, 1]` is the aggregate FinBERT sentiment score. Bearish sentiment increases Monte Carlo weight (tail risk capture); bullish increases trend-following weights.

### Walk-Forward Backtesting
```python
from backtesting import WalkForwardBacktester

bt = WalkForwardBacktester("AAPL", lookback_years=3, forecast_horizon=21)
results = bt.run(models=['naive', 'arima', 'monte_carlo', 'exp_smoothing'])
summary = bt.get_summary()         # Per-model metrics
sig = bt.get_significance_tests()  # Pairwise statistical tests
```

### Ablation Studies
```python
from ablation_runner import AblationRunner

runner = AblationRunner(tickers=["AAPL", "MSFT", "NVDA"])
runner.run_suite(configs=["full_ensemble", "no_sentiment", "naive_baseline"])
comparison = runner.get_averaged_comparison()  # Averaged across tickers
```

## 📈 Sample Portfolio

The dashboard ships with a comprehensive sample portfolio spanning:
- **13 asset categories** (Public Equity, Private Equity, Real Estate, Gold, Crypto, etc.)
- **6 currencies** (USD, INR, EUR, GBP, JPY, AED)
- **55+ holdings** with realistic valuations
- **11 liability positions** (mortgages, loans, credit cards)

## 🔐 Security

- **Authentication**: streamlit-authenticator with cookie-based sessions
- **Data Storage**: SQLAlchemy ORM with SQLite (local, encrypted at rest via OS-level encryption)
- **Input Validation**: Schema validation on CSV imports

## 📄 Citation

```bibtex
@software{family_office_tracker_2026,
  title={Ensemble Intelligence for Multi-Asset Portfolio Management},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```

## 📝 License

This project is developed as a semester capstone. All rights reserved.
