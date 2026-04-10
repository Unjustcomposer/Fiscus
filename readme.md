# 🏛️ Family Office Portfolio Dashboard

An industry-grade Family Office Portfolio Dashboard designed to track, analyze, and forecast high-net-worth portfolios. Built with **Python**, **Streamlit**, and **SQLAlchemy**, this data analytics capstone project replaces fragmented spreadsheets with a rigorous mathematical backend, **deep learning engine**, and an intuitive, interactive user interface.

---

## ✨ Key Features

### 📉 Advanced Risk Analytics (Public Equities)
Calculates real-time financial metrics using `yfinance` and `numpy`:
- **Sharpe Ratio** & **Beta** (correlated against the S&P 500)
- **Annualized Volatility**
- **Historical Value at Risk (VaR 95%)**
- **Pearson Correlation Heatmap** for visualizing asset co-movement.

### 🔮 Predictive Forecasting (Monte Carlo)
Features a Geometric Brownian Motion simulator that processes current net worth, expected blended return, and blended volatility to run **10,000 parallel realities** mapping paths over the next 5 to 30 years, returning a percentile-based probability matrix.

### 🤖 Deep Learning Engine
A dedicated neural network-powered analytics module featuring three models:

#### 📈 LSTM Stock Price Forecasting
A **stacked LSTM (Long Short-Term Memory)** recurrent neural network built with PyTorch that:
- Learns temporal price patterns from 2 years of historical data
- Uses a 60-day sliding window architecture (LSTM 128 → LSTM 64 → Dense 32 → Output)
- Produces multi-day price forecasts via auto-regressive rollout
- Displays 95% confidence bands and reports RMSE/MAPE metrics

#### 📰 Transformer-Based Sentiment Analysis (FinBERT)
Replaces traditional lexicon-based NLP with a **pre-trained FinBERT transformer** model:
- Deep neural network fine-tuned on millions of financial documents
- Understands context, negation, and domain-specific financial language
- Returns true probability distributions (positive/negative/neutral) per headline
- Integrated into the AI Advisory report with VADER as silent fallback

#### 🔍 Autoencoder Anomaly Detection
A **deep autoencoder** (symmetric encoder-decoder) that:
- Extracts 10 statistical features per 30-day rolling window per asset
- Learns "normal" portfolio behavior patterns during training
- Flags holdings with high reconstruction error as anomalous risk shifts
- Color-coded risk levels: 🟢 Normal → 🟠 Watch → 🟡 Elevated → 🔴 Critical

### 🧠 NLP-Powered AI Advisory
Utilizes the `FinBERT` deep learning model (with `vaderSentiment` NLP fallback) to evaluate live financial RSS feeds. The AI calculates an aggregated semantic score (Bullish/Bearish/Neutral) and dynamically adjusts investment recommendations and risk posturing based on global macroeconomic sentiment.

### ⚖️ Automated Target Rebalancing
An Integrated Investment Policy Statement (IPS) engine that compares current asset values against target allocations, delivering exact dollar-value recommendations for required buys and sells to eliminate portfolio drift.

### 💾 Relational Database Backend
Data is structured and securely stored via an **SQLite** database using **SQLAlchemy** ORM, ensuring high data integrity, quick CRUD operations across physical assets, private equity, and crypto, and a scalable ETL pipeline.

---

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/Unjustcomposer/Family_Office_Portfolio_Tracker.git
   cd Family_Office_Portfolio_Tracker
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   > **Note:** PyTorch and Transformers are large packages (~2-3 GB). The first `pip install` will take several minutes. The FinBERT model (~400MB) downloads automatically on first use.

3. **Run the application**
   ```bash
   streamlit run app.py
   ```
   *The application will automatically launch in your default web browser at `http://localhost:8501`.*

---

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Python, SQLAlchemy, SQLite |
| **Deep Learning** | PyTorch (LSTM, Autoencoder), HuggingFace Transformers (FinBERT) |
| **Data** | yfinance, pandas, NumPy, scikit-learn |
| **NLP** | FinBERT (primary), VADER Sentiment (fallback) |

---
*Built as a comprehensive Data Analytics, Finance & Deep Learning project.*
