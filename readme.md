# 🏛️ Family Office Portfolio Dashboard

An industry-grade Family Office Portfolio Dashboard designed to track, analyze, and forecast high-net-worth portfolios. Built with **Python**, **Streamlit**, and **SQLAlchemy**, this data analytics capstone project replaces fragmented spreadsheets with a rigorous mathematical backend and an intuitive, interactive user interface.

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

### 🧠 NLP-Powered AI Advisory
Utilizes the `vaderSentiment` NLP library to evaluate live financial RSS feeds. The AI calculates an aggregated semantic score (Bullish/Bearish/Neutral) and dynamically adjusts investment recommendations and risk posturing based on global macroeconomic sentiment.

### ⚖️ Automated Target Rebalancing
An Integrated Investment Policy Statement (IPS) engine that compares current asset values against target allocations, delivering exact dollar-value recommendations for required buys and sells to eliminate portfolio drift.

### 💾 Relational Database Backend
Data is structured and securely stored via an **SQLite** database using **SQLAlchemy** ORM, ensuring high data integrity, quick CRUD operations across physical assets, private equity, and crypto, and a scalable ETL pipeline.

---

## 🚀 Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/family-office-dashboard.git
   cd family-office-dashboard
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   streamlit run app.py
   ```
   *The application will automatically launch in your default web browser at `http://localhost:8501`.*

---
*Built as a comprehensive Data Analytics & Finance project.*