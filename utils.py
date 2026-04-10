# utils.py — Family Office Portfolio Dashboard
# Core data layer: persistence, CRUD, calculations, snapshots, XIRR

import pandas as pd
import json
import os
from datetime import datetime

# ============================================================================
# CONSTANTS
# ============================================================================
DATA_FILE = "portfolio_data.json"
SNAPSHOTS_FILE = "snapshots.json"

ASSET_CATEGORIES = [
    "Public Equity",
    "Indian Equity",
    "Private Equity",
    "Gold & Precious Metals",
    "Loans (Given)",
    "Art & Collectibles",
    "Real Estate",
    "Fixed Income & Bonds",
    "Cash & Equivalents",
    "Cryptocurrency",
    "Hedge Funds",
    "Forex Management",
    "Other Assets",
]
LIABILITY_CATEGORIES = [
    "Mortgage",
    "Business Loan",
    "Personal Loan",
    "Credit Card Debt",
    "Margin Loan",
    "Other Liabilities",
]
CURRENCIES = ["USD", "INR", "EUR", "GBP", "JPY", "AED"]

FX_TO_USD = {"USD": 1.0, "INR": 0.012, "EUR": 1.08, "GBP": 1.27, "JPY": 0.0067, "AED": 0.27}


import database

# ============================================================================
# PERSISTENCE — Load / Save to SQL
# ============================================================================
def save_portfolio(df: pd.DataFrame):
    """Persist the portfolio DataFrame to SQLite database."""
    database.save_holdings_df(df)

def load_portfolio() -> pd.DataFrame:
    """Load portfolio from SQLite database. Returns empty DataFrame if not found."""
    df = database.get_holdings_df()
    if df.empty:
        return create_empty_dataframe()
    for col in ["Cost Basis", "Current Value", "Quantity"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

# ============================================================================
# SNAPSHOTS — Historical Net Worth Tracking
# ============================================================================
def save_snapshot(df: pd.DataFrame):
    """Record today's Net Worth, Assets, and Liabilities as a snapshot."""
    summary = calculate_portfolio_summary(df)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "net_worth": round(summary["net_worth"], 2),
        "total_assets": round(summary["total_assets"], 2),
        "total_liabilities": round(summary["total_liabilities"], 2),
    }
    database.save_snapshot(
        entry["date"], 
        entry["net_worth"], 
        entry["total_assets"], 
        entry["total_liabilities"]
    )
    return entry

def load_snapshots() -> list:
    """Proxy for backward compatibility."""
    df = database.get_snapshots_df()
    if df.empty:
        return []
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df.to_dict("records")

def get_snapshots_df() -> pd.DataFrame:
    """Return snapshots as a tidy DataFrame from SQL."""
    return database.get_snapshots_df()


# ============================================================================
# DATAFRAME SCHEMA
# ============================================================================
def create_empty_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "Name", "Ticker", "Side", "Category", "Currency", "Quantity",
        "Cost Basis", "Current Value", "Date Added", "Notes"
    ])


# ============================================================================
# CRUD — Add, Update, Delete
# ============================================================================
def add_holding(df, name, side, category, quantity, cost_basis,
                current_value, notes="", currency="USD", ticker="") -> pd.DataFrame:
    """Append a new holding with a unique ID."""
    new_id = int(df["id"].max() + 1) if not df.empty and "id" in df.columns and df["id"].notna().any() else 1
    new_row = {
        "id": new_id,
        "Name": name,
        "Ticker": ticker or "",
        "Side": side,
        "Category": category,
        "Currency": currency,
        "Quantity": float(quantity),
        "Cost Basis": float(cost_basis),
        "Current Value": float(current_value),
        "Date Added": datetime.now().strftime("%Y-%m-%d"),
        "Notes": notes or "",
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def update_holding(df, holding_id, **kwargs) -> pd.DataFrame:
    """Update fields of an existing holding by its ID."""
    mask = df["id"] == holding_id
    for k, v in kwargs.items():
        if k in df.columns:
            df.loc[mask, k] = v
    return df


def delete_holding(df, holding_id) -> pd.DataFrame:
    """Remove a holding by its ID."""
    return df[df["id"] != holding_id].reset_index(drop=True)


# ============================================================================
# CURRENCY CONVERSION
# ============================================================================
def to_base_currency(value: float, currency: str, base_curr: str) -> float:
    """Convert a value from its native currency to the target base currency."""
    rate_usd = FX_TO_USD.get(currency, 1.0)
    base_usd = FX_TO_USD.get(base_curr, 1.0)
    return value * (rate_usd / base_usd)

def normalize_to_base(df: pd.DataFrame, base_curr: str = "USD") -> pd.DataFrame:
    """Add base currency normalised columns to the DataFrame."""
    df = df.copy()
    df["Cost Basis Base"] = df.apply(lambda r: to_base_currency(r["Cost Basis"], r.get("Currency", "USD"), base_curr), axis=1)
    df["Current Value Base"] = df.apply(lambda r: to_base_currency(r["Current Value"], r.get("Currency", "USD"), base_curr), axis=1)
    return df


# ============================================================================
# PORTFOLIO CALCULATIONS
# ============================================================================
def calculate_portfolio_summary(df: pd.DataFrame, base_curr: str = "USD") -> dict:
    """Return core KPIs, always in base currency."""
    if df.empty:
        return dict(total_assets=0, total_liabilities=0, net_worth=0,
                    total_invested=0, total_return=0, return_pct=0)
    df_base = normalize_to_base(df, base_curr)
    assets = df_base[df_base["Side"] == "Asset"]
    liabs = df_base[df_base["Side"] == "Liability"]
    total_assets = assets["Current Value Base"].sum()
    total_invested = assets["Cost Basis Base"].sum()
    total_liabilities = liabs["Current Value Base"].sum()
    net_worth = total_assets - total_liabilities
    total_return = total_assets - total_invested
    return_pct = (total_return / total_invested * 100) if total_invested > 0 else 0
    return dict(total_assets=total_assets, total_liabilities=total_liabilities,
                net_worth=net_worth, total_invested=total_invested,
                total_return=total_return, return_pct=return_pct)


def get_allocation_summary(df: pd.DataFrame, side="Asset", base_curr: str = "USD") -> pd.DataFrame:
    """Group by Category, aggregated in base currency."""
    if df.empty:
        return pd.DataFrame()
    df_base = normalize_to_base(df, base_curr)
    filtered = df_base[df_base["Side"] == side]
    if filtered.empty:
        return pd.DataFrame()
    summary = filtered.groupby("Category").agg(
        **{
            "Current Value": ("Current Value Base", "sum"),
            "Cost Basis": ("Cost Basis Base", "sum"),
            "Holdings": ("Name", "count"),
        }
    ).reset_index()
    summary["Gain/Loss"] = summary["Current Value"] - summary["Cost Basis"]
    summary["Return %"] = (summary["Gain/Loss"] / summary["Cost Basis"] * 100).round(2)
    return summary.sort_values("Current Value", ascending=False)


def get_holding_performance(df: pd.DataFrame, side="Asset", base_curr: str = "USD") -> pd.DataFrame:
    """Per-holding gain/loss in base currency."""
    if df.empty:
        return pd.DataFrame()
    df_base = normalize_to_base(df, base_curr)
    filtered = df_base[df_base["Side"] == side].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["Gain/Loss"] = filtered["Current Value Base"] - filtered["Cost Basis Base"]
    filtered["Return %"] = (filtered["Gain/Loss"] / filtered["Cost Basis Base"] * 100).round(2)
    return filtered[["id", "Name", "Category", "Currency", "Cost Basis", "Current Value",
                      "Cost Basis Base", "Current Value Base", "Gain/Loss", "Return %"]]\
        .sort_values("Current Value Base", ascending=False)


# ============================================================================
# XIRR — Annualised return for time-weighted assets
# ============================================================================
def compute_xirr(cost_basis: float, current_value: float, date_added_str: str) -> float:
    """
    Simplified IRR: ((current_value / cost_basis) ^ (1/years)) - 1
    Returns annualised return as a percentage.
    """
    try:
        date_added = pd.to_datetime(date_added_str)
        years = (datetime.now() - date_added).days / 365.25
        if years < 0.01 or cost_basis <= 0:
            return 0.0
        ratio = current_value / cost_basis
        irr = (ratio ** (1 / years) - 1) * 100
        return round(irr, 2)
    except Exception:
        return 0.0


# ============================================================================
# SAMPLE DATA GENERATOR
# ============================================================================
def generate_sample_portfolio() -> pd.DataFrame:
    df = create_empty_dataframe()
    assets = [
        # ── PUBLIC EQUITY — US Large Cap ──
        ("Apple Inc.", "AAPL", "Public Equity", 800, 120000, 156000, "Core tech holding", "USD", "2021-03-15"),
        ("Microsoft Corp.", "MSFT", "Public Equity", 400, 100000, 168000, "Cloud + AI play", "USD", "2020-11-10"),
        ("NVIDIA Corp.", "NVDA", "Public Equity", 300, 45000, 126000, "GPU/AI dominance", "USD", "2022-01-20"),
        ("Amazon.com", "AMZN", "Public Equity", 600, 90000, 114000, "E-commerce + AWS", "USD", "2021-06-05"),
        ("Alphabet Inc.", "GOOGL", "Public Equity", 250, 62500, 87500, "Search + Cloud", "USD", "2021-09-01"),
        ("Tesla Inc.", "TSLA", "Public Equity", 500, 125000, 95000, "EV sector, volatile", "USD", "2021-11-15"),
        ("Meta Platforms", "META", "Public Equity", 200, 50000, 96000, "Social + Metaverse", "USD", "2022-06-01"),
        ("JPMorgan Chase", "JPM", "Public Equity", 350, 49000, 73500, "Banking leader", "USD", "2020-04-01"),
        ("Johnson & Johnson", "JNJ", "Public Equity", 400, 60000, 62000, "Healthcare stable", "USD", "2019-08-15"),
        ("Berkshire Hathaway B", "BRK-B", "Public Equity", 100, 25000, 46200, "Value investing", "USD", "2018-12-01"),
        # ── PUBLIC EQUITY — Growth / Momentum ──
        ("Palantir Technologies", "PLTR", "Public Equity", 2000, 20000, 48600, "AI/Data analytics", "USD", "2023-01-10"),
        ("AMD", "AMD", "Public Equity", 500, 37500, 52500, "Semiconductor", "USD", "2022-05-20"),
        ("Costco Wholesale", "COST", "Public Equity", 50, 25000, 45000, "Retail defensive", "USD", "2021-02-14"),
        # ── INDIAN EQUITY ──
        ("Reliance Industries", "RELIANCE.NS", "Indian Equity", 2000, 20000000, 28000000, "Indian conglomerate", "INR", "2020-03-10"),
        ("HDFC Bank", "HDFCBANK.NS", "Indian Equity", 3000, 4500000, 5100000, "Premier bank", "INR", "2021-01-15"),
        ("Infosys Ltd.", "INFY", "Indian Equity", 1500, 2250000, 2700000, "IT services", "INR", "2020-06-20"),
        ("TCS", "TCS.NS", "Indian Equity", 800, 2800000, 3200000, "IT services giant", "INR", "2019-09-01"),
        ("Bajaj Finance", "BAJFINANCE.NS", "Indian Equity", 500, 3500000, 4100000, "NBFC leader", "INR", "2021-04-01"),
        # ── PRIVATE EQUITY ──
        ("Acme Growth Fund III", "", "Private Equity", 1, 250000, 340000, "Series B stage fund, 2.1x MOIC", "USD", "2021-09-01"),
        ("HealthTech Ventures LP", "", "Private Equity", 1, 150000, 115000, "Early-stage healthtech, J-curve", "USD", "2022-07-15"),
        ("Sequoia Capital India Fund", "", "Private Equity", 1, 500000, 720000, "India venture fund", "USD", "2020-01-15"),
        ("Tiger Global Private Fund", "", "Private Equity", 1, 300000, 255000, "Growth equity, marked down", "USD", "2021-12-01"),
        ("Real Estate PE Fund II", "", "Private Equity", 1, 200000, 278000, "Commercial RE fund", "USD", "2019-06-01"),
        # ── GOLD & PRECIOUS METALS ──
        ("Gold Bars (2kg)", "GLD", "Gold & Precious Metals", 10, 400000, 520000, "Physical gold in vault", "USD", "2019-11-01"),
        ("Silver Coins (500oz)", "SLV", "Gold & Precious Metals", 500, 12500, 15800, "Collectible silver", "USD", "2020-05-20"),
        ("Gold Sovereign Bonds", "", "Gold & Precious Metals", 100, 5000000, 6800000, "SGB 2.5% coupon", "INR", "2020-09-01"),
        # ── LOANS (GIVEN) ──
        ("Loan to XYZ Corp", "", "Loans (Given)", 1, 500000, 540000, "12% annual, matures 2027", "USD", "2023-01-01"),
        ("Personal Loan – Family", "", "Loans (Given)", 1, 200000, 200000, "Interest-free, repayable 2028", "USD", "2023-06-01"),
        ("Bridge Loan – Startup", "", "Loans (Given)", 1, 75000, 78000, "15% convertible note", "USD", "2024-03-01"),
        # ── ART & COLLECTIBLES ──
        ("Monet 'Water Lilies' Print", "", "Art & Collectibles", 1, 80000, 135000, "Authenticated, insured at Christie's", "USD", "2018-04-10"),
        ("Vintage Rolex Daytona 1969", "", "Art & Collectibles", 1, 25000, 52000, "Paul Newman dial, excellent", "USD", "2019-02-15"),
        ("Wine Collection (Bordeaux)", "", "Art & Collectibles", 48, 36000, 58000, "2005-2015 vintages, temp-controlled", "EUR", "2017-10-01"),
        ("Rare Stamps Portfolio", "", "Art & Collectibles", 1, 15000, 19500, "British Commonwealth, graded", "GBP", "2016-03-20"),
        # ── REAL ESTATE ──
        ("Downtown Office Space, Mumbai", "", "Real Estate", 1, 125000000, 165000000, "Rental ₹4.5L/mo, BKC", "INR", "2015-08-01"),
        ("Beach Villa, Goa", "", "Real Estate", 1, 65000000, 82000000, "Vacation + Airbnb ₹1.2L/mo", "INR", "2017-12-01"),
        ("Manhattan Studio Apartment", "", "Real Estate", 1, 650000, 720000, "Rental $3,200/mo, Upper West Side", "USD", "2019-05-15"),
        ("London Flat, Kensington", "", "Real Estate", 1, 450000, 510000, "Rental £2,800/mo", "GBP", "2018-09-01"),
        ("Dubai Marina Apartment", "", "Real Estate", 1, 1500000, 1850000, "Rental 8,500 AED/mo", "AED", "2022-03-01"),
        # ── FIXED INCOME & BONDS ──
        ("US Treasury Bonds 10Y", "", "Fixed Income & Bonds", 200, 200000, 195000, "4.25% coupon, maturity 2033", "USD", "2022-10-01"),
        ("Corporate FD (HDFC)", "", "Fixed Income & Bonds", 1, 5000000, 5650000, "7.5% locked 3yr", "INR", "2021-03-01"),
        ("Vanguard Total Bond ETF", "BND", "Fixed Income & Bonds", 500, 37500, 35800, "Broad bond exposure", "USD", "2023-01-15"),
        ("SBI Fixed Deposit", "", "Fixed Income & Bonds", 1, 10000000, 11200000, "6.8% 5yr FD", "INR", "2022-06-01"),
        ("Municipal Bonds (CA)", "", "Fixed Income & Bonds", 100, 100000, 98500, "Tax-exempt, 3.5%", "USD", "2023-04-01"),
        # ── CASH & EQUIVALENTS ──
        ("USD Savings (Chase)", "", "Cash & Equivalents", 1, 450000, 450000, "Chase Private Client", "USD", "2020-01-01"),
        ("INR Current Account", "", "Cash & Equivalents", 1, 8000000, 8000000, "HDFC Bank", "INR", "2020-01-01"),
        ("EUR Money Market Fund", "", "Cash & Equivalents", 1, 120000, 122400, "HSBC EUR MM", "EUR", "2023-07-01"),
        ("GBP Savings Account", "", "Cash & Equivalents", 1, 85000, 87500, "Barclays", "GBP", "2022-01-01"),
        # ── CRYPTOCURRENCY ──
        ("Bitcoin (BTC)", "BTC-USD", "Cryptocurrency", 3, 90000, 285000, "Cold wallet, Ledger", "USD", "2020-10-15"),
        ("Ethereum (ETH)", "ETH-USD", "Cryptocurrency", 25, 50000, 45000, "Staked on Lido", "USD", "2021-01-20"),
        ("Solana (SOL)", "SOL-USD", "Cryptocurrency", 500, 15000, 67500, "DeFi exposure", "USD", "2022-09-01"),
        # ── HEDGE FUNDS ──
        ("Bridgewater Pure Alpha II", "", "Hedge Funds", 1, 500000, 545000, "Macro strategy, quarterly liquidity", "USD", "2020-06-01"),
        ("Citadel Wellington Fund", "", "Hedge Funds", 1, 300000, 378000, "Multi-strategy", "USD", "2021-03-15"),
        ("Two Sigma Absolute Return", "", "Hedge Funds", 1, 250000, 282500, "Quant strategies", "USD", "2022-01-01"),
        # ── FOREX MANAGEMENT ──
        ("USD/INR Forward Contract", "", "Forex Management", 1, 500000, 515000, "Hedging INR exposure, 6mo", "USD", "2024-06-01"),
        ("EUR/USD Position", "", "Forex Management", 1, 200000, 196000, "Tactical short EUR", "USD", "2024-09-01"),
        # ── OTHER ASSETS ──
        ("Patent Portfolio (5 patents)", "", "Other Assets", 5, 150000, 225000, "Tech patents, licensed to 3 firms", "USD", "2019-01-15"),
        ("Life Insurance (Cash Value)", "", "Other Assets", 1, 200000, 265000, "Whole life, Northwestern Mutual", "USD", "2015-01-01"),
    ]
    for idx, (name, ticker, cat, qty, cost, val, notes, curr, date) in enumerate(assets, 1):
        row = {
            "id": idx, "Name": name, "Ticker": ticker,
            "Side": "Asset", "Category": cat,
            "Currency": curr, "Quantity": float(qty),
            "Cost Basis": float(cost), "Current Value": float(val),
            "Date Added": date, "Notes": notes,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    liabilities = [
        # ── MORTGAGES ──
        ("Primary Home Mortgage", "Mortgage", 1, 1200000, 920000, "4.5% fixed 30yr, Wells Fargo", "USD", "2015-08-01"),
        ("Mumbai Office Mortgage", "Mortgage", 1, 50000000, 38000000, "8.2% floating, SBI", "INR", "2015-08-01"),
        ("London Flat Mortgage", "Mortgage", 1, 280000, 195000, "3.8% fixed, HSBC UK", "GBP", "2018-09-01"),
        # ── BUSINESS LOANS ──
        ("Office Property Loan", "Business Loan", 1, 500000, 380000, "6% variable, Bank of America", "USD", "2015-08-15"),
        ("Dubai Property Loan", "Business Loan", 1, 900000, 750000, "5.5% fixed, Emirates NBD", "AED", "2022-03-01"),
        # ── PERSONAL LOANS ──
        ("Car Loan (Mercedes S-Class)", "Personal Loan", 1, 85000, 52000, "EMI $1,400/mo, 4.2%", "USD", "2022-03-01"),
        ("Education Loan", "Personal Loan", 1, 120000, 95000, "3.5% subsidized, Sallie Mae", "USD", "2020-09-01"),
        # ── CREDIT CARDS ──
        ("AMEX Centurion", "Credit Card Debt", 1, 25000, 25000, "Monthly revolving balance", "USD", "2024-01-01"),
        ("HDFC Infinia CC", "Credit Card Debt", 1, 350000, 350000, "Monthly balance", "INR", "2024-06-01"),
        # ── MARGIN / OTHER ──
        ("Margin Loan (IBKR)", "Margin Loan", 1, 150000, 150000, "Interactive Brokers, 5.8%", "USD", "2023-09-01"),
        ("Tax Liability 2025", "Other Liabilities", 1, 180000, 180000, "Estimated federal + state", "USD", "2025-04-15"),
    ]
    for idx, (name, cat, qty, cost, val, notes, curr, date) in enumerate(liabilities, len(assets) + 1):
        row = {
            "id": idx, "Name": name, "Ticker": "",
            "Side": "Liability", "Category": cat,
            "Currency": curr, "Quantity": float(qty),
            "Cost Basis": float(cost), "Current Value": float(val),
            "Date Added": date, "Notes": notes,
        }
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df
