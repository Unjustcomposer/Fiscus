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
        .sort_values("Gain/Loss", ascending=False)


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
        ("Apple Inc.", "AAPL", "Public Equity", 500, 75000, 95000, "Tech large-cap", "USD", "2022-01-15"),
        ("Tesla Inc.", "TSLA", "Public Equity", 200, 50000, 48000, "EV sector", "USD", "2021-06-01"),
        ("Reliance Industries", "", "Public Equity", 1000, 10000000, 12000000, "Indian conglomerate", "INR", "2020-03-10"),
        ("Acme Growth Fund III", "", "Private Equity", 1, 250000, 310000, "Series B stage fund", "USD", "2021-09-01"),
        ("HealthTech Ventures", "", "Private Equity", 1, 100000, 85000, "Early-stage healthtech", "USD", "2022-07-15"),
        ("Gold Bars (1kg)", "GLD", "Gold & Precious Metals", 5, 300000, 375000, "Physical gold", "USD", "2019-11-01"),
        ("Silver Coins", "SLV", "Gold & Precious Metals", 200, 5000, 6200, "Collectible silver", "USD", "2020-05-20"),
        ("Loan to XYZ Corp", "", "Loans (Given)", 1, 500000, 520000, "12% annual, matures 2027", "USD", "2023-01-01"),
        ("Personal Loan – Family", "", "Loans (Given)", 1, 200000, 200000, "Interest-free", "USD", "2023-06-01"),
        ("Monet Print", "", "Art & Collectibles", 1, 80000, 120000, "Authenticated, insured", "USD", "2018-04-10"),
        ("Vintage Rolex Daytona", "", "Art & Collectibles", 1, 25000, 45000, "1969, excellent condition", "USD", "2019-02-15"),
        ("Downtown Office Space", "", "Real Estate", 1, 125000000, 150000000, "Rental ₹4L/mo", "INR", "2015-08-01"),
        ("Beach Villa, Goa", "", "Real Estate", 1, 65000000, 79000000, "Vacation property", "INR", "2017-12-01"),
        ("US Treasury Bonds", "", "Fixed Income & Bonds", 100, 100000, 102000, "10Y T-bonds", "USD", "2022-10-01"),
        ("Corporate FD", "", "Fixed Income & Bonds", 1, 5000000, 5360000, "7.5% locked 3yr", "INR", "2021-03-01"),
        ("Savings Account", "", "Cash & Equivalents", 1, 350000, 350000, "HDFC Bank", "USD", "2020-01-01"),
        ("Bitcoin (BTC)", "BTC-USD", "Cryptocurrency", 2, 60000, 72000, "Cold wallet", "USD", "2020-10-15"),
        ("Ethereum (ETH)", "ETH-USD", "Cryptocurrency", 15, 30000, 27000, "Staked", "USD", "2021-01-20"),
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
        ("Home Mortgage", "Mortgage", 1, 1200000, 980000, "4.5% fixed", "USD", "2015-08-01"),
        ("Office Property Loan", "Business Loan", 1, 500000, 420000, "6% variable", "USD", "2015-08-15"),
        ("Car Loan", "Personal Loan", 1, 45000, 28000, "EMI $850/mo", "USD", "2022-03-01"),
        ("AMEX Platinum", "Credit Card Debt", 1, 15000, 15000, "Monthly balance", "USD", "2024-01-01"),
        ("Margin Loan", "Margin Loan", 1, 50000, 50000, "Interactive Brokers", "USD", "2023-09-01"),
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