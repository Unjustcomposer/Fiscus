# target_allocation.py — Investment Policy Statement (IPS) Engine
# Manages target allocations and generates rebalancing recommendations

import json
import os
import pandas as pd
import utils

IPS_FILE = "ips_targets.json"

DEFAULT_TARGETS = {
    "Public Equity": 30.0,
    "Private Equity": 15.0,
    "Real Estate": 20.0,
    "Gold & Precious Metals": 10.0,
    "Fixed Income & Bonds": 10.0,
    "Cash & Equivalents": 5.0,
    "Loans (Given)": 5.0,
    "Art & Collectibles": 3.0,
    "Cryptocurrency": 2.0,
}


def load_targets() -> dict:
    """Load IPS target allocations from disk."""
    if os.path.exists(IPS_FILE):
        try:
            with open(IPS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_TARGETS.copy()


def save_targets(targets: dict):
    """Persist IPS targets to disk."""
    with open(IPS_FILE, "w") as f:
        json.dump(targets, f, indent=2)


def compute_rebalancing(portfolio_df: pd.DataFrame, targets: dict) -> pd.DataFrame:
    """
    Compare current allocation vs targets.
    Returns a DataFrame with:
      Category, Target %, Current %, Drift %, Dollar Drift, Action, Trade Amount
    """
    if portfolio_df.empty:
        return pd.DataFrame()

    # Get current allocation in USD
    alloc = utils.get_allocation_summary(portfolio_df, "Asset")
    total_assets = alloc["Current Value"].sum() if not alloc.empty else 0
    if total_assets == 0:
        return pd.DataFrame()

    # All categories we care about (union of targets + current holdings)
    all_cats = set(targets.keys()) | set(alloc["Category"].tolist() if not alloc.empty else [])

    rows = []
    for cat in sorted(all_cats):
        target_pct = targets.get(cat, 0.0)
        current_row = alloc[alloc["Category"] == cat] if not alloc.empty else pd.DataFrame()
        current_val = float(current_row["Current Value"].values[0]) if not current_row.empty else 0.0
        current_pct = (current_val / total_assets * 100) if total_assets > 0 else 0.0
        drift_pct = current_pct - target_pct
        target_val = (target_pct / 100) * total_assets
        dollar_drift = current_val - target_val

        # Determine action and urgency
        abs_drift = abs(drift_pct)
        if abs_drift < 2.0:
            action = "✅ On Target"
            urgency = "Low"
        elif drift_pct > 0:
            action = "📉 Reduce"
            urgency = "High" if abs_drift > 10 else "Medium"
        else:
            action = "📈 Increase"
            urgency = "High" if abs_drift > 10 else "Medium"

        rows.append({
            "Category": cat,
            "Target %": round(target_pct, 1),
            "Current %": round(current_pct, 1),
            "Drift %": round(drift_pct, 1),
            "Current Value ($)": round(current_val, 0),
            "Target Value ($)": round(target_val, 0),
            "Dollar Drift ($)": round(dollar_drift, 0),
            "Action": action,
            "Urgency": urgency,
        })

    return pd.DataFrame(rows).sort_values("Drift %", key=abs, ascending=False)


def get_rebalancing_trades(rebal_df: pd.DataFrame) -> tuple[list, list]:
    """Split rebalancing into buys and sells."""
    if rebal_df.empty:
        return [], []
    sells = rebal_df[rebal_df["Dollar Drift ($)"] > 200].to_dict("records")
    buys = rebal_df[rebal_df["Dollar Drift ($)"] < -200].to_dict("records")
    return buys, sells


def validate_targets(targets: dict) -> tuple[bool, str]:
    """Check that targets sum to 100%."""
    total = sum(targets.values())
    if abs(total - 100.0) > 0.1:
        return False, f"Targets sum to {total:.1f}% — must equal 100%."
    return True, ""
