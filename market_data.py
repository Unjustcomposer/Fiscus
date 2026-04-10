# market_data.py — Live Market Price Fetcher
# Uses yfinance to fetch real-time prices for public equities, ETFs, crypto

import yfinance as yf
import json
import os
from datetime import datetime, timedelta

PRICE_CACHE_FILE = "price_cache.json"
CACHE_TTL_MINUTES = 15  # Don't hammer Yahoo Finance — cache for 15 min


def _load_cache() -> dict:
    if os.path.exists(PRICE_CACHE_FILE):
        try:
            with open(PRICE_CACHE_FILE) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict):
    with open(PRICE_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def _is_cache_fresh(entry: dict) -> bool:
    if not entry or "fetched_at" not in entry:
        return False
    fetched_at = datetime.fromisoformat(entry["fetched_at"])
    return datetime.now() - fetched_at < timedelta(minutes=CACHE_TTL_MINUTES)


def fetch_price(ticker: str) -> dict:
    """
    Fetch the latest price for a ticker symbol.
    Returns a dict: {price, currency, name, change_pct, source}
    Returns None if ticker is invalid or fetch fails.
    """
    ticker = ticker.strip().upper()
    cache = _load_cache()

    # Return cached value if fresh
    if ticker in cache and _is_cache_fresh(cache[ticker]):
        return cache[ticker]

    try:
        t = yf.Ticker(ticker)
        info = t.info

        # yfinance returns different keys depending on the asset type
        price = (
            info.get("regularMarketPrice")
            or info.get("currentPrice")
            or info.get("navPrice")
            or info.get("ask")
        )
        if not price:
            # Try fast_info as fallback
            fi = t.fast_info
            price = getattr(fi, "last_price", None)

        if not price:
            return None

        currency = info.get("currency", "USD")
        name = info.get("longName") or info.get("shortName") or ticker
        prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose") or price
        change_pct = ((price - prev_close) / prev_close * 100) if prev_close else 0.0

        result = {
            "ticker": ticker,
            "price": round(float(price), 4),
            "currency": currency,
            "name": name,
            "change_pct": round(change_pct, 2),
            "fetched_at": datetime.now().isoformat(),
            "source": "Yahoo Finance",
        }
        cache[ticker] = result
        _save_cache(cache)
        return result

    except Exception as e:
        return {"error": str(e), "ticker": ticker}


def fetch_prices_bulk(tickers: list) -> dict:
    """Fetch multiple tickers. Returns {ticker: price_dict}."""
    results = {}
    for ticker in tickers:
        if ticker and ticker.strip():
            results[ticker.strip().upper()] = fetch_price(ticker.strip())
    return results


def update_portfolio_prices(df):
    """
    Find all holdings with a Ticker column and update their Current Value
    using live market prices. Returns (updated_df, update_log).
    """
    if df.empty or "Ticker" not in df.columns:
        return df, []

    df = df.copy()
    log = []

    for idx, row in df.iterrows():
        ticker = str(row.get("Ticker", "")).strip()
        if not ticker or ticker.lower() in ("", "nan", "none", "-"):
            continue

        qty = float(row.get("Quantity", 1))
        data = fetch_price(ticker)

        if data and "price" in data and not data.get("error"):
            old_val = float(row["Current Value"])
            new_val = round(data["price"] * qty, 2)
            df.at[idx, "Current Value"] = new_val
            log.append({
                "name": row["Name"],
                "ticker": ticker,
                "price": data["price"],
                "currency": data["currency"],
                "old_value": old_val,
                "new_value": new_val,
                "change_pct": data["change_pct"],
            })
        elif data and data.get("error"):
            log.append({
                "name": row["Name"],
                "ticker": ticker,
                "error": data["error"],
            })

    return df, log

def update_fx_rates():
    """Update live FX rates using yfinance, writing directly to utils.FX_TO_USD."""
    import utils
    log = []
    for curr in utils.CURRENCIES:
        if curr == "USD":
            continue
        ticker = f"{curr}USD=X"
        data = fetch_price(ticker)
        
        if not data or not data.get("price") or data.get("error"):
            # try fallback
            inv_ticker = f"USD{curr}=X"
            inv_data = fetch_price(inv_ticker)
            if inv_data and inv_data.get("price") and not inv_data.get("error"):
                old_rate = utils.FX_TO_USD.get(curr, 1.0)
                new_rate = 1.0 / inv_data["price"]
                utils.FX_TO_USD[curr] = new_rate
                log.append({"currency": curr, "old": old_rate, "new": new_rate})
        elif data and data.get("price") and not data.get("error"):
            old_rate = utils.FX_TO_USD.get(curr, 1.0)
            utils.FX_TO_USD[curr] = data["price"]
            log.append({"currency": curr, "old": old_rate, "new": data["price"]})
            
    return log
