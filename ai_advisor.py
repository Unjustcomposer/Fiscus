# ai_advisor.py — Family Office Portfolio AI Advisory Engine
# Fetches real RSS headlines + generates intelligent rule-based advisory

import random
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()


# ============================================================================
# REAL NEWS FEED — Reuters / MarketWatch RSS
# ============================================================================
RSS_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",
    "https://feeds.reuters.com/reuters/businessNews",
    "https://rss.nytimes.com/services/xml/rss/nyt/Business.xml",
    "https://feeds.bbci.co.uk/news/business/rss.xml",
]

FALLBACK_HEADLINES = [
    "Global markets show resilience amid central bank rate uncertainty.",
    "Tech sector leads equity gains as AI infrastructure spending rises.",
    "Gold rallies to historic highs as investors seek safe-haven assets.",
    "Private credit market sees record inflows as banks tighten lending.",
    "Real estate commercial sector faces headwinds; residential holds firm.",
]


def fetch_news_headlines(max_items: int = 5) -> list[str]:
    """
    Attempt to fetch real headlines from RSS feeds.
    Falls back gracefully to local headlines if offline.
    """
    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            if feed.entries:
                return [e.title for e in feed.entries[:max_items]]
        except Exception:
            continue
    return random.sample(FALLBACK_HEADLINES, min(max_items, len(FALLBACK_HEADLINES)))


# ============================================================================
# RULE-BASED ADVISORY ENGINE
# ============================================================================
LIQUIDATION_REASONS = [
    "has achieved outsized gains relative to purchase price — prime for partial profit-taking.",
    "represents a concentrated position. Industry standards suggest capping single assets at 10% AUM.",
    "has outperformed its category peers and may be entering a valuation ceiling.",
    "is highly illiquid; consider trimming to strengthen the overall liquidity buffer.",
]

INVESTMENT_IDEAS = [
    {"name": "Global Infrastructure ETF (e.g. IGF)", "category": "Public Equity",
     "reason": "Strong dividend yield + government infrastructure spending tailwinds. Excellent inflation hedge."},
    {"name": "Short-Duration Corporate Bonds", "category": "Fixed Income & Bonds",
     "reason": "Attractive real yields with minimal duration risk while rate trajectory clarifies."},
    {"name": "Venture Debt / Private Credit Fund", "category": "Private Equity",
     "reason": "High IRR as startups struggle to raise equity; family offices well-positioned for this."},
    {"name": "Physical Silver (ETF or bars)", "category": "Gold & Precious Metals",
     "reason": "Historically undervalued vs gold. Rising industrial demand from EV and solar sectors."},
    {"name": "Prime Logistics Real Estate (REIT)", "category": "Real Estate",
     "reason": "E-commerce secular growth ensures sustained demand for high-quality warehouse space."},
    {"name": "Invoice Discounting Platform", "category": "Loans (Given)",
     "reason": "15-18% annualised yield. Short tenor (30-90 days) keeps liquidity risk manageable."},
    {"name": "Currency Carry Trade / Yield Optimization", "category": "Forex Management",
     "reason": "Exploit positive interest rate differentials between major central bank currencies."},
    {"name": "Long USD / Emerging Markets Hedge", "category": "Forex Management",
     "reason": "Protect purchasing power during periods of heightened global risk-off sentiment."}
]


def generate_advisory_report(portfolio_df, headlines: list[str] = None, base_curr: str = "USD") -> str:
    """
    Generate a comprehensive advisory report based on portfolio state and news context.
    Returns a markdown-formatted string.
    """
    if portfolio_df.empty:
        return "⚠️ **Portfolio is empty.** Please add holdings to receive advisory."

    # Import here to avoid circular imports
    import utils

    df_base = utils.normalize_to_base(portfolio_df, base_curr)
    summary = utils.calculate_portfolio_summary(portfolio_df, base_curr)

    assets = df_base[df_base["Side"] == "Asset"].copy()
    liabs = df_base[df_base["Side"] == "Liability"].copy()

    if assets.empty:
        return "⚠️ **No assets tracked.** Add investments to receive advisory."

    # Cash metrics
    cash = assets[assets["Category"] == "Cash & Equivalents"]["Current Value Base"].sum()
    cash_pct = (cash / summary["total_assets"] * 100) if summary["total_assets"] > 0 else 0

    # Fetch news
    if headlines is None:
        headlines = fetch_news_headlines(5)

    # Best gainer
    assets["Gain"] = assets["Current Value Base"] - assets["Cost Basis Base"]
    top_gainer = assets.nlargest(1, "Gain").iloc[0]

    # Biggest concentration
    from utils import get_allocation_summary
    alloc = get_allocation_summary(portfolio_df, "Asset", base_curr)
    total_val = alloc["Current Value"].sum()
    alloc["Pct"] = alloc["Current Value"] / total_val * 100
    overweight = alloc[alloc["Pct"] > 30]

    # Build report
    report = []

    # Calculate NLP Sentiment — Deep Learning (FinBERT) with VADER fallback
    try:
        from dl_sentiment import get_analyzer
        deep_analyzer = get_analyzer()
        sentiment_data = deep_analyzer.get_market_sentiment(headlines)
        avg_sentiment = sentiment_data['avg_score']
        sentiment_label = sentiment_data['label']
        model_used = sentiment_data['model']
    except Exception:
        # Silent fallback to VADER
        compound_scores = [analyzer.polarity_scores(h)['compound'] for h in headlines]
        avg_sentiment = sum(compound_scores) / len(compound_scores) if compound_scores else 0
        if avg_sentiment > 0.15:
            sentiment_label = "Bullish 📈"
        elif avg_sentiment < -0.15:
            sentiment_label = "Bearish 📉"
        else:
            sentiment_label = "Neutral ⚖️"
        model_used = "VADER (Lexicon Fallback)"

    # Section 1: News context
    report.append("### 🌐 Global Market Context")
    report.append(f"**Overall NLP Sentiment Score:** {avg_sentiment:.2f} ({sentiment_label})")
    report.append(f"**Model:** {model_used}")
    report.append("")
    report.append("*Real-time headlines informing this analysis:*")
    for h in headlines:
        report.append(f"> 📰 {h}")
    report.append("")

    # Section 2: Liquidity
    sym = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "AED": "د.إ"}.get(base_curr, "$")
    report.append("### 💧 Liquidity Assessment")
    report.append(f"**Cash & Equivalents:** {sym}{cash:,.0f} ({cash_pct:.1f}% of total assets)")
    if cash_pct < 5:
        report.append("🔴 **Critical:** Cash buffer is dangerously low (<5%). Consider liquidating some short-term positions to build reserves for market opportunities.")
    elif cash_pct < 10:
        report.append("🟡 **Warning:** Cash below the recommended 10% threshold. Minor rebalancing advised.")
    elif cash_pct > 30:
        report.append("🟡 **Cash drag:** You are holding >30% in cash. Inflation is actively eroding this. Consider deploying into the suggestions below.")
    else:
        report.append("🟢 **Healthy:** Liquidity is within the recommended 10-20% Family Office range.")
    report.append("")

    # Section 3: Concentration risk
    report.append("### ⚖️ Concentration Risk")
    if not overweight.empty:
        for _, row in overweight.iterrows():
            report.append(f"🔴 **{row['Category']}** is over-concentrated at **{row['Pct']:.1f}%** of AUM. Target: <30%.")
    else:
        report.append("🟢 No single asset class exceeds the 30% concentration threshold. Portfolio is well-diversified.")
    report.append("")

    # Section 4: Liquidation candidate — Data-Driven Scoring
    # Score = gain_pct × concentration_pct (higher = stronger sell candidate)
    report.append("### 📉 Liquidation Recommendation")
    total_asset_val = assets["Current Value Base"].sum()
    if total_asset_val > 0 and not assets.empty:
        assets_scored = assets.copy()
        assets_scored["gain_pct"] = (
            (assets_scored["Current Value Base"] - assets_scored["Cost Basis Base"])
            / assets_scored["Cost Basis Base"].replace(0, 1) * 100
        )
        assets_scored["concentration_pct"] = (
            assets_scored["Current Value Base"] / total_asset_val * 100
        )
        # Only consider gainers
        gainers = assets_scored[assets_scored["gain_pct"] > 5].copy()
        if not gainers.empty:
            gainers["liquidation_score"] = (
                gainers["gain_pct"] * gainers["concentration_pct"] / 100
            )
            top_candidate = gainers.nlargest(1, "liquidation_score").iloc[0]
            gain_pct = top_candidate["gain_pct"]
            conc_pct = top_candidate["concentration_pct"]
            report.append(f"**Consider trimming:** {top_candidate['Name']} (*{top_candidate['Category']}*)")
            report.append(f"- Unrealised Gain: **{sym}{top_candidate['Current Value Base'] - top_candidate['Cost Basis Base']:,.0f}** ({gain_pct:+.1f}%)")
            report.append(f"- Portfolio Concentration: **{conc_pct:.1f}%** of total assets")
            # Generate specific reason from data
            reasons = []
            if gain_pct > 50:
                reasons.append(f"has achieved outsized gains of {gain_pct:+.0f}% — prime for partial profit-taking")
            if conc_pct > 10:
                reasons.append(f"represents {conc_pct:.1f}% of AUM, exceeding the 10% single-asset concentration limit")
            if gain_pct > 20 and conc_pct > 5:
                reasons.append("exhibits both high returns and meaningful portfolio weight — risk/reward favors trimming")
            reason_text = "; ".join(reasons) if reasons else "has performed well and may benefit from profit-taking"
            report.append(f"- Rationale: This holding {reason_text}. A 20-30% trim is recommended.")
        else:
            report.append("No strong liquidation candidates identified. Most positions are near or below cost basis.")
    else:
        report.append("No strong liquidation candidates identified at this time.")
    report.append("")

    # Section 5: Investment suggestions — Portfolio-Gap-Driven
    # Rank INVESTMENT_IDEAS by which categories have the largest negative
    # drift from IPS targets, with sentiment-adjusted prioritization.
    report.append("### 📈 Investment Opportunities")
    
    # Mathematical adjustment of text based on sentiment score
    if avg_sentiment < -0.15:
        report.append(f"Given the **Bearish** market sentiment ({avg_sentiment:.2f}), the AI model emphasizes defensive posturing and capital preservation. Based on your {sym}{cash:,.0f} available cash:")
    elif avg_sentiment > 0.15:
        report.append(f"Given the **Bullish** market sentiment ({avg_sentiment:.2f}), the AI model is optimizing for growth capture. Based on your {sym}{cash:,.0f} available cash:")
    else:
        report.append(f"With macro sentiment appearing **Neutral** ({avg_sentiment:.2f}), the AI model maintains standard allocation logic. Based on your {sym}{cash:,.0f} available cash:")
    
    # Compute portfolio allocation gaps vs IPS targets
    try:
        import target_allocation
        targets = target_allocation.load_targets()
        rebal_df = target_allocation.compute_rebalancing(portfolio_df, targets)
        
        if not rebal_df.empty:
            # Get underweight categories (negative drift = need to buy)
            underweight = rebal_df[rebal_df["Drift %"] < -1.0].sort_values("Drift %")
            underweight_cats = set(underweight["Category"].tolist())
        else:
            underweight_cats = set()
    except Exception:
        underweight_cats = set()
    
    # Score each investment idea by portfolio gap relevance + sentiment
    defensive_cats = {"Fixed Income & Bonds", "Gold & Precious Metals", "Cash & Equivalents"}
    growth_cats = {"Public Equity", "Private Equity", "Cryptocurrency"}
    
    scored_ideas = []
    for idea in INVESTMENT_IDEAS:
        score = 0.0
        # Portfolio gap bonus: ideas in underweight categories score higher
        if idea["category"] in underweight_cats:
            score += 5.0
        # Sentiment adjustment
        if avg_sentiment < -0.15 and idea["category"] in defensive_cats:
            score += 3.0  # Bearish → prefer defensive
        elif avg_sentiment > 0.15 and idea["category"] in growth_cats:
            score += 3.0  # Bullish → prefer growth
        # Base diversity score
        score += 1.0
        scored_ideas.append((score, idea))
    
    # Sort by score descending, take top 3
    scored_ideas.sort(key=lambda x: x[0], reverse=True)
    top_ideas = [idea for _, idea in scored_ideas[:3]]
    
    for i, inv in enumerate(top_ideas, 1):
        gap_note = ""
        if inv["category"] in underweight_cats:
            gap_note = " *(fills portfolio gap)*"
        report.append(f"{i}. **{inv['name']}** — *{inv['category']}*{gap_note}")
        report.append(f"   > {inv['reason']}")
    report.append("")

    # Section: Forex Advisory
    forex_holdings = assets[assets["Category"] == "Forex Management"]
    if not forex_holdings.empty:
        report.append("### 💱 Forex Management Advisory")
        report.append("Based on current macro volatility and your active forex exposures:")
        for _, row in forex_holdings.iterrows():
            report.append(f"- **{row['Name']}** ({row['Currency']}): Consider setting strict stop-losses given recent FX cross-rate swings. Carry trade returns may face headwinds if central bank rates pivot.")
        report.append("")

    # Liability note
    leverage = (summary["total_liabilities"] / summary["net_worth"]) if summary["net_worth"] > 0 else 0
    if leverage > 0.5:
        report.append("### ⚠️ Leverage Alert")
        report.append(f"Total liabilities are **{leverage*100:.0f}%** of Net Worth. This is elevated. Prioritize debt reduction before deploying additional capital.")
        report.append("")

    report.append("---")
    report.append("*This report is AI-rule-based and uses live news feeds for context. It is for informational purposes only and does not constitute financial advice. Always consult your Investment Committee before executing transactions.*")

    return "\n".join(report)
