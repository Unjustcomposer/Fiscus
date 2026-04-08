# app.py — Family Office Portfolio Dashboard (Industry Grade)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import utils
import ai_advisor
import market_data
import target_allocation
import analytics_engine
import numpy as np

# ============================================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================================
st.set_page_config(
    page_title="Family Office Portfolio",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# THEME CSS
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
}
h2, h3 { font-weight: 600 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%);
    border: 1px solid rgba(102,126,234,0.25);
    border-radius: 14px;
    padding: 18px 22px;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(102,126,234,0.18);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.75;
}
[data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 700 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #11192e 60%, #0c2340 100%);
}
[data-testid="stSidebar"] * { color: #d5d5e8 !important; }

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    padding: 8px 18px;
    font-weight: 500;
    font-size: 0.9rem;
}

[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

.stButton > button {
    border-radius: 9px;
    font-weight: 600;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 16px rgba(0,0,0,0.2);
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.35), transparent);
    margin: 1.5rem 0;
}

[data-testid="stForm"] {
    border: 1px solid rgba(102,126,234,0.22);
    border-radius: 14px;
    padding: 26px;
    background: linear-gradient(135deg, rgba(102,126,234,0.04) 0%, rgba(118,75,162,0.04) 100%);
}

.gain { color: #2ecc71; font-weight: 600; }
.loss { color: #e74c3c; font-weight: 600; }

#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE — load from disk once
# ============================================================================
if "portfolio" not in st.session_state:
    st.session_state.portfolio = utils.load_portfolio()
if "advisory_report" not in st.session_state:
    st.session_state.advisory_report = None
if "advisory_headlines" not in st.session_state:
    st.session_state.advisory_headlines = None

# ============================================================================
# HELPERS
# ============================================================================
def fmt(val):
    try:
        val = float(val)
    except Exception:
        return str(val)
    prefix = "-$" if val < 0 else "$"
    v = abs(val)
    if v >= 1_000_000:
        return f"{prefix}{v/1_000_000:,.2f}M"
    if v >= 1_000:
        return f"{prefix}{v:,.0f}"
    return f"{prefix}{v:,.2f}"


def save():
    """Persist current portfolio to disk."""
    utils.save_portfolio(st.session_state.portfolio)


def chart_layout(**kwargs):
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(t=30, b=30, l=20, r=20),
    )
    defaults.update(kwargs)   # caller kwargs always win
    return defaults

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown("## 🏛️ Family Office")
    st.markdown("**Portfolio Dashboard**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "💼 Manage Holdings",
         "📊 Analytics", "🔮 Risk & Forecasting", "⚖️ Rebalancing", "📑 Reports & KPIs", "🧠 AI Advisory",
         "💾 Data Management"],
        index=0
    )
    st.markdown("---")

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🎲 Sample Data", use_container_width=True):
            st.session_state.portfolio = utils.generate_sample_portfolio()
            save()
            st.rerun()
    with col_s2:
        if st.button("📸 Snapshot", use_container_width=True, help="Record today's Net Worth"):
            if not st.session_state.portfolio.empty:
                utils.save_snapshot(st.session_state.portfolio)
                st.success("Snapshot saved!", icon="✅")
            else:
                st.warning("Portfolio is empty.")

    if st.button("🔄 Sync Live Prices", use_container_width=True, help="Update via Yahoo Finance"):
        with st.spinner("Fetching market data..."):
            updated_df, log = market_data.update_portfolio_prices(st.session_state.portfolio)
            st.session_state.portfolio = updated_df
            save()
            updated_count = len([x for x in log if 'error' not in x])
            if updated_count > 0:
                st.session_state._price_log = f"✅ Updated prices for {updated_count} holdings."
            else:
                st.session_state._price_log = "ℹ️ No prices updated. (Check tickers)"
            st.rerun()

    if getattr(st.session_state, '_price_log', None):
        st.caption(st.session_state._price_log)
        st.session_state._price_log = None

    st.markdown("---")
    summary_side = utils.calculate_portfolio_summary(st.session_state.portfolio)
    st.metric("Net Worth", fmt(summary_side["net_worth"]))
    st.metric("Total Assets", fmt(summary_side["total_assets"]))
    st.metric("Total Liabilities", fmt(summary_side["total_liabilities"]))
    st.caption("All values displayed in USD.")

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.title("🏠 Portfolio Dashboard")

    summary = utils.calculate_portfolio_summary(st.session_state.portfolio)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("🏛️ Net Worth", fmt(summary["net_worth"]))
    m2.metric("📈 Total Assets", fmt(summary["total_assets"]))
    m3.metric("📉 Total Liabilities", fmt(summary["total_liabilities"]))
    m4.metric("💹 Unrealised Return", fmt(summary["total_return"]),
              delta=f"{summary['return_pct']:+.1f}%")
    leverage = (summary["total_liabilities"] / summary["net_worth"]) if summary["net_worth"] > 0 else 0
    m5.metric("⚖️ Leverage Ratio", f"{leverage:.2f}x")

    st.markdown("---")

    if st.session_state.portfolio.empty:
        st.info("Portfolio is empty. Load sample data from the sidebar or use ➕ Add Holding.")
    else:
        col_l, col_r = st.columns(2)

        # --- LEFT: Asset Sunburst ---
        with col_l:
            st.subheader("Asset Allocation")
            assets_df = st.session_state.portfolio[st.session_state.portfolio["Side"] == "Asset"].copy()
            if not assets_df.empty:
                assets_df["Root"] = "Assets"
                assets_df["Current Value USD"] = utils.normalize_to_usd(assets_df)["Current Value USD"]
                fig = px.sunburst(
                    assets_df, path=["Root", "Category", "Name"],
                    values="Current Value USD",
                    color="Current Value USD",
                    color_continuous_scale=["#2c2c54", "#667eea", "#764ba2", "#d0bbff"],
                    hover_data={"Current Value USD": ":$,.0f"},
                )
                fig.update_traces(
                    textinfo="label+percent entry",
                    hovertemplate="<b>%{label}</b><br>Value: $%{value:,.0f}<extra></extra>",
                    marker=dict(line=dict(width=0))
                )
                fig.update_layout(height=400, coloraxis_showscale=False, **chart_layout())
                st.plotly_chart(fig, use_container_width=True)
                st.caption("✨ Click a category to drill down.")

        # --- RIGHT: Donut ---
        with col_r:
            st.subheader("Assets vs Liabilities")
            fig_donut = go.Figure(go.Pie(
                labels=["Assets", "Liabilities"],
                values=[max(summary["total_assets"], 0), max(summary["total_liabilities"], 0)],
                hole=0.65,
                marker_colors=["#667eea", "#e74c3c"],
                textinfo="label+percent",
                textposition="outside",
                hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<extra></extra>",
                marker=dict(line=dict(width=0))
            ))
            fig_donut.add_annotation(text="<b>Debt Ratio</b>", x=0.5, y=0.58,
                                     font_size=13, showarrow=False, font_color="#c0c0d0")
            fig_donut.add_annotation(
                text=f"{leverage*100:.1f}%", x=0.5, y=0.42, font_size=30,
                showarrow=False, font_color="#e74c3c" if leverage > 0.4 else "#2ecc71")
            fig_donut.update_layout(height=400, showlegend=False, **chart_layout())
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption("✨ Debt as % of total assets.")

        # --- Holdings Tables ---
        st.markdown("---")
        st.subheader("📋 All Holdings")
        tab_a, tab_l = st.tabs(["💼 Assets", "📋 Liabilities"])

        def render_holdings_table(side):
            df_side = st.session_state.portfolio[st.session_state.portfolio["Side"] == side].copy()
            if df_side.empty:
                st.info(f"No {side.lower()}s recorded.")
                return
            df_usd = utils.normalize_to_usd(df_side)
            df_usd["Gain/Loss ($)"] = (df_usd["Current Value USD"] - df_usd["Cost Basis USD"]).round(2)
            df_usd["Return %"] = ((df_usd["Gain/Loss ($)"] / df_usd["Cost Basis USD"]) * 100).round(1)
            display = df_usd[["Name", "Category", "Currency", "Cost Basis", "Current Value",
                               "Gain/Loss ($)", "Return %", "Date Added", "Notes"]].copy()
            for c in ["Cost Basis", "Current Value"]:
                display[c] = display[c].apply(lambda x: f"{x:,.2f}")
            st.dataframe(display, use_container_width=True, hide_index=True)

        with tab_a:
            render_holdings_table("Asset")
        with tab_l:
            render_holdings_table("Liability")

        # --- Net Worth History ---
        snaps_df = utils.get_snapshots_df()
        if not snaps_df.empty:
            st.markdown("---")
            st.subheader("📈 Net Worth History")
            fig_nw = go.Figure()
            fig_nw.add_trace(go.Scatter(
                x=snaps_df["date"], y=snaps_df["net_worth"],
                mode="lines+markers", name="Net Worth",
                line=dict(color="#667eea", width=2.5),
                fill="tozeroy", fillcolor="rgba(102,126,234,0.12)"
            ))
            fig_nw.add_trace(go.Scatter(
                x=snaps_df["date"], y=snaps_df["total_assets"],
                mode="lines", name="Assets", line=dict(color="#2ecc71", width=1.5, dash="dot")))
            fig_nw.add_trace(go.Scatter(
                x=snaps_df["date"], y=snaps_df["total_liabilities"],
                mode="lines", name="Liabilities", line=dict(color="#e74c3c", width=1.5, dash="dot")))
            fig_nw.update_layout(height=300, yaxis_title="USD",
                                  legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                                  **chart_layout())
            st.plotly_chart(fig_nw, use_container_width=True)

# ============================================================================
# PAGE: MANAGE HOLDINGS
# ============================================================================
elif page == "💼 Manage Holdings":
    st.title("💼 Manage Holdings")
    st.markdown("Record a new asset/liability, or update current values and remove active holdings.")

    tab_add, tab_edit = st.tabs(["➕ Add New Holding", "✏️ Edit / Delete"])

    with tab_add:
        with st.form("add_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                side = st.selectbox("Type *", ["Asset", "Liability"])
            with c2:
                cats = utils.ASSET_CATEGORIES if side == "Asset" else utils.LIABILITY_CATEGORIES
                category = st.selectbox("Category *", cats)
        
            c_name, c_tick = st.columns([2, 1])
            with c_name:
                name = st.text_input("Name / Identifier *", placeholder="e.g. Apple Inc., Gold Bars")
            with c_tick:
                ticker = st.text_input("Ticker", placeholder="AAPL, GLD", help="Optional. Used for Live Prices.")
        
            c3, c4, c5, c6 = st.columns(4)
            with c3:
                currency = st.selectbox("Currency", utils.CURRENCIES)
            with c4:
                quantity = st.number_input("Quantity", min_value=0.0001, value=1.0, format="%.4f")
            with c5:
                cost_label = "Cost Basis *" if side == "Asset" else "Original Principal *"
                cost_basis = st.number_input(cost_label, min_value=0.01, step=1000.0, format="%.2f")
            with c6:
                val_label = "Current Value *" if side == "Asset" else "Outstanding Balance *"
                current_value = st.number_input(val_label, min_value=0.0, step=1000.0, format="%.2f")
        
            notes = st.text_area("Notes", placeholder="Optional notes")
        
            submitted = st.form_submit_button("💾 Add to Portfolio", use_container_width=True)
            if submitted:
                if not name.strip():
                    st.error("Please enter a name.")
                elif cost_basis <= 0:
                    st.error("Cost basis must be greater than zero.")
                else:
                    st.session_state.portfolio = utils.add_holding(
                        st.session_state.portfolio, name.strip(), side, category,
                        quantity, cost_basis, current_value, notes, currency, ticker=ticker.strip()
                    )
                    save()
                    st.success(f"✅ {'📈' if side == 'Asset' else '📉'} **{name}** added successfully!")
        
        st.markdown("---")
        s = utils.calculate_portfolio_summary(st.session_state.portfolio)
        c1, c2, c3 = st.columns(3)
        c1.metric("Net Worth", fmt(s["net_worth"]))
        c2.metric("Total Assets", fmt(s["total_assets"]))
        c3.metric("Total Liabilities", fmt(s["total_liabilities"]))

    with tab_edit:
        if st.session_state.portfolio.empty:
            st.info("No holdings to edit.")
        else:
            df = st.session_state.portfolio
            side_filter = st.radio("Show", ["All", "Assets", "Liabilities"], horizontal=True)
            search = st.text_input("🔍 Search by name", placeholder="Type to filter...")
        
            if side_filter == "Assets":
                df = df[df["Side"] == "Asset"]
            elif side_filter == "Liabilities":
                df = df[df["Side"] == "Liability"]
            if search:
                df = df[df["Name"].str.contains(search, case=False, na=False)]
        
            if df.empty:
                st.info("No holdings match your filter.")
            else:
                for _, row in df.iterrows():
                    with st.expander(f"{'📈' if row['Side']=='Asset' else '📉'} {row['Name']} — {row['Category']} ({row.get('Currency','USD')})"):
                        col1, col2, col3 = st.columns([2, 2, 1])
                        with col1:
                            new_val = st.number_input(
                                f"Current Value ({row.get('Currency','USD')})",
                                value=float(row["Current Value"]),
                                step=100.0,
                                key=f"val_{row['id']}"
                            )
                            new_ticker = st.text_input("Ticker", value=str(row.get("Ticker", "")), key=f"tick_{row['id']}")
                        with col2:
                            new_cost = st.number_input(
                                f"Cost Basis ({row.get('Currency','USD')})",
                                value=float(row["Cost Basis"]),
                                step=100.0,
                                key=f"cost_{row['id']}"
                            )
                            new_notes = st.text_input("Notes", value=str(row["Notes"]), key=f"notes_{row['id']}")
                        with col3:
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("💾 Save", key=f"save_{row['id']}", use_container_width=True):
                                st.session_state.portfolio = utils.update_holding(
                                    st.session_state.portfolio, row["id"],
                                    current_value=new_val,
                                    cost_basis=new_cost,
                                    notes=new_notes,
                                    Ticker=new_ticker.strip()
                                )
                                save()
                                st.success("Updated!")
                                st.rerun()
                            st.markdown("<br>", unsafe_allow_html=True)
                            if st.button("🗑️ Delete", key=f"del_{row['id']}", use_container_width=True, type="primary"):
                                st.session_state.portfolio = utils.delete_holding(
                                    st.session_state.portfolio, row["id"])
                                save()
                                st.success("Deleted!")
                                st.rerun()

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
elif page == "📊 Analytics":
    st.title("📊 Portfolio Analytics")

    if st.session_state.portfolio.empty:
        st.warning("No data. Add holdings first.")
    else:
        tab1, tab2, tab3, tab4 = st.tabs(["🏛️ Overview", "📈 Assets", "📉 Liabilities", "⏱️ XIRR"])

        # OVERVIEW
        with tab1:
            summary = utils.calculate_portfolio_summary(st.session_state.portfolio)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Worth", fmt(summary["net_worth"]))
            c2.metric("Total Assets", fmt(summary["total_assets"]))
            c3.metric("Total Liabilities", fmt(summary["total_liabilities"]))
            c4.metric("Return on Assets", f"{summary['return_pct']:+.1f}%")
            st.markdown("---")

            # Waterfall
            alloc_a = utils.get_allocation_summary(st.session_state.portfolio, "Asset")
            alloc_l = utils.get_allocation_summary(st.session_state.portfolio, "Liability")
            labels, values, measures = [], [], []
            for _, r in alloc_a.iterrows():
                labels.append(r["Category"]); values.append(r["Current Value"]); measures.append("relative")
            for _, r in alloc_l.iterrows():
                labels.append(r["Category"]); values.append(-r["Current Value"]); measures.append("relative")
            labels.append("Net Worth"); values.append(summary["net_worth"]); measures.append("total")

            fig_wf = go.Figure(go.Waterfall(
                x=labels, y=values, measure=measures,
                increasing=dict(marker_color="#667eea"),
                decreasing=dict(marker_color="#e74c3c"),
                totals=dict(marker_color="#2ecc71"),
                text=[fmt(abs(v)) for v in values], textposition="outside",
                connector=dict(line=dict(color="rgba(102,126,234,0.3)", width=1))
            ))
            fig_wf.update_layout(title="Net Worth Waterfall", height=420,
                                  xaxis_tickangle=-45, showlegend=False,
                                  **chart_layout(margin=dict(t=50, b=120, l=20, r=20)))
            st.plotly_chart(fig_wf, use_container_width=True)

        # ASSETS
        with tab2:
            perf = utils.get_holding_performance(st.session_state.portfolio, "Asset")
            if perf.empty:
                st.info("No asset data.")
            else:
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in perf["Gain/Loss"]]
                fig = go.Figure(go.Bar(
                    x=perf["Name"], y=perf["Gain/Loss"],
                    marker_color=colors,
                    text=perf["Gain/Loss"].apply(lambda v: fmt(v)),
                    textposition="outside"
                ))
                fig.update_layout(title="Unrealised Gain/Loss per Asset (USD)",
                                   height=450, xaxis_tickangle=-45,
                                   **chart_layout(margin=dict(t=50, b=140, l=20, r=20)))
                st.plotly_chart(fig, use_container_width=True)

                alloc = utils.get_allocation_summary(st.session_state.portfolio, "Asset")
                fig2 = px.bar(alloc, x="Category", y="Current Value",
                              color="Return %", color_continuous_scale="RdYlGn",
                              color_continuous_midpoint=0,
                              text=alloc["Current Value"].apply(fmt))
                fig2.update_layout(title="Asset Value by Category",
                                    height=420, xaxis_tickangle=-45,
                                    **chart_layout(margin=dict(t=50, b=100, l=20, r=20)))
                fig2.update_traces(textposition="outside")
                st.plotly_chart(fig2, use_container_width=True)

        # LIABILITIES
        with tab3:
            liab_perf = utils.get_holding_performance(st.session_state.portfolio, "Liability")
            if liab_perf.empty:
                st.info("No liabilities. Debt-free! 🎉")
            else:
                liab_alloc = utils.get_allocation_summary(st.session_state.portfolio, "Liability")
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.bar(liab_alloc, x="Category", y="Current Value",
                                 color="Category",
                                 color_discrete_sequence=px.colors.sequential.Reds,
                                 text=liab_alloc["Current Value"].apply(fmt))
                    fig.update_layout(title="Outstanding by Category", height=420,
                                       showlegend=False,
                                       **chart_layout(margin=dict(t=50, b=80, l=20, r=20)))
                    fig.update_traces(textposition="outside")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig2 = go.Figure([
                        go.Bar(name="Principal", x=liab_perf["Name"],
                               y=liab_perf["Cost Basis USD"], marker_color="rgba(231,76,60,0.35)"),
                        go.Bar(name="Outstanding", x=liab_perf["Name"],
                               y=liab_perf["Current Value USD"], marker_color="#e74c3c"),
                    ])
                    fig2.update_layout(title="Principal vs Outstanding", barmode="group",
                                        height=420, xaxis_tickangle=-40,
                                        legend=dict(orientation="h", y=-0.3, x=0.5, xanchor="center"),
                                        **chart_layout(margin=dict(t=50, b=130, l=20, r=20)))
                    st.plotly_chart(fig2, use_container_width=True)

        # XIRR
        with tab4:
            st.subheader("⏱️ Annualised Return (XIRR)")
            st.markdown("Annualised return per asset, accounting for how long it has been held.")
            assets_df = st.session_state.portfolio[st.session_state.portfolio["Side"] == "Asset"].copy()
            if assets_df.empty:
                st.info("No assets.")
            else:
                assets_df["XIRR %"] = assets_df.apply(
                    lambda r: utils.compute_xirr(float(r["Cost Basis"]), float(r["Current Value"]), r["Date Added"]),
                    axis=1
                )
                assets_df["Held Since"] = assets_df["Date Added"]
                assets_df["Years Held"] = assets_df["Date Added"].apply(
                    lambda d: round((datetime.now() - pd.to_datetime(d)).days / 365.25, 1) if d else 0
                )
                display = assets_df[["Name", "Category", "Held Since", "Years Held",
                                      "Cost Basis", "Current Value", "XIRR %"]].copy()
                for c in ["Cost Basis", "Current Value"]:
                    display[c] = display[c].apply(lambda x: f"${float(x):,.2f}")
                display["XIRR %"] = display["XIRR %"].apply(lambda x: f"{x:+.2f}%")
                st.dataframe(display, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: RISK & FORECASTING
# ============================================================================
elif page == "🔮 Risk & Forecasting":
    st.title("🔮 Advanced Risk & Forecasting")
    st.markdown("Perform predictive analytics, analyze public market volatility, and forecast via Monte Carlo.")

    if st.session_state.portfolio.empty:
        st.warning("No data to analyze.")
    else:
        tab1, tab2 = st.tabs(["📉 Risk Metrics (Public Equities)", "📈 Monte Carlo Simulation"])
        
        with tab1:
            st.subheader("Public Equity Risk Profiler")
            st.markdown("Retrieves 1-year historical pricing to compute Beta vs S&P 500, Volatility, Sharpe Ratio, and VaR.")
            if st.button("🔄 Compute Risk Metrics (via yfinance)"):
                with st.spinner("Downloading historical data & computing covariances..."):
                    metrics, corr = analytics_engine.get_risk_metrics(st.session_state.portfolio)
                    if metrics.empty:
                        st.error("Not enough valid public equity tickers found (e.g. AAPL, TSLA) to run risk metrics. Please ensure you've added valid Tickers to your holdings.")
                    else:
                        st.session_state._risk_metrics = metrics
                        st.session_state._risk_corr = corr
            
            if getattr(st.session_state, "_risk_metrics", None) is not None:
                metrics = st.session_state._risk_metrics
                st.dataframe(metrics, use_container_width=True, hide_index=True)
                
                st.markdown("---")
                st.subheader("🔗 Asset Correlation Heatmap")
                corr = st.session_state._risk_corr
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                     title="Pearson Correlation (1 Year)")
                fig_corr.update_layout(**chart_layout())
                st.plotly_chart(fig_corr, use_container_width=True)

        with tab2:
            st.subheader("Monte Carlo Net Worth Projection")
            st.markdown("Run 10,000 geometric brownian motion pathways targeting the next 5-20 years.")
            
            c1, c2, c3 = st.columns(3)
            with c1:
                summary = utils.calculate_portfolio_summary(st.session_state.portfolio)
                nw = summary.get("net_worth", 0)
                st.metric("Current Net Worth", fmt(nw))
            with c2:
                mu = st.number_input("Expected Blended Return (%)", value=8.5, step=0.5) / 100
            with c3:
                vol = st.number_input("Expected Blended Volatility (%)", value=12.0, step=0.5) / 100
                
            years = st.slider("Forecast Horizon (Years)", min_value=5, max_value=30, value=10, step=1)
            
            if nw <= 0:
                st.error("Net worth must be positive to run the simulation.")
            else:
                if st.button("🚀 Run Monte Carlo", type="primary"):
                    with st.spinner("Simulating 10,000 parallel realities..."):
                        paths = analytics_engine.run_monte_carlo(nw, mu, vol, years=years, sim_count=10000)
                        
                        # Calculate percentiles for charting
                        percentiles = [5, 25, 50, 75, 95]
                        percentile_df = pd.DataFrame(index=paths.index)
                        
                        for p in percentiles:
                            percentile_df[f"{p}th Percentile"] = np.percentile(paths, p, axis=1)
                            
                        # Plot
                        fig_mc = go.Figure()
                        
                        colors = {5: "rgba(231,76,60,0.4)", 25: "rgba(243,156,18,0.6)", 
                                  50: "#2ecc71", 75: "rgba(52,152,219,0.6)", 95: "rgba(155,89,182,0.4)"}
                        
                        for p in percentiles:
                            fig_mc.add_trace(go.Scatter(
                                x=percentile_df.index, y=percentile_df[f"{p}th Percentile"],
                                mode="lines", name=f"{p}th Pct",
                                line=dict(color=colors[p], width=3 if p==50 else 1.5, dash="solid" if p==50 else "dot")
                            ))
                            
                        fig_mc.update_layout(title="Monte Carlo Projected Net Worth Distribution",
                                             xaxis_title="Years from Now", yaxis_title="USD",
                                             height=500, **chart_layout())
                        st.plotly_chart(fig_mc, use_container_width=True)
                        
                        final_vals = paths.iloc[-1]
                        st.markdown("### 📊 Outcomes at Year " + str(years))
                        m1, m2, m3, m4 = st.columns(4)
                        m1.metric("Bear Market (5th pct)", fmt(np.percentile(final_vals, 5)))
                        m2.metric("Base Case (Median)", fmt(np.median(final_vals)))
                        m3.metric("Bull Market (95th pct)", fmt(np.percentile(final_vals, 95)))
                        prob_dbl = (final_vals > nw * 2).sum() / 10000 * 100
                        m4.metric("Prob. of Doubling NW", f"{prob_dbl:.1f}%")

# ============================================================================
# PAGE: REBALANCING
# ============================================================================
elif page == "⚖️ Rebalancing":
    st.title("⚖️ Target Allocation & Rebalancing")
    st.markdown("Set target weightings for your assets and generate a rebalancing plan.")

    targets = target_allocation.load_targets()
    
    with st.expander("⚙️ Edit Target Allocations", expanded=False):
        with st.form("targets_form"):
            st.markdown("Ensure your total allocation equals 100%.")
            updated_targets = {}
            cats = utils.ASSET_CATEGORIES
            cols = st.columns(3)
            for i, cat in enumerate(cats):
                col = cols[i % 3]
                default_val = targets.get(cat, 0.0)
                updated_targets[cat] = col.number_input(f"{cat} (%)", min_value=0.0, max_value=100.0, value=float(default_val), step=1.0)
            
            if st.form_submit_button("Save Targets"):
                valid, msg = target_allocation.validate_targets(updated_targets)
                if not valid:
                    st.error(msg)
                else:
                    target_allocation.save_targets(updated_targets)
                    st.success("Targets updated!")
                    st.rerun()

    valid, msg = target_allocation.validate_targets(targets)
    if not valid:
        st.warning(f"⚠️ Your targets do not sum to 100%. {msg} Edit targets above.")
    
    rebal_df = target_allocation.compute_rebalancing(st.session_state.portfolio, targets)
    
    if rebal_df.empty:
        st.info("Not enough asset data to calculate rebalancing.")
    else:
        st.subheader("📊 Portfolio Drift Analysis")
        st.dataframe(rebal_df, use_container_width=True, hide_index=True)
        
        buys, sells = target_allocation.get_rebalancing_trades(rebal_df)
        
        st.markdown("---")
        st.subheader("🛠️ Recommended Actions")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📉 Suggested Buys")
            if not buys:
                st.success("No assets need increasing.")
            for b in buys:
                st.info(f"**Increase {b['Category']}**\n\nTarget: {b['Target %']}% | Current: {b['Current %']}%\n\nAction: **Buy ${abs(b['Dollar Drift ($)']):,.0f}**", icon="🔼")
                
        with c2:
            st.markdown("#### 📈 Suggested Sells")
            if not sells:
                st.success("No assets need trimming.")
            for s in sells:
                st.warning(f"**Reduce {s['Category']}**\n\nTarget: {s['Target %']}% | Current: {s['Current %']}%\n\nAction: **Sell ${s['Dollar Drift ($)']:,.0f}**", icon="🔽")

# ============================================================================
# PAGE: REPORTS & KPIs
# ============================================================================
elif page == "📑 Reports & KPIs":
    st.title("📑 KPI Reports")
    st.markdown("A consolidated view of all key performance indicators.")

    if st.session_state.portfolio.empty:
        st.warning("No data. Add holdings first.")
    else:
        summary = utils.calculate_portfolio_summary(st.session_state.portfolio)
        leverage = (summary["total_liabilities"] / summary["net_worth"]) if summary["net_worth"] > 0 else 0
        debt_to_assets = (summary["total_liabilities"] / summary["total_assets"]) if summary["total_assets"] > 0 else 0

        # Core KPIs row 1
        st.subheader("1. Core Financial Metrics")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Gross Net Worth", fmt(summary["net_worth"]))
        k2.metric("Total Capital Deployed", fmt(summary["total_invested"]))
        k3.metric("Unrealised Return", fmt(summary["total_return"]),
                  delta=f"{summary['return_pct']:+.1f}%")
        k4.metric("Leverage Ratio", f"{leverage:.2f}x")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("Total Assets (USD)", fmt(summary["total_assets"]))
        k6.metric("Total Liabilities (USD)", fmt(summary["total_liabilities"]))
        k7.metric("Debt-to-Assets", f"{debt_to_assets*100:.1f}%")
        holdings_count = len(st.session_state.portfolio)
        asset_count = len(st.session_state.portfolio[st.session_state.portfolio["Side"] == "Asset"])
        k8.metric("Total Holdings", f"{holdings_count} ({asset_count} assets)")

        # Asset class table
        st.markdown("---")
        st.subheader("2. Asset Class Performance")
        alloc_a = utils.get_allocation_summary(st.session_state.portfolio, "Asset")
        if not alloc_a.empty:
            total_av = alloc_a["Current Value"].sum()
            alloc_a["Allocation %"] = (alloc_a["Current Value"] / total_av * 100).round(1)
            rpt = alloc_a.rename(columns={"Current Value": "Market Value (USD)",
                                           "Cost Basis": "Capital Deployed (USD)",
                                           "Holdings": "Count"}).copy()
            for c in ["Market Value (USD)", "Capital Deployed (USD)", "Gain/Loss"]:
                rpt[c] = rpt[c].apply(lambda x: f"${x:,.0f}")
            rpt["Return %"] = rpt["Return %"].apply(lambda x: f"{x:+.2f}%")
            rpt["Allocation %"] = rpt["Allocation %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(rpt, use_container_width=True, hide_index=True)

        # Liabilities table
        st.markdown("---")
        st.subheader("3. Liability Summary")
        alloc_l = utils.get_allocation_summary(st.session_state.portfolio, "Liability")
        if not alloc_l.empty:
            lrpt = alloc_l.rename(columns={"Current Value": "Outstanding (USD)",
                                            "Cost Basis": "Original Principal (USD)",
                                            "Holdings": "Count"})[
                ["Category", "Count", "Outstanding (USD)", "Original Principal (USD)"]].copy()
            for c in ["Outstanding (USD)", "Original Principal (USD)"]:
                lrpt[c] = lrpt[c].apply(lambda x: f"${x:,.0f}")
            st.dataframe(lrpt, use_container_width=True, hide_index=True)
        else:
            st.success("🎉 No active liabilities.")

        # Net Worth History
        snaps_df = utils.get_snapshots_df()
        if not snaps_df.empty:
            st.markdown("---")
            st.subheader("4. Net Worth Over Time")
            fig_nw = go.Figure()
            fig_nw.add_trace(go.Scatter(x=snaps_df["date"], y=snaps_df["net_worth"],
                                         mode="lines+markers", name="Net Worth",
                                         line=dict(color="#667eea", width=2.5),
                                         fill="tozeroy", fillcolor="rgba(102,126,234,0.12)"))
            fig_nw.update_layout(height=320, yaxis_title="USD",
                                  **chart_layout(margin=dict(t=30, b=30, l=20, r=20)))
            st.plotly_chart(fig_nw, use_container_width=True)

        st.markdown("---")
        st.button("🖨️ Print Report", help="Press Ctrl+P in your browser to print this page.")

# ============================================================================
# PAGE: AI ADVISORY
# ============================================================================
elif page == "🧠 AI Advisory":
    st.title("🧠 AI Portfolio Advisor")
    st.markdown("Live news-powered analysis and recommendations for your portfolio.")

    if st.session_state.portfolio.empty:
        st.warning("Portfolio is empty. Add holdings to receive advice.")
    else:
        summary = utils.calculate_portfolio_summary(st.session_state.portfolio)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Net Worth", fmt(summary["net_worth"]))
        c2.metric("Total Assets", fmt(summary["total_assets"]))
        c3.metric("Total Liabilities", fmt(summary["total_liabilities"]))
        cash_df = st.session_state.portfolio[
            (st.session_state.portfolio["Side"] == "Asset") &
            (st.session_state.portfolio["Category"] == "Cash & Equivalents")
        ]
        total_cash = cash_df["Current Value"].astype(float).sum() if not cash_df.empty else 0.0
        c4.metric("💵 Available Cash", fmt(total_cash))

        st.markdown("---")
        if st.button("🚀 Generate Advisory Report", use_container_width=True, type="primary"):
            with st.spinner("🧠 Fetching live news and analysing portfolio..."):
                headlines = ai_advisor.fetch_news_headlines(5)
                report = ai_advisor.generate_advisory_report(st.session_state.portfolio, headlines)
                st.session_state.advisory_report = report
                st.session_state.advisory_headlines = headlines

        if st.session_state.advisory_report:
            st.markdown("---")
            st.subheader("📋 Advisory Report")
            st.markdown(st.session_state.advisory_report)

            # Concentration Heatmap
            st.markdown("---")
            st.subheader("🎯 Concentration Risk Heatmap")
            alloc = utils.get_allocation_summary(st.session_state.portfolio, "Asset")
            if not alloc.empty:
                total_v = alloc["Current Value"].sum()
                alloc["Pct"] = (alloc["Current Value"] / total_v * 100).round(1)
                fig_heat = go.Figure(go.Bar(
                    x=alloc["Category"], y=alloc["Pct"],
                    marker_color=["#e74c3c" if p > 30 else "#f39c12" if p > 20 else "#2ecc71"
                                  for p in alloc["Pct"]],
                    text=alloc["Pct"].apply(lambda x: f"{x:.1f}%"),
                    textposition="outside",
                ))
                fig_heat.add_hline(y=30, line_dash="dash", line_color="red",
                                    annotation_text="Concentration Limit (30%)")
                fig_heat.add_hline(y=20, line_dash="dot", line_color="orange",
                                    annotation_text="Watch Level (20%)")
                fig_heat.update_layout(
                    height=370, xaxis_tickangle=-35, yaxis_title="% of Total Assets",
                    showlegend=False,
                    **chart_layout(margin=dict(t=50, b=100, l=20, r=20))
                )
                st.plotly_chart(fig_heat, use_container_width=True)
                st.caption("🟢 Healthy (<20%)  🟡 Watch (20-30%)  🔴 Overconcentrated (>30%)")

# ============================================================================
# PAGE: DATA MANAGEMENT
# ============================================================================
elif page == "💾 Data Management":
    st.title("💾 Data Management")
    st.markdown("Export, import, or reset your portfolio data.")

    tab1, tab2, tab3 = st.tabs(["📥 Export", "📤 Import", "🗑️ Reset"])

    with tab1:
        st.subheader("Export Portfolio")
        if st.session_state.portfolio.empty:
            st.info("Nothing to export.")
        else:
            st.dataframe(st.session_state.portfolio.head(10), use_container_width=True, hide_index=True)
            csv = st.session_state.portfolio.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                data=csv,
                file_name=f"family_office_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        # Snapshot export
        snaps = utils.get_snapshots_df()
        if not snaps.empty:
            st.markdown("---")
            st.subheader("Export Snapshot History")
            st.download_button(
                "⬇️ Download Snapshots CSV",
                data=snaps.to_csv(index=False),
                file_name=f"snapshots_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

    with tab2:
        st.subheader("Import Portfolio")
        st.warning("⚠️ Importing will replace your current portfolio.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                imp = pd.read_csv(uploaded)
                required = ["Name", "Side", "Category", "Quantity", "Cost Basis", "Current Value"]
                if all(c in imp.columns for c in required):
                    if "id" not in imp.columns:
                        imp.insert(0, "id", range(1, len(imp) + 1))
                    st.dataframe(imp.head(), use_container_width=True, hide_index=True)
                    if st.button("✅ Confirm Import", use_container_width=True):
                        st.session_state.portfolio = imp
                        save()
                        st.success(f"Imported {len(imp)} holdings!")
                        st.rerun()
                else:
                    st.error(f"Missing columns. Required: {', '.join(required)}")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab3:
        st.subheader("Reset Portfolio")
        if st.session_state.portfolio.empty:
            st.info("Already empty.")
        else:
            st.warning(f"⚠️ {len(st.session_state.portfolio)} holdings will be permanently deleted.")
            if st.checkbox("I understand this is irreversible"):
                if st.button("🗑️ Delete All", type="primary", use_container_width=True):
                    st.session_state.portfolio = utils.create_empty_dataframe()
                    save()
                    st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#666;padding:16px;font-size:12px;'>"
    "🏛️ Family Office Portfolio Dashboard &nbsp;|&nbsp; "
    "Assets · Liabilities · Net Worth · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True
)