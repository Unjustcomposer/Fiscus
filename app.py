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
if "base_currency" not in st.session_state:
    st.session_state.base_currency = "INR"

# ============================================================================
# HELPERS
# ============================================================================
def fmt(val):
    try:
        val = float(val)
    except Exception:
        return str(val)
    base_curr = st.session_state.get("base_currency", "USD")
    sym = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "AED": "د.إ"}.get(base_curr, "$")
    
    prefix = f"-{sym}" if val < 0 else sym
    v = abs(val)
    
    if base_curr == "INR":
        # Indian numbering system
        s = f"{v:,.2f}"
        parts = s.split(".")
        main = parts[0].replace(",", "")
        if len(main) <= 3:
            res = main
        else:
            res = main[-3:]
            main = main[:-3]
            while len(main) > 0:
                res = main[-2:] + "," + res
                main = main[:-2]
        return f"{prefix}{res}.{parts[1]}"
    else:
        # Western system
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
         "📊 Analytics", "🧠 Intelligence Hub", "⚖️ Rebalancing", "📑 Reports & KPIs", "🧠 AI Advisory",
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
        with st.spinner("Fetching market data and live FX rates..."):
            updated_df, log = market_data.update_portfolio_prices(st.session_state.portfolio)
            fx_log = market_data.update_fx_rates()
            st.session_state.portfolio = updated_df
            save()
            updated_count = len([x for x in log if 'error' not in x])
            msg = []
            if updated_count > 0:
                msg.append(f"Updated {updated_count} holdings.")
            if fx_log:
                msg.append(f"Updated {len(fx_log)} FX rates.")
            if msg:
                st.session_state._price_log = "✅ " + " | ".join(msg)
            else:
                st.session_state._price_log = "ℹ️ No prices updated. (Check tickers)"
            st.rerun()

    if getattr(st.session_state, '_price_log', None):
        st.caption(st.session_state._price_log)
        st.session_state._price_log = None

    st.markdown("---")
    st.session_state.base_currency = st.selectbox("Base Currency", utils.CURRENCIES, index=utils.CURRENCIES.index(st.session_state.get("base_currency", "INR")))
    
    st.markdown("---")
    st.markdown("#### 🛠️ View Controls")
    st.session_state.top_n = st.slider("Show Top Drivers", min_value=3, max_value=20, value=st.session_state.get("top_n", 5))
    
    all_cats = sorted(utils.ASSET_CATEGORIES + utils.LIABILITY_CATEGORIES)
    st.session_state.cat_filter = st.multiselect("Filter by Category", all_cats, default=[])

    summary_side = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
    st.metric("Net Worth", fmt(summary_side["net_worth"]))
    st.metric("Total Assets", fmt(summary_side["total_assets"]))
    st.metric("Total Liabilities", fmt(summary_side["total_liabilities"]))
    st.caption(f"All values displayed in {st.session_state.base_currency}.")

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
if page == "🏠 Dashboard":
    st.title("🏠 Portfolio Dashboard")

    summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
    
    # Apply category filter if set
    df_filtered = st.session_state.portfolio.copy()
    if st.session_state.get("cat_filter"):
        df_filtered = df_filtered[df_filtered["Category"].isin(st.session_state.cat_filter)]
        summary = utils.calculate_portfolio_summary(df_filtered, st.session_state.base_currency)

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

        # --- LEFT: Asset Treemap (Square Chart) ---
        with col_l:
            st.subheader("Asset Allocation (Treemap)")
            assets_df = df_filtered[df_filtered["Side"] == "Asset"].copy()
            if not assets_df.empty:
                assets_df["Root"] = "Assets"
                assets_df = utils.normalize_to_base(assets_df, st.session_state.base_currency)
                fig = px.treemap(
                    assets_df, path=["Root", "Category", "Name"],
                    values="Current Value Base",
                    color="Current Value Base",
                    color_continuous_scale=["#1a1a2e", "#667eea", "#764ba2"],
                )
                fig.update_layout(height=400, coloraxis_showscale=False, **chart_layout())
                st.plotly_chart(fig, use_container_width=True)
                st.caption("✨ Size indicates relative market value.")

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
            df_side = df_filtered[df_filtered["Side"] == side].copy()
            if df_side.empty:
                st.info(f"No {side.lower()}s recorded.")
                return
            df_usd = utils.normalize_to_base(df_side, st.session_state.base_currency)
            df_usd["Gain/Loss (Base)"] = (df_usd["Current Value Base"] - df_usd["Cost Basis Base"]).round(2)
            df_usd["Return %"] = ((df_usd["Gain/Loss (Base)"] / df_usd["Cost Basis Base"]) * 100).round(1)
            
            # Sort by Current Value Base Descending
            df_usd = df_usd.sort_values("Current Value Base", ascending=False)
            top_n = st.session_state.get("top_n", 5)
            
            display = df_usd[["Name", "Category", "Currency", "Cost Basis", "Current Value",
                               "Gain/Loss (Base)", "Return %", "Date Added", "Notes"]].head(top_n).copy()
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
            st.subheader("📈 Net Worth Growth History")
            fig_nw = go.Figure()
            
            # Use smoother lines and area fill
            fig_nw.add_trace(go.Scatter(
                x=snaps_df["date"], y=snaps_df["net_worth"],
                mode="lines+markers", name="Net Worth",
                line=dict(color="#667eea", width=4, shape="spline"),
                fill="tozeroy", fillcolor="rgba(102,126,234,0.15)"
            ))
            
            fig_nw.add_trace(go.Scatter(
                x=snaps_df["date"], y=snaps_df["total_assets"],
                mode="lines", name="Total Assets", 
                line=dict(color="#2ecc71", width=2, dash="dot", shape="spline")))
            
            fig_nw.add_trace(go.Scatter(
                x=snaps_df["date"], y=snaps_df["total_liabilities"],
                mode="lines", name="Total Liabilities", 
                line=dict(color="#e74c3c", width=2, dash="dot", shape="spline")))
            
            fig_nw.update_layout(
                height=400, 
                yaxis_title=st.session_state.base_currency,
                hovermode="x unified",
                legend=dict(orientation="h", y=-0.25, x=0.5, xanchor="center"),
                **chart_layout(margin=dict(t=30, b=50, l=20, r=20))
            )
            st.plotly_chart(fig_nw, use_container_width=True)

# ============================================================================
# PAGE: MANAGE HOLDINGS
# ============================================================================
elif page == "💼 Manage Holdings":
    st.title("💼 Manage Holdings")
    st.markdown("Record a new asset/liability, or update current values and remove active holdings.")

    tab_add, tab_edit = st.tabs(["➕ Add New Holding", "✏️ Edit / Delete"])

    with tab_add:
        # Move type selection outside the form for reactivity
        side = st.radio("Type *", ["Asset", "Liability"], horizontal=True)
        
        with st.form("add_form", clear_on_submit=True):
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
        s = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
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
            summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Net Worth", fmt(summary["net_worth"]))
            c2.metric("Total Assets", fmt(summary["total_assets"]))
            c3.metric("Total Liabilities", fmt(summary["total_liabilities"]))
            c4.metric("Return on Assets", f"{summary['return_pct']:+.1f}%")
            st.markdown("---")

            # Waterfall
            alloc_a = utils.get_allocation_summary(st.session_state.portfolio, "Asset", st.session_state.base_currency)
            alloc_l = utils.get_allocation_summary(st.session_state.portfolio, "Liability", st.session_state.base_currency)
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
            perf = utils.get_holding_performance(st.session_state.portfolio, "Asset", st.session_state.base_currency)
            if perf.empty:
                st.info("No asset data.")
            else:
                top_n = st.session_state.get("top_n", 5)
                # Sort by absolute gain/loss for top drivers
                perf["AbsGain"] = perf["Gain/Loss"].abs()
                perf = perf.sort_values("AbsGain", ascending=False).head(top_n)
                
                colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in perf["Gain/Loss"]]
                fig = go.Figure(go.Bar(
                    x=perf["Name"], y=perf["Gain/Loss"],
                    marker_color=colors,
                    text=perf["Gain/Loss"].apply(lambda v: fmt(v)),
                    textposition="outside"
                ))
                fig.update_layout(title=f"Top {top_n} P&L Drivers (Base)",
                                   height=450, xaxis_tickangle=-45,
                                   **chart_layout(margin=dict(t=50, b=140, l=20, r=20)))
                st.plotly_chart(fig, use_container_width=True)

                alloc = utils.get_allocation_summary(st.session_state.portfolio, "Asset", st.session_state.base_currency)
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
            liab_perf = utils.get_holding_performance(st.session_state.portfolio, "Liability", st.session_state.base_currency)
            if liab_perf.empty:
                st.info("No liabilities. Debt-free! 🎉")
            else:
                top_n = st.session_state.get("top_n", 5)
                liab_perf = liab_perf.sort_values("Current Value Base", ascending=False).head(top_n)
                
                liab_alloc = utils.get_allocation_summary(st.session_state.portfolio, "Liability", st.session_state.base_currency)
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
                               y=liab_perf["Cost Basis Base"], marker_color="rgba(231,76,60,0.35)"),
                        go.Bar(name="Outstanding", x=liab_perf["Name"],
                               y=liab_perf["Current Value Base"], marker_color="#e74c3c"),
                    ])
                    fig2.update_layout(title=f"Top {top_n} Debt Drivers", barmode="group",
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
                assets_df["XIRR_raw"] = assets_df.apply(
                    lambda r: utils.compute_xirr(float(r["Cost Basis"]), float(r["Current Value"]), r["Date Added"]),
                    axis=1
                )
                assets_df["XIRR %"] = assets_df["XIRR_raw"].apply(lambda x: f"{x:+.2f}%")
                assets_df["Held Since"] = assets_df["Date Added"]
                assets_df["Years Held"] = assets_df["Date Added"].apply(
                    lambda d: round((datetime.now() - pd.to_datetime(d)).days / 365.25, 1) if d else 0
                )
                
                # Sort by XIRR raw descending
                display = assets_df.sort_values("XIRR_raw", ascending=False)[
                    ["Name", "Category", "Held Since", "Years Held", "Cost Basis", "Current Value", "XIRR %"]
                ].copy()
                
                for c in ["Cost Basis", "Current Value"]:
                    display[c] = display[c].apply(lambda x: fmt(float(x)))
                st.dataframe(display, use_container_width=True, hide_index=True)

# ============================================================================
# PAGE: INTELLIGENCE HUB
# ============================================================================
elif page == "🧠 Intelligence Hub":
    st.title("🧠 Intelligence Hub")
    st.markdown("Unified risk analytics, ensemble forecasting, sentiment analysis, and actionable intelligence — all in one place.")

    if st.session_state.portfolio.empty:
        st.warning("No data. Add holdings first.")
    else:
        hub_tabs = st.tabs([
            "🤖 Auto Analysis",
            "📊 Portfolio Risk",
            "🔮 Ensemble Forecast",
            "📈 Monte Carlo (NW)",
            "📰 Market Sentiment",
            "🔍 Anomaly Scanner",
            "💡 Intelligence Summary",
        ])

        # Get tickers
        try:
            from dl_forecaster import get_portfolio_tickers
            available_tickers = get_portfolio_tickers(st.session_state.portfolio)
        except ImportError:
            available_tickers = []

        # ── TAB 0: AUTO ANALYSIS (Smart Model Selection) ───────────────
        with hub_tabs[0]:
            st.subheader("🤖 Automatic Analysis — Smart Model Selection")
            st.markdown("""
            The system **evaluates each ticker's data profile** — volatility regime, trend strength,
            data length, tail risk — then **automatically picks the best model** and generates results.
            No manual configuration needed.
            """)
            st.markdown("---")

            if not available_tickers:
                st.info("Add holdings with ticker symbols (e.g. AAPL, TSLA, GOOGL) to enable auto-analysis.")
            else:
                st.markdown(f"**Portfolio tickers:** {', '.join(available_tickers)}")
                auto_days = st.slider("Forecast Horizon (Days)", 7, 60, 30, key="auto_days")

                if st.button("🚀 Run Auto Analysis", type="primary", use_container_width=True, key="auto_btn"):
                    from intelligence_engine import EnsembleForecaster, InsightGenerator, AutoAnalyzer
                    forecaster = EnsembleForecaster()
                    gen = InsightGenerator()
                    auto_results = []
                    auto_insights = []
                    progress = st.progress(0, text="Starting auto analysis...")

                    for idx_a, ticker in enumerate(available_tickers):
                        progress.progress((idx_a + 1) / len(available_tickers),
                                          text=f"Analyzing {ticker} — evaluating data & selecting model...")
                        result = forecaster.smart_forecast(ticker, forecast_days=auto_days)
                        auto_results.append(result)
                        insight = gen.generate_ticker_insight(result)
                        auto_insights.append(insight)

                    progress.empty()
                    st.session_state._auto_results = auto_results
                    st.session_state._auto_insights = auto_insights
                    st.session_state._auto_summary = gen.generate_portfolio_summary(auto_insights)

                if getattr(st.session_state, '_auto_results', None):
                    results = st.session_state._auto_results
                    insights = st.session_state._auto_insights

                    # Portfolio summary
                    st.markdown(st.session_state._auto_summary)

                    # Per-ticker cards with model selection reasoning
                    st.markdown("---")
                    st.subheader("🔬 Per-Ticker Analysis & Model Selection")

                    for i, (result, insight) in enumerate(zip(results, insights)):
                        if not result.get('success'):
                            st.warning(f"⚠️ **{result.get('ticker', '?')}** — {result.get('error', 'Failed')}")
                            continue

                        ticker = result['ticker']
                        con = result['consensus']
                        analysis = result.get('auto_analysis', {})
                        mode = result.get('selection_mode', 'unknown')
                        reason = result.get('selection_reason', '')
                        signal = result['signal']

                        signal_color = "#2ecc71" if "BUY" in signal else "#e74c3c" if "SELL" in signal else "#f39c12"

                        with st.expander(f"{signal}  **{ticker}** — Target ${con['price']:,.2f} ({con['return_pct']:+.1f}%)  |  _{reason}_", expanded=(i == 0)):
                            # Data Profile
                            if analysis:
                                dp1, dp2, dp3, dp4 = st.columns(4)
                                dp1.metric("Data Points", f"{analysis.get('data_points', '?')} days")
                                dp2.metric("Volatility", f"{analysis.get('ann_volatility', '?')}%",
                                           delta=analysis.get('vol_regime', '').upper())
                                dp3.metric("Trend Strength", f"{analysis.get('trend_strength', '?')}")
                                dp4.metric("Kurtosis", f"{analysis.get('kurtosis', '?')}",
                                           delta="Fat Tails" if analysis.get('has_fat_tails') else "Normal")

                                # Model score comparison
                                st.markdown("**Model Suitability Scores:**")
                                model_scores = analysis.get('model_scores', {})
                                best = analysis.get('best_model', '')
                                score_cols = st.columns(3)
                                for j, (mname, mscore) in enumerate(model_scores.items()):
                                    icon = {"lstm": "🧠", "monte_carlo": "🎲", "exp_smoothing": "📈"}.get(mname, "📊")
                                    mlabel = {"lstm": "LSTM", "monte_carlo": "Monte Carlo", "exp_smoothing": "Exp. Smoothing"}.get(mname, mname)
                                    is_best = mname == best
                                    badge = " ✅ SELECTED" if is_best else ""
                                    score_cols[j].metric(f"{icon} {mlabel}{badge}", f"{mscore:.1f}")

                                # Selection reasons
                                reasons = analysis.get('reasons', [])
                                if reasons:
                                    st.markdown("**Why this model:**")
                                    for r_txt in reasons:
                                        st.markdown(f"  - {r_txt}")

                            st.markdown("---")

                            # Results
                            if result['models']:
                                for mname, mdata in result['models'].items():
                                    mlabel = {"lstm": "🧠 LSTM", "monte_carlo": "🎲 Monte Carlo", "exp_smoothing": "📈 Exp. Smoothing"}.get(mname, mname)
                                    r1, r2, r3 = st.columns(3)
                                    r1.metric(f"{mlabel} Target", f"${mdata['end_price']:,.2f}")
                                    if 'mape' in mdata:
                                        r2.metric("Accuracy (MAPE)", f"{mdata['mape']:.1f}%")
                                    if 'prob_up' in mdata:
                                        r2.metric("Prob. of Gain", f"{mdata['prob_up']:.0f}%")
                                    if 'p5' in mdata and 'p95' in mdata:
                                        r3.metric("90% Range", f"${mdata['p5']:,.2f} – ${mdata['p95']:,.2f}")
                                    elif 'rmse' in mdata:
                                        r3.metric("RMSE", f"${mdata['rmse']:,.2f}")

                            # Insights
                            st.markdown("**Insights:**")
                            for line in insight['insights']:
                                st.markdown(f"- {line}")

        # ── TAB 1: PORTFOLIO RISK DASHBOARD ────────────────────────────
        with hub_tabs[1]:
            st.subheader("📊 Portfolio Risk Dashboard")
            st.markdown("Comprehensive risk profiling: Beta, Sharpe, Sortino, VaR, CVaR, Max Drawdown, and composite Risk Score.")

            if st.button("🔄 Compute Risk Metrics", key="risk_btn"):
                with st.spinner("Downloading 1-year data & computing risk metrics..."):
                    metrics, corr = analytics_engine.get_risk_metrics(st.session_state.portfolio)
                    if metrics.empty:
                        st.error("No valid tickers found. Add holdings with ticker symbols.")
                    else:
                        st.session_state._risk_metrics = metrics
                        st.session_state._risk_corr = corr

            if getattr(st.session_state, "_risk_metrics", None) is not None:
                metrics = st.session_state._risk_metrics.copy()

                # Risk score gauges
                st.markdown("---")
                st.subheader("⚡ Risk Scores")
                risk_cols = st.columns(min(len(metrics), 6))
                for i, (_, row) in enumerate(metrics.iterrows()):
                    col = risk_cols[i % len(risk_cols)]
                    score = row["Risk Score"]
                    color = "#2ecc71" if score < 30 else "#f39c12" if score < 60 else "#e74c3c"
                    label = "Low" if score < 30 else "Moderate" if score < 60 else "High"
                    col.markdown(
                        f"<div style='text-align:center;padding:12px;border-radius:12px;"
                        f"border:2px solid {color};background:rgba({','.join(str(int(color.lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.1)'>"
                        f"<div style='font-size:0.8rem;opacity:0.7'>{row['Ticker']}</div>"
                        f"<div style='font-size:2rem;font-weight:700;color:{color}'>{score}</div>"
                        f"<div style='font-size:0.75rem;color:{color}'>{label} Risk</div></div>",
                        unsafe_allow_html=True
                    )

                # Full metrics table
                st.markdown("---")
                st.subheader("📋 Detailed Risk Metrics")
                display_metrics = metrics.sort_values("Risk Score", ascending=False)
                st.dataframe(display_metrics, use_container_width=True, hide_index=True)

                # Correlation heatmap
                st.markdown("---")
                st.subheader("🔗 Asset Correlation Heatmap")
                corr = st.session_state._risk_corr
                fig_corr = px.imshow(corr, text_auto=True, aspect="auto",
                                     color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                     title="Pearson Correlation (1 Year)")
                fig_corr.update_layout(**chart_layout())
                st.plotly_chart(fig_corr, use_container_width=True)

                # Risk vs Return chart
                st.markdown("---")
                st.subheader("📊 Risk vs Return")
                fig_rr = go.Figure()
                risk_colors = ["#2ecc71" if s < 30 else "#f39c12" if s < 60 else "#e74c3c" for s in metrics["Risk Score"]]
                fig_rr.add_trace(go.Scatter(
                    x=metrics["Volatility (%)"], y=metrics["Expected Return (%)"],
                    mode="markers+text", text=metrics["Ticker"], textposition="top center",
                    marker=dict(size=metrics["Risk Score"].clip(15, 60), color=risk_colors,
                                line=dict(width=2, color="white")),
                    hovertemplate="<b>%{text}</b><br>Vol: %{x:.1f}%<br>Return: %{y:.1f}%<extra></extra>"
                ))
                fig_rr.update_layout(title="Risk-Return Scatter", xaxis_title="Annualized Volatility (%)",
                                     yaxis_title="Expected Return (%)", height=450, **chart_layout())
                st.plotly_chart(fig_rr, use_container_width=True)
                st.caption("Bubble size = Risk Score. 🟢 Low Risk · 🟡 Moderate · 🔴 High Risk")

        # ── TAB 2: ENSEMBLE FORECAST ───────────────────────────────────
        with hub_tabs[2]:
            st.subheader("🔮 Ensemble Forecast — Auto-Voted Consensus")
            st.markdown("""
            Runs **3 models** simultaneously — LSTM Neural Network, Monte Carlo GBM, and
            Exponential Smoothing — then **auto-votes** a consensus forecast weighted by
            each model's accuracy. The result is a confidence-rated signal.
            """)
            st.markdown("---")

            if not available_tickers:
                st.info("No tickers found. Add holdings with ticker symbols (e.g. AAPL, TSLA).")
            else:
                col_t, col_d, col_e = st.columns(3)
                with col_t:
                    ens_ticker = st.selectbox("Select Ticker", available_tickers, key="ens_ticker")
                with col_d:
                    ens_days = st.slider("Forecast Days", 7, 60, 30, key="ens_days")
                with col_e:
                    ens_epochs = st.slider("LSTM Epochs", 10, 60, 30, step=10, key="ens_epochs",
                                           help="Lower = faster, Higher = more accurate")

                run_lstm = st.checkbox("Include LSTM (slower but more accurate)", value=True, key="ens_lstm")

                if st.button("🚀 Run Ensemble Forecast", type="primary", use_container_width=True, key="ens_btn"):
                    with st.spinner(f"🧠 Running ensemble on {ens_ticker}... (this may take 30-90 seconds)"):
                        from intelligence_engine import EnsembleForecaster
                        forecaster = EnsembleForecaster()
                        result = forecaster.forecast(ens_ticker, ens_days, ens_epochs, run_lstm=run_lstm)
                        if result['success']:
                            st.session_state._ensemble_result = result
                        else:
                            st.error(f"❌ {result.get('error', 'Unknown error')}")

                if getattr(st.session_state, '_ensemble_result', None):
                    r = st.session_state._ensemble_result
                    con = r['consensus']

                    # Signal banner
                    st.markdown("---")
                    signal_color = "#2ecc71" if "BUY" in r['signal'] else "#e74c3c" if "SELL" in r['signal'] else "#f39c12"
                    st.markdown(
                        f"<div style='text-align:center;padding:20px;border-radius:14px;"
                        f"border:2px solid {signal_color};background:rgba({','.join(str(int(signal_color.lstrip('#')[j:j+2],16)) for j in (0,2,4))},0.12)'>"
                        f"<div style='font-size:2.5rem;font-weight:700'>{r['signal']}</div>"
                        f"<div style='font-size:1.1rem;margin-top:8px'>"
                        f"{r['ticker']} — Target: <b>${con['price']:,.2f}</b> ({con['return_pct']:+.1f}%) "
                        f"| Agreement: <b>{con['agreement_score']}%</b> "
                        f"| Vote: {con['up_votes']}/{con['total_votes']} bullish</div></div>",
                        unsafe_allow_html=True
                    )

                    # Model comparison
                    st.markdown("---")
                    st.subheader("📊 Model Comparison")
                    mod_cols = st.columns(len(r['models']))
                    for idx, (name, data) in enumerate(r['models'].items()):
                        with mod_cols[idx]:
                            icon = {"lstm": "🧠", "monte_carlo": "🎲", "exp_smoothing": "📈"}.get(name, "📊")
                            mlabel = {"lstm": "LSTM Neural Net", "monte_carlo": "Monte Carlo GBM", "exp_smoothing": "Holt Smoothing"}.get(name, name)
                            weight = r['model_weights'].get(name, 0)
                            st.metric(f"{icon} {mlabel}", f"${data['end_price']:,.2f}", delta=f"Weight: {weight:.0%}")
                            if 'mape' in data:
                                st.caption(f"MAPE: {data['mape']:.1f}% | RMSE: ${data.get('rmse', 0):,.2f}")
                            if 'prob_up' in data:
                                st.caption(f"P(gain): {data['prob_up']:.0f}% | Range: ${data['p5']:,.2f}–${data['p95']:,.2f}")

                    # Ensemble chart
                    st.markdown("---")
                    st.subheader("🔮 Forecast Visualization")
                    fig_ens = go.Figure()
                    if 'lstm' in r['models'] and 'historical' in r['models']['lstm']:
                        hist = r['models']['lstm']['historical']
                        recent = hist.tail(90)
                        fig_ens.add_trace(go.Scatter(x=recent["Date"], y=recent["Close"],
                            mode="lines", name="Historical", line=dict(color="#667eea", width=2)))

                    model_colors = {"lstm": "#2ecc71", "monte_carlo": "#e67e22", "exp_smoothing": "#3498db"}
                    model_labels = {"lstm": "LSTM", "monte_carlo": "Monte Carlo (Median)", "exp_smoothing": "Holt Smoothing"}
                    for name, data in r['models'].items():
                        if 'forecast' in data:
                            dates = data.get('dates', list(range(len(data['forecast']))))
                            fig_ens.add_trace(go.Scatter(x=dates, y=data['forecast'],
                                mode="lines", name=model_labels.get(name, name),
                                line=dict(color=model_colors.get(name, "#999"), width=2.5)))
                        elif 'percentiles' in data:
                            pcts = data['percentiles']
                            x_range = list(range(len(pcts[50])))
                            fig_ens.add_trace(go.Scatter(x=x_range, y=pcts[50],
                                mode="lines", name="MC Median", line=dict(color="#e67e22", width=2.5)))
                            fig_ens.add_trace(go.Scatter(x=x_range + x_range[::-1],
                                y=pcts[95] + pcts[5][::-1],
                                fill="toself", fillcolor="rgba(230,126,34,0.1)",
                                line=dict(color="rgba(0,0,0,0)"), name="MC 90% Band", showlegend=True))

                    fig_ens.update_layout(title=f"{r['ticker']} — Ensemble Forecast",
                        xaxis_title="Date / Day", yaxis_title="Price ($)", height=500, hovermode="x unified",
                        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
                        **chart_layout(margin=dict(t=50, b=60, l=20, r=20)))
                    st.plotly_chart(fig_ens, use_container_width=True)

                    # Auto-generated insights
                    st.markdown("---")
                    st.subheader("💡 Auto-Generated Insights")
                    from intelligence_engine import InsightGenerator
                    gen = InsightGenerator()
                    risk_row = None
                    if getattr(st.session_state, "_risk_metrics", None) is not None:
                        rm = st.session_state._risk_metrics
                        match = rm[rm["Ticker"] == r['ticker']]
                        if not match.empty:
                            risk_row = match.iloc[0].to_dict()
                    insight = gen.generate_ticker_insight(r, risk_row=risk_row)
                    for line in insight['insights']:
                        st.markdown(f"- {line}")

        # ── TAB 3: MONTE CARLO (NET WORTH) ─────────────────────────────
        with hub_tabs[3]:
            st.subheader("📈 Monte Carlo Net Worth Projection")
            st.markdown("Run 10,000 geometric brownian motion pathways to project portfolio net worth.")
            summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
            nw = summary.get("net_worth", 0)
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Current Net Worth", fmt(nw))
            with c2:
                mu = st.number_input("Expected Return (%)", value=8.5, step=0.5, key="mc_mu") / 100
            with c3:
                vol = st.number_input("Expected Volatility (%)", value=12.0, step=0.5, key="mc_vol") / 100
            years = st.slider("Forecast Horizon (Years)", min_value=5, max_value=30, value=10, key="mc_years")
            if nw <= 0:
                st.error("Net worth must be positive.")
            elif st.button("🚀 Run Monte Carlo", type="primary", key="mc_btn"):
                with st.spinner("Simulating 10,000 parallel realities..."):
                    paths = analytics_engine.run_monte_carlo(nw, mu, vol, years=years, sim_count=10000)
                    percentiles = [5, 25, 50, 75, 95]
                    percentile_df = pd.DataFrame(index=paths.index)
                    for p in percentiles:
                        percentile_df[f"{p}th Percentile"] = np.percentile(paths, p, axis=1)
                    fig_mc = go.Figure()
                    mc_colors = {5: "rgba(231,76,60,0.4)", 25: "rgba(243,156,18,0.6)",
                              50: "#2ecc71", 75: "rgba(52,152,219,0.6)", 95: "rgba(155,89,182,0.4)"}
                    for p in percentiles:
                        fig_mc.add_trace(go.Scatter(
                            x=percentile_df.index, y=percentile_df[f"{p}th Percentile"],
                            mode="lines", name=f"{p}th Pct",
                            line=dict(color=mc_colors[p], width=3 if p == 50 else 1.5,
                                      dash="solid" if p == 50 else "dot")))
                    fig_mc.update_layout(title="Monte Carlo Projected Net Worth",
                        xaxis_title="Years", yaxis_title=st.session_state.base_currency,
                        height=500, **chart_layout())
                    st.plotly_chart(fig_mc, use_container_width=True)
                    final_vals = paths.iloc[-1]
                    st.markdown(f"### 📊 Outcomes at Year {years}")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Bear (5th pct)", fmt(np.percentile(final_vals, 5)))
                    m2.metric("Base (Median)", fmt(np.median(final_vals)))
                    m3.metric("Bull (95th pct)", fmt(np.percentile(final_vals, 95)))
                    prob_dbl = (final_vals > nw * 2).sum() / 10000 * 100
                    m4.metric("Prob. of Doubling", f"{prob_dbl:.1f}%")

        # ── TAB 4: MARKET SENTIMENT ────────────────────────────────────
        with hub_tabs[4]:
            st.subheader("📰 Transformer-Based Sentiment Analysis")
            st.markdown("Uses **FinBERT** — a deep transformer pre-trained on financial text — to analyze market sentiment.")
            st.markdown("---")
            if st.button("🚀 Run Sentiment Analysis", type="primary", use_container_width=True, key="sent_btn"):
                with st.spinner("🧠 Loading transformer & analyzing headlines..."):
                    try:
                        from dl_sentiment import get_analyzer
                        analyzer_dl = get_analyzer()
                        headlines = ai_advisor.fetch_news_headlines(8)
                        sentiment_data = analyzer_dl.get_market_sentiment(headlines)
                        st.session_state._deep_sentiment = sentiment_data
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
            if getattr(st.session_state, '_deep_sentiment', None):
                data = st.session_state._deep_sentiment
                st.markdown("---")
                sm1, sm2, sm3, sm4 = st.columns(4)
                sm1.metric("Overall Sentiment", data['label'])
                sm2.metric("Composite Score", f"{data['avg_score']:.3f}")
                sm3.metric("Model Used", data['model'])
                sm4.metric("Headlines Analyzed", str(data['headline_count']))
                st.markdown("---")
                col_chart, col_breakdown = st.columns([2, 1])
                with col_chart:
                    st.subheader("Per-Headline Scores")
                    results = data['results']
                    h_text = [rv['text'][:60] + '...' if len(rv['text']) > 60 else rv['text'] for rv in results]
                    scores = [rv['score'] for rv in results]
                    s_colors = ['#2ecc71' if s > 0.15 else '#e74c3c' if s < -0.15 else '#f39c12' for s in scores]
                    fig_sent = go.Figure(go.Bar(y=h_text, x=scores, orientation='h', marker_color=s_colors,
                        text=[f"{s:+.3f}" for s in scores], textposition="outside"))
                    fig_sent.add_vline(x=0, line_dash="dash", line_color="white", line_width=1)
                    fig_sent.update_layout(height=max(300, len(results) * 55), xaxis_title="Score",
                        xaxis=dict(range=[-1.1, 1.1]), **chart_layout(margin=dict(t=20, b=40, l=300, r=60)))
                    st.plotly_chart(fig_sent, use_container_width=True)
                with col_breakdown:
                    st.subheader("Distribution")
                    fig_pie = go.Figure(go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                        values=[data['positive_count'], data['negative_count'], data['neutral_count']],
                        marker_colors=['#2ecc71', '#e74c3c', '#f39c12'], hole=0.5, textinfo='label+value'))
                    fig_pie.update_layout(height=300, showlegend=False, **chart_layout())
                    st.plotly_chart(fig_pie, use_container_width=True)

        # ── TAB 5: ANOMALY SCANNER ─────────────────────────────────────
        with hub_tabs[5]:
            st.subheader("🔍 Autoencoder Anomaly Detection")
            st.markdown("Trains a **deep autoencoder** on historical return patterns to flag anomalous holdings.")
            st.markdown("---")
            col_ae1, col_ae2 = st.columns(2)
            with col_ae1:
                ae_epochs = st.slider("Training Epochs", 50, 200, 100, step=25, key="ae_epochs")
            with col_ae2:
                st.info("Requires at least 3 holdings with valid ticker symbols.")
            if st.button("🚀 Run Anomaly Scanner", type="primary", use_container_width=True, key="anom_btn"):
                with st.spinner("🧠 Training autoencoder..."):
                    try:
                        from dl_anomaly import PortfolioAnomalyDetector
                        detector = PortfolioAnomalyDetector()
                        anom_result = detector.analyze_portfolio(st.session_state.portfolio, epochs=ae_epochs)
                        st.session_state._anomaly_result = anom_result
                    except Exception as e:
                        st.error(f"❌ Failed: {e}")
            if getattr(st.session_state, '_anomaly_result', None):
                anom_result = st.session_state._anomaly_result
                if not anom_result['success']:
                    st.error(f"❌ {anom_result['error']}")
                else:
                    results_df = anom_result['results']
                    st.markdown("---")
                    am1, am2, am3 = st.columns(3)
                    anomaly_count = len(results_df[results_df['Anomaly'] == '⚠️ YES'])
                    am1.metric("Assets Analyzed", str(anom_result['tickers_analyzed']))
                    am2.metric("Anomalies Detected", str(anomaly_count),
                               delta=f"{anomaly_count} alerts" if anomaly_count > 0 else "All normal", delta_color="inverse")
                    am3.metric("Training Samples", f"{anom_result['training_samples']:,}")
                    st.markdown("---")
                    st.dataframe(results_df, use_container_width=True, hide_index=True)
                    st.markdown("---")
                    fig_ae = go.Figure()
                    ae_colors = []
                    for _, arow in results_df.iterrows():
                        if '🔴' in arow['Risk Level']: ae_colors.append('#e74c3c')
                        elif '🟡' in arow['Risk Level']: ae_colors.append('#f39c12')
                        elif '🟠' in arow['Risk Level']: ae_colors.append('#e67e22')
                        else: ae_colors.append('#2ecc71')
                    fig_ae.add_trace(go.Bar(x=results_df['Name'], y=results_df['Reconstruction Error'],
                        marker_color=ae_colors, text=results_df['Risk Level'], textposition='outside'))
                    fig_ae.add_hline(y=anom_result['threshold'], line_dash="dash", line_color="red",
                        annotation_text=f"Threshold ({anom_result['threshold']:.4f})", annotation_position="top right")
                    fig_ae.update_layout(title="Reconstruction Error by Asset", xaxis_title="Asset",
                        yaxis_title="Error (MSE)", height=450, xaxis_tickangle=-35,
                        **chart_layout(margin=dict(t=50, b=120, l=20, r=20)))
                    st.plotly_chart(fig_ae, use_container_width=True)
                    st.caption("🟢 Normal · 🟠 Watch · 🟡 Elevated · 🔴 Critical")

        # ── TAB 6: INTELLIGENCE SUMMARY ────────────────────────────────
        with hub_tabs[6]:
            st.subheader("💡 Intelligence Summary")
            st.markdown("Run all models across your portfolio for consolidated, actionable intelligence.")
            st.markdown("---")
            if not available_tickers:
                st.info("Add holdings with ticker symbols to generate intelligence.")
            else:
                st.markdown(f"**Tickers available:** {', '.join(available_tickers)}")
                skip_lstm = st.checkbox("Skip LSTM (faster analysis)", value=True, key="sum_skip_lstm")
                if st.button("🚀 Generate Full Intelligence Report", type="primary", use_container_width=True, key="intel_btn"):
                    from intelligence_engine import EnsembleForecaster, InsightGenerator
                    forecaster = EnsembleForecaster()
                    gen = InsightGenerator()
                    all_insights = []
                    progress = st.progress(0, text="Analyzing portfolio...")
                    for idx_t, ticker in enumerate(available_tickers):
                        progress.progress((idx_t + 1) / len(available_tickers), text=f"Analyzing {ticker}...")
                        ens_result = forecaster.forecast(ticker, forecast_days=30, epochs=20, run_lstm=not skip_lstm)
                        risk_row = None
                        if getattr(st.session_state, "_risk_metrics", None) is not None:
                            rm = st.session_state._risk_metrics
                            match = rm[rm["Ticker"] == ticker]
                            if not match.empty:
                                risk_row = match.iloc[0].to_dict()
                        insight = gen.generate_ticker_insight(ens_result, risk_row=risk_row)
                        all_insights.append(insight)
                    progress.empty()
                    st.session_state._intel_insights = all_insights
                    st.session_state._intel_summary = gen.generate_portfolio_summary(all_insights)
                if getattr(st.session_state, '_intel_summary', None):
                    st.markdown(st.session_state._intel_summary)
                    st.markdown("---")
                    st.subheader("📋 Per-Ticker Intelligence")
                    for ins in st.session_state._intel_insights:
                        with st.expander(f"{ins['signal']} **{ins['ticker']}** — Target ${ins.get('target_price', 0):,.2f} ({ins.get('expected_return', 0):+.1f}%)"):
                            for line in ins['insights']:
                                st.markdown(f"- {line}")

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
        summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
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
        k5.metric("Total Assets (Base)", fmt(summary["total_assets"]))
        k6.metric("Total Liabilities (Base)", fmt(summary["total_liabilities"]))
        k7.metric("Debt-to-Assets", f"{debt_to_assets*100:.1f}%")
        holdings_count = len(st.session_state.portfolio)
        asset_count = len(st.session_state.portfolio[st.session_state.portfolio["Side"] == "Asset"])
        k8.metric("Total Holdings", f"{holdings_count} ({asset_count} assets)")

        # Asset class table
        st.markdown("---")
        st.subheader("2. Asset Class Performance")
        alloc_a = utils.get_allocation_summary(st.session_state.portfolio, "Asset", st.session_state.base_currency)
        if not alloc_a.empty:
            total_av = alloc_a["Current Value"].sum()
            alloc_a["Allocation %"] = (alloc_a["Current Value"] / total_av * 100).round(1)
            rpt = alloc_a.rename(columns={"Current Value": "Market Value (Base)",
                                           "Cost Basis": "Capital Deployed (Base)",
                                           "Holdings": "Count"}).copy()
            for c in ["Market Value (Base)", "Capital Deployed (Base)", "Gain/Loss"]:
                rpt[c] = rpt[c].apply(lambda x: f"${x:,.0f}")
            rpt["Return %"] = rpt["Return %"].apply(lambda x: f"{x:+.2f}%")
            rpt["Allocation %"] = rpt["Allocation %"].apply(lambda x: f"{x:.1f}%")
            st.dataframe(rpt, use_container_width=True, hide_index=True)

        # Liabilities table
        st.markdown("---")
        st.subheader("3. Liability Summary")
        alloc_l = utils.get_allocation_summary(st.session_state.portfolio, "Liability", st.session_state.base_currency)
        if not alloc_l.empty:
            lrpt = alloc_l.rename(columns={"Current Value": "Outstanding (Base)",
                                            "Cost Basis": "Original Principal (Base)",
                                            "Holdings": "Count"})[
                ["Category", "Count", "Outstanding (Base)", "Original Principal (Base)"]].copy()
            for c in ["Outstanding (Base)", "Original Principal (Base)"]:
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
            fig_nw.update_layout(height=320, yaxis_title=st.session_state.base_currency,
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
        summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
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
                report = ai_advisor.generate_advisory_report(st.session_state.portfolio, headlines, st.session_state.base_currency)
                st.session_state.advisory_report = report
                st.session_state.advisory_headlines = headlines

        if st.session_state.advisory_report:
            st.markdown("---")
            st.subheader("📋 Advisory Report")
            st.markdown(st.session_state.advisory_report)

            # Concentration Heatmap
            st.markdown("---")
            st.subheader("🎯 Concentration Risk Heatmap")
            alloc = utils.get_allocation_summary(st.session_state.portfolio, "Asset", st.session_state.base_currency)
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
            st.dataframe(st.session_state.portfolio, use_container_width=True, hide_index=True)
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
