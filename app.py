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
from ui_utils import inject_theme, fmt, chart_layout, bar_colors, COLORS


# ============================================================================
# CACHED MODEL LOADERS — Avoid reloading 400MB+ models per session
# ============================================================================
@st.cache_resource
def _get_finbert_analyzer():
    """Singleton loader for FinBERT transformer model."""
    from dl_sentiment import get_analyzer
    return get_analyzer()

@st.cache_resource
def _get_ensemble_forecaster():
    """Singleton loader for EnsembleForecaster."""
    from intelligence_engine import EnsembleForecaster
    return EnsembleForecaster()

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
inject_theme()

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
# AUTH CHECK — Gate the entire dashboard
# ============================================================================
try:
    from auth import check_auth
    if not check_auth():
        st.stop()
except ImportError:
    pass  # Auth module not installed, skip

# ============================================================================
# HELPERS
# ============================================================================
def save():
    """Persist current portfolio to disk."""
    utils.save_portfolio(st.session_state.portfolio)

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
         "📊 Analytics", "🔬 Research & AI Hub", "⚖️ Rebalancing", "📑 Reports & KPIs",
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
            fig_wf.update_layout(title="Net Worth Waterfall", height=450,
                                  xaxis=dict(tickangle=-45, automargin=True),
                                  showlegend=False,
                                  **chart_layout(margin=dict(t=50, b=150, l=20, r=20)))
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
                                   height=450, 
                                   xaxis=dict(tickangle=-45, automargin=True),
                                   **chart_layout(margin=dict(t=50, b=160, l=20, r=20)))
                st.plotly_chart(fig, use_container_width=True)

                alloc = utils.get_allocation_summary(st.session_state.portfolio, "Asset", st.session_state.base_currency)
                fig2 = px.bar(alloc, x="Category", y="Current Value",
                              color="Return %", color_continuous_scale="RdYlGn",
                              color_continuous_midpoint=0,
                              text=alloc["Current Value"].apply(fmt))
                fig2.update_layout(title="Asset Value by Category",
                                    height=450, 
                                    xaxis=dict(tickangle=-45, automargin=True),
                                    **chart_layout(margin=dict(t=50, b=150, l=20, r=20)))
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
                                        height=450, 
                                        xaxis=dict(tickangle=-45, automargin=True),
                                        legend=dict(orientation="h", y=-0.5, x=0.5, xanchor="center"),
                                        **chart_layout(margin=dict(t=50, b=180, l=20, r=20)))
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
# PAGE: RESEARCH & AI HUB (consolidated Intelligence + Backtesting + Advisory)
# ============================================================================
elif page == "🔬 Research & AI Hub":
    from research_hub import render as render_research_hub
    # Get available tickers
    rh_tickers = []
    if not st.session_state.portfolio.empty and "Ticker" in st.session_state.portfolio.columns:
        rh_tickers = [
            t.strip().upper()
            for t in st.session_state.portfolio[
                (st.session_state.portfolio["Side"] == "Asset") &
                (st.session_state.portfolio["Ticker"].str.strip() != "")
            ]["Ticker"].unique()
            if t and str(t).lower() not in ("nan", "none", "")
        ]
    render_research_hub(rh_tickers, chart_layout, fmt)


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
