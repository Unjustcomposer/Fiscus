# pages/page_research_hub.py — Unified Research & AI Command Center
# Consolidates Intelligence Hub, Backtesting, and AI Advisory into
# a single page with three top-level tabs.

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from ui_utils import COLORS, fmt, chart_layout, bar_colors


def render(available_tickers: list, chart_layout_fn, fmt_fn):
    """
    Render the consolidated Research & AI Hub page.

    Parameters:
        available_tickers: List of ticker symbols from portfolio
        chart_layout_fn:   The chart_layout() helper from app.py
        fmt_fn:            The fmt() currency formatter
    """

    st.title("🔬 Research & AI Hub")
    st.markdown(
        "Unified command center — forecasting intelligence, model validation, "
        "and strategic advisory in one place."
    )

    # ── THREE TOP-LEVEL TABS ──────────────────────────────────────────
    top_tabs = st.tabs([
        "🧠 Intelligence Engine",
        "🧪 Model Validation",
        "💡 Strategic Advisory",
    ])

    # ==================================================================
    # TAB 1: INTELLIGENCE ENGINE (former Intelligence Hub)
    # ==================================================================
    with top_tabs[0]:
        _render_intelligence_engine(available_tickers, chart_layout_fn, fmt_fn)

    # ==================================================================
    # TAB 2: MODEL VALIDATION (Backtesting + Ablation)
    # ==================================================================
    with top_tabs[1]:
        _render_model_validation(available_tickers, chart_layout_fn, fmt_fn)

    # ==================================================================
    # TAB 3: STRATEGIC ADVISORY (former AI Advisory)
    # ==================================================================
    with top_tabs[2]:
        _render_strategic_advisory(chart_layout_fn, fmt_fn)


# ======================================================================
# SECTION 1: INTELLIGENCE ENGINE
# ======================================================================
def _render_intelligence_engine(available_tickers, chart_layout_fn, fmt_fn):
    """Render the full Intelligence Engine with sub-tabs."""

    import ai_advisor
    import analytics_engine
    import utils

    st.subheader("🧠 Intelligence Engine")
    st.markdown(
        "Unified risk analytics, ensemble forecasting, sentiment analysis, "
        "and actionable intelligence."
    )

    if st.session_state.portfolio.empty:
        st.warning("No data. Add holdings first.")
        return

    hub_tabs = st.tabs([
        "🤖 Auto Analysis",
        "📊 Portfolio Risk",
        "🔮 Ensemble Forecast",
        "📈 Monte Carlo (NW)",
        "📰 Market Sentiment",
        "🔍 Anomaly Scanner",
        "💡 Intelligence Summary",
    ])

    # ── TAB 0: AUTO ANALYSIS ──────────────────────────────────────
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
            auto_days = st.slider("Forecast Horizon (Days)", 7, 60, 30, key="rh_auto_days")

            if st.button("🚀 Run Auto Analysis", type="primary", use_container_width=True, key="rh_auto_btn"):
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
                    reason = result.get('selection_reason', '')
                    signal = result['signal']

                    with st.expander(f"{signal}  **{ticker}** — Target ${con['price']:,.2f} ({con['return_pct']:+.1f}%)  |  _{reason}_", expanded=(i == 0)):
                        # Meta-Selection Details
                        meta = result.get('meta_selection', {})
                        if meta:
                            m1, m2, m3 = st.columns(3)
                            m1.metric("Selection Method", meta.get('method', 'Heuristic').title())
                            m2.metric("Optimal Model", meta.get('best_model', 'N/A').replace('_', ' ').title())
                            m3.metric("Confidence", f"{meta.get('confidence', 0)*100:.1f}%")
                            st.markdown("---")

                        # Data Profile
                        analysis = meta.get('heuristic_result', {}) if meta.get('method') == 'heuristic' else meta.get('meta_result', {}).get('features', {})
                        
                        if analysis:
                            dp1, dp2, dp3, dp4 = st.columns(4)
                            if meta.get('method') == 'heuristic':
                                dp1.metric("Data Points", f"{analysis.get('data_points', '?')} days")
                                dp2.metric("Volatility", f"{analysis.get('ann_volatility', '?')}%",
                                           delta=analysis.get('vol_regime', '').upper())
                                dp3.metric("Trend Strength", f"{analysis.get('trend_strength', '?')}")
                                dp4.metric("Kurtosis", f"{analysis.get('kurtosis', '?')}",
                                           delta="Fat Tails" if analysis.get('has_fat_tails') else "Normal")
                            else:
                                # Meta-Classifier features
                                dp1.metric("Volatility", f"{analysis.get('ann_volatility', 0)*100:.1f}%")
                                dp2.metric("Trend Strength", f"{analysis.get('trend_strength', 0):.4f}")
                                dp3.metric("Hurst Exponent", f"{analysis.get('hurst_exponent', 0):.2f}")
                                dp4.metric("Max Drawdown", f"{analysis.get('max_drawdown', 0)*100:.1f}%")

                            # Model score comparison (only if heuristic)
                            if meta.get('method') == 'heuristic':
                                st.markdown("**Model Suitability Scores:**")
                                model_scores = analysis.get('model_scores', {})
                                best = analysis.get('best_model', '')
                                score_cols = st.columns(3)
                                for j, (mname, mscore) in enumerate(model_scores.items()):
                                    from intelligence_engine import AutoAnalyzer
                                    icon = {"lstm": "🧠", "monte_carlo": "🎲", "exp_smoothing": "📈"}.get(mname, "📊")
                                    mlabel = {"lstm": "LSTM", "monte_carlo": "Monte Carlo", "exp_smoothing": "Exp. Smoothing"}.get(mname, mname)
                                    is_best = mname == best
                                    badge = " ✅ SELECTED" if is_best else ""
                                    score_cols[j].metric(f"{icon} {mlabel}{badge}", f"{mscore:.1f}")

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

    # ── TAB 1: PORTFOLIO RISK ─────────────────────────────────────
    with hub_tabs[1]:
        st.subheader("📊 Portfolio Risk Dashboard")
        st.markdown("Comprehensive risk profiling: Beta, Sharpe, Sortino, VaR, CVaR, Max Drawdown, and composite Risk Score.")

        if st.button("🔄 Compute Risk Metrics", key="rh_risk_btn"):
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
            fig_corr.update_layout(**chart_layout_fn())
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
                                 yaxis_title="Expected Return (%)", height=450, **chart_layout_fn())
            st.plotly_chart(fig_rr, use_container_width=True)
            st.caption("Bubble size = Risk Score. 🟢 Low Risk · 🟡 Moderate · 🔴 High Risk")

    # ── TAB 2: ENSEMBLE FORECAST ──────────────────────────────────
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
                ens_ticker = st.selectbox("Select Ticker", available_tickers, key="rh_ens_ticker")
            with col_d:
                ens_days = st.slider("Forecast Days", 7, 60, 30, key="rh_ens_days")
            with col_e:
                ens_epochs = st.slider("LSTM Epochs", 10, 60, 30, step=10, key="rh_ens_epochs",
                                       help="Lower = faster, Higher = more accurate")

            run_lstm = st.checkbox("Include LSTM (slower but more accurate)", value=True, key="rh_ens_lstm")

            if st.button("🚀 Run Ensemble Forecast", type="primary", use_container_width=True, key="rh_ens_btn"):
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
                    **chart_layout_fn(margin=dict(t=50, b=60, l=20, r=20)))
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

    # ── TAB 3: MONTE CARLO (NET WORTH) ────────────────────────────
    with hub_tabs[3]:
        st.subheader("📈 Monte Carlo Net Worth Projection")
        st.markdown("Run 10,000 geometric brownian motion pathways to project portfolio net worth.")
        import utils
        summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
        nw = summary.get("net_worth", 0)
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Current Net Worth", fmt_fn(nw))
        with c2:
            mu = st.number_input("Expected Return (%)", value=8.5, step=0.5, key="rh_mc_mu") / 100
        with c3:
            vol = st.number_input("Expected Volatility (%)", value=12.0, step=0.5, key="rh_mc_vol") / 100
        years = st.slider("Forecast Horizon (Years)", min_value=5, max_value=30, value=10, key="rh_mc_years")
        if nw <= 0:
            st.error("Net worth must be positive.")
        elif st.button("🚀 Run Monte Carlo", type="primary", key="rh_mc_btn"):
            with st.spinner("Simulating 10,000 parallel realities..."):
                import analytics_engine
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
                    height=500, **chart_layout_fn())
                st.plotly_chart(fig_mc, use_container_width=True)
                final_vals = paths.iloc[-1]
                st.markdown(f"### 📊 Outcomes at Year {years}")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Bear (5th pct)", fmt_fn(np.percentile(final_vals, 5)))
                m2.metric("Base (Median)", fmt_fn(np.median(final_vals)))
                m3.metric("Bull (95th pct)", fmt_fn(np.percentile(final_vals, 95)))
                prob_dbl = (final_vals > nw * 2).sum() / 10000 * 100
                m4.metric("Prob. of Doubling", f"{prob_dbl:.1f}%")

    # ── TAB 4: MARKET SENTIMENT ───────────────────────────────────
    with hub_tabs[4]:
        st.subheader("📰 Transformer-Based Sentiment Analysis")
        st.markdown("Uses **FinBERT** — a deep transformer pre-trained on financial text — to analyze market sentiment.")
        st.markdown("---")
        if st.button("🚀 Run Sentiment Analysis", type="primary", use_container_width=True, key="rh_sent_btn"):
            with st.spinner("🧠 Loading transformer & analyzing headlines..."):
                try:
                    from dl_sentiment import get_analyzer
                    import ai_advisor
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
                    xaxis=dict(range=[-1.1, 1.1]), **chart_layout_fn(margin=dict(t=20, b=40, l=300, r=60)))
                st.plotly_chart(fig_sent, use_container_width=True)
            with col_breakdown:
                st.subheader("Distribution")
                fig_pie = go.Figure(go.Pie(labels=['Positive', 'Negative', 'Neutral'],
                    values=[data['positive_count'], data['negative_count'], data['neutral_count']],
                    marker_colors=['#2ecc71', '#e74c3c', '#f39c12'], hole=0.5, textinfo='label+value'))
                fig_pie.update_layout(height=300, showlegend=False, **chart_layout_fn())
                st.plotly_chart(fig_pie, use_container_width=True)

    # ── TAB 5: ANOMALY SCANNER ────────────────────────────────────
    with hub_tabs[5]:
        st.subheader("🔍 Autoencoder Anomaly Detection")
        st.markdown("Trains a **deep autoencoder** on historical return patterns to flag anomalous holdings.")
        st.markdown("---")
        col_ae1, col_ae2 = st.columns(2)
        with col_ae1:
            ae_epochs = st.slider("Training Epochs", 50, 200, 100, step=25, key="rh_ae_epochs")
        with col_ae2:
            st.info("Requires at least 3 holdings with valid ticker symbols.")
        if st.button("🚀 Run Anomaly Scanner", type="primary", use_container_width=True, key="rh_anom_btn"):
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
                    **chart_layout_fn(margin=dict(t=50, b=120, l=20, r=20)))
                st.plotly_chart(fig_ae, use_container_width=True)
                st.caption("🟢 Normal · 🟠 Watch · 🟡 Elevated · 🔴 Critical")

    # ── TAB 6: INTELLIGENCE SUMMARY ───────────────────────────────
    with hub_tabs[6]:
        st.subheader("💡 Intelligence Summary")
        st.markdown("Run all models across your portfolio for consolidated, actionable intelligence.")
        st.markdown("---")
        if not available_tickers:
            st.info("Add holdings with ticker symbols to generate intelligence.")
        else:
            st.markdown(f"**Tickers available:** {', '.join(available_tickers)}")
            skip_lstm = st.checkbox("Skip LSTM (faster analysis)", value=True, key="rh_sum_skip_lstm")
            if st.button("🚀 Generate Full Intelligence Report", type="primary", use_container_width=True, key="rh_intel_btn"):
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


# ======================================================================
# SECTION 2: MODEL VALIDATION (Backtesting + Ablation)
# ======================================================================
def _render_model_validation(available_tickers, chart_layout_fn, fmt_fn):
    """Render the Model Validation section — delegates to page_backtesting logic."""

    st.subheader("🧪 Model Validation — Backtesting & Ablation")
    st.markdown(
        "Walk-forward backtesting with expanding windows, baseline comparisons, "
        "and statistical significance testing for research-grade evaluation."
    )

    bt_tabs = st.tabs([
        "📊 Walk-Forward Backtest",
        "🔬 Ablation Studies",
        "📈 Results Viewer",
    ])

    # ── TAB 1: Walk-Forward Backtest ──────────────────────────────
    with bt_tabs[0]:
        st.subheader("📊 Walk-Forward Backtesting Engine")
        st.markdown("""
        Evaluates model performance using **expanding-window** walk-forward testing.
        Each window trains on all data up to time *t*, then forecasts *h* days ahead
        and compares against actuals. This process repeats monthly.
        """)
        st.markdown("---")

        if not available_tickers:
            st.info("No tickers found. Add holdings with ticker symbols.")
            return

        col1, col2, col3 = st.columns(3)
        with col1:
            bt_ticker = st.selectbox(
                "Select Ticker", available_tickers, key="rh_bt_ticker")
        with col2:
            bt_horizon = st.slider(
                "Forecast Horizon (days)", 5, 60, 21,
                key="rh_bt_horizon",
                help="Number of days to forecast ahead per window")
        with col3:
            bt_years = st.slider(
                "History (years)", 2, 5, 3,
                key="rh_bt_years",
                help="Years of historical data to use")

        col4, col5 = st.columns(2)
        with col4:
            bt_step = st.slider(
                "Step Size (days)", 5, 42, 21,
                key="rh_bt_step",
                help="Days between evaluation windows")
        with col5:
            bt_models = st.multiselect(
                "Models to Evaluate",
                ['naive', 'random_walk', 'arima', 'buy_and_hold',
                 'sma', 'monte_carlo', 'exp_smoothing'],
                default=['naive', 'random_walk', 'arima',
                         'monte_carlo', 'exp_smoothing'],
                key="rh_bt_models")

        if st.button("🚀 Run Backtest", type="primary",
                      use_container_width=True, key="rh_bt_run"):
            _run_backtest(
                bt_ticker, bt_models, bt_horizon, bt_step, bt_years,
                chart_layout_fn)

        # Show cached results
        if getattr(st.session_state, '_bt_summary', None) is not None:
            _display_backtest_results(chart_layout_fn)

    # ── TAB 2: Ablation Studies ───────────────────────────────────
    with bt_tabs[1]:
        st.subheader("🔬 Ablation Studies")
        st.markdown("""
        Systematically evaluate the contribution of each component
        by selectively disabling them and measuring the impact on
        forecasting accuracy.
        """)
        st.markdown("---")

        if not available_tickers:
            st.info("No tickers found.")
            return

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            ab_tickers = st.multiselect(
                "Tickers for Ablation",
                available_tickers,
                default=available_tickers[:3],
                key="rh_ab_tickers",
                help="Select 2-5 tickers for comprehensive evaluation")
        with col_a2:
            ab_configs = st.multiselect(
                "Ablation Configurations",
                [
                    'naive_baseline', 'random_walk_baseline',
                    'arima_baseline', 'buy_hold_baseline',
                    'mc_only', 'ets_only',
                    'full_ensemble', 'no_sentiment',
                ],
                default=[
                    'naive_baseline', 'arima_baseline',
                    'mc_only', 'ets_only',
                    'full_ensemble', 'no_sentiment',
                ],
                key="rh_ab_configs")

        if st.button("🚀 Run Ablation Suite", type="primary",
                      use_container_width=True, key="rh_ab_run"):
            _run_ablation(ab_tickers, ab_configs, chart_layout_fn)

        if getattr(st.session_state, '_ab_comparison', None) is not None:
            _display_ablation_results(chart_layout_fn)

    # ── TAB 3: Results Viewer ─────────────────────────────────────
    with bt_tabs[2]:
        st.subheader("📈 Saved Results")
        st.markdown("View and export previously computed backtest results.")
        st.markdown("---")

        has_bt = getattr(st.session_state, '_bt_predictions', None) is not None
        has_ab = getattr(st.session_state, '_ab_comparison', None) is not None

        if not has_bt and not has_ab:
            st.info("No results yet. Run a backtest or ablation study first.")
            return

        if has_bt:
            st.subheader("Backtest Predictions Log")
            preds = st.session_state._bt_predictions
            st.dataframe(preds, use_container_width=True, hide_index=True)

            csv = preds.to_csv(index=False)
            st.download_button(
                "⬇️ Download Backtest Results CSV",
                data=csv,
                file_name="backtest_results.csv",
                mime="text/csv",
                use_container_width=True)

        if has_ab:
            st.markdown("---")
            st.subheader("Ablation Comparison")
            ab_df = st.session_state._ab_comparison
            st.dataframe(ab_df, use_container_width=True, hide_index=True)

            csv = ab_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Ablation Results CSV",
                data=csv,
                file_name="ablation_results.csv",
                mime="text/csv",
                use_container_width=True)

        # Export LaTeX
        if has_bt:
            st.markdown("---")
            st.subheader("📄 LaTeX Export")
            try:
                from evaluation import EvaluationReport, format_latex_table
                report = EvaluationReport()
                summary = st.session_state._bt_summary
                for _, row in summary.iterrows():
                    report.results.append(row.to_dict())
                latex = format_latex_table(report, "Walk-Forward Backtest Results")
                st.code(latex, language="latex")
            except Exception as e:
                st.warning(f"LaTeX export failed: {e}")


# ======================================================================
# SECTION 3: STRATEGIC ADVISORY
# ======================================================================
def _render_strategic_advisory(chart_layout_fn, fmt_fn):
    """Render the Strategic Advisory section."""

    import ai_advisor
    import utils

    st.subheader("💡 Strategic Advisory")
    st.markdown("Live news-powered analysis and data-driven recommendations for your portfolio.")

    if st.session_state.portfolio.empty:
        st.warning("Portfolio is empty. Add holdings to receive advice.")
        return

    summary = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Worth", fmt_fn(summary["net_worth"]))
    c2.metric("Total Assets", fmt_fn(summary["total_assets"]))
    c3.metric("Total Liabilities", fmt_fn(summary["total_liabilities"]))
    cash_df = st.session_state.portfolio[
        (st.session_state.portfolio["Side"] == "Asset") &
        (st.session_state.portfolio["Category"] == "Cash & Equivalents")
    ]
    total_cash = cash_df["Current Value"].astype(float).sum() if not cash_df.empty else 0.0
    c4.metric("💵 Available Cash", fmt_fn(total_cash))

    st.markdown("---")
    if st.button("🚀 Generate Advisory Report", use_container_width=True, type="primary", key="rh_adv_btn"):
        with st.spinner("🧠 Fetching live news and analysing portfolio..."):
            headlines = ai_advisor.fetch_news_headlines(5)
            report = ai_advisor.generate_advisory_report(st.session_state.portfolio, headlines, st.session_state.base_currency)
            st.session_state.advisory_report = report
            st.session_state.advisory_headlines = headlines

    if st.session_state.get("advisory_report"):
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
                **chart_layout_fn(margin=dict(t=50, b=100, l=20, r=20))
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            st.caption("🟢 Healthy (<20%)  🟡 Watch (20-30%)  🔴 Overconcentrated (>30%)")


# ======================================================================
# PRIVATE HELPERS — Backtesting
# ======================================================================

def _run_backtest(ticker, models, horizon, step, years, chart_layout_fn):
    """Execute walk-forward backtest."""
    from backtesting import WalkForwardBacktester

    progress = st.progress(0, text=f"Initializing backtest for {ticker}...")

    def update_progress(pct, msg):
        progress.progress(min(pct, 1.0), text=msg)

    try:
        bt = WalkForwardBacktester(
            ticker=ticker,
            lookback_years=years,
            forecast_horizon=horizon,
            step_size=step,
            min_train_days=252,
        )

        predictions = bt.run(
            models=models,
            progress_callback=update_progress)

        if predictions.empty:
            progress.empty()
            st.error("No predictions generated. Check data availability.")
            return

        summary = bt.get_summary()
        sig_tests = bt.get_significance_tests()

        st.session_state._bt_predictions = predictions
        st.session_state._bt_summary = summary
        st.session_state._bt_sig_tests = sig_tests
        st.session_state._bt_ticker = ticker

        progress.empty()
        st.success(
            f"✅ Backtest complete! {len(predictions)} predictions across "
            f"{predictions['window'].nunique()} windows.")

    except Exception as e:
        progress.empty()
        st.error(f"❌ Backtest failed: {e}")


def _display_backtest_results(chart_layout_fn):
    """Display cached backtest results."""
    summary = st.session_state._bt_summary
    predictions = st.session_state._bt_predictions
    ticker = st.session_state.get('_bt_ticker', '')

    # Summary table
    st.markdown("---")
    st.subheader(f"📋 Model Comparison — {ticker}")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Highlight best
    if not summary.empty:
        best_rmse = summary.loc[summary['RMSE'].idxmin()]
        best_da = summary.loc[summary['DA (%)'].idxmax()]
        c1, c2, c3 = st.columns(3)
        c1.metric("🏆 Best RMSE", best_rmse['Model'],
                   delta=f"RMSE: {best_rmse['RMSE']:.2f}")
        c2.metric("🎯 Best DA", best_da['Model'],
                   delta=f"DA: {best_da['DA (%)']:.1f}%")
        if 'Profit Factor' in summary.columns:
            pf_valid = summary[summary['Profit Factor'] < float('inf')]
            if not pf_valid.empty:
                best_pf = pf_valid.loc[pf_valid['Profit Factor'].idxmax()]
                c3.metric("💰 Best Profit Factor", best_pf['Model'],
                           delta=f"PF: {best_pf['Profit Factor']:.2f}")

    # Chart: RMSE comparison
    st.markdown("---")
    st.subheader("📊 RMSE by Model")
    fig_rmse = go.Figure()
    colors = ['#667eea', '#764ba2', '#f093fb', '#4fd1c5',
              '#f6ad55', '#fc8181', '#68d391', '#63b3ed']
    for i, (_, row) in enumerate(summary.iterrows()):
        fig_rmse.add_trace(go.Bar(
            x=[row['Model']], y=[row['RMSE']],
            name=row['Model'],
            marker_color=colors[i % len(colors)],
            text=f"{row['RMSE']:.2f}",
            textposition='outside'))
    fig_rmse.update_layout(
        title="Root Mean Squared Error by Model",
        yaxis_title="RMSE ($)",
        height=400, showlegend=False,
        **chart_layout_fn())
    st.plotly_chart(fig_rmse, use_container_width=True)

    # Chart: DA comparison
    st.subheader("🎯 Directional Accuracy by Model")
    fig_da = go.Figure()
    for i, (_, row) in enumerate(summary.iterrows()):
        color = '#2ecc71' if row['DA (%)'] > 55 else '#f39c12' if row['DA (%)'] > 50 else '#e74c3c'
        fig_da.add_trace(go.Bar(
            x=[row['Model']], y=[row['DA (%)']],
            marker_color=color,
            text=f"{row['DA (%)']:.1f}%",
            textposition='outside'))
    fig_da.add_hline(y=50, line_dash="dash", line_color="white",
                      annotation_text="Random Baseline (50%)")
    fig_da.update_layout(
        title="Directional Accuracy (higher = better)",
        yaxis_title="DA (%)", yaxis=dict(range=[0, 100]),
        height=400, showlegend=False,
        **chart_layout_fn())
    st.plotly_chart(fig_da, use_container_width=True)

    # Statistical significance tests
    sig_tests = st.session_state.get('_bt_sig_tests', [])
    if sig_tests:
        st.markdown("---")
        st.subheader("📐 Statistical Significance Tests")
        st.markdown(
            "Pairwise comparison of prediction errors. "
            "p < 0.05 indicates statistically significant difference.")

        test_rows = []
        for t in sig_tests:
            tt = t['paired_ttest']
            wt = t['wilcoxon']
            dm = t['diebold_mariano']
            test_rows.append({
                'Model A': t['model_a'],
                'Model B': t['model_b'],
                'MAE_A': t['mean_error_a'],
                'MAE_B': t['mean_error_b'],
                't-test p': f"{tt['p_value']:.4f}",
                'Wilcoxon p': f"{wt['p_value']:.4f}",
                'DM p': f"{dm['p_value']:.4f}",
                'Significant': '✅' if tt['significant_5pct'] else '❌',
                'Better': dm['better_model'],
            })
        st.dataframe(pd.DataFrame(test_rows),
                      use_container_width=True, hide_index=True)


def _run_ablation(tickers, configs, chart_layout_fn):
    """Execute ablation study suite."""
    from ablation_runner import AblationRunner

    progress = st.progress(0, text="Initializing ablation suite...")

    def update_progress(pct, msg):
        progress.progress(min(pct, 1.0), text=msg)

    try:
        runner = AblationRunner(
            tickers=tickers,
            forecast_horizon=21,
            step_size=21,
            min_train_days=252,
            lookback_years=3,
        )

        runner.run_suite(
            configs=configs,
            progress_callback=update_progress)

        comparison = runner.get_averaged_comparison()
        sig_tests = runner.run_significance_tests()

        st.session_state._ab_comparison = comparison
        st.session_state._ab_sig_tests = sig_tests

        progress.empty()
        st.success(f"✅ Ablation complete! {len(configs)} configurations "
                    f"× {len(tickers)} tickers evaluated.")

    except Exception as e:
        progress.empty()
        st.error(f"❌ Ablation failed: {e}")


def _display_ablation_results(chart_layout_fn):
    """Display ablation study results."""
    comparison = st.session_state._ab_comparison

    st.markdown("---")
    st.subheader("📋 Ablation Comparison (Averaged Across Tickers)")
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    # RMSE comparison chart
    if not comparison.empty and 'RMSE' in comparison.columns:
        st.markdown("---")
        st.subheader("📊 RMSE by Configuration")
        valid = comparison.dropna(subset=['RMSE']).sort_values('RMSE')
        fig = go.Figure()
        colors = ['#2ecc71' if 'ensemble' in c else
                  '#3498db' if 'only' in c else
                  '#e74c3c' for c in valid['Config']]
        fig.add_trace(go.Bar(
            x=valid['Config'],
            y=valid['RMSE'],
            marker_color=colors,
            text=valid['RMSE'].round(2),
            textposition='outside'))
        fig.update_layout(
            title="RMSE by Ablation Config (lower = better)",
            yaxis_title="RMSE", height=450, xaxis_tickangle=-35,
            **chart_layout_fn())
        st.plotly_chart(fig, use_container_width=True)

    # Significance
    sig_tests = st.session_state.get('_ab_sig_tests', [])
    if sig_tests:
        st.markdown("---")
        st.subheader("📐 Significance: Ensemble vs Baseline")
        rows = []
        for t in sig_tests:
            rows.append({
                'Ticker': t['ticker'],
                'Baseline MAE': t['baseline_mean_error'],
                'Ensemble MAE': t['test_mean_error'],
                'Improvement': f"{t['improvement_pct']:+.1f}%",
                'p-value (t-test)': f"{t['paired_ttest']['p_value']:.4f}",
                'Significant': '✅' if t['paired_ttest']['significant_5pct'] else '❌',
            })
        st.dataframe(pd.DataFrame(rows),
                      use_container_width=True, hide_index=True)
