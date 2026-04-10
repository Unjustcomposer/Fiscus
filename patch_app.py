import codecs
import re

app_path = r"c:\Users\khand\Desktop\mini_project\app.py"
with codecs.open(app_path, "r", "utf-8") as f:
    content = f.read()

# 1. Base Currency Session State
content = content.replace(
    'if "advisory_headlines" not in st.session_state:\n    st.session_state.advisory_headlines = None',
    'if "advisory_headlines" not in st.session_state:\n    st.session_state.advisory_headlines = None\nif "base_currency" not in st.session_state:\n    st.session_state.base_currency = "INR"'
)

# 2. fmt function
fmt_old = """def fmt(val):
    try:
        val = float(val)
    except Exception:
        return str(val)
    prefix = "-$" if val < 0 else "$"
    v = abs(val)"""

fmt_new = """def fmt(val):
    try:
        val = float(val)
    except Exception:
        return str(val)
    base_curr = st.session_state.get("base_currency", "USD")
    sym = {"USD": "$", "INR": "₹", "EUR": "€", "GBP": "£", "JPY": "¥", "AED": "د.إ"}.get(base_curr, "$")
    prefix = f"-{sym}" if val < 0 else sym
    v = abs(val)"""
content = content.replace(fmt_old, fmt_new)

# 3. Sidebar
sidebar_old = """    summary_side = utils.calculate_portfolio_summary(st.session_state.portfolio)
    st.metric("Net Worth", fmt(summary_side["net_worth"]))
    st.metric("Total Assets", fmt(summary_side["total_assets"]))
    st.metric("Total Liabilities", fmt(summary_side["total_liabilities"]))
    st.caption("All values displayed in USD.")"""

sidebar_new = """    st.session_state.base_currency = st.selectbox("Base Currency", utils.CURRENCIES, index=utils.CURRENCIES.index(st.session_state.get("base_currency", "INR")))
    summary_side = utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)
    st.metric("Net Worth", fmt(summary_side["net_worth"]))
    st.metric("Total Assets", fmt(summary_side["total_assets"]))
    st.metric("Total Liabilities", fmt(summary_side["total_liabilities"]))
    st.caption(f"All values displayed in {st.session_state.base_currency}.")"""
content = content.replace(sidebar_old, sidebar_new)

# 4. Global Replacements
content = content.replace('utils.calculate_portfolio_summary(st.session_state.portfolio)', 'utils.calculate_portfolio_summary(st.session_state.portfolio, st.session_state.base_currency)')

content = content.replace('Current Value USD', 'Current Value Base')
content = content.replace('Cost Basis USD', 'Cost Basis Base')

content = content.replace('utils.normalize_to_usd(assets_df)', 'utils.normalize_to_base(assets_df, st.session_state.base_currency)')
content = content.replace('utils.normalize_to_usd(df_side)', 'utils.normalize_to_base(df_side, st.session_state.base_currency)')

content = content.replace('utils.get_allocation_summary(st.session_state.portfolio, "Asset")', 'utils.get_allocation_summary(st.session_state.portfolio, "Asset", st.session_state.base_currency)')
content = content.replace('utils.get_allocation_summary(st.session_state.portfolio, "Liability")', 'utils.get_allocation_summary(st.session_state.portfolio, "Liability", st.session_state.base_currency)')
content = content.replace('utils.get_holding_performance(st.session_state.portfolio, "Asset")', 'utils.get_holding_performance(st.session_state.portfolio, "Asset", st.session_state.base_currency)')
content = content.replace('utils.get_holding_performance(st.session_state.portfolio, "Liability")', 'utils.get_holding_performance(st.session_state.portfolio, "Liability", st.session_state.base_currency)')

# 5. Dashboard table USD text to Base
content = content.replace('Unrealised Gain/Loss per Asset (USD)', 'Unrealised Gain/Loss per Asset')
content = content.replace('Total Assets (USD)', 'Total Assets (Base)')
content = content.replace('Total Liabilities (USD)', 'Total Liabilities (Base)')
content = content.replace('Market Value (USD)', 'Market Value (Base)')
content = content.replace('Capital Deployed (USD)', 'Capital Deployed (Base)')
content = content.replace('Outstanding (USD)', 'Outstanding (Base)')
content = content.replace('Original Principal (USD)', 'Original Principal (Base)')

# 6. Form bug fix
form_old = """    with tab_add:
        with st.form("add_form", clear_on_submit=True):
            c1, c2 = st.columns(2)
            with c1:
                side = st.selectbox("Type *", ["Asset", "Liability"])
            with c2:
                cats = utils.ASSET_CATEGORIES if side == "Asset" else utils.LIABILITY_CATEGORIES
                category = st.selectbox("Category *", cats)"""

form_new = """    with tab_add:
        side = st.radio("Type *", ["Asset", "Liability"], horizontal=True)
        with st.form("add_form", clear_on_submit=True):
            cats = utils.ASSET_CATEGORIES if side == "Asset" else utils.LIABILITY_CATEGORIES
            category = st.selectbox("Category *", cats)"""
content = content.replace(form_old, form_new)

# Re-save
with codecs.open(app_path, "w", "utf-8") as f:
    f.write(content)

print("Patch applied.")
