# ui_utils.py — Centralized UI Constants, Formatters & Chart Helpers
# Extracted from app.py to enforce Single Responsibility Principle.

import streamlit as st

# ============================================================================
# THEME CONSTANTS
# ============================================================================
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "gain": "#2ecc71",
    "loss": "#e74c3c",
    "warning": "#f39c12",
    "info": "#3498db",
    "bg_dark": "#0e1117",
    "sidebar_top": "#0d0d1a",
    "sidebar_mid": "#11192e",
    "sidebar_bot": "#0c2340",
    "text_muted": "#c0c0d0",
    "border": "rgba(102,126,234,0.25)",
}

GRADIENTS = {
    "primary": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    "card_bg": "linear-gradient(135deg, rgba(102,126,234,0.1) 0%, rgba(118,75,162,0.1) 100%)",
    "sidebar": "linear-gradient(180deg, #0d0d1a 0%, #11192e 60%, #0c2340 100%)",
    "divider": "linear-gradient(90deg, transparent, rgba(102,126,234,0.35), transparent)",
}

CURRENCY_SYMBOLS = {
    "USD": "$",
    "INR": "\u20b9",
    "EUR": "\u20ac",
    "GBP": "\u00a3",
    "JPY": "\u00a5",
    "AED": "\u062f.\u0625",
}


# ============================================================================
# THEME CSS — Injected once per session
# ============================================================================
THEME_CSS = """
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
"""


def inject_theme():
    """Inject the premium theme CSS into the Streamlit page. Call once."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ============================================================================
# FORMATTERS
# ============================================================================
def fmt(val, base_currency: str = None):
    """
    Format a numeric value as a currency string.

    Supports INR (Indian numbering: lakhs/crores) and Western (M/K) formats.
    If base_currency is None, reads from st.session_state.base_currency.
    """
    try:
        val = float(val)
    except Exception:
        return str(val)

    if base_currency is None:
        base_currency = st.session_state.get("base_currency", "USD")

    sym = CURRENCY_SYMBOLS.get(base_currency, "$")
    prefix = f"-{sym}" if val < 0 else sym
    v = abs(val)

    if base_currency == "INR":
        # Indian numbering system (lakhs / crores)
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


def fmt_pct(val, decimals: int = 1) -> str:
    """Format a percentage value with sign."""
    try:
        return f"{float(val):+.{decimals}f}%"
    except Exception:
        return str(val)


def fmt_ratio(val, decimals: int = 2) -> str:
    """Format a ratio (e.g. leverage)."""
    try:
        return f"{float(val):.{decimals}f}x"
    except Exception:
        return str(val)


# ============================================================================
# CHART LAYOUT HELPER
# ============================================================================
def chart_layout(**kwargs):
    """
    Standard Plotly layout dict for the premium dark theme.
    
    Caller kwargs always override defaults, enabling per-chart customization.
    """
    defaults = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter"),
        margin=dict(t=30, b=30, l=20, r=20),
    )
    defaults.update(kwargs)
    return defaults


def bar_colors(values):
    """Return green/red color list based on positive/negative values."""
    return [COLORS["gain"] if v >= 0 else COLORS["loss"] for v in values]
