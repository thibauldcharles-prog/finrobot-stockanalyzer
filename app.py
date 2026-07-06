"""
FinRobot Stock Analyzer — Streamlit Web App
No API key required. Analysis powered by built-in rules engine.
Run: streamlit run webapp/app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date as _date
import time

st.set_page_config(
    page_title="FinRobot Stock Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container { padding-top: 3.5rem; padding-bottom: 1rem; }
/* keep headings from being clipped under the fixed top toolbar */
h1, h2, h3 { line-height: 1.3 !important; padding-top: 0.15rem; }
[data-testid="stHeader"] { background: rgba(13,17,23,0.6); }
[data-testid="stMetric"] {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 14px 18px;
}
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; }
[data-testid="stMetricLabel"] { font-size: 0.72rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.6px; color: #8b949e; }
[data-testid="stSidebar"] { border-right: 1px solid #30363d; }
[data-testid="stTabs"] button { font-weight: 600; font-size: 0.9rem; }
.news-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 8px; padding: 12px 16px; margin-bottom: 10px;
}
.news-title { font-size: 0.9rem; font-weight: 500; color: #e6edf3; }
.news-pub   { font-size: 0.78rem; color: #8b949e; margin-top: 4px; }
.section-header {
    font-size: 0.85rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; color: #79c0ff;
    border-bottom: 1px solid #30363d; padding-bottom: 6px; margin-bottom: 12px;
}
.pick-card {
    background: #161b22; border: 1px solid #30363d;
    border-radius: 10px; padding: 16px 20px; margin-bottom: 14px;
}
</style>
""", unsafe_allow_html=True)


# ── Formatting helpers ────────────────────────────────────────────────────────

def fmt_num(n, prefix="$") -> str:
    if n is None: return "N/A"
    try:
        n = float(n)
        if abs(n) >= 1e12: return f"{prefix}{n/1e12:.2f}T"
        if abs(n) >= 1e9:  return f"{prefix}{n/1e9:.2f}B"
        if abs(n) >= 1e6:  return f"{prefix}{n/1e6:.2f}M"
        if abs(n) >= 1e3:  return f"{prefix}{n/1e3:.1f}K"
        return f"{prefix}{n:,.2f}"
    except: return "N/A"

def fmt_pct(n) -> str:
    if n is None: return "N/A"
    try: return f"{float(n)*100:.2f}%"
    except: return "N/A"

def fmt_val(n, d=2) -> str:
    if n is None: return "N/A"
    try: return f"{float(n):,.{d}f}"
    except: return "N/A"


# ── Data fetching ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_info(symbol: str) -> dict:
    return yf.Ticker(symbol).info

@st.cache_data(ttl=300, show_spinner=False)
def fetch_history(symbol: str, period: str) -> pd.DataFrame:
    return yf.Ticker(symbol).history(period=period)

@st.cache_data(ttl=300, show_spinner=False)
def fetch_financials(symbol: str):
    t = yf.Ticker(symbol)
    return t.financials, t.balance_sheet, t.cashflow

@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(symbol: str) -> list:
    try:
        raw = yf.Ticker(symbol).news or []
        result = []
        for item in raw[:8]:
            if "content" in item and isinstance(item["content"], dict):
                c = item["content"]
                result.append({
                    "title":     c.get("title", "Untitled"),
                    "publisher": (c.get("provider") or {}).get("displayName", ""),
                    "date":      c.get("pubDate", "")[:10],
                })
            else:
                result.append({
                    "title":     item.get("title", "Untitled"),
                    "publisher": item.get("publisher", ""),
                    "date":      "",
                })
        return result
    except:
        return []


# ── Chart ─────────────────────────────────────────────────────────────────────

def _rsi(series, period=14):
    d = series.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + g / l.replace(0, float("nan")))

def build_chart(hist: pd.DataFrame, symbol: str) -> go.Figure:
    if hist.empty:
        return go.Figure().update_layout(template="plotly_dark", title="No data")
    h = hist.copy()
    h.index = pd.to_datetime(h.index)
    if h.index.tz is not None:
        h.index = h.index.tz_localize(None)
    close = h["Close"]
    has_ohlc = all(c in h.columns for c in ["Open","High","Low"])
    has_rsi  = len(h) >= 15
    rows = 3 if has_rsi else 2
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.58,0.22,0.20][:rows],
        vertical_spacing=0.03,
        subplot_titles=([symbol,"Volume","RSI 14"] if has_rsi else [symbol,"Volume"]),
    )
    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=h.index, open=h["Open"], high=h["High"], low=h["Low"], close=close,
            increasing_fillcolor="#3fb950", increasing_line_color="#3fb950",
            decreasing_fillcolor="#f85149", decreasing_line_color="#f85149",
            showlegend=False, name="OHLC",
        ), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(
            x=h.index, y=close, mode="lines",
            line=dict(color="#388bfd", width=1.8),
            fill="tozeroy", fillcolor="rgba(56,139,253,0.08)", name="Close",
        ), row=1, col=1)
    for w, col, lbl in [(20,"#3fb950","MA 20"),(50,"#d29922","MA 50"),(200,"#da8b55","MA 200")]:
        if len(h) >= w:
            fig.add_trace(go.Scatter(
                x=h.index, y=close.rolling(w).mean(), mode="lines",
                line=dict(color=col, width=1.3, dash="dot"), name=lbl, opacity=0.85,
            ), row=1, col=1)
    if len(h) >= 20:
        mid = close.rolling(20).mean(); std = close.rolling(20).std()
        fig.add_trace(go.Scatter(x=h.index, y=mid+2*std, mode="lines",
            line=dict(color="rgba(56,139,253,0.3)", width=0.8), showlegend=False, name="BB+"),
            row=1, col=1)
        fig.add_trace(go.Scatter(x=h.index, y=mid-2*std, mode="lines",
            line=dict(color="rgba(56,139,253,0.3)", width=0.8),
            fill="tonexty", fillcolor="rgba(56,139,253,0.05)", showlegend=False, name="BB-"),
            row=1, col=1)
    opens = h.get("Open", close)
    colors = ["#3fb950" if c >= o else "#f85149" for c, o in zip(close, opens)]
    fig.add_trace(go.Bar(x=h.index, y=h["Volume"], marker_color=colors,
        showlegend=False, opacity=0.75, name="Volume"), row=2, col=1)
    if has_rsi:
        rsi = _rsi(close)
        fig.add_trace(go.Scatter(x=h.index, y=rsi, mode="lines",
            line=dict(color="#c9a0dc", width=1.5), name="RSI 14"), row=3, col=1)
        for lv, cl in [(70,"#f85149"),(30,"#3fb950")]:
            fig.add_hline(y=lv, line_dash="dash", line_color=cl, opacity=0.6, row=3, col=1)
        fig.update_yaxes(range=[0,100], row=3, col=1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor="#0d1117", plot_bgcolor="#161b22",
        margin=dict(l=0,r=0,t=40,b=0), xaxis_rangeslider_visible=False, height=680,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    bgcolor="rgba(22,27,34,0.8)"),
    )
    fig.update_yaxes(gridcolor="#30363d", gridwidth=0.5)
    fig.update_xaxes(gridcolor="#30363d", gridwidth=0.5, showgrid=False)
    return fig


# ── Financial table ───────────────────────────────────────────────────────────

def format_financial_df(df):
    if df is None or df.empty:
        return pd.DataFrame({"Note": ["No data available."]})
    out = df.copy()
    out.columns = [c.strftime("%Y-%m-%d") if hasattr(c,"strftime") else str(c) for c in out.columns]
    def _fmt(v):
        if v is None or (isinstance(v, float) and pd.isna(v)): return "—"
        try:
            n = float(v)
            if abs(n) >= 1e12: return f"${n/1e12:.2f}T"
            if abs(n) >= 1e9:  return f"${n/1e9:.2f}B"
            if abs(n) >= 1e6:  return f"${n/1e6:.2f}M"
            return f"${n:,.0f}"
        except: return str(v)
    # pandas 3.0 removed DataFrame.applymap (use .map); older pandas only has applymap.
    try:
        return out.map(_fmt)
    except AttributeError:
        return out.applymap(_fmt)


# ══════════════════════════════════════════════════════════════════════════════
# RULE-BASED ANALYSIS ENGINE  (no API key needed)
# ══════════════════════════════════════════════════════════════════════════════

_SECTOR_PE = {
    "Technology": 28, "Healthcare": 24, "Consumer Cyclical": 22,
    "Consumer Defensive": 20, "Financial Services": 14, "Industrials": 20,
    "Energy": 15, "Utilities": 16, "Real Estate": 35,
    "Basic Materials": 17, "Communication Services": 22,
}

def _g(d, *keys, default=None):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                if isinstance(v, float) and pd.isna(v): continue
            except: pass
            return v
    return default

def _ps(v):
    if v is None: return "N/A"
    try: return f"{float(v)*100:.1f}%"
    except: return "N/A"

def _df_row(df, *names):
    if df is None or df.empty: return []
    for name in names:
        if name in df.index:
            vals = []
            for v in df.loc[name]:
                if v is not None:
                    try:
                        if not pd.isna(v): vals.append(float(v))
                    except: pass
            return vals
    return []

def _col_years(df):
    if df is None or df.empty: return []
    return [c.strftime("%Y") if hasattr(c,"strftime") else str(c)[:4] for c in df.columns]

def _verdict(score):
    if score >= 75: return "🟢 **STRONG BUY**"
    if score >= 60: return "🟢 **BUY**"
    if score >= 45: return "🟡 **HOLD**"
    if score >= 30: return "🟠 **UNDERPERFORM**"
    return "🔴 **SELL / AVOID**"

def _score_stock(info):
    score = 0
    bench = _SECTOR_PE.get(info.get("sector",""), 22)
    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe:
        try:
            f = float(pe)
            if f > 0:
                r = f / bench
                score += 25 if r<0.7 else 18 if r<0.9 else 12 if r<1.1 else 6 if r<1.4 else 2
        except: pass
    pm = info.get("profitMargins")
    if pm:
        try:
            f = float(pm)
            score += 20 if f>0.15 else 14 if f>0.08 else 7 if f>0 else 0
        except: pass
    roe = info.get("returnOnEquity")
    if roe:
        try:
            f = float(roe)
            score += 20 if f>0.2 else 13 if f>0.1 else 5 if f>0 else 0
        except: pass
    rg = info.get("revenueGrowth")
    if rg:
        try:
            f = float(rg)
            score += 20 if f>0.15 else 12 if f>0.05 else 5 if f>0 else 0
        except: pass
    de = info.get("debtToEquity")
    if de:
        try:
            f = float(de)
            score += 10 if f<30 else 7 if f<80 else 3 if f<150 else 0
        except: pass
    target = info.get("targetMeanPrice")
    price  = _g(info, "currentPrice", "regularMarketPrice")
    if target and price:
        try:
            up = (float(target)-float(price))/float(price)*100
            score += 5 if up>30 else 3 if up>10 else 0
        except: pass
    return min(score, 100)

def _value_at_price_score(info: dict) -> int:
    """How attractive is this stock at TODAY'S price?
    Weights FCF yield, earnings yield, upside to target, growth & balance sheet."""
    score = 0

    # Earnings yield (1/PE) — how cheap are earnings relative to price paid
    pe = info.get("forwardPE") or info.get("trailingPE")
    if pe:
        try:
            f = float(pe)
            if f > 0:
                ey = 100.0 / f
                score += 25 if ey > 8 else 18 if ey > 5 else 10 if ey > 3 else 4
        except: pass

    # FCF yield (free cash flow / market cap) — real cash returned per $ of price
    fcf = info.get("freeCashflow")
    mkt = info.get("marketCap")
    if fcf and mkt:
        try:
            fy = float(fcf) / float(mkt) * 100
            score += 25 if fy > 8 else 18 if fy > 5 else 12 if fy > 3 else 5 if fy > 0 else 0
        except: pass

    # Upside to analyst consensus target — market mis-pricing signal
    target = info.get("targetMeanPrice")
    price  = _g(info, "currentPrice", "regularMarketPrice")
    if target and price:
        try:
            up = (float(target) - float(price)) / float(price) * 100
            score += 20 if up > 25 else 14 if up > 10 else 8 if up > 5 else 3 if up > 0 else 0
        except: pass

    # Revenue growth — paying today's price for tomorrow's bigger business
    rg = info.get("revenueGrowth")
    if rg:
        try:
            f = float(rg)
            score += 15 if f > 0.15 else 10 if f > 0.05 else 5 if f > 0 else 0
        except: pass

    # Balance sheet safety — debt risk erodes value of current price
    de = info.get("debtToEquity")
    cr = info.get("currentRatio")
    bs = 0
    if de:
        try:
            f = float(de)
            bs += 8 if f < 30 else 5 if f < 80 else 2 if f < 150 else 0
        except: pass
    if cr:
        try:
            f = float(cr)
            bs += 7 if f > 2 else 5 if f > 1.5 else 2 if f > 1 else 0
        except: pass
    score += bs

    return min(score, 100)


# ── Geopolitical scoring ──────────────────────────────────────────────────────
# Encodes current macro/political environment (as of 2025-2026).
# Adjusts combined score by -15 … +15 based on country + industry exposure.

_GEO_COUNTRY_ADJ = {
    # Strong positives — stable democracies, safe-haven status, resource security
    "United States": 5, "Canada": 4, "Australia": 4, "Switzerland": 4,
    "Norway": 4, "Sweden": 3, "Denmark": 3, "Finland": 3,
    "India": 3,        # neutral geopolitics + high growth
    "Japan": 3,        # stable democracy, tech powerhouse, NATO partner
    "Singapore": 3,    # neutral hub, rule of law
    "Netherlands": 2, "Germany": 1, "United Kingdom": 1, "France": 1,
    "South Korea": 1,  # strong economy, North Korea risk
    "Brazil": -1,      # commodity rich but political volatility
    "Mexico": -2,      # nearshoring benefit vs tariff uncertainty
    "Israel": -3,      # ongoing regional conflict
    "Turkey": -3,      # geopolitical instability, currency risk
    "Argentina": -5,   # economic/political instability
    "Taiwan": -3,      # strong tech sector, but military tension risk
    "Hong Kong": -6,   # eroding autonomy, China regulatory reach
    "China": -9,       # US-China trade war, regulatory risk, delisting threat
    "Russia": -15,     # sanctions, war, international isolation
}

_GEO_INDUSTRY_ADJ = {
    # Tailwinds — defense, reshoring, energy security, critical materials
    "Aerospace & Defense":              12,
    "Defense":                          12,
    "Semiconductors":                    7,  # CHIPS Act, US reshoring drive
    "Semiconductor Equipment":           7,
    "Uranium":                           6,  # nuclear energy renaissance
    "Solar":                             5,  # energy independence, IRA
    "Utilities—Renewable Energy":        5,
    "Renewable Utilities":               5,
    "Oil & Gas E&P":                     4,  # energy security premium
    "Oil & Gas Midstream":               3,
    "Oil & Gas Refining & Marketing":    2,
    "Steel":                             3,  # US tariff protection
    "Aluminum":                          2,
    "Copper":                            2,  # electrification + reshoring
    "Gold":                              1,  # safe-haven demand (kept small — avoids gold-heavy lists)
    "Biotechnology":                     2,  # healthcare security focus
    "Drug Manufacturers—General":        2,
    "Drug Manufacturers—Specialty & Generic": 2,
    # Headwinds — China supply chain, tariff exposure, discretionary risk
    "Consumer Electronics":             -6,  # China mfg + tariffs
    "Electronic Components":            -4,
    "Auto Manufacturers":               -5,  # tariff exposure (Mexico/Canada/China)
    "Auto Parts":                       -5,
    "Specialty Retail":                 -3,  # China import exposure
    "Department Stores":                -3,
    "Apparel Manufacturing":            -5,
    "Apparel Retail":                   -4,
    "Luxury Goods":                     -4,  # China consumer slowdown
    "Footwear & Accessories":           -3,
    "Internet Retail":                  -2,
    "Furnishings Fixtures & Appliances":-3,
    "Tools & Accessories":              -2,
}

# Sector-level nudge when industry isn't in the table
_GEO_SECTOR_ADJ = {
    "Industrials": 3,          # reshoring, infrastructure spending
    "Energy": 2,               # energy security premium
    "Basic Materials": 1,      # commodity demand from reshoring (trimmed to avoid mining-heavy lists)
    "Healthcare": 2,           # healthcare security, pandemic prep
    "Consumer Defensive": 1,   # stable regardless of geo
    "Consumer Cyclical": -2,   # tariff/import sensitivity
    "Technology": 0,           # mixed — some reshoring benefit, some China risk
    "Financial Services": 1,
    "Communication Services": 0,
    "Utilities": 0,
    "Real Estate": 0,
}


def _geo_adjustment(info: dict) -> tuple:
    """Returns (adjustment int -15…+15, short label str, reason str)."""
    country  = info.get("country", "")
    industry = info.get("industry", "")
    sector   = info.get("sector", "")

    adj  = _GEO_COUNTRY_ADJ.get(country, 0)
    adj += _GEO_INDUSTRY_ADJ.get(
        industry,
        _GEO_SECTOR_ADJ.get(sector, 0)   # fall back to sector nudge
    )
    adj = max(-15, min(15, adj))

    # Traffic-light label
    if adj >= 10:    label = "🟢 Strong geo tailwind"
    elif adj >= 5:   label = "🟢 Geo tailwind"
    elif adj >= 1:   label = "🟡 Mild geo positive"
    elif adj == 0:   label = "⚪ Geo neutral"
    elif adj >= -4:  label = "🟡 Mild geo risk"
    elif adj >= -8:  label = "🔴 Geo headwind"
    else:            label = "🔴 High geo risk"

    # Specific reason tag
    if "Defense" in industry or "Aerospace" in industry:
        reason = "NATO/defense spending↑"
    elif "Semiconductor" in industry:
        reason = "CHIPS Act reshoring"
    elif "Solar" in industry or "Renewable" in industry or "Uranium" in industry:
        reason = "energy independence drive"
    elif "Oil" in industry or "Gas" in industry:
        reason = "energy security premium"
    elif "Steel" in industry or "Aluminum" in industry or "Copper" in industry:
        reason = "tariff-protected materials"
    elif country in ("China", "Hong Kong"):
        reason = "US-China trade/regulatory risk"
    elif country == "Russia":
        reason = "sanctions & isolation risk"
    elif "Auto" in industry:
        reason = "auto tariff exposure"
    elif "Apparel" in industry or "Consumer Electronics" in industry:
        reason = "import tariff headwind"
    elif "Luxury" in industry:
        reason = "China consumer slowdown"
    elif country == "India":
        reason = "neutral geopolitics, strong growth"
    else:
        reason = country if country else "diversified exposure"

    return adj, label, reason


def _score_etf(info: dict) -> int:
    """Score an ETF 0-100 based on returns, expense ratio, yield and AUM."""
    score = 0
    # 3-year average annualised return (30 pts)
    r3 = info.get("threeYearAverageReturn")
    if r3:
        try:
            f = float(r3)
            score += 30 if f > 0.15 else 22 if f > 0.10 else 15 if f > 0.05 else 8 if f > 0 else 0
        except: pass
    # 5-year average annualised return (20 pts)
    r5 = info.get("fiveYearAverageReturn")
    if r5:
        try:
            f = float(r5)
            score += 20 if f > 0.12 else 15 if f > 0.08 else 10 if f > 0.04 else 5 if f > 0 else 0
        except: pass
    # YTD return (15 pts)
    ytd = info.get("ytdReturn")
    if ytd:
        try:
            f = float(ytd)
            score += 15 if f > 0.15 else 10 if f > 0.05 else 5 if f > 0 else 0
        except: pass
    # Expense ratio — lower is better (25 pts)
    er = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
    if er:
        try:
            f = float(er)
            score += 25 if f < 0.001 else 20 if f < 0.005 else 15 if f < 0.01 else 10 if f < 0.02 else 5 if f < 0.05 else 2
        except: pass
    # Dividend yield (10 pts)
    yld = info.get("yield") or info.get("trailingAnnualDividendYield")
    if yld:
        try:
            f = float(yld)
            score += 10 if f > 0.03 else 7 if f > 0.015 else 4 if f > 0.005 else 0
        except: pass
    return min(score, 100)


def _diversify_by_sector(results: list, top_n: int, score_key: str = "combined_score",
                         max_per_sector=None) -> list:
    """Select top_n results while capping how many come from any single sector, so the
    shortlist spans different markets instead of piling into one (e.g. mining/gold).
    If there aren't enough distinct sectors, remaining slots are back-filled with the
    next-highest-scoring names."""
    if not results:
        return []
    ranked = sorted(results, key=lambda x: x.get(score_key, 0), reverse=True)
    if max_per_sector is None:
        max_per_sector = max(2, top_n // 5)   # 10→2, 15→3, 20→4, 30→6 → forces variety
    picked, leftovers, counts = [], [], {}
    for r in ranked:
        sec = r.get("sector") or "Unknown"
        if counts.get(sec, 0) < max_per_sector:
            picked.append(r)
            counts[sec] = counts.get(sec, 0) + 1
        else:
            leftovers.append(r)
    out = picked[:top_n]
    if len(out) < top_n:                       # not enough sectors — backfill by score
        out = out + leftovers[: top_n - len(out)]
    return out


def run_local_analysis(analysis_type, symbol, info, income, balance, cashflow):
    sector = info.get("sector","")
    name   = info.get("longName") or info.get("shortName") or symbol
    fns = {
        "overview":      lambda: _la_overview(symbol, name, sector, info),
        "income":        lambda: _la_income(symbol, name, info, income),
        "balance_sheet": lambda: _la_balance(symbol, name, info, balance),
        "cash_flow":     lambda: _la_cashflow(symbol, name, info, cashflow),
        "risk":          lambda: _la_risk(symbol, name, sector, info),
        "thesis":        lambda: _la_thesis(symbol, name, sector, info),
    }
    fn = fns.get(analysis_type)
    return fn() if fn else "Unknown analysis type."


def _la_overview(symbol, name, sector, info):
    price  = _g(info, "currentPrice", "regularMarketPrice")
    hi52   = info.get("fiftyTwoWeekHigh")
    lo52   = info.get("fiftyTwoWeekLow")
    pe_fwd = info.get("forwardPE")
    pe_tr  = info.get("trailingPE")
    pb     = info.get("priceToBook")
    ev_eb  = info.get("enterpriseToEbitda")
    roe    = info.get("returnOnEquity")
    pm     = info.get("profitMargins")
    gm     = info.get("grossMargins")
    om     = info.get("operatingMargins")
    de     = info.get("debtToEquity")
    cr     = info.get("currentRatio")
    rg     = info.get("revenueGrowth")
    eg     = info.get("earningsGrowth")
    target = info.get("targetMeanPrice")
    rec    = (info.get("recommendationKey") or "N/A").upper()
    n_ana  = info.get("numberOfAnalystOpinions","N/A")
    bench  = _SECTOR_PE.get(sector, 22)
    score  = 0
    findings = []

    pe = pe_fwd or pe_tr
    if pe:
        try:
            f = float(pe)
            if f <= 0:
                findings.append("🔴 P/E negative — company currently unprofitable")
            else:
                r = f / bench
                if r < 0.7:   score+=25; findings.append(f"✅ P/E {f:.1f}x — significant discount to {bench}x sector avg")
                elif r < 0.9:  score+=20; findings.append(f"✅ P/E {f:.1f}x — modest discount to {bench}x sector avg")
                elif r < 1.15: score+=14; findings.append(f"⚪ P/E {f:.1f}x — in line with {bench}x sector avg (fair value)")
                elif r < 1.4:  score+=8;  findings.append(f"⚠️ P/E {f:.1f}x — premium to {bench}x sector avg")
                else:          score+=3;  findings.append(f"🔴 P/E {f:.1f}x — large premium to {bench}x sector avg")
        except: pass
    if pb:
        try:
            f = float(pb)
            if f<1:   score+=5; findings.append(f"✅ P/B {f:.2f}x — trading below book value")
            elif f<3: score+=3; findings.append(f"⚪ P/B {f:.2f}x — reasonable")
            else:     score+=1; findings.append(f"⚠️ P/B {f:.2f}x — premium to book")
        except: pass
    if pm:
        try:
            f = float(pm)
            if f>0.25:   score+=15; findings.append(f"✅ Net margin {f*100:.1f}% — exceptional profitability")
            elif f>0.12:  score+=12; findings.append(f"✅ Net margin {f*100:.1f}% — solid profitability")
            elif f>0.05:  score+=8;  findings.append(f"⚪ Net margin {f*100:.1f}% — decent margins")
            elif f>0:     score+=4;  findings.append(f"⚠️ Net margin {f*100:.1f}% — thin margins")
            else:         findings.append(f"🔴 Net margin {f*100:.1f}% — loss-making")
        except: pass
    if roe:
        try:
            f = float(roe)
            if f>0.25:   score+=15; findings.append(f"✅ ROE {f*100:.1f}% — excellent return on equity")
            elif f>0.12:  score+=10; findings.append(f"⚪ ROE {f*100:.1f}% — adequate returns")
            elif f>0:     score+=5;  findings.append(f"⚠️ ROE {f*100:.1f}% — below-average returns")
            else:         findings.append(f"🔴 ROE {f*100:.1f}% — destroying equity value")
        except: pass
    if rg:
        try:
            f = float(rg)
            if f>0.20:   score+=10; findings.append(f"✅ Revenue growth {f*100:.1f}% — strong expansion")
            elif f>0.05:  score+=7;  findings.append(f"⚪ Revenue growth {f*100:.1f}% — moderate")
            elif f>0:     score+=3;  findings.append(f"⚠️ Revenue growth {f*100:.1f}% — sluggish")
            else:         score+=1;  findings.append(f"🔴 Revenue declining {f*100:.1f}%")
        except: pass
    if eg:
        try:
            f = float(eg)
            if f>0.15:  score+=10; findings.append(f"✅ Earnings growth {f*100:.1f}% — strong momentum")
            elif f>0:    score+=6;  findings.append(f"⚪ Earnings growth {f*100:.1f}%")
            else:        score+=1;  findings.append(f"⚠️ Earnings declining {f*100:.1f}%")
        except: pass
    if de:
        try:
            f = float(de)
            if f<30:    score+=10; findings.append(f"✅ D/E {f:.0f}% — strong balance sheet")
            elif f<100:  score+=7;  findings.append(f"⚪ D/E {f:.0f}% — manageable leverage")
            elif f<200:  score+=3;  findings.append(f"⚠️ D/E {f:.0f}% — elevated leverage")
            else:        findings.append(f"🔴 D/E {f:.0f}% — high debt load")
        except: pass
    if cr:
        try:
            f = float(cr)
            if f>2:   findings.append(f"✅ Current ratio {f:.2f} — strong liquidity")
            elif f>1:  findings.append(f"⚪ Current ratio {f:.2f} — adequate liquidity")
            else:      findings.append(f"🔴 Current ratio {f:.2f} — liquidity concern")
        except: pass
    if target and price:
        try:
            up = (float(target)-float(price))/float(price)*100
            if up>20:   score+=8; findings.append(f"✅ Analyst target ${float(target):.2f} = {up:.0f}% upside ({n_ana} analysts)")
            elif up>5:   score+=5; findings.append(f"⚪ Analyst target ${float(target):.2f} = {up:.0f}% upside ({n_ana} analysts)")
            elif up>-5:  score+=3; findings.append(f"⚪ Analyst target ${float(target):.2f} ≈ current price")
            else:        score+=1; findings.append(f"⚠️ Analyst target ${float(target):.2f} = {up:.0f}% downside")
        except: pass

    pos52 = ""
    if price and hi52 and lo52:
        try:
            pos = (float(price)-float(lo52)) / max(float(hi52)-float(lo52), 0.01) * 100
            pos52 = f"52-week position: **{pos:.0f}%** from low  (${lo52:.2f} — ${hi52:.2f})"
        except: pass

    desc = info.get("longBusinessSummary","No description available.")
    if len(desc) > 700: desc = desc[:700]+"…"

    out = [
        f"## 🏢 {name} ({symbol}) — Company Overview",
        f"**{sector or 'N/A'}** · {info.get('industry','N/A')} · {info.get('country','N/A')}",
        pos52, "",
        "### Business Summary", desc, "",
        "### Valuation Snapshot",
        "| Metric | Value | Sector Benchmark |", "|---|---|---|",
        f"| Forward P/E | {fmt_val(pe_fwd)} | {bench}x |",
        f"| Trailing P/E | {fmt_val(pe_tr)} | — |",
        f"| Price / Book | {fmt_val(pb)} | — |",
        f"| EV / EBITDA | {fmt_val(ev_eb)} | — |",
        f"| Gross Margin | {_ps(gm)} | — |",
        f"| Operating Margin | {_ps(om)} | — |",
        f"| Net Margin | {_ps(pm)} | — |",
        f"| ROE | {_ps(roe)} | — |",
        "", "### Key Findings",
    ] + [f"- {f}" for f in findings] + [
        "", "---",
        f"### Overall Score: **{score} / 100** → {_verdict(score)}",
        f"*Analyst consensus: {rec} · {n_ana} analysts · Mean target: ${fmt_val(target) if target else 'N/A'}*",
    ]
    return "\n".join(str(x) for x in out)


def _la_income(symbol, name, info, df):
    years = _col_years(df)
    rev   = _df_row(df, "Total Revenue")
    gp    = _df_row(df, "Gross Profit")
    oi    = _df_row(df, "Operating Income", "EBIT")
    ni    = _df_row(df, "Net Income")
    findings = []; score = 0

    if len(rev)>=2 and rev[1]:
        rg = (rev[0]-rev[1])/abs(rev[1])
        if rg>0.15:   score+=25; findings.append(f"✅ Revenue grew {rg*100:.1f}% YoY — strong growth")
        elif rg>0.03:  score+=18; findings.append(f"⚪ Revenue grew {rg*100:.1f}% YoY — moderate")
        elif rg>0:     score+=10; findings.append(f"⚠️ Revenue grew {rg*100:.1f}% YoY — near-flat")
        else:          score+=3;  findings.append(f"🔴 Revenue declined {rg*100:.1f}% YoY")
        if len(rev)>=3 and rev[2]:
            rg2 = (rev[1]-rev[2])/abs(rev[2])
            findings.append(f"⚪ Prior-year growth {rg2*100:.1f}% → trend {'accelerating ✅' if rg>rg2 else 'decelerating ⚠️'}")
    if rev and gp and rev[0]:
        gm = gp[0]/rev[0]
        if gm>0.5:    score+=20; findings.append(f"✅ Gross margin {gm*100:.1f}% — very strong")
        elif gm>0.3:   score+=15; findings.append(f"⚪ Gross margin {gm*100:.1f}% — healthy")
        elif gm>0.15:  score+=8;  findings.append(f"⚠️ Gross margin {gm*100:.1f}% — below average")
        else:          score+=3;  findings.append(f"🔴 Gross margin {gm*100:.1f}% — low-margin business")
        if len(gp)>1 and len(rev)>1 and rev[1]:
            gm1 = gp[1]/rev[1]
            findings.append(f"⚪ Gross margin {'expanding ✅' if gm>gm1 else 'contracting ⚠️'} vs {gm1*100:.1f}% prior year")
    if rev and ni and rev[0]:
        nm = ni[0]/rev[0]
        if nm>0.15:   score+=20; findings.append(f"✅ Net margin {nm*100:.1f}% — excellent")
        elif nm>0.07:  score+=14; findings.append(f"⚪ Net margin {nm*100:.1f}% — solid")
        elif nm>0:     score+=7;  findings.append(f"⚠️ Net margin {nm*100:.1f}% — thin")
        else:          findings.append(f"🔴 Net margin {nm*100:.1f}% — loss-making")
    if len(oi)>=2 and oi[1]:
        oig = (oi[0]-oi[1])/abs(oi[1])
        if oig>0.1:   score+=15; findings.append(f"✅ Operating income grew {oig*100:.1f}% — leverage working")
        elif oig>0:    score+=8;  findings.append(f"⚪ Operating income grew {oig*100:.1f}%")
        else:          findings.append(f"⚠️ Operating income declined {oig*100:.1f}%")

    rows = []
    for i, yr in enumerate(years[:4]):
        rows.append(f"| {yr} | {fmt_num(rev[i]) if i<len(rev) else '—'} | {fmt_num(gp[i]) if i<len(gp) else '—'} | {fmt_num(oi[i]) if i<len(oi) else '—'} | {fmt_num(ni[i]) if i<len(ni) else '—'} |")

    return "\n".join([
        f"## 📊 Income Statement Analysis — {name} ({symbol})", "",
        "### Annual Figures",
        "| Year | Revenue | Gross Profit | Operating Income | Net Income |",
        "|---|---|---|---|---|",
    ] + rows + ["", "### Key Findings"] + [f"- {f}" for f in findings] + [
        "", "---", f"### Income Quality Score: **{min(score,100)} / 100** → {_verdict(min(score,100))}",
    ])


def _la_balance(symbol, name, info, df):
    years = _col_years(df)
    cash  = _df_row(df, "Cash And Cash Equivalents",
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Short Term Investments")
    ta    = _df_row(df, "Total Assets")
    td    = _df_row(df, "Total Debt", "Long Term Debt")
    eq    = _df_row(df, "Stockholders Equity", "Total Equity Gross Minority Interest")
    findings = []; score = 0

    de = info.get("debtToEquity"); cr = info.get("currentRatio"); qr = info.get("quickRatio")
    if de:
        try:
            f = float(de)
            if f<30:    score+=25; findings.append(f"✅ D/E {f:.0f}% — very low leverage")
            elif f<80:   score+=20; findings.append(f"✅ D/E {f:.0f}% — manageable leverage")
            elif f<150:  score+=12; findings.append(f"⚠️ D/E {f:.0f}% — elevated, watch coverage")
            elif f<300:  score+=5;  findings.append(f"🔴 D/E {f:.0f}% — high leverage")
            else:        findings.append(f"🔴 D/E {f:.0f}% — dangerously levered")
        except: pass
    if cr:
        try:
            f = float(cr)
            if f>2:    score+=20; findings.append(f"✅ Current ratio {f:.2f} — strong liquidity")
            elif f>1.5: score+=15; findings.append(f"✅ Current ratio {f:.2f} — healthy")
            elif f>1:   score+=8;  findings.append(f"⚪ Current ratio {f:.2f} — adequate")
            else:       findings.append(f"🔴 Current ratio {f:.2f} — liquidity squeeze risk")
        except: pass
    if qr:
        try:
            f = float(qr)
            if f>1.5:  score+=10; findings.append(f"✅ Quick ratio {f:.2f} — excellent")
            elif f>1:   score+=6;  findings.append(f"⚪ Quick ratio {f:.2f} — adequate")
            else:       findings.append(f"⚠️ Quick ratio {f:.2f} — lean liquid assets")
        except: pass
    if len(cash)>=2 and cash[1]:
        cg = (cash[0]-cash[1])/abs(cash[1])
        if cg>0.1:  score+=15; findings.append(f"✅ Cash grew {cg*100:.1f}% YoY")
        elif cg>0:   score+=8;  findings.append(f"⚪ Cash grew {cg*100:.1f}% YoY")
        else:        findings.append(f"⚠️ Cash declined {cg*100:.1f}% YoY")
    if len(td)>=2 and td[1]:
        dg = (td[0]-td[1])/abs(td[1])
        if dg<-0.1:  score+=10; findings.append(f"✅ Debt reduced {abs(dg)*100:.1f}% YoY — deleveraging")
        elif dg<0.05: score+=5;  findings.append(f"⚪ Debt stable")
        else:         findings.append(f"⚠️ Debt grew {dg*100:.1f}% YoY")

    rows = []
    for i, yr in enumerate(years[:4]):
        rows.append(f"| {yr} | {fmt_num(cash[i]) if i<len(cash) else '—'} | {fmt_num(ta[i]) if i<len(ta) else '—'} | {fmt_num(td[i]) if i<len(td) else '—'} | {fmt_num(eq[i]) if i<len(eq) else '—'} |")

    return "\n".join([
        f"## 🏦 Balance Sheet Analysis — {name} ({symbol})", "",
        "### Balance Sheet Summary",
        "| Year | Cash | Total Assets | Total Debt | Equity |",
        "|---|---|---|---|---|",
    ] + rows + ["", "### Key Findings"] + [f"- {f}" for f in findings] + [
        "", "---", f"### Balance Sheet Score: **{min(score,100)} / 100** → {_verdict(min(score,100))}",
    ])


def _la_cashflow(symbol, name, info, df):
    years = _col_years(df)
    ocf   = _df_row(df, "Operating Cash Flow", "Cash Flow From Operations")
    capex = _df_row(df, "Capital Expenditure", "Purchase Of Property Plant And Equipment")
    div   = _df_row(df, "Common Stock Dividend Paid", "Payment Of Dividends")
    mktcap = info.get("marketCap"); rev = info.get("totalRevenue")
    findings = []; score = 0

    fcf_list = []
    for i in range(min(len(ocf), len(capex))):
        cap = capex[i]
        if isinstance(cap, (int,float)) and cap < 0: cap = -cap
        fcf_list.append(ocf[i] - cap)

    if fcf_list:
        fcf = fcf_list[0]
        if fcf > 0: score+=30; findings.append(f"✅ Free Cash Flow: {fmt_num(fcf)} — cash generative")
        else:        findings.append(f"🔴 Free Cash Flow: {fmt_num(fcf)} — burning cash")
        if mktcap and mktcap > 0:
            try:
                fy = fcf/mktcap*100
                if fy>5:   score+=20; findings.append(f"✅ FCF yield {fy:.1f}% — attractive")
                elif fy>2:  score+=12; findings.append(f"⚪ FCF yield {fy:.1f}% — decent")
                elif fy>0:  score+=5;  findings.append(f"⚠️ FCF yield {fy:.1f}% — thin")
                else:       findings.append(f"🔴 Negative FCF yield")
            except: pass
        if rev and rev > 0:
            try:
                fm = fcf/rev*100
                if fm>15:  score+=20; findings.append(f"✅ FCF margin {fm:.1f}% — exceptional cash conversion")
                elif fm>8:  score+=14; findings.append(f"⚪ FCF margin {fm:.1f}% — solid")
                elif fm>0:  score+=7;  findings.append(f"⚠️ FCF margin {fm:.1f}% — thin")
            except: pass
        if len(fcf_list)>=2 and fcf_list[1]:
            fg = (fcf_list[0]-fcf_list[1])/abs(fcf_list[1])
            if fg>0.1:  score+=15; findings.append(f"✅ FCF grew {fg*100:.1f}% YoY")
            elif fg>0:   score+=8;  findings.append(f"⚪ FCF grew {fg*100:.1f}% YoY")
            else:        findings.append(f"⚠️ FCF declined {fg*100:.1f}% YoY")
    if ocf and capex:
        try:
            ratio = abs(capex[0])/ocf[0] if ocf[0] else 0
            if ratio<0.15:  score+=15; findings.append(f"✅ CapEx {ratio*100:.0f}% of OCF — asset-light model")
            elif ratio<0.4:  score+=10; findings.append(f"⚪ CapEx {ratio*100:.0f}% of OCF — moderate")
            else:            score+=4;  findings.append(f"⚠️ CapEx {ratio*100:.0f}% of OCF — capital-intensive")
        except: pass

    rows = []
    for i, yr in enumerate(years[:4]):
        rows.append(f"| {yr} | {fmt_num(ocf[i]) if i<len(ocf) else '—'} | {fmt_num(capex[i]) if i<len(capex) else '—'} | {fmt_num(fcf_list[i]) if i<len(fcf_list) else '—'} | {fmt_num(div[i]) if i<len(div) else '—'} |")

    return "\n".join([
        f"## 💸 Cash Flow Analysis — {name} ({symbol})", "",
        "### Cash Flow Table",
        "| Year | Operating CF | CapEx | Free Cash Flow | Dividends |",
        "|---|---|---|---|---|",
    ] + rows + ["", "### Key Findings"] + [f"- {f}" for f in findings] + [
        "", "---", f"### Cash Flow Score: **{min(score,100)} / 100** → {_verdict(min(score,100))}",
    ])


def _la_risk(symbol, name, sector, info):
    beta  = info.get("beta"); de = info.get("debtToEquity")
    cr    = info.get("currentRatio"); pm = info.get("profitMargins")
    short = info.get("shortPercentOfFloat"); rg = info.get("revenueGrowth")
    price = _g(info,"currentPrice","regularMarketPrice"); hi52 = info.get("fiftyTwoWeekHigh")

    def _r(v, t1, t2, rev=False):
        if v is None: return "⚪ N/A"
        f = float(v)
        if not rev: return "🟢 Low" if f<=t1 else "🟡 Medium" if f<=t2 else "🔴 High"
        else:       return "🟢 Low" if f>=t1 else "🟡 Medium" if f>=t2 else "🔴 High"

    rows = [
        ("Market Risk",    beta,  _r(beta,0.8,1.3),   f"Beta {float(beta):.2f}" if beta else "N/A"),
        ("Financial Risk", de,    _r(de,50,150),       f"D/E {float(de):.0f}%" if de else "N/A"),
        ("Business Risk",  pm,    _r(pm,0.08,0,rev=True), f"Net margin {_ps(pm)}"),
        ("Liquidity Risk", cr,    _r(cr,1.5,1.0,rev=True), f"Current ratio {float(cr):.2f}" if cr else "N/A"),
        ("Short Interest", short, _r(short,0.05,0.15), f"Short float {_ps(short)}"),
        ("Growth Risk",    rg,    _r(rg,0.05,0,rev=True),  f"Revenue growth {_ps(rg)}"),
    ]
    reds    = sum(1 for *_,r,_ in rows if "🔴" in r)
    yellows = sum(1 for *_,r,_ in rows if "🟡" in r)
    if reds>=3:      overall = "🔴 **VERY HIGH** — multiple serious risk flags"
    elif reds>=2:     overall = "🔴 **HIGH** — significant risks identified"
    elif yellows>=3:  overall = "🟡 **MEDIUM** — several moderate risks"
    elif yellows>=1:  overall = "🟡 **LOW-MEDIUM** — manageable profile"
    else:             overall = "🟢 **LOW** — solid risk profile"

    dd = ""
    if price and hi52:
        try: dd = f"\n- Drawdown from 52-week high: **{(float(hi52)-float(price))/float(hi52)*100:.1f}%**"
        except: pass

    out = [f"## ⚠️ Risk Assessment — {name} ({symbol})", "",
           "| Category | Rating | Detail |", "|---|---|---|"]
    for label, _, rating, detail in rows:
        out.append(f"| {label} | {rating} | {detail} |")
    out += ["", "### Notes",
            f"- Sector: **{sector or 'N/A'}** — sector-specific regulatory/cyclical risks apply" + dd,
            "", "---", f"### Overall Risk Rating: {overall}"]
    return "\n".join(out)


def _la_thesis(symbol, name, sector, info):
    price  = _g(info,"currentPrice","regularMarketPrice")
    target = info.get("targetMeanPrice")
    pe     = info.get("forwardPE") or info.get("trailingPE")
    bench  = _SECTOR_PE.get(sector, 22)
    pm = info.get("profitMargins"); roe = info.get("returnOnEquity")
    rg = info.get("revenueGrowth"); de  = info.get("debtToEquity")
    beta = info.get("beta"); short = info.get("shortPercentOfFloat")
    rec  = (info.get("recommendationKey") or "N/A").upper()
    n_ana = info.get("numberOfAnalystOpinions","N/A")
    eps  = info.get("trailingEps")

    bull_t = bear_t = base_t = None
    if pe and eps and price:
        try:
            f = float(pe); e = float(eps)
            if e > 0:
                bull_t = f * 1.25 * e * 1.15
                bear_t = f * 0.80 * e * 0.90
                base_t = float(target) if target else f * e * 1.05
        except: pass

    def tgt(v): return f"${v:.2f}" if v else (f"${float(target):.2f}" if target else "N/A")
    def up(v):
        if v and price:
            try: return f" ({(v-float(price))/float(price)*100:+.0f}%)"
            except: pass
        return ""

    bull_r = []
    if rg and float(rg)>0.1:   bull_r.append(f"Strong revenue growth at {_ps(rg)} YoY")
    if pm and float(pm)>0.1:   bull_r.append(f"High-quality margins — net margin {_ps(pm)}")
    if roe and float(roe)>0.15: bull_r.append(f"Excellent capital returns — ROE {_ps(roe)}")
    if de and float(de)<60:     bull_r.append("Clean balance sheet supports buybacks & dividends")
    if target and price and float(target)>float(price)*1.1:
        bull_r.append(f"Analysts see {(float(target)-float(price))/float(price)*100:.0f}% upside")
    while len(bull_r) < 3: bull_r.append("Sector tailwinds and durable competitive advantages")

    bear_r = []
    if pe and float(pe)>bench*1.3:   bear_r.append(f"Elevated valuation ({fmt_val(pe)}x P/E vs {bench}x sector avg)")
    if rg and float(rg)<0.03:        bear_r.append(f"Slowing top-line growth ({_ps(rg)} YoY)")
    if de and float(de)>150:         bear_r.append(f"High leverage D/E {float(de):.0f}% limits flexibility")
    if short and float(short)>0.08:  bear_r.append(f"Elevated short interest ({_ps(short)}) signals skepticism")
    if beta and float(beta)>1.5:     bear_r.append(f"High beta ({float(beta):.2f}) amplifies downside in risk-off")
    while len(bear_r) < 3: bear_r.append("Macro headwinds and potential multiple compression")

    if rec in ("STRONGBUY","STRONG_BUY"): final_rec = "⭐ **Strong Buy**"
    elif rec=="BUY":          final_rec = "✅ **Buy**"
    elif rec=="HOLD":         final_rec = "⚪ **Hold**"
    elif rec in ("UNDERPERFORM","SELL"): final_rec = "🔴 **Sell / Underperform**"
    else:                     final_rec = "⚪ **Hold / Neutral**"

    lines = [
        f"## 🎯 Investment Thesis — {name} ({symbol})",
        f"**{sector or 'N/A'}**  ·  Current price: **${float(price):,.2f}**" if price else "",
        "", "---",
        "### 🐂 Bull Case", f"**Target: {tgt(bull_t)}{up(bull_t)}**", "",
    ]
    for i,r in enumerate(bull_r[:3]): lines.append(f"{i+1}. {r}")
    lines += ["", "---", "### 🐻 Bear Case", f"**Target: {tgt(bear_t)}{up(bear_t)}**", ""]
    for i,r in enumerate(bear_r[:3]): lines.append(f"{i+1}. {r}")
    lines += [
        "", "---",
        "### 📊 Base Case (12-Month)", f"**Target: {tgt(base_t)}{up(base_t)}**", "",
        f"Expects the stock to trade near sector-average multiples ({bench}x P/E) with "
        f"{'steady revenue growth' if (rg or 0)>0.03 else 'margin improvement'} as the primary driver.",
        "", "---",
        f"### Recommendation: {final_rec}",
        f"*{n_ana} analyst consensus → {rec}  ·  Mean target: ${fmt_val(target) if target else 'N/A'}*",
        "", "**Key Catalysts to Watch:**",
        "- Quarterly earnings vs. consensus estimates",
        f"- Revenue growth trajectory (currently {_ps(rg)})",
        f"- Margin trends (current net margin {_ps(pm)})",
        "- Guidance revisions from management",
    ]
    return "\n".join(l for l in lines if l is not None)


# ══════════════════════════════════════════════════════════════════════════════
# STOCK SCREENER
# ══════════════════════════════════════════════════════════════════════════════

# ── Extended ticker universe ──────────────────────────────────────────────────
# Supplements the S&P 500 (fetched dynamically) with ~550 additional tickers.

_EXTRA_TICKERS = [
    # Cloud / SaaS / Software
    "PLTR","SNOW","DDOG","NET","CRWD","ZS","OKTA","MDB","CFLT","GTLB",
    "MNDY","BILL","ASAN","S","IOT","HUBS","TWLO","ZI","PCTY","PAYC",
    "APPN","FIVN","TOST","BRZE","SMAR","ESTC","MANH","NCNO","DOCN","NICE",
    "PRFT","ALRM","BAND","PEGA","AZPN","PTC","EPAM","GLOB","JAMF","TASK",
    "ALKT","WEAVE","FSLY","WK","EGHT","LPSN","BIGC","SPSC","API","GTLB",
    # Semiconductors (smaller / mid)
    "WOLF","AMBA","SLAB","DIOD","MTSI","FORM","ONTO","COHU","POWI","SITM",
    "ALGM","CRUS","SMTC","AEHR","IIVI","UCTT","ICHR","MKSI","LSCC","OSIS",
    "ACLS","RMBS","PDFS","NVEC","AOSL","HIMX","CAMT","IPGP","SWKS","QRVO",
    # Fintech / Crypto / Insurance / Lending
    "COIN","HOOD","SOFI","AFRM","UPST","OPEN","TREE","LMND","ROOT","TRUP",
    "COOP","UWMC","PFSI","CURO","GDOT","MGNI","NU","STNE","PAGS","PSFE",
    "PRAA","ENVA","WRLD","CACC","SLM","NAVI","ECPG","EZPW","RCII","LPLA",
    "COWN","MQ","STEP","ACGL","RNR","RYAN","KINS","HGTY","RDFN","COMP",
    # EV / Clean Mobility / Space
    "RIVN","LCID","NIO","XPEV","LI","FSR","WKHS","NKLA","PTRA",
    "JOBY","ACHR","LILM","BLDE","SPCE","RKLB","ASTS","IONQ","ARQQ","SOUN",
    # Consumer Platform / Sharing Economy
    "ABNB","DASH","LYFT","RBLX","SNAP","PINS","BMBL","ANGI","YELP","TRIP",
    "CARS","OSTK","ETSY","POSH","REAL","RENT","WISH","XPOF","GOOS","RVLV",
    # Retail / Consumer Brands
    "FIVE","OLLI","BJ","PSMT","ARKO","CASY","MUSA","WING","SHAK","JACK",
    "FAT","CBRL","DENN","DINE","BYND","SMPL","NOMD","HIMS","ELF","OLPX",
    "NTRP","HELE","PRGO","LWAY","CHWY","PETS","WOOF","SKX","CROX","VRA",
    "CPRI","TPR","PVH","HBI","ONON","BKE","URBN","EXPR","BIRD","PLBY",
    "RH","WSM","ARHAUS","PRPL","SNBR","LESL","BIRD","DRVN","FRGE","WW",
    # Healthcare / Biotech
    "RXRX","SEER","BEAM","EDIT","NTLA","CRSP","PACB","TXG","NTRA","ACAD",
    "SAGE","SRPT","BLUE","FOLD","BMRN","RARE","ALNY","ARWR","IONS","NBIX",
    "HALO","EXEL","INCY","NKTR","KYMR","ARVN","KRTX","PRAX","IMVT","KRYS",
    "MDGL","VKTX","DAWN","IOVA","NVAX","INO","VXRT","ARCT","HZNP","SGEN",
    "RCKT","MGTX","QURE","AVXL","DRNA","DVAX","AGEN","ONCE","RGNX","DNLI",
    "GRPH","CDNA","VERA","TALK","FATE","PRLD","IMGO","RVMD","KALA","DICE",
    # Healthcare Services / Devices
    "TDOC","ONEM","PHR","ACCD","AMWL","SGFY","GDRX","EXAS","SDGR","STAA",
    "NVCR","OMCL","AXNX","NXGN","SWAV","MDRX","VEEV","ICLR","MEDP","HLNE",
    "GMED","NUVA","TMDX","MMSI","OFIX","IART","CNMD","HAYW","ITGR","NVST",
    "IRTC","INSP","AORT","NARI","ATRC","QDEL","ALGN","LNTH","NVCR","IRTC",
    # Energy — Oil & Gas
    "AR","RRC","EQT","CNX","CTRA","SM","NOG","SWN","GPOR","MNRL",
    "ESTE","SBOW","CRK","REI","CDEV","CRC","MTDR","VTLE","CIVI","DKL",
    "SUN","HFC","PARR","CAPL","CVR","DINO","PBF","TRGP","AM","HESM",
    # Clean Energy / Renewables
    "PLUG","FCEL","BLDP","CWEN","CLNE","GEVO","NOVA","RUN","SPWR","ARRY",
    "CSIQ","JKS","DAQO","MAXN","FSLR","BE","AMRC","HASI","NEP","SHLS",
    "STEM","HYZON","BEEM","SUNW","PECK","ORA","WEST","ALTM","NRGV","FLUX",
    # Defense / Aerospace
    "KTOS","AVAV","BWXT","MRCY","DRS","CACI","SAIC","MAXR","LDOS","PSN",
    "MOOG","HEICO","TGI","KAMAN","TPIC","ASEI","VSE","ESLT","TDY","AJRD",
    # Materials / Mining / Metals
    "MP","LAC","LTHM","PLL","CLF","STLD","CMC","ZEUS","GEF","SLVM",
    "AUY","PAAS","HL","SSRM","SILV","AG","MUX","NGD","GSS","MAG",
    "KGC","AU","HMY","BTG","SAND","WPM","AEM","AGI","TREX","HWKN",
    "AA","CENX","KALU","ARNC","MTRN","CSTM","MATV","STRL","USCR","MLM",
    # International ADRs — Asia Pacific
    "TSM","BABA","JD","PDD","BIDU","SE","GRAB","TCOM","FUTU","TIGR",
    "WIT","HDB","IBN","INFY","TTM","WNS","CTSH","ERIC","NOK","ASML",
    "SAP","NMR","MFG","SMFG","KB","SHG","SONY","TM","HMC","MUFG",
    # International ADRs — Europe
    "SHEL","BP","GSK","AZN","HSBC","BCS","LYG","RELX","UL","RIO",
    "ABB","ORAN","TEF","VOD","BT","WPP","PHG","ING","UBS","CS",
    "CNI","CP","ENB","TRP","SU","CVE","IMO","BNS","BMO","RY",
    # International ADRs — Latin America
    "VALE","XP","ITUB","BBD","BRFS","ABEV","STNE","PAGS","PAM","VIST",
    "NU","BBAR","GGAL","SID","GGB","LOMA","CEPU","TGS","YPF","MELI",
    # REITs
    "COLD","IIPR","AIV","UE","ROIC","STAG","GTY","NTST","CLDT","RHP",
    "SHO","PK","APLE","SAFE","TRNO","EGP","REXR","VRE","SITC","BRT",
    "GMRE","NXRT","INN","GOOD","ILPT","NNN","WPC","LAND","PINE","EPRT",
    "BRSP","DBRG","CTO","ALEX","PLYM","MODV","GIPR","FCRE","IIPR","NLCP",
    # Gaming / Entertainment / Media
    "DKNG","PENN","SEAS","CNK","AMC","IMAX","EPR","MTN","EVRI","BYD",
    "SGMS","ACEL","GAN","GNOG","GMBL","CHDN","BALY","FUBO","AMCX","PARA",
    "WBD","FOXA","FOX","SIRI","FWONA","FWONK","NWSA","NWS","IHRT","LUMN",
    # Regional Banks
    "WAL","OZK","CUBI","BANR","FULT","BOKF","SNV","IBOC","FFIN","FNB",
    "UCBI","SFNC","CATY","TBK","HAFC","TRMK","WSBC","FBMS","CVBF","WAFD",
    "WTFC","WSFS","TOWN","SBCF","HTLF","HOMB","FFBC","NBTB","CTBI","BHLB",
    "EFSC","LCNB","FBIZ","BSVN","KRNY","ESSA","MVBF","FFNW","BANF","HTBK",
    "PPBI","PFIS","BANC","HOPE","CVB","PNFP","TCBK","NBTB","CORE","SFBS",
    # Misc / Industrial / Other
    "RGEN","ITRI","TRMB","LFUS","EXPO","RBC","GFF","LSCC","OSIS","UCTT",
    "ICHR","PSFE","OPFI","EZCORP","ENVA","WRLD","ECPG","RCII","UHAL","FWRD",
    "XPOF","PTON","SKLZ","DKNG","OPEN","NKTR","SMAR","SPWH","BOOT","YETI",
]


# ── International market pools ────────────────────────────────────────────────
# Prices are in local currency (GBp for .L, EUR for .DE/.PA, AUD for .AX, etc.)

_FTSE100 = [
    "III.L","AAF.L","AAL.L","ABF.L","ADM.L","AHT.L","ANTO.L","AUTO.L","AV.L",
    "AZN.L","BA.L","BARC.L","BATS.L","BDEV.L","BEZ.L","BLND.L","BNZL.L",
    "BP.L","BRBY.L","BT-A.L","CPG.L","CNA.L","CRH.L","DCC.L","DGE.L",
    "DLG.L","DPLM.L","ENT.L","EXPN.L","EZJ.L","FERG.L","FLTR.L","FRES.L",
    "GLEN.L","GSK.L","HLMA.L","HLN.L","HIK.L","HSX.L","HSBA.L","IAG.L",
    "IHG.L","IMB.L","IMI.L","INF.L","ICP.L","JD.L","KGF.L","LAND.L",
    "LGEN.L","LLOY.L","LRE.L","LSEG.L","MKS.L","MNDI.L","MNG.L","MRO.L",
    "NG.L","NWG.L","NXT.L","OCDO.L","PHNX.L","PRU.L","PSON.L","PSN.L",
    "REL.L","RIO.L","RKT.L","RMV.L","ROR.L","RR.L","RS1.L","SBRY.L",
    "SDR.L","SGRO.L","SGE.L","SHEL.L","SMIN.L","SN.L","SPX.L","SSE.L",
    "STAN.L","STJ.L","SVT.L","TW.L","TSCO.L","ULVR.L","UU.L","VOD.L",
    "WPP.L","WTB.L","AZN.L","GSK.L","EXPN.L",
]

_DAX = [
    "ADS.DE","AIR.DE","ALV.DE","BAS.DE","BAYN.DE","BEI.DE","BMW.DE","BNR.DE",
    "CBK.DE","CON.DE","1COV.DE","DB1.DE","DBK.DE","DHL.DE","DTG.DE","DTE.DE",
    "EOAN.DE","ENR.DE","FME.DE","FRE.DE","G1A.DE","HEI.DE","HEN3.DE","HNR1.DE",
    "IFX.DE","MBG.DE","MRK.DE","MTX.DE","MUV2.DE","PAH3.DE","P911.DE","QIA.DE",
    "RHM.DE","RWE.DE","SAP.DE","SHL.DE","SIE.DE","SRT3.DE","SY1.DE","VOW3.DE",
    "VNA.DE","ZAL.DE",
]

_CAC40 = [
    "AI.PA","AIR.PA","ACA.PA","BNP.PA","BN.PA","CA.PA","CAP.PA","CS.PA",
    "DG.PA","DSY.PA","EL.PA","ENGI.PA","ERF.PA","GLE.PA","HO.PA","KER.PA",
    "LR.PA","MC.PA","ML.PA","MT.PA","OR.PA","ORA.PA","PUB.PA","RI.PA",
    "RMS.PA","RNO.PA","SAF.PA","SAN.PA","SGO.PA","STM.PA","SU.PA","TEP.PA",
    "TTE.PA","VIE.PA","VIV.PA","WLN.PA",
]

_AEX = [
    "ASML.AS","HEIA.AS","REN.AS","INGA.AS","PHIA.AS","AD.AS","MT.AS",
    "WKL.AS","NN.AS","ABN.AS","AKZA.AS","DSM.AS","IMCD.AS","RAND.AS",
    "UNA.AS","VPK.AS","BESI.AS","LIGHT.AS","TKWY.AS",
]

_ASX200 = [
    "BHP.AX","CBA.AX","CSL.AX","NAB.AX","WBC.AX","ANZ.AX","WES.AX","WOW.AX",
    "MQG.AX","RIO.AX","FMG.AX","TLS.AX","ALL.AX","GMG.AX","TCL.AX","WDS.AX",
    "NST.AX","EVN.AX","S32.AX","MIN.AX","WHC.AX","AMC.AX","COL.AX","CPU.AX",
    "CWY.AX","DXS.AX","IAG.AX","IEL.AX","IPL.AX","LLC.AX","LYC.AX","MPL.AX",
    "ORG.AX","ORI.AX","PLS.AX","REA.AX","RHC.AX","SEK.AX","SHL.AX","STO.AX",
    "SUL.AX","WPR.AX","XRO.AX","QAN.AX","ASX.AX","NWS.AX","TWE.AX","JHG.AX",
    "NHC.AX","CHC.AX",
]

_TSX = [
    "RY.TO","TD.TO","BNS.TO","BMO.TO","CM.TO","CNR.TO","ENB.TO","TRP.TO",
    "SU.TO","CVE.TO","IMO.TO","CNQ.TO","CP.TO","ATD.TO","L.TO","MFC.TO",
    "SLF.TO","GWO.TO","IAG.TO","FFH.TO","BCE.TO","T.TO","SHOP.TO","BN.TO",
    "BAM.TO","WN.TO","ABX.TO","AEM.TO","K.TO","AGI.TO","EQX.TO","WPM.TO",
    "FNV.TO","SSRM.TO","TECK-B.TO","FM.TO","CCO.TO","NXE.TO","MRU.TO",
    "DOL.TO","TFI.TO","CTC-A.TO","RCI-B.TO","QBR-B.TO","FTS.TO","EMA.TO",
    "CPX.TO","IFC.TO","H.TO","CAR-UN.TO","SIA.TO","PKI.TO","GIB-A.TO",
    "STN.TO","WSP.TO","AQN.TO","BIP-UN.TO","BEP-UN.TO","CSU.TO","OTEX.TO",
]

_NIFTY50 = [
    "RELIANCE.NS","TCS.NS","HDFCBANK.NS","ICICIBANK.NS","INFY.NS",
    "HINDUNILVR.NS","ITC.NS","SBIN.NS","BHARTIARTL.NS","KOTAKBANK.NS",
    "BAJFINANCE.NS","LT.NS","AXISBANK.NS","ASIANPAINT.NS","MARUTI.NS",
    "SUNPHARMA.NS","TITAN.NS","WIPRO.NS","ULTRACEMCO.NS","POWERGRID.NS",
    "HCLTECH.NS","ONGC.NS","NTPC.NS","TATAMOTORS.NS","JSWSTEEL.NS",
    "TATASTEEL.NS","TECHM.NS","NESTLEIND.NS","CIPLA.NS","BPCL.NS",
    "DRREDDY.NS","GRASIM.NS","HEROMOTOCO.NS","HINDALCO.NS","DIVISLAB.NS",
    "COALINDIA.NS","BAJAJFINSV.NS","ADANIPORTS.NS","EICHERMOT.NS",
    "BRITANNIA.NS","SBILIFE.NS","INDUSINDBK.NS","APOLLOHOSP.NS",
    "TATACONSUM.NS","BAJAJ-AUTO.NS","HDFCLIFE.NS","ADANIENT.NS",
    "PIDILITIND.NS","SHREECEM.NS","ICICIPRULI.NS",
]

_NIKKEI = [
    "7203.T","6758.T","9432.T","6861.T","4519.T","8306.T","9983.T","8035.T",
    "6954.T","9984.T","4063.T","6981.T","7267.T","6902.T","7974.T","4661.T",
    "8316.T","8411.T","8058.T","8031.T","6503.T","9022.T","9433.T","4502.T",
    "6098.T","6367.T","4543.T","8801.T","2502.T","4911.T","9201.T","9202.T",
    "8802.T","6301.T","7751.T","6702.T","4523.T","6645.T","8630.T","7011.T",
    "4568.T","7741.T","6971.T","8766.T","9020.T",
]

_HANGSENG = [
    "0700.HK","9988.HK","3690.HK","1299.HK","0941.HK","2318.HK","1398.HK",
    "0939.HK","3988.HK","0388.HK","0005.HK","0011.HK","9999.HK","1024.HK",
    "9618.HK","6618.HK","2382.HK","1810.HK","0992.HK","0016.HK","0001.HK",
    "0002.HK","0003.HK","0006.HK","0066.HK","0823.HK","1113.HK","2020.HK",
    "9633.HK","6862.HK","0027.HK","1928.HK","0762.HK","0728.HK","0914.HK",
]

# ── ETF universe ─────────────────────────────────────────────────────────────
_ETF_UNIVERSE = [
    # US broad market
    "SPY","IVV","VOO","QQQ","VTI","IWM","MDY","IJR","RSP","SCHB","ITOT",
    # International broad
    "VEA","VWO","EFA","EEM","IEFA","ACWI","VT","VXUS","SPDW","SPEM","IEMG",
    # Country ETFs
    "EWJ","EWG","EWU","EWC","EWA","EWZ","INDA","MCHI","EWH","EWS",
    "EWP","EWQ","EWI","EWT","EWY","EWL","EWD","EWN","EZA","EPOL",
    # Sector — US
    "XLK","XLV","XLF","XLE","XLI","XLC","XLY","XLP","XLU","XLRE","XLB",
    "VGT","VHT","VFH","VDE","VIS","VOX","VCR","VDC","VPU","VAW",
    # Thematic / growth
    "ARKK","ARKG","ARKF","ARKQ","ARKW",
    "SOXX","SMH","IBB","XBI","KOMP","SKYY","CLOU","WCLD",
    "ICLN","TAN","QCLN","ACES",
    # Defense & industrial
    "ITA","XAR","DFEN","PPA",
    # Dividend & income
    "VYM","SCHD","HDV","DVY","SDY","DGRO","NOBL","SPHD","JEPI","JEPQ",
    # Factor
    "QUAL","MTUM","VLUE","USMV","DSTL",
    # Bonds
    "BND","AGG","TLT","IEF","SHY","HYG","LQD","VCIT","BNDX","EMB","TIPS",
    # Commodities / inflation hedge
    "GLD","IAU","SLV","GDX","GDXJ","USO","PDBC","DJP",
    # Real estate
    "VNQ","IYR","SCHH","REM",
    # AI / semiconductor themes
    "BOTZ","IRBO","AIQ","WTAI","CHAT","THNQ",
    # Multi-asset / balanced
    "AOR","AOM","AOA","NTSX","GAA",
]

_REGION_MAP = {
    "United States":      None,           # S&P 500 + _EXTRA_TICKERS
    "United Kingdom":     _FTSE100,
    "Germany":            _DAX,
    "France":             _CAC40,
    "Europe":             _FTSE100 + _DAX + _CAC40 + _AEX,
    "Australia":          _ASX200,
    "Canada":             _TSX,
    "India":              _NIFTY50,
    "Japan":              _NIKKEI,
    "Hong Kong / China":  _HANGSENG,
}


@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sp500() -> list:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        return df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sp400() -> list:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_400_companies")[0]
        col = next((c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()), df.columns[1])
        return df[col].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_sp600() -> list:
    try:
        df = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_600_companies")[0]
        col = next((c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower()), df.columns[1])
        return df[col].str.replace(".", "-", regex=False).tolist()
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def _fetch_russell2000() -> list:
    try:
        # iShares Russell 2000 holdings page as a more reliable source
        dfs = pd.read_html("https://en.wikipedia.org/wiki/Russell_2000_Index")
        for df in dfs:
            cols = [c.lower() for c in df.columns]
            if any("ticker" in c or "symbol" in c for c in cols):
                col = next(c for c in df.columns if "ticker" in c.lower() or "symbol" in c.lower())
                return df[col].dropna().str.replace(".", "-", regex=False).tolist()
    except Exception:
        pass
    return []


@st.cache_data(ttl=86400, show_spinner=False)
def get_ticker_universe(region: str = "All") -> list:
    sp500  = _fetch_sp500()
    sp400  = _fetch_sp400()
    sp600  = _fetch_sp600()
    r2000  = _fetch_russell2000()
    us     = list(dict.fromkeys(sp500 + sp400 + sp600 + r2000 + _EXTRA_TICKERS))
    if region == "All":
        intl = (_FTSE100 + _DAX + _CAC40 + _AEX +
                _ASX200 + _TSX + _NIFTY50 + _NIKKEI + _HANGSENG)
        return list(dict.fromkeys(us + intl))
    pool = _REGION_MAP.get(region)
    return us if pool is None else list(dict.fromkeys(pool))


def _fetch_info_raw(sym: str) -> dict:
    time.sleep(0.05)   # gentle throttle — 50 ms per thread avoids bulk rate-limit triggers
    for attempt in range(4):
        try:
            t    = yf.Ticker(sym)
            info = t.info or {}
            # Supplement missing price from fast_info if needed
            if info and not info.get("currentPrice") and not info.get("regularMarketPrice"):
                try:
                    fi = t.fast_info
                    p  = getattr(fi, "last_price", None) or getattr(fi, "previous_close", None)
                    if p:
                        info["currentPrice"] = float(p)
                except Exception:
                    pass
            return info
        except Exception as e:
            msg = str(e).lower()
            if any(k in msg for k in ("rate", "429", "too many", "ratelimit")):
                time.sleep(min(30, 3 * 2 ** attempt))   # 3 → 6 → 12 → 24 s
            else:
                return {}
    return {}


@st.cache_data(ttl=7200, show_spinner=False)
def _get_info_cached(sym: str) -> dict:
    """Per-stock .info cache (2 h TTL) — prevents re-triggering rate limits on repeated scans."""
    return _fetch_info_raw(sym)


@st.cache_data(ttl=300, show_spinner=False)
def _bulk_price_download(symbols_key: str, symbols: tuple) -> dict:
    """Batch-fetch closing prices via yf.download() — much less rate-limited than .info."""
    result  = {}
    chunk   = 400
    sym_list = list(symbols)
    for i in range(0, len(sym_list), chunk):
        batch = sym_list[i:i + chunk]
        try:
            df = yf.download(batch, period="5d", progress=False,
                             auto_adjust=True)
            if df.empty:
                continue
            close = df.get("Close", df)
            if isinstance(close, pd.DataFrame):
                for s in batch:
                    try:
                        col = close[s].dropna()
                        if not col.empty:
                            result[s] = float(col.iloc[-1])
                    except Exception:
                        pass
            else:
                if len(batch) == 1:
                    col = close.dropna()
                    if not col.empty:
                        result[batch[0]] = float(col.iloc[-1])
        except Exception:
            pass
    return result


def run_screener(max_price, min_score: int, top_n: int,
                 region: str, sector_filter: str,
                 progress_bar=None, status_text=None, diversify: bool = True) -> tuple:
    universe = get_ticker_universe(region)
    total    = len(universe)
    results  = []

    # ── Phase 1: bulk price download (fast, rarely rate-limited) ─────────────
    if status_text is not None:
        status_text.text(f"Phase 1/2 — downloading prices for {total:,} stocks…")
    if progress_bar is not None:
        progress_bar.progress(0.05)

    prices = _bulk_price_download(region, tuple(universe))

    # Filter by price before hitting the slower .info endpoint
    if max_price is not None:
        candidates = [s for s in universe if 0 < prices.get(s, 0) <= max_price]
    else:
        candidates = [s for s in universe if prices.get(s, 0) > 0]

    n_cand = len(candidates)
    if status_text is not None:
        status_text.text(f"Phase 2/2 — fetching fundamentals for {n_cand:,} price-filtered stocks…")

    # ── Phase 2: fetch .info only for price-filtered candidates ──────────────
    def _score(sym: str):
        info  = _get_info_cached(sym) or {}
        price = _g(info, "currentPrice", "regularMarketPrice") or prices.get(sym)
        if not price:
            return None
        try:
            price = float(price)
        except (ValueError, TypeError):
            return None
        if price <= 0:
            return None
        if max_price is not None and price > max_price:
            return None
        if sector_filter != "All" and info.get("sector", "") != sector_filter:
            return None
        score = _score_stock(info)
        if score < min_score:
            return None
        target   = info.get("targetMeanPrice")
        currency = info.get("currency", "USD")
        return {
            "symbol":   sym,
            "name":     info.get("shortName") or info.get("longName") or sym,
            "price":    price,
            "currency": currency,
            "score":    score,
            "sector":   info.get("sector", "N/A"),
            "pe":       info.get("forwardPE") or info.get("trailingPE"),
            "pm":       info.get("profitMargins"),
            "rg":       info.get("revenueGrowth"),
            "target":   target,
            "upside":   ((float(target) - price) / price * 100) if target else None,
            "rec":      (info.get("recommendationKey") or "N/A").upper(),
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_score, sym): sym for sym in candidates}
        for future in as_completed(futures):
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(0.1 + 0.9 * completed / max(n_cand, 1))
            if status_text is not None:
                status_text.text(
                    f"Phase 2/2 — {completed:,}/{n_cand:,} fundamentals fetched  ·  "
                    f"{len(results)} candidates found"
                )
            r = future.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    top = _diversify_by_sector(results, top_n, "score") if diversify else results[:top_n]
    return top, total, len(results)


def run_value_radar(top_n: int, min_combined: int = 0, region: str = "United States",
                    progress_bar=None, status_text=None, diversify: bool = True) -> tuple:
    """Scan stocks and rank by quality + value-at-price + geopolitical score."""
    universe = get_ticker_universe(region)
    total    = len(universe)
    results  = []

    def _fetch(sym: str):
        info  = _get_info_cached(sym) or {}
        price = _g(info, "currentPrice", "regularMarketPrice") or prices.get(sym)
        if not price:
            return None
        try:
            price = float(price)
        except (ValueError, TypeError):
            return None
        if price <= 0:
            return None

        quality  = _score_stock(info)
        value    = _value_at_price_score(info)
        geo_adj, geo_label, geo_reason = _geo_adjustment(info)
        # Combined = average of quality & value, then nudged by geo
        combined = max(0, min(100, round((quality + value) / 2) + geo_adj))
        if combined < min_combined:
            return None

        fcf_yield = None
        fcf = info.get("freeCashflow"); mkt = info.get("marketCap")
        if fcf and mkt:
            try: fcf_yield = float(fcf) / float(mkt) * 100
            except: pass

        earnings_yield = None
        pe = info.get("forwardPE") or info.get("trailingPE")
        if pe:
            try:
                f = float(pe)
                if f > 0: earnings_yield = 100.0 / f
            except: pass

        upside = None
        target = info.get("targetMeanPrice")
        if target and price:
            try: upside = (float(target) - price) / price * 100
            except: pass

        return {
            "symbol":         sym,
            "name":           info.get("shortName") or info.get("longName") or sym,
            "price":          price,
            "combined_score": combined,
            "quality_score":  quality,
            "value_score":    value,
            "geo_adj":        geo_adj,
            "geo_label":      geo_label,
            "geo_reason":     geo_reason,
            "fcf_yield":      fcf_yield,
            "earnings_yield": earnings_yield,
            "upside":         upside,
            "sector":         info.get("sector", "N/A"),
            "country":        info.get("country", ""),
            "currency":       info.get("currency", "USD"),
            "rec":            (info.get("recommendationKey") or "N/A").upper(),
            "pe":             pe,
            "pm":             info.get("profitMargins"),
            "rg":             info.get("revenueGrowth"),
        }

    # Phase 1: bulk price download to skip symbols with no market data
    if status_text is not None:
        status_text.text(f"Phase 1/2 — bulk price download for {total:,} symbols…")
    if progress_bar is not None:
        progress_bar.progress(0.05)
    prices = _bulk_price_download(region, tuple(universe))
    candidates = [s for s in universe if prices.get(s, 0) > 0]
    n_cand = len(candidates)
    if status_text is not None:
        status_text.text(f"Phase 2/2 — fetching fundamentals for {n_cand:,} stocks…")

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch, sym): sym for sym in candidates}
        for future in as_completed(futures):
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(0.05 + 0.95 * completed / max(n_cand, 1))
            if status_text is not None:
                status_text.text(
                    f"Phase 2/2 — {completed:,}/{n_cand:,}  ·  {len(results)} value picks found"
                )
            r = future.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda x: x["combined_score"], reverse=True)
    top = _diversify_by_sector(results, top_n, "combined_score") if diversify else results[:top_n]
    return top, total


def run_etf_screener(top_n: int = 10,
                     progress_bar=None, status_text=None) -> list:
    """Scan _ETF_UNIVERSE and rank by ETF score."""
    total   = len(_ETF_UNIVERSE)
    results = []

    def _fetch(sym: str):
        info = _get_info_cached(sym)
        if not info:
            return None
        if info.get("quoteType", "").upper() not in ("ETF", "MUTUALFUND", ""):
            return None
        price = _g(info, "currentPrice", "regularMarketPrice", "navPrice")
        if not price:
            return None
        try:
            price = float(price)
        except (ValueError, TypeError):
            return None
        if price <= 0:
            return None
        score = _score_etf(info)
        er  = info.get("annualReportExpenseRatio") or info.get("netExpenseRatio")
        yld = info.get("yield") or info.get("trailingAnnualDividendYield")
        return {
            "symbol":    sym,
            "name":      info.get("shortName") or info.get("longName") or sym,
            "price":     price,
            "score":     score,
            "category":  info.get("category") or info.get("fundFamily") or "ETF",
            "er":        er,
            "yield":     yld,
            "r3y":       info.get("threeYearAverageReturn"),
            "r5y":       info.get("fiveYearAverageReturn"),
            "ytd":       info.get("ytdReturn"),
            "aum":       info.get("totalAssets"),
            "is_etf":    True,
        }

    completed = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(_fetch, sym): sym for sym in _ETF_UNIVERSE}
        for future in as_completed(futures):
            completed += 1
            if progress_bar is not None:
                progress_bar.progress(completed / total)
            if status_text is not None:
                status_text.text(f"Scanning ETFs… {completed}/{total}")
            r = future.result()
            if r is not None:
                results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_n]


@st.cache_data(ttl=3600, show_spinner=False)
def get_daily_picks(n: int = 10) -> dict:
    """Scans global stock universe + ETF universe. Returns {"stocks": [...], "etfs": [...]}. Cached 1 h."""
    # ── Stocks (all global markets) ───────────────────────────────────────────
    raw, _ = run_value_radar(n, 0, region="All")
    stocks = []
    for p in raw:
        sym = p["symbol"]
        try:
            info   = _get_info_cached(sym) or {}
            sector = info.get("sector", "")
            nm     = info.get("longName") or info.get("shortName") or sym
            p["overview_text"] = _la_overview(sym, nm, sector, info)
            p["thesis_text"]   = _la_thesis(sym, nm, sector, info)
        except Exception:
            p["overview_text"] = "Analysis unavailable."
            p["thesis_text"]   = "Analysis unavailable."
        stocks.append(p)

    # ── ETFs ──────────────────────────────────────────────────────────────────
    etfs = run_etf_screener(n)

    return {"stocks": stocks, "etfs": etfs}


# ── Session state ─────────────────────────────────────────────────────────────

for _k, _v in {
    "symbol": "", "loaded": False, "analyses": {},
    "vr_picks": None, "vr_total": 0,
    "picks_results": None, "picks_total": 0, "picks_found": 0,
    "daily_picks": None,
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📈 FinRobot")
    st.markdown("<span style='color:#8b949e;font-size:0.8rem;'>AI-Powered Stock Analyzer</span>",
                unsafe_allow_html=True)
    st.divider()

    ticker_input = st.text_input(
        "Ticker Symbol", placeholder="AAPL, MSFT, NVDA…",
        value=st.session_state.symbol,
    ).strip().upper()

    analyze_clicked = st.button("🔍  Analyze", use_container_width=True, type="primary")

    if analyze_clicked and ticker_input:
        st.session_state.symbol   = ticker_input
        st.session_state.loaded   = False
        st.session_state.analyses = {}
        fetch_info.clear(); fetch_history.clear()
        fetch_financials.clear(); fetch_news.clear()

    st.divider()
    st.markdown("**Chart Period**")
    period_map = {"1 Month":"1mo","3 Months":"3mo","6 Months":"6mo",
                  "1 Year":"1y","2 Years":"2y","5 Years":"5y"}
    period_label    = st.selectbox("Period", list(period_map.keys()), index=3,
                                   label_visibility="collapsed")
    selected_period = period_map[period_label]
    st.divider()
    st.markdown(
        "<span style='color:#484f58;font-size:0.75rem;'>"
        "Data: Yahoo Finance (free)<br>"
        "Analysis: Built-in rules engine<br>"
        "No API key needed</span>",
        unsafe_allow_html=True,
    )


# ── Tabs always visible ────────────────────────────────────────────────────────

sym                                      = st.session_state.symbol
info = hist = income = balance           = cashflow = news = None
name = sector = industry = country       = None
price = prev = chg = chg_pct = chg_str  = None

_header_slot = st.empty()   # company header injected here when a stock is loaded

tab_ov, tab_chart, tab_fin, tab_ai, tab_picks, tab_vr, tab_daily = st.tabs([
    "  📊 Overview  ",
    "  📈 Price Chart  ",
    "  🏦 Financials  ",
    "  🤖 AI Analysis  ",
    "  💡 Stock Picks  ",
    "  🎯 Value Radar  ",
    "  📅 Daily Top 10  ",
])

if not sym:
    with tab_ov:
        st.markdown("## Welcome to FinRobot Stock Analyzer")
        st.markdown(
            "Enter a ticker symbol in the sidebar and click **Analyze** to get started.\n\n"
            "**What you get:**\n"
            "- 📊 Live price data, key metrics & company overview\n"
            "- 📈 Interactive candlestick chart with MA20/50/200, Bollinger Bands & RSI\n"
            "- 🏦 Annual income statement, balance sheet & cash flow\n"
            "- 🤖 6 built-in AI analyses — **no API key required**\n"
            "- 💡 Smart Stock Picks: surfaces low-price stocks with strong fundamentals\n\n"
            "**Example tickers:** `AAPL` `MSFT` `NVDA` `TSLA` `AMZN` `GOOGL`"
        )
    with tab_chart:
        st.info("Enter a ticker in the sidebar and click **Analyze** to view the price chart.")
    with tab_fin:
        st.info("Enter a ticker in the sidebar and click **Analyze** to view financial statements.")
    with tab_ai:
        st.info("Enter a ticker in the sidebar and click **Analyze** to run AI analysis.")
else:
    with st.spinner(f"Loading data for **{sym}**…"):
        try:
            info = fetch_info(sym)
            if not info or (not info.get("shortName") and not info.get("longName") and not info.get("symbol")):
                st.error(f"Ticker **{sym}** not found. Check the symbol and try again.")
                info = None
            else:
                hist                      = fetch_history(sym, selected_period)
                income, balance, cashflow = fetch_financials(sym)
                news                      = fetch_news(sym)
                st.session_state.loaded   = True
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            info = None

    if info:
        name     = info.get("longName") or info.get("shortName") or sym
        sector   = info.get("sector","N/A")
        industry = info.get("industry","N/A")
        country  = info.get("country","N/A")
        price    = info.get("currentPrice") or info.get("regularMarketPrice")
        prev     = info.get("previousClose") or info.get("regularMarketPreviousClose")
        chg      = price - prev if (price and prev) else None
        chg_pct  = chg / prev * 100 if chg is not None else None
        chg_str  = f"{chg:+.2f} ({chg_pct:+.2f}%)" if chg is not None else "N/A"

        with _header_slot.container():
            col_title, col_price = st.columns([3,1])
            with col_title:
                st.markdown(f"## {name} &nbsp; `{sym}`")
                st.markdown(
                    f"<span style='color:#8b949e;font-size:0.85rem;'>{sector} &nbsp;·&nbsp; {industry} &nbsp;·&nbsp; {country}</span>",
                    unsafe_allow_html=True)
            with col_price:
                if price: st.metric("Current Price", f"${price:,.2f}", chg_str)
            st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# OVERVIEW TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_ov:
    if info:
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        with c1: st.metric("Current Price", f"${price:,.2f}" if price else "N/A", chg_str if chg else None)
        with c2: st.metric("Market Cap", fmt_num(info.get("marketCap")))
        with c3: st.metric("Forward P/E", fmt_val(info.get("forwardPE")))
        with c4: st.metric("EPS (TTM)", fmt_val(info.get("trailingEps")))
        with c5:
            dy = info.get("dividendYield")
            st.metric("Div. Yield", f"{dy*100:.2f}%" if dy else "N/A")
        with c6: st.metric("Beta", fmt_val(info.get("beta")))
        st.divider()

        left_col, right_col = st.columns([1,1], gap="large")
        with left_col:
            st.markdown('<div class="section-header">Key Metrics</div>', unsafe_allow_html=True)
            metrics = {
                "52-Week High":       f"${info.get('fiftyTwoWeekHigh','N/A')}",
                "52-Week Low":        f"${info.get('fiftyTwoWeekLow','N/A')}",
                "Trailing P/E":       fmt_val(info.get("trailingPE")),
                "P/B Ratio":          fmt_val(info.get("priceToBook")),
                "EV/EBITDA":          fmt_val(info.get("enterpriseToEbitda")),
                "Profit Margin":      fmt_pct(info.get("profitMargins")),
                "Gross Margin":       fmt_pct(info.get("grossMargins")),
                "Operating Margin":   fmt_pct(info.get("operatingMargins")),
                "ROE":                fmt_pct(info.get("returnOnEquity")),
                "ROA":                fmt_pct(info.get("returnOnAssets")),
                "Debt / Equity":      fmt_val(info.get("debtToEquity")),
                "Current Ratio":      fmt_val(info.get("currentRatio")),
                "Revenue (TTM)":      fmt_num(info.get("totalRevenue")),
                "Revenue Growth":     fmt_pct(info.get("revenueGrowth")),
                "Earnings Growth":    fmt_pct(info.get("earningsGrowth")),
                "Free Cash Flow":     fmt_num(info.get("freeCashflow")),
                "Short % of Float":   fmt_pct(info.get("shortPercentOfFloat")),
                "Shares Outstanding": fmt_num(info.get("sharesOutstanding"), prefix=""),
                "Avg Volume (10d)":   fmt_num(info.get("averageVolume10days"), prefix=""),
                "Analyst Target":     f"${info.get('targetMeanPrice','N/A')}",
                "Analyst Rating":     (info.get("recommendationKey") or "N/A").upper(),
                "# of Analysts":      str(info.get("numberOfAnalystOpinions","N/A")),
            }
            df_m = pd.DataFrame({"Metric":list(metrics.keys()),"Value":list(metrics.values())}).set_index("Metric")
            st.dataframe(df_m, use_container_width=True, height=540)

        with right_col:
            st.markdown('<div class="section-header">Business Description</div>', unsafe_allow_html=True)
            desc = info.get("longBusinessSummary","No description available.")
            st.markdown(f"<div style='color:#8b949e;font-size:0.875rem;line-height:1.65;'>{desc}</div>",
                        unsafe_allow_html=True)
            st.divider()
            st.markdown('<div class="section-header">Recent News</div>', unsafe_allow_html=True)
            if not news: st.caption("No recent news available.")
            for item in news:
                st.markdown(
                    f'<div class="news-card">'
                    f'<div class="news-title">{item["title"]}</div>'
                    f'<div class="news-pub">{item["publisher"]}'
                    + (f' &nbsp;·&nbsp; {item["date"]}' if item.get("date") else "")
                    + f'</div></div>',
                    unsafe_allow_html=True,
                )


# ═══════════════════════════════════════════════════════════════════════════════
# CHART TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_chart:
    if info and hist is not None:
        if hist.empty:
            st.warning("No price history available for this ticker.")
        else:
            st.plotly_chart(build_chart(hist, sym), use_container_width=True, config={
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d","select2d"],
                "scrollZoom": True,
            })
            c1,c2,c3,c4,c5 = st.columns(5)
            cl = hist["Close"]
            c1.metric("Period Open",   f"${cl.iloc[0]:,.2f}")
            c2.metric("Period Close",  f"${cl.iloc[-1]:,.2f}")
            c3.metric("Period High",   f"${hist['High'].max():,.2f}")
            c4.metric("Period Low",    f"${hist['Low'].min():,.2f}")
            c5.metric("Period Return", f"{(cl.iloc[-1]/cl.iloc[0]-1)*100:+.2f}%")


# ═══════════════════════════════════════════════════════════════════════════════
# FINANCIALS TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_fin:
    if info:
        st.caption("Annual figures — most recent year shown first. Values in USD.")
        ft1, ft2, ft3 = st.tabs(["Income Statement","Balance Sheet","Cash Flow"])
        with ft1: st.dataframe(format_financial_df(income),   use_container_width=True, height=700)
        with ft2: st.dataframe(format_financial_df(balance),  use_container_width=True, height=700)
        with ft3: st.dataframe(format_financial_df(cashflow), use_container_width=True, height=700)


# ═══════════════════════════════════════════════════════════════════════════════
# AI ANALYSIS TAB
# ═══════════════════════════════════════════════════════════════════════════════

ANALYSIS_TYPES = {
    "overview":      ("🏢 Company Overview",  "Full valuation, health, risks & verdict"),
    "income":        ("📊 Income Statement",  "Revenue trends, margins, EPS analysis"),
    "balance_sheet": ("🏦 Balance Sheet",     "Liquidity, leverage, equity assessment"),
    "cash_flow":     ("💸 Cash Flow",         "OCF quality, FCF, capital allocation"),
    "risk":          ("⚠️ Risk Assessment",   "Market, business, financial & ESG risks"),
    "thesis":        ("🎯 Investment Thesis", "Bull/Bear/Base cases with price targets"),
}

with tab_ai:
    if info:
        st.markdown(
            f"<span style='color:#8b949e;font-size:0.82rem;'>"
            f"No API key required · Built-in rules engine · Stock: {sym}</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        btn_cols = st.columns(3)
        for i, (key, (label, tooltip)) in enumerate(ANALYSIS_TYPES.items()):
            with btn_cols[i % 3]:
                if st.button(label, key=f"btn_{key}", use_container_width=True, help=tooltip):
                    with st.spinner(f"Analyzing {label}…"):
                        try:
                            result = run_local_analysis(key, sym, info, income, balance, cashflow)
                            st.session_state.analyses[key] = (label, result)
                        except Exception as e:
                            st.session_state.analyses[key] = (label, f"Error: {e}")
        st.divider()
        if st.session_state.analyses:
            for key, (label, result) in st.session_state.analyses.items():
                with st.expander(f"**{label}**", expanded=True):
                    st.markdown(result)
                    st.download_button("⬇ Download", data=result,
                        file_name=f"{sym}_{key}_analysis.txt", mime="text/plain",
                        key=f"dl_{key}")
        else:
            st.markdown(
                "<div style='color:#484f58;text-align:center;padding:40px;'>"
                "Click an analysis button above to generate insights.</div>",
                unsafe_allow_html=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# STOCK PICKS TAB
# ═══════════════════════════════════════════════════════════════════════════════

_SECTORS = [
    "All","Technology","Healthcare","Financial Services",
    "Consumer Cyclical","Consumer Defensive","Industrials",
    "Energy","Utilities","Real Estate","Basic Materials",
    "Communication Services",
]

_REGIONS = [
    "All Markets","United States","United Kingdom","Europe",
    "Germany","France","Canada","Australia","India","Japan","Hong Kong / China",
]

# Map UI label → _REGION_MAP key (None means "All")
_REGION_LABEL_MAP = {
    "All Markets":        "All",
    "United States":      "United States",
    "United Kingdom":     "United Kingdom",
    "Europe":             "Europe",
    "Germany":            "Germany",
    "France":             "France",
    "Canada":             "Canada",
    "Australia":          "Australia",
    "India":              "India",
    "Japan":              "Japan",
    "Hong Kong / China":  "Hong Kong / China",
}

# Currencies that are not USD — used to format prices correctly in results
_LOCAL_CCY = {
    "GBp": "p", "GBP": "£", "EUR": "€", "AUD": "A$", "CAD": "C$",
    "INR": "₹", "JPY": "¥", "HKD": "HK$",
}

def _fmt_price(price: float, currency: str) -> str:
    sym = _LOCAL_CCY.get(currency, "$")
    if currency == "JPY":
        return f"{sym}{price:,.0f}"
    return f"{sym}{price:,.2f}"

with tab_picks:
    st.markdown("### 💡 Smart Stock Picks")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "Scans 1,500+ stocks across global markets in parallel and ranks picks by "
        "fundamental quality score (valuation · profitability · growth · balance sheet). "
        "Prices shown in each stock's <b>local currency</b>."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    # Row 1 — quantity filters
    ca, cb, cc = st.columns(3)
    with ca:
        _price_opts = [10, 25, 50, 100, 250, 1000, None]
        max_price = st.selectbox(
            "Max price (local currency)",
            _price_opts, index=3,
            format_func=lambda x: "No limit" if x is None else f"Under {x}",
        )
    with cb:
        min_score = st.selectbox("Min quality score", [0, 30, 45, 60], index=0,
                                 format_func=lambda x: f"Score ≥ {x}")
    with cc:
        top_n = st.selectbox("Results to show", [5, 10, 20, 50], index=1,
                             format_func=lambda x: f"Top {x}")

    # Row 2 — market / sector filters
    ce, cf = st.columns(2)
    with ce:
        region_label  = st.selectbox("Region / Market", _REGIONS, index=0)
        region_key    = _REGION_LABEL_MAP[region_label]
    with cf:
        sector_filter = st.selectbox("Sector", _SECTORS, index=0)

    if region_key != "United States" and region_key != "All":
        st.caption(
            "ℹ️ International stocks trade in local currencies. "
            "The price filter applies in that currency (e.g. ¥ for Japan, £p for UK)."
        )

    sp_diversify = st.checkbox(
        "🌐 Diversify across sectors (spread picks over different markets)",
        value=True, key="sp_div",
        help="Caps how many picks come from any single sector so the list isn't dominated by one industry.",
    )
    run_screen = st.button("🔎  Scan Global Stocks", type="primary", use_container_width=True)

    if run_screen:
        _pb  = st.progress(0)
        _st  = st.empty()
        _picks, _total_scanned, _total_found = run_screener(
            max_price, min_score, top_n, region_key, sector_filter, _pb, _st,
            diversify=sp_diversify,
        )
        _pb.empty()
        _st.empty()
        st.session_state["picks_results"] = _picks
        st.session_state["picks_total"]   = _total_scanned
        st.session_state["picks_found"]   = _total_found
        st.rerun()

    _sp_picks  = st.session_state.get("picks_results")
    _sp_total  = st.session_state.get("picks_total", 0)
    _sp_found  = st.session_state.get("picks_found", 0)

    if _sp_picks is None:
        st.markdown(
            "<div style='color:#484f58;text-align:center;padding:60px;'>"
            "Set your filters above and click <b>Scan Global Stocks</b> to discover opportunities.</div>",
            unsafe_allow_html=True,
        )
    elif not _sp_picks:
        if _sp_found == 0 and _sp_total > 0:
            st.error(
                f"**No stocks returned data** across {_sp_total:,} candidates. "
                "This is almost always a **Yahoo Finance rate limit** — too many requests "
                "in a short period. Wait 10–15 minutes, then try again."
            )
        else:
            st.warning(
                f"No stocks found matching your criteria across {_sp_total:,} scanned. "
                "Try setting **Min quality score = Score ≥ 0** or relaxing other filters."
            )
    else:
        st.success(
            f"Scanned **{_sp_total:,}** stocks  ·  "
            f"**{_sp_found}** matched criteria  ·  "
            f"Showing top **{len(_sp_picks)}**"
        )
        st.markdown("")
        for rank, p in enumerate(_sp_picks, 1):
            v          = _verdict(p["score"]).replace("**", "")
            upside_str = f"  ·  Analyst upside: **{p['upside']:+.0f}%**" if p.get("upside") is not None else ""
            try:
                pe_str = f"{float(p['pe']):.1f}x" if p.get("pe") else "N/A"
            except (TypeError, ValueError):
                pe_str = "N/A"
            price_str  = _fmt_price(p["price"], p.get("currency", "USD"))
            ccy_badge  = (
                f'<span style="color:#8b949e;font-size:0.75rem;"> {p["currency"]}</span>'
                if p.get("currency", "USD") != "USD" else ""
            )
            html = (
                f'<div class="pick-card">'
                f'<b style="font-size:1.1rem;color:#e6edf3;">#{rank} &nbsp; {p["symbol"]} — {p["name"]}</b>'
                f'<span style="float:right;color:#8b949e;font-size:0.85rem;">{p.get("sector","N/A")}</span><br>'
                f'<span style="font-size:1.3rem;font-weight:700;color:#388bfd;">{price_str}</span>'
                f'{ccy_badge}'
                f'&nbsp;&nbsp;<span style="color:#8b949e;font-size:0.85rem;">Score: '
                f'<b style="color:#e6edf3;">{p["score"]}/100</b> · {v}{upside_str}</span><br>'
                f'<span style="color:#8b949e;font-size:0.82rem;">'
                f'P/E: {pe_str} &nbsp;|&nbsp; '
                f'Net Margin: {_ps(p.get("pm"))} &nbsp;|&nbsp; '
                f'Rev Growth: {_ps(p.get("rg"))} &nbsp;|&nbsp; '
                f'Analyst: {p.get("rec","N/A")}</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
            if st.button(f"📊 Analyze {p['symbol']}", key=f"pick_{p['symbol']}_{rank}"):
                st.session_state.symbol   = p["symbol"]
                st.session_state.loaded   = False
                st.session_state.analyses = {}
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# VALUE RADAR TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_vr:
    st.markdown("### 🎯 Value Radar — Best Stocks at Today's Price")
    st.markdown(
        "<span style='color:#8b949e;font-size:0.85rem;'>"
        "Ranks stocks (US or the region you pick) by a <b>Combined Score</b> built from three dimensions:<br>"
        "① <b>Quality</b> — margins, ROE, revenue growth, debt/equity &nbsp;·&nbsp; "
        "② <b>Value-at-Price</b> — FCF yield, earnings yield, analyst upside, balance sheet &nbsp;·&nbsp; "
        "③ <b>Geopolitical</b> — current macro/political tailwinds &amp; headwinds "
        "(NATO defense spending, CHIPS Act reshoring, US-China tensions, tariff exposure, "
        "energy independence drive, sanctions risk)."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    vr_c1, vr_c2, vr_c3, vr_c4 = st.columns(4)
    with vr_c1:
        vr_top_n = st.selectbox("Results to show", [10, 15, 20, 30], index=1,
                                format_func=lambda x: f"Top {x}", key="vr_topn")
    with vr_c2:
        vr_min = st.selectbox("Min combined score", [0, 30, 40, 50, 60], index=0,
                              format_func=lambda x: f"Score ≥ {x}", key="vr_min")
    with vr_c3:
        vr_region_label = st.selectbox("Region / Market", _REGIONS, index=0, key="vr_region")
        vr_region = _REGION_LABEL_MAP[vr_region_label]
    with vr_c4:
        st.markdown("")
        st.markdown("")
        vr_run = st.button("🎯  Run Value Radar", type="primary",
                           use_container_width=True, key="vr_run")

    vr_diversify = st.checkbox(
        "🌐 Diversify across sectors (spread picks over different markets, not just the top-scoring niche)",
        value=True, key="vr_div",
    )
    st.markdown(
        "<span style='color:#484f58;font-size:0.78rem;'>"
        "ℹ️  No price filter — all price ranges are included so you see the genuine best value. "
        "With diversify on, no single sector can dominate the shortlist."
        "</span>",
        unsafe_allow_html=True,
    )
    st.markdown("")

    if vr_run:
        _vr_pb = st.progress(0)
        _vr_st = st.empty()
        _picks, _total = run_value_radar(vr_top_n, vr_min, vr_region, _vr_pb, _vr_st,
                                         diversify=vr_diversify)
        _vr_pb.empty()
        _vr_st.empty()
        st.session_state["vr_picks"] = _picks
        st.session_state["vr_total"] = _total
        st.rerun()

    _vr_picks = st.session_state.get("vr_picks")
    _vr_total = st.session_state.get("vr_total", 0)

    if _vr_picks is None:
        st.markdown(
            "<div style='color:#484f58;text-align:center;padding:60px;'>"
            "Click <b>Run Value Radar</b> to find the best-value stocks at today's prices.</div>",
            unsafe_allow_html=True,
        )
    elif not _vr_picks:
        st.error(
            "**No stocks returned data.** "
            "This is almost always a **Yahoo Finance rate limit** — too many requests in a short period. "
            "Wait 10–15 minutes and try again. Per-stock results are cached for 2 hours once fetched."
        )
    else:
        _sectors = {}
        for _p in _vr_picks:
            _s = _p.get("sector") or "Unknown"
            _sectors[_s] = _sectors.get(_s, 0) + 1
        _sector_summary = " · ".join(f"{k} ({v})" for k, v in
                                     sorted(_sectors.items(), key=lambda kv: -kv[1]))
        st.success(
            f"Scanned **{_vr_total:,}** stocks  ·  "
            f"Showing top **{len(_vr_picks)}** by Combined Score  ·  "
            f"**{len(_sectors)} sectors** represented"
        )
        st.caption(f"📊 Sector mix: {_sector_summary}")
        st.markdown("")

        for rank, p in enumerate(_vr_picks, 1):
            cs  = p["combined_score"]
            qs  = p["quality_score"]
            vs  = p["value_score"]
            ga  = p["geo_adj"]
            try:
                pe_s = f"{float(p['pe']):.1f}x" if p.get("pe") else "N/A"
            except (TypeError, ValueError):
                pe_s = "N/A"

            cs_color  = "#3fb950" if cs >= 60 else "#d29922" if cs >= 45 else "#8b949e"
            geo_color = "#3fb950" if ga > 0 else "#f85149" if ga < 0 else "#8b949e"
            geo_sign  = f"+{ga}" if ga > 0 else str(ga)

            fcf_s = f"{p['fcf_yield']:.1f}%" if p.get("fcf_yield") is not None else "N/A"
            ey_s  = f"{p['earnings_yield']:.1f}%" if p.get("earnings_yield") is not None else "N/A"
            up_s  = f"{p['upside']:+.0f}%" if p.get("upside") is not None else "N/A"
            up_color = "#3fb950" if (p.get("upside") or 0) > 10 else \
                       "#d29922" if (p.get("upside") or 0) > 0 else "#f85149"

            try:
                price_disp = f"${p['price']:,.2f}"
            except (TypeError, ValueError):
                price_disp = "N/A"

            html = (
                f'<div class="pick-card">'
                f'<b style="font-size:1.05rem;color:#e6edf3;">#{rank} &nbsp; {p["symbol"]} — {p["name"]}</b>'
                f'<span style="float:right;color:#8b949e;font-size:0.82rem;">'
                f'{p.get("sector","N/A")} &nbsp;·&nbsp; {p.get("country","")}</span><br>'
                f'<span style="font-size:1.25rem;font-weight:700;color:#388bfd;">{price_disp}</span>'
                f'&nbsp;&nbsp;'
                f'<span style="font-size:0.95rem;font-weight:700;color:{cs_color};">Combined {cs}/100</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;">'
                f' &nbsp;Q:{qs} &nbsp;V:{vs} &nbsp;'
                f'<span style="color:{geo_color};">Geo:{geo_sign}</span>'
                f'</span><br>'
                f'<span style="font-size:0.8rem;color:{geo_color};">'
                f'{p.get("geo_label","")} &nbsp;—&nbsp; {p.get("geo_reason","")}'
                f'</span><br>'
                f'<span style="color:#8b949e;font-size:0.82rem;">'
                f'FCF Yield: <b style="color:#e6edf3;">{fcf_s}</b> &nbsp;|&nbsp; '
                f'Earnings Yield: <b style="color:#e6edf3;">{ey_s}</b> &nbsp;|&nbsp; '
                f'Analyst Upside: <b style="color:{up_color};">{up_s}</b> &nbsp;|&nbsp; '
                f'P/E: {pe_s} &nbsp;|&nbsp; '
                f'Net Margin: {_ps(p.get("pm"))} &nbsp;|&nbsp; '
                f'Rev Growth: {_ps(p.get("rg"))} &nbsp;|&nbsp; '
                f'Rating: {p.get("rec","N/A")}'
                f'</span>'
                f'</div>'
            )
            st.markdown(html, unsafe_allow_html=True)
            if st.button(f"📊 Deep-Dive {p['symbol']}", key=f"vr_{p['symbol']}_{rank}"):
                st.session_state.symbol   = p["symbol"]
                st.session_state.loaded   = False
                st.session_state.analyses = {}
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# DAILY TOP 10 TAB
# ═══════════════════════════════════════════════════════════════════════════════

with tab_daily:
    st.markdown("### 📅 Daily Top 10 — Global Stocks & ETFs")
    today_str = _date.today().strftime("%B %d, %Y")
    hdr_col, btn_col = st.columns([4, 1])
    with hdr_col:
        st.markdown(
            f"<span style='color:#8b949e;font-size:0.85rem;'>"
            f"Analysis for <b>{today_str}</b> &nbsp;·&nbsp; "
            f"Global stocks ranked by Quality · Value-at-Price · Geopolitics &nbsp;·&nbsp; "
            f"ETFs ranked by returns, expense ratio &amp; yield &nbsp;·&nbsp; "
            f"Cached daily — instant after first load"
            f"</span>",
            unsafe_allow_html=True,
        )
    with btn_col:
        _gen = st.button("✨ Generate", key="daily_gen",
                         help="Run today's global scan (~60s)", use_container_width=True)
    st.markdown("")

    # Lazy scan — only runs on button click, never automatically at startup.
    # (An automatic startup scan crashes resource-limited hosts like Streamlit Cloud.)
    if _gen:
        get_daily_picks.clear()
        with st.spinner(
            "Building today's picks… scanning global stocks + ETFs. "
            "Takes ~60 s — cached for the day after that."
        ):
            st.session_state["daily_picks"] = get_daily_picks(10)
        st.rerun()

    _daily = st.session_state.get("daily_picks")

    if _daily is None:
        st.info(
            "Click **✨ Generate** (top-right) to build today's Top 10 global stocks & ETFs. "
            "The scan takes ~60 seconds and is then cached for the day."
        )
        daily_stocks, daily_etfs = [], []
    else:
        daily_stocks = _daily.get("stocks", [])
        daily_etfs   = _daily.get("etfs",   [])

    # ── STOCKS SECTION ────────────────────────────────────────────────────────
    st.markdown(
        "<div class='section-header'>📈 Top 10 Global Stocks</div>",
        unsafe_allow_html=True,
    )
    if not daily_stocks:
        st.warning("No stock picks available. Click 🔄 Refresh to try again.")
    else:
        st.markdown(
            f"<span style='color:#484f58;font-size:0.78rem;'>"
            f"Ranked by Combined Score (quality + value-at-price + geopolitical adjustment) "
            f"across US, Europe, Canada, Australia, India, Japan &amp; Asia.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        for rank, p in enumerate(daily_stocks, 1):
            cs = p["combined_score"]
            qs = p["quality_score"]
            vs = p["value_score"]
            ga = p["geo_adj"]

            cs_color  = "#3fb950" if cs >= 60 else "#d29922" if cs >= 45 else "#8b949e"
            geo_color = "#3fb950" if ga > 0 else "#f85149" if ga < 0 else "#8b949e"
            geo_sign  = f"+{ga}" if ga > 0 else str(ga)

            fcf_s  = f"{p['fcf_yield']:.1f}%"     if p["fcf_yield"]      is not None else "N/A"
            ey_s   = f"{p['earnings_yield']:.1f}%" if p["earnings_yield"] is not None else "N/A"
            up_s   = f"{p['upside']:+.0f}%"        if p["upside"]         is not None else "N/A"
            up_col = "#3fb950" if (p["upside"] or 0) > 10 else "#d29922" if (p["upside"] or 0) > 0 else "#f85149"
            pe_s   = f"{float(p['pe']):.1f}x"      if p["pe"] else "N/A"
            price_s = _fmt_price(p["price"], p.get("currency", "USD"))
            ccy_badge = (
                f'<span style="color:#8b949e;font-size:0.75rem;"> {p["currency"]}</span>'
                if p.get("currency", "USD") != "USD" else ""
            )

            st.markdown(
                f'<div class="pick-card">'
                f'<b style="font-size:1.1rem;color:#e6edf3;">#{rank} &nbsp; {p["symbol"]} — {p["name"]}</b>'
                f'<span style="float:right;color:#8b949e;font-size:0.82rem;">'
                f'{p["sector"]} &nbsp;·&nbsp; {p.get("country","")}</span><br>'
                f'<span style="font-size:1.3rem;font-weight:700;color:#388bfd;">{price_s}</span>'
                f'{ccy_badge}&nbsp;&nbsp;'
                f'<span style="font-size:1rem;font-weight:700;color:{cs_color};">Combined {cs}/100</span>'
                f'<span style="color:#8b949e;font-size:0.8rem;"> &nbsp;'
                f'Q:{qs} &nbsp;V:{vs} &nbsp;'
                f'<span style="color:{geo_color};">Geo:{geo_sign}</span></span><br>'
                f'<span style="font-size:0.8rem;color:{geo_color};">'
                f'{p["geo_label"]} &nbsp;—&nbsp; {p["geo_reason"]}</span><br>'
                f'<span style="color:#8b949e;font-size:0.82rem;">'
                f'FCF Yield: <b style="color:#e6edf3;">{fcf_s}</b> &nbsp;|&nbsp; '
                f'Earnings Yield: <b style="color:#e6edf3;">{ey_s}</b> &nbsp;|&nbsp; '
                f'Analyst Upside: <b style="color:{up_col};">{up_s}</b> &nbsp;|&nbsp; '
                f'P/E: {pe_s} &nbsp;|&nbsp; '
                f'Net Margin: {_ps(p["pm"])} &nbsp;|&nbsp; '
                f'Rev Growth: {_ps(p["rg"])} &nbsp;|&nbsp; '
                f'Rating: {p["rec"]}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            exp_a, exp_b, btn_c = st.columns([2, 2, 1])
            with exp_a:
                with st.expander(f"🏢 Overview — {p['symbol']}"):
                    st.markdown(p.get("overview_text", ""))
            with exp_b:
                with st.expander(f"🎯 Investment Thesis — {p['symbol']}"):
                    st.markdown(p.get("thesis_text", ""))
            with btn_c:
                st.markdown("<div style='padding-top:8px;'></div>", unsafe_allow_html=True)
                if st.button("📊 Deep-Dive", key=f"daily_dd_{p['symbol']}_{rank}",
                             use_container_width=True):
                    st.session_state.symbol   = p["symbol"]
                    st.session_state.loaded   = False
                    st.session_state.analyses = {}
                    st.rerun()
            st.markdown("")

    # ── ETF SECTION ───────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        "<div class='section-header'>📦 Top 10 ETFs</div>",
        unsafe_allow_html=True,
    )
    if not daily_etfs:
        st.warning("No ETF picks available. Click 🔄 Refresh to try again.")
    else:
        st.markdown(
            f"<span style='color:#484f58;font-size:0.78rem;'>"
            f"Ranked by ETF score: 3yr/5yr returns, expense ratio, dividend yield. "
            f"Covers broad market, international, sectors, thematic, bonds &amp; commodities.</span>",
            unsafe_allow_html=True,
        )
        st.markdown("")
        for rank, e in enumerate(daily_etfs, 1):
            sc    = e["score"]
            sc_cl = "#3fb950" if sc >= 60 else "#d29922" if sc >= 45 else "#8b949e"

            r3_s  = f"{e['r3y']*100:.1f}% / yr"  if e.get("r3y")  is not None else "N/A"
            r5_s  = f"{e['r5y']*100:.1f}% / yr"  if e.get("r5y")  is not None else "N/A"
            ytd_s = f"{e['ytd']*100:.1f}%"        if e.get("ytd")  is not None else "N/A"
            er_s  = f"{e['er']*100:.2f}%"         if e.get("er")   is not None else "N/A"
            yld_s = f"{e['yield']*100:.2f}%"      if e.get("yield") is not None else "N/A"
            aum_s = fmt_num(e.get("aum"), prefix="") if e.get("aum") else "N/A"

            st.markdown(
                f'<div class="pick-card">'
                f'<b style="font-size:1.1rem;color:#e6edf3;">#{rank} &nbsp; {e["symbol"]} — {e["name"]}</b>'
                f'<span style="float:right;color:#8b949e;font-size:0.82rem;">{e.get("category","ETF")}</span><br>'
                f'<span style="font-size:1.3rem;font-weight:700;color:#388bfd;">${e["price"]:,.2f}</span>'
                f'&nbsp;&nbsp;'
                f'<span style="font-size:1rem;font-weight:700;color:{sc_cl};">ETF Score {sc}/100</span><br>'
                f'<span style="color:#8b949e;font-size:0.82rem;">'
                f'3yr Return: <b style="color:#e6edf3;">{r3_s}</b> &nbsp;|&nbsp; '
                f'5yr Return: <b style="color:#e6edf3;">{r5_s}</b> &nbsp;|&nbsp; '
                f'YTD: <b style="color:#e6edf3;">{ytd_s}</b> &nbsp;|&nbsp; '
                f'Expense Ratio: <b style="color:#e6edf3;">{er_s}</b> &nbsp;|&nbsp; '
                f'Yield: <b style="color:#e6edf3;">{yld_s}</b> &nbsp;|&nbsp; '
                f'AUM: {aum_s}'
                f'</span></div>',
                unsafe_allow_html=True,
            )
            if st.button("📊 Analyze", key=f"daily_etf_{e['symbol']}_{rank}",
                         use_container_width=False):
                st.session_state.symbol   = e["symbol"]
                st.session_state.loaded   = False
                st.session_state.analyses = {}
                st.rerun()
            st.markdown("")
