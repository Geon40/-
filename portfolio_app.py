
# -*- coding: utf-8 -*-
# Streamlit Portfolio ROI Tracker
# How to run locally:
#   1) pip install streamlit yfinance pandas numpy matplotlib
#   2) streamlit run portfolio_app.py
#
# CSV format:
#   Ticker,Quantity,AverageCost,Currency
#   AAPL,10,150,USD
#   005930.KS,5,72000,KRW
#   7203.T,3,2200,JPY
#
# Notes:
# - Prices are fetched from Yahoo Finance via yfinance and are usually delayed.
# - FX rates are pulled from Yahoo as well (e.g., USDKRW=X). You can override rates manually.
# - Base currency is configurable (default: KRW).

import io
import time
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Portfolio ROI Tracker", layout="wide")

st.title("📈 증권연구회 투자부 Portfolio ROI Tracker (YFinance, Geon40)")

with st.sidebar:
    st.header("⚙️ Settings")
    base_currency = st.selectbox("Base Currency", ["KRW", "USD", "JPY", "EUR"], index=0)
    refresh_sec = st.number_input("Auto-refresh (seconds)", min_value=0, max_value=3600, value=0, step=5,
                                  help="Set to 0 to disable auto-refresh.")
    use_auto_fx = st.checkbox("Auto FX from Yahoo", value=True)
    st.caption("If unchecked, you can set custom FX rates below.")

    # Custom FX overrides
    custom_fx = {}
    fx_currencies = ["USD", "KRW", "JPY", "EUR"]
    fx_currencies = [c for c in fx_currencies if c != base_currency]
    if not use_auto_fx:
        st.subheader("Custom FX Rates")
        st.caption(f"Enter how many {base_currency} per 1 unit of the foreign currency.")
        for c in fx_currencies:
            custom_fx[c] = st.number_input(f"{c}→{base_currency}", min_value=0.0, value=0.0, step=0.01)

    st.markdown("---")
    st.caption("💡 Tip: Upload your CSV or edit the sample table below.")


# Sample data
sample_df = pd.DataFrame({
    "Ticker": ["AAPL", "005930.KS", "7203.T"],
    "Quantity": [10, 5, 3],
    "AverageCost": [150.0, 72000.0, 2200.0],
    "Currency": ["USD", "KRW", "JPY"],
})

# --- Load portfolio data (GDrive default) ---
DEFAULT_URL = https://drive.google.com/uc?id=1LQOigngRqO_N9FZLWjUTQtqkq-Bd-VMP



uploaded = st.file_uploader("Upload portfolio CSV", type=["csv", "xlsx"])

try:
    if uploaded is not None:
        # 업로드된 파일 우선 적용
        if uploaded.name.lower().endswith(".xlsx"):
            portfolio_df = pd.read_excel(uploaded)
        else:
            portfolio_df = pd.read_csv(uploaded)
    else:
        # 업로드가 없으면 구글드라이브 CSV 자동 로드
        portfolio_df = pd.read_csv(DEFAULT_URL)
except Exception as e:
    st.error(f"Failed to read portfolio file: {e}")
    st.stop()

# Data cleaning
required_cols = ["Ticker", "Quantity", "AverageCost", "Currency"]
missing = [c for c in required_cols if c not in portfolio_df.columns]
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

portfolio_df["Ticker"] = portfolio_df["Ticker"].astype(str).str.strip()
portfolio_df["Currency"] = portfolio_df["Currency"].astype(str).str.upper().str.strip()
portfolio_df["Quantity"] = pd.to_numeric(portfolio_df["Quantity"], errors="coerce").fillna(0.0)
portfolio_df["AverageCost"] = pd.to_numeric(portfolio_df["AverageCost"], errors="coerce").fillna(0.0)

if (portfolio_df["Quantity"] < 0).any():
    st.warning("Some quantities are negative — treating as short positions.")

# Build list of tickers
tickers = portfolio_df["Ticker"].unique().tolist()

@st.cache_data(ttl=120)
def fetch_prices(ticker_list):
    if not ticker_list:
        return pd.DataFrame()
    # Use yfinance to get last and previous close
    data = yf.download(ticker_list, period="5d", interval="1d", group_by="ticker", auto_adjust=False, progress=False)
    # Normalize to multiindex (ticker, field)
    prices = {}
    for t in ticker_list:
        try:
            df = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            last_close = df["Close"].dropna().iloc[-1]
            prev_close = df["Close"].dropna().iloc[-2] if len(df["Close"].dropna()) >= 2 else np.nan
            prices[t] = {"last": float(last_close), "prev_close": float(prev_close) if not np.isnan(prev_close) else np.nan}
        except Exception:
            prices[t] = {"last": np.nan, "prev_close": np.nan}
    return pd.DataFrame(prices).T.reset_index().rename(columns={"index": "Ticker"})

@st.cache_data(ttl=300)
def fetch_fx(base_ccy, needed_ccys):
    rates = {}
    for c in needed_ccys:
        if c == base_ccy:
            rates[c] = 1.0
            continue
        pair = f"{c}{base_ccy}=X"  # e.g., USDKRW=X
        try:
            df = yf.download(pair, period="5d", interval="1d", progress=False)
            rate = float(df["Close"].dropna().iloc[-1])
        except Exception:
            rate = np.nan
        rates[c] = rate
    return rates

# Fetch prices
with st.spinner("Fetching latest prices..."):
    price_df = fetch_prices(tickers)

# Merge back
merged = portfolio_df.merge(price_df, on="Ticker", how="left")

# FX conversion
needed_fx = sorted(merged["Currency"].unique().tolist())
fx_rates = {base_currency: 1.0}

if use_auto_fx:
    with st.spinner("Fetching FX rates..."):
        auto_rates = fetch_fx(base_currency, needed_fx)
        fx_rates.update(auto_rates)

# Apply custom overrides
for c, v in custom_fx.items():
    if v > 0:
        fx_rates[c] = v

# Any missing FX gets 0 -> later treated as NaN
for c in needed_fx:
    if c not in fx_rates:
        fx_rates[c] = np.nan

def map_fx(curr):
    if curr == base_currency:
        return 1.0
    return fx_rates.get(curr, np.nan)

merged["FX_to_Base"] = merged["Currency"].map(map_fx)

# Compute P/L
merged["MarketPrice"] = merged["last"]
merged["PrevClose"] = merged["prev_close"]
merged["MarketValue_Local"] = merged["MarketPrice"] * merged["Quantity"]
merged["CostBasis_Local"] = merged["AverageCost"] * merged["Quantity"]

# Convert to base
merged["MarketValue_Base"] = merged["MarketValue_Local"] * merged["FX_to_Base"]
merged["CostBasis_Base"] = merged["CostBasis_Local"] * merged["FX_to_Base"]

merged["Unrealized_PL_Base"] = merged["MarketValue_Base"] - merged["CostBasis_Base"]
merged["Return_%"] = np.where(merged["CostBasis_Base"] != 0, merged["Unrealized_PL_Base"] / merged["CostBasis_Base"] * 100.0, np.nan)

# Daily change
merged["Daily_%"] = np.where(
    ~merged["PrevClose"].isna() & (merged["PrevClose"] != 0),
    (merged["MarketPrice"] - merged["PrevClose"]) / merged["PrevClose"] * 100.0,
    np.nan,
)

# Aggregate
tot_mv = merged["MarketValue_Base"].sum(min_count=1)
tot_cost = merged["CostBasis_Base"].sum(min_count=1)
tot_pl = merged["Unrealized_PL_Base"].sum(min_count=1)
tot_ret = (tot_pl / tot_cost * 100.0) if (tot_cost and tot_cost != 0) else np.nan

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Market Value", f"{tot_mv:,.0f} {base_currency}")
col2.metric("Total Cost Basis", f"{tot_cost:,.0f} {base_currency}")
col3.metric("Unrealized P/L", f"{tot_pl:,.0f} {base_currency}")
col4.metric("Portfolio Return", (f"{tot_ret:.2f} %" if pd.notna(tot_ret) else "—"))

st.markdown("### Holdings")

display_cols = [
    "Ticker","Quantity","Currency","AverageCost",
    "MarketPrice","PrevClose","Daily_%",
    "FX_to_Base","MarketValue_Base","CostBasis_Base","Unrealized_PL_Base","Return_%"
]
display_df = merged[display_cols].copy()

num_cols = ["Quantity","AverageCost","MarketPrice","PrevClose","Daily_%","FX_to_Base",
            "MarketValue_Base","CostBasis_Base","Unrealized_PL_Base","Return_%"]

for c in num_cols:
    if c in display_df.columns:
        display_df[c] = pd.to_numeric(display_df[c], errors="coerce")

st.dataframe(display_df.style.format({
    "Quantity": "{:,.4f}",
    "AverageCost": "{:,.4f}",
    "MarketPrice": "{:,.4f}",
    "PrevClose": "{:,.4f}",
    "Daily_%": "{:+.2f} %",
    "FX_to_Base": "{:,.4f}",
    "MarketValue_Base": "{:,.0f}",
    "CostBasis_Base": "{:,.0f}",
    "Unrealized_PL_Base": "{:,.0f}",
    "Return_%": "{:+.2f} %",
}))

# Weight chart
st.markdown("### Allocation by Position")
weights = merged.assign(MV=merged["MarketValue_Base"])
if tot_mv and tot_mv > 0:
    weights["Weight_%"] = weights["MV"] / tot_mv * 100.0
    fig = plt.figure()
    plt.pie(weights["Weight_%"].fillna(0.0), labels=weights["Ticker"], autopct="%1.1f%%", startangle=140)
    st.pyplot(fig)

# P/L bar chart
st.markdown("### Unrealized P/L by Ticker")
fig2 = plt.figure()
plt.bar(merged["Ticker"], merged["Unrealized_PL_Base"].fillna(0.0))
plt.xticks(rotation=45, ha="right")
plt.ylabel(f"Unrealized P/L ({base_currency})")
st.pyplot(fig2)

st.caption("Prices and FX from Yahoo Finance via yfinance; values may be delayed. Use at your own discretion.")

# Auto-refresh
if refresh_sec and refresh_sec > 0:
    st.experimental_singleton.clear()  # best-effort
    st.cache_data.clear()
    st.experimental_rerun()
