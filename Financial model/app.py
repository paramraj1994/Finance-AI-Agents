import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Financial Model – Profit & Loss",
    layout="wide"
)

st.title("📊 Financial Model – Profit & Loss Statement")
st.caption("Historical P&L + Projections (2025 as Base Year)")

# -------------------------------------------------
# Sidebar – Currency
# -------------------------------------------------
st.sidebar.header("💱 Currency Settings")

base_currency = st.sidebar.selectbox("Base Currency", ["INR", "USD", "EUR"])
display_currency = st.sidebar.selectbox("Display Currency", ["INR", "USD", "EUR"])

fx_rates = {
    "INR": {"INR": 1, "USD": 0.012, "EUR": 0.011},
    "USD": {"INR": 83, "USD": 1, "EUR": 0.92},
    "EUR": {"INR": 90, "USD": 1.09, "EUR": 1}
}

fx = fx_rates[base_currency][display_currency]
currency_symbol = {"INR": "₹", "USD": "$", "EUR": "€"}[display_currency]

# -------------------------------------------------
# Upload Excel
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload your historical P&L Excel file",
    type=["xlsx"]
)

if uploaded_file is None:
    st.stop()

df = pd.read_excel(uploaded_file)

st.subheader("📄 Uploaded Historical Data")
st.dataframe(df)

# -------------------------------------------------
# Column Mapping (CRITICAL FIX)
# -------------------------------------------------
st.subheader("🧭 Map Excel Columns")

columns = df.columns.tolist()

year_col = st.selectbox("Select Year column", columns)
revenue_col = st.selectbox("Select Revenue column", columns)
cost_col = st.selectbox("Select Cost column", columns)

# Ensure numeric
df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
df[revenue_col] = pd.to_numeric(df[revenue_col], errors="coerce")
df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

# -------------------------------------------------
# Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Projection Assumptions")

revenue_growth = st.sidebar.slider("Revenue Growth (%)", 0.0, 0.5, 0.12, 0.01)
cost_grow_
