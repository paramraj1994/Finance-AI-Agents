import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Financial Model – Profit & Loss",
    layout="wide"
)

st.title("📊 Financial Model – Profit & Loss Statement")
st.caption("Historical P&L + Projections based on 2025")

# -------------------------------------------------
# Sidebar – Currency Conversion
# -------------------------------------------------
st.sidebar.header("💱 Currency Settings")

base_currency = st.sidebar.selectbox(
    "Base Currency (Excel)",
    ["INR", "USD", "EUR"]
)

display_currency = st.sidebar.selectbox(
    "Display Currency",
    ["INR", "USD", "EUR"]
)

# Simple conversion matrix (editable)
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
    st.info("Please upload an Excel file to continue.")
    st.stop()

df = pd.read_excel(uploaded_file)

# -------------------------------------------------
# Display Historical Data
# -------------------------------------------------
st.subheader("📄 Historical Financials (As Uploaded)")
st.dataframe(df)

# -------------------------------------------------
# Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Projection Assumptions")

revenue_growth = st.sidebar.slider(
    "Revenue Growth (%)", 0.0, 0.5, 0.12, 0.01
)

cost_growth = st.sidebar.slider(
    "Cost Growth (%)", 0.0, 0.5, 0.08, 0.01
)

tax_rate = st.sidebar.slider(
    "Tax Rate (%)", 0.1, 0.4, 0.25, 0.01
)

projection_years = st.sidebar.number_input(
    "Projection Years",
    min_value=1,
    max_value=10,
    value=3
)

# -------------------------------------------------
# Extract 2025 Base Values
# EXPECTED Excel FORMAT:
# Columns: Year | Revenue | Cost
# -------------------------------------------------
if 2025 not in df["Year"].values:
    st.error("Year 2025 not found in uploaded Excel.")
    st.stop()

base_row = df[df["Year"] == 2025].iloc[0]

base_revenue = base_
