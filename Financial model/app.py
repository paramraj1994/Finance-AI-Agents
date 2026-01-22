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
st.caption("Historical results + projections (Base year: 2025)")

# -------------------------------------------------
# Sidebar – Currency
# -------------------------------------------------
st.sidebar.header("💱 Currency")

base_currency = st.sidebar.selectbox(
    "Base Currency (Excel)",
    ["INR", "USD", "EUR"]
)

display_currency = st.sidebar.selectbox(
    "Display Currency",
    ["INR", "USD", "EUR"]
)

fx_rates = {
    "INR": {"INR": 1, "USD": 0.012, "EUR": 0.011},
    "USD": {"INR": 83, "USD": 1, "EUR": 0.92},
    "EUR": {"INR": 90, "USD": 1.09, "EUR": 1},
}

fx = fx_rates[base_currency][display_currency]
currency_symbol = {"INR": "₹", "USD": "$", "EUR": "€"}[display_currency]

# -------------------------------------------------
# Upload Excel
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload historical financial Excel file",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

df = pd.read_excel(uploaded_file)

st.subheader("📄 Uploaded Historical Data")
st.dataframe(df)

# -------------------------------------------------
# Column Mapping
# -------------------------------------------------
st.subheader("🧭 Column Mapping")

columns = df.columns.tolist()

year_col = st.selectbox("Select Year column", columns)
revenue_col = st.selectbox("Select Revenue column", columns)
cost_col = st.selectbox("Select Cost column", columns)

# -------------------------------------------------
# Clean & Prepare Data
# -------------------------------------------------
df[year_col] = df[year_col].astype(str)

df[revenue_col] = (
    df[revenue_col]
    .astype(str)
    .str.replace(",", "")
    .astype(float)
)

df[cost_col] = (
    df[cost_col]
    .astype(str)
    .str.replace(",", "")
    .astype(float)
)

# -------------------------------------------------
# Debug Visibility (DO NOT REMOVE)
# -------------------------------------------------
st.subheader("🔍 Data Check")
st.write("Detected Years:", df[year_col].unique())

# -------------------------------------------------
# Sidebar – Assumptions
# -------------------------------------------------
st.sidebar.header("📐 Projection Assumptions")

revenue_growth = st.sidebar.slider(
    "Revenue Growth (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.12,
    step=0.01
)

cost_growth = st.sidebar.slider(
    "Cost Growth (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.08,
    step=0.01
)

tax_rate = st.sidebar.slider(
    "Tax Rate (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.25,
    step=0.01
)

projection_years = st.sidebar.number_input(
    "Number of Projection Years",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)

# -------------------------------------------------
# Base Year Logic (SAFE)
# -------------------------------------------------
if "2025" not in df[year_col].values:
    st.error("❌ Base year 2025 not found in selected Year column.")
    st.stop()

base_row = df[df[year_col] == "2025"].iloc[0]

base_revenue = base_row[revenue_col]
base_cost = base_row[cost_col]

# -------------------------------------------------
# Historical P&L
# -------------------------------------------------
st.subheader("📑 Historical Profit & Loss")

historical_df = pd.DataFrame({
    "Revenue": df[revenue_col] * fx,
    "Cost": df[cost_col] * fx,
    "EBITDA": (df[revenue_col] - df[cost_col]) * fx
}, index=df[year_col])

st.dataframe(
    historical_df.style
    .format(f"{currency_symbol} {{:,.0f}}")
    .set_properties(**{"text-align": "right"})
)

# -------------------------------------------------
# Build Projections
# -------------------------------------------------
years = [str(2025 + i) for i in range(1, projection_years + 1)]

revenue = base_revenue
cost = base_cost

pnl = {
    "Revenue": [],
    "Cost": [],
    "EBITDA": [],
    "Tax": [],
    "Profit After Tax": []
}

for _ in years:
    revenue *= (1 + revenue_growth)
    cost *= (1 + cost_growth)

    ebitda = revenue - cost
    tax = max(0, ebitda * tax_rate)
    pat = ebitda - tax

    pnl["Revenue"].append(revenue * fx)
    pnl["Cost"].append(cost * fx)
    pnl["EBITDA"].append(ebitda * fx)
    pnl["Tax"].append(tax * fx)
    pnl["Profit After Tax"].append(pat * fx)

projection_df = pd.DataFrame(pnl, index=years).T

# -------------------------------------------------
# Display Projected P&L
# -------------------------------------------------
st.subheader("📈 Projected Profit & Loss Statement")

st.dataframe(
    projection_df.style
    .format(f"{currency_symbol} {{:,.0f}}")
    .set_properties(**{"text-align": "right"})
)

# -------------------------------------------------
# Financial Trend Chart
# -------------------------------------------------
st.subheader("📊 Revenue, EBITDA & PAT Trend")

fig, ax = plt.subplots()

ax.plot(projection_df.columns, projection_df.loc["Revenue"], marker="o", label="Revenue")
ax.plot(projection_df.columns, projection_df.loc["EBITDA"], marker="o", label="EBITDA")
ax.plot(projection_df.columns, projection_df.loc["Profit After Tax"], marker="o", label="PAT")

ax.set_ylabel(f"Amount ({display_currency})")
ax.set_xlabel("Year")
ax.legend()
ax.grid(True)

st.pyplot(fig)
