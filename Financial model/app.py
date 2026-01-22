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
st.caption("Upload historical financials and generate projected P&L dynamically")

# -------------------------------------------------
# Upload Excel
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload your historical financial Excel file",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

df = pd.read_excel(uploaded_file)

st.subheader("📄 Uploaded Historical Data (Preview)")
st.dataframe(df)

# -------------------------------------------------
# Sidebar Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Projection Assumptions")

revenue_growth = st.sidebar.slider(
    "Revenue Growth (%)", 0.0, 0.5, 0.12, 0.01
)

ebitda_margin = st.sidebar.slider(
    "EBITDA Margin (%)", 0.05, 0.6, 0.25, 0.01
)

tax_rate = st.sidebar.slider(
    "Tax Rate (%)", 0.1, 0.4, 0.25, 0.01
)

projection_years = st.sidebar.number_input(
    "Number of Projection Years",
    min_value=1,
    max_value=10,
    value=3
)

# -------------------------------------------------
# Identify Historical Revenue
# -------------------------------------------------
st.subheader("📌 Base Inputs")

base_revenue = st.number_input(
    "Last Historical Year Revenue",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

last_year = st.number_input(
    "Last Historical Year",
    min_value=2000,
    max_value=2100,
    value=2025
)

# -------------------------------------------------
# Build P&L Model
# -------------------------------------------------
years = [str(last_year + i) for i in range(1, projection_years + 1)]

pnl_data = {
    "Revenue": [],
    "EBITDA": [],
    "Tax": [],
    "Profit After Tax": []
}

revenue = base_revenue

for _ in years:
    revenue = revenue * (1 + revenue_growth)
    ebitda = revenue * ebitda_margin
    tax = ebitda * tax_rate
    pat = ebitda - tax

    pnl_data["Revenue"].append(revenue)
    pnl_data["EBITDA"].append(ebitda)
    pnl_data["Tax"].append(tax)
    pnl_data["Profit After Tax"].append(pat)

pnl_df = pd.DataFrame(pnl_data, index=years).T

# -------------------------------------------------
# Display P&L Statement
# -------------------------------------------------
st.subheader("📑 Projected Profit & Loss Statement")

st.dataframe(
    pnl_df.style
    .format("{:,.0f}")
    .set_properties(**{"text-align": "right"})
)

# -------------------------------------------------
# Charts Section
# -------------------------------------------------
st.subheader("📈 Financial Trend Analysis")

fig, ax = plt.subplots()
ax.plot(pnl_df.columns, pnl_df.loc["Revenue"], marker="o", label="Revenue")
ax.plot(pnl_df.columns, pnl_df.loc["EBITDA"], marker="o", label="EBITDA")
ax.plot(pnl_df.columns, pnl_df.loc["Profit After Tax"], marker="o", label="PAT")

ax.set_xlabel("Year")
ax.set_ylabel("Amount")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------------------------
# Download P&L
# -------------------------------------------------
st.subheader("⬇️ Download P&L Statement")

csv = pnl_df.reset_index().rename(columns={"index": "Line Item"}).to_csv(index=False)

st.download_button(
    label="Download P&L as CSV",
    data=csv,
    file_name="Profit_and_Loss_Statement.csv",
    mime="text/csv"
)
