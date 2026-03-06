import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="FP&A Dashboard", layout="wide")

st.title("📊 FP&A Monthly Reporting Dashboard")

st.sidebar.header("Upload Financial Data")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv","xlsx"]
)

if uploaded_file is None:
    st.info("Upload a dataset to begin analysis.")
    st.stop()

# Load file
if uploaded_file.name.endswith(".csv"):
    df = pd.read_csv(uploaded_file)
else:
    df = pd.read_excel(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df)

# Required columns check
required_cols = ["Month","Revenue","Expense"]

for col in required_cols:
    if col not in df.columns:
        st.error(f"Missing column: {col}")
        st.stop()

df["Profit"] = df["Revenue"] - df["Expense"]

# ==============================
# KPI SECTION
# ==============================

st.subheader("Key Financial Metrics")

col1,col2,col3 = st.columns(3)

col1.metric(
    "Total Revenue",
    f"{df['Revenue'].sum():,.0f}"
)

col2.metric(
    "Total Expense",
    f"{df['Expense'].sum():,.0f}"
)

col3.metric(
    "Total Profit",
    f"{df['Profit'].sum():,.0f}"
)

# ==============================
# MONTHLY TREND
# ==============================

st.subheader("Monthly Financial Trend")

fig = px.line(
    df,
    x="Month",
    y=["Revenue","Expense","Profit"],
    markers=True
)

st.plotly_chart(fig, use_container_width=True)

# ==============================
# VARIANCE ANALYSIS
# ==============================

st.subheader("Budget vs Actual Variance")

if "Budget_Revenue" in df.columns:

    df["Revenue Variance"] = df["Revenue"] - df["Budget_Revenue"]

    fig2 = px.bar(
        df,
        x="Month",
        y="Revenue Variance",
        title="Revenue Variance"
    )

    st.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Add column 'Budget_Revenue' to enable variance analysis.")

# ==============================
# SENSITIVITY ANALYSIS
# ==============================

st.subheader("Profit Sensitivity Analysis")

revenue_change = st.slider(
    "Revenue Change %",
    -50,
    50,
    0
)

expense_change = st.slider(
    "Expense Change %",
    -50,
    50,
    0
)

adj_revenue = df["Revenue"].sum() * (1 + revenue_change/100)
adj_expense = df["Expense"].sum() * (1 + expense_change/100)

adj_profit = adj_revenue - adj_expense

st.metric(
    "Adjusted Profit",
    f"{adj_profit:,.0f}"
)

# ==============================
# DATA DOWNLOAD
# ==============================

st.subheader("Download Processed Data")

st.download_button(
    "Download Analysis",
    df.to_csv(index=False),
    "financial_analysis.csv",
    "text/csv"
)
