import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model.loader import load_financials
from model.assumptions import default_assumptions
from model.financial_model import build_model

st.set_page_config(page_title="Financial Model", layout="wide")

st.title("📊 Interactive Financial Model")

# Load data
file_path = "data/Chatgpt raw file.xlsx"
df = load_financials(file_path)

st.subheader("📂 Input Financial Data")
st.dataframe(df)

# Assumptions
st.sidebar.header("🔧 Model Assumptions")

assumptions = default_assumptions()
assumptions["revenue_growth"] = st.sidebar.slider(
    "Revenue Growth (%)", 0.0, 0.5, assumptions["revenue_growth"]
)

assumptions["ebitda_margin"] = st.sidebar.slider(
    "EBITDA Margin (%)", 0.05, 0.6, assumptions["ebitda_margin"]
)

assumptions["tax_rate"] = st.sidebar.slider(
    "Tax Rate (%)", 0.1, 0.4, assumptions["tax_rate"]
)

# Base revenue input
base_revenue = st.number_input(
    "Base Year Revenue",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

forecast_years = ["2026", "2027", "2028"]

# Build model
model_df = build_model(base_revenue, forecast_years, assumptions)

st.subheader("📈 Financial Projections")
st.dataframe(model_df.style.format("{:,.0f}"))

# Charts
st.subheader("📉 Visualizations")

fig, ax = plt.subplots()
ax.plot(model_df["Year"], model_df["Revenue"], marker="o", label="Revenue")
ax.plot(model_df["Year"], model_df["EBITDA"], marker="o", label="EBITDA")
ax.plot(model_df["Year"], model_df["PAT"], marker="o", label="PAT")
ax.legend()
ax.set_title("Financial Performance")

st.pyplot(fig)
