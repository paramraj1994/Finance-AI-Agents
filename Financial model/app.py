import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Detailed Valuation", layout="wide")
st.title("📊 Financial Model – Detailed Valuation Model")
st.caption("P&L • Cash Flow • Fully Explained DCF Valuation")

# -------------------------------------------------
# CSS: Center align all tables
# -------------------------------------------------
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td {
        text-align: center !important;
        vertical-align: middle !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Sidebar Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Key Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
base_revenue = st.sidebar.number_input("Last Historical Revenue", value=1120.0, step=100.0)

tax_rate = st.sidebar.slider("Tax Rate (%)", 0.15, 0.40, 0.25, 0.01)
reinvestment_rate = st.sidebar.slider("Reinvestment (% of PAT)", 0.0, 0.50, 0.10, 0.01)

discount_rate = st.sidebar.slider("Discount Rate (WACC)", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

# -------------------------------------------------
# Scenario Definitions
# -------------------------------------------------
scenarios = {
    "Bear": {"growth": 0.07, "margin": 0.20},
    "Base": {"growth": 0.12, "margin": 0.25},
    "Bull": {"growth": 0.18, "margin": 0.30},
}

st.subheader("📌 Scenario Assumptions")
assumptions_df = pd.DataFrame(scenarios).T
assumptions_df["growth"] = assumptions_df["growth"].map(lambda x: f"{x:.1%}")
assumptions_df["margin"] = assumptions_df["margin"].map(lambda x: f"{x:.1%}")
st.dataframe(assumptions_df)

selected = st.radio("🎯 Select Scenario", scenarios.keys(), horizontal=True)
growth = scenarios[selected]["growth"]
margin = scenarios[selected]["margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# -------------------------------------------------
# Build Financial Model
# -------------------------------------------------
revenue = base_revenue
data = {"Revenue": [], "EBITDA": [], "Tax": [], "PAT": [], "FCF": []}

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * (1 - reinvestment_rate)

    data["Revenue"].append(revenue)
    data["EBITDA"].append(ebitda)
    data["Tax"].append(tax)
    data["PAT"].append(pat)
    data["FCF"].append(fcf)

model_df = pd.DataFrame(data, index=years)

# -------------------------------------------------
# Profit & Loss (Vertical)
# -------------------------------------------------
st.subheader("📑 Profit & Loss Statement")

pnl = pd.DataFrame({
    "Formula": [
        "Revenueₜ = Revenueₜ₋₁ × (1 + Growth)",
        "EBITDA = Revenue × EBITDA Margin",
        "Tax = EBITDA × Tax Rate",
        "PAT = EBITDA − Tax"
    ],
    "Year 1": [model_df.iloc[0]["Revenue"], model_df.iloc[0]["EBITDA"],
               model_df.iloc[0]["Tax"], model_df.iloc[0]["PAT"]],
}).set_index(pd.Index(["Revenue", "EBITDA", "Tax", "PAT"]))

for i, y in enumerate(years):
    pnl[y] = model_df.iloc[i][["Revenue", "EBITDA", "Tax", "PAT"]].values

st.dataframe(pnl.round(0))

# -------------------------------------------------
# Cash Flow Statement
# -------------------------------------------------
st.subheader("💵 Cash Flow Statement")

cf = pd.DataFrame({
    "Formula": [
        "From P&L",
        "Reinvestment = PAT × rate",
        "FCF = PAT − Reinvestment"
    ],
    "Year 1": [
        model_df.iloc[0]["PAT"],
        -reinvestment_rate * model_df.iloc[0]["PAT"],
        model_df.iloc[0]["FCF"]
    ]
}, index=["PAT", "Reinvestment", "FCF"])

for i, y in enumerate(years):
    cf[y] = [
        model_df.iloc[i]["PAT"],
        -reinvestment_rate * model_df.iloc[i]["PAT"],
        model_df.iloc[i]["FCF"]
    ]

st.dataframe(cf.round(0))

# -------------------------------------------------
# DCF CALCULATION — DETAILED
# -------------------------------------------------
st.subheader("💰 Discounted Cash Flow Valuation")

discount_factors = []
pv_fcf = []

for i in range(projection_years):
    df = 1 / ((1 + discount_rate) ** (i + 1))
    discount_factors.append(df)
    pv_fcf.append(model_df["FCF"].iloc[i] * df)

dcf_table = pd.DataFrame({
    "Free Cash Flow": model_df["FCF"].values,
    "Discount Factor": discount_factors,
    "PV of FCF": pv_fcf
}, index=years)

st.markdown("### 🔹 Present Value of Explicit Cash Flows")
st.dataframe(dcf_table.round(2))

# -------------------------------------------------
# TERMINAL VALUE — FULL BREAKDOWN
# -------------------------------------------------
st.markdown("### 🔹 Terminal Value Calculation")

last_fcf = model_df["FCF"].iloc[-1]

terminal_value = (
    last_fcf * (1 + terminal_growth)
    / (discount_rate - terminal_growth)
)

pv_terminal = terminal_value * discount_factors[-1]

terminal_table = pd.DataFrame({
    "Value": [
        last_fcf,
        terminal_growth,
        discount_rate,
        terminal_value,
        pv_terminal
    ]
}, index=[
    "Final Year FCF",
    "Terminal Growth Rate",
    "Discount Rate (WACC)",
    "Terminal Value = FCF × (1+g)/(WACC−g)",
    "PV of Terminal Value"
])

st.dataframe(terminal_table.round(4))

# -------------------------------------------------
# ENTERPRISE VALUE
# -------------------------------------------------
st.subheader("🏁 Enterprise Value")

ev_table = pd.DataFrame({
    "Amount": [
        sum(pv_fcf),
        pv_terminal,
        sum(pv_fcf) + pv_terminal
    ]
}, index=[
    "PV of Explicit FCFs",
    "PV of Terminal Value",
    "Enterprise Value"
])

st.dataframe(ev_table.round(0))
st.metric("Enterprise Value", f"{(sum(pv_fcf) + pv_terminal):,.0f}")

# -------------------------------------------------
# Explanation
# -------------------------------------------------
with st.expander("🧠 Valuation Explanation"):
    st.markdown(f"""
**Step 1 — Project Cash Flows**  
Free Cash Flow is derived from PAT after reinvestment.

**Step 2 — Discount Cash Flows**  
Each FCF is discounted using WACC = **{discount_rate:.0%}**

**Step 3 — Terminal Value**  
Calculated using Gordon Growth Model:

Terminal Value =  
FCF × (1 + g) / (WACC − g)

where g = **{terminal_growth:.0%}**

**Step 4 — Enterprise Value**  
Enterprise Value =  
PV of explicit FCFs + PV of terminal value
""")
