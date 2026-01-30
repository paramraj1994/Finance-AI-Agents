import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Explained Statements", layout="wide")
st.title("📊 Financial Model – Explained Financial Statements")
st.caption("Scenario-based | Vertical Statements | Fully Explained Calculations")

# -------------------------------------------------
# Sidebar – Global Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Global Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.15, 0.40, 0.25, 0.01)
discount_rate = st.sidebar.slider("Discount Rate (WACC)", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

base_revenue = st.sidebar.number_input(
    "Last Historical Revenue",
    min_value=0.0,
    value=1120.0,
    step=100.0
)

# -------------------------------------------------
# Scenario Definitions
# -------------------------------------------------
scenarios = {
    "Bear": {"Revenue Growth": 0.07, "EBITDA Margin": 0.20},
    "Base": {"Revenue Growth": 0.12, "EBITDA Margin": 0.25},
    "Bull": {"Revenue Growth": 0.18, "EBITDA Margin": 0.30},
}

# -------------------------------------------------
# Scenario Assumptions Table
# -------------------------------------------------
st.subheader("📌 Scenario Assumptions")

assumptions_df = pd.DataFrame(scenarios).T
st.dataframe(
    assumptions_df.style
    .format("{:.1%}")
    .set_properties(**{"text-align": "center"})
)

# -------------------------------------------------
# Scenario Toggle
# -------------------------------------------------
selected_scenario = st.radio(
    "🎯 Select Scenario",
    options=list(scenarios.keys()),
    horizontal=True
)

growth = scenarios[selected_scenario]["Revenue Growth"]
ebitda_margin = scenarios[selected_scenario]["EBITDA Margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# -------------------------------------------------
# Financial Model
# -------------------------------------------------
revenue = base_revenue
model = {
    "Revenue": [],
    "EBITDA": [],
    "Tax": [],
    "PAT": [],
    "FCF": []
}

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * ebitda_margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * 0.9

    model["Revenue"].append(revenue)
    model["EBITDA"].append(ebitda)
    model["Tax"].append(tax)
    model["PAT"].append(pat)
    model["FCF"].append(fcf)

model_df = pd.DataFrame(model, index=years)

# -------------------------------------------------
# Trend Chart
# -------------------------------------------------
st.subheader(f"📈 {selected_scenario} Case – Financial Trends")

fig, ax = plt.subplots()
ax.plot(model_df.index, model_df["Revenue"], marker="o", label="Revenue")
ax.plot(model_df.index, model_df["EBITDA"], marker="o", label="EBITDA")
ax.plot(model_df.index, model_df["PAT"], marker="o", label="PAT")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# =================================================
# TABLE 1: PROFIT & LOSS (EXPLAINED)
# =================================================
st.subheader("📑 Profit & Loss Statement (Explained)")

pnl_vertical = pd.DataFrame({
    "Revenue": model_df["Revenue"],
    "EBITDA Margin (%)": ebitda_margin * 100,
    "EBITDA": model_df["EBITDA"],
    "Tax Rate (%)": tax_rate * 100,
    "Tax": model_df["Tax"],
    "Profit After Tax (PAT)": model_df["PAT"],
}).T

st.dataframe(
    pnl_vertical.style
    .format("{:,.0f}", subset=["Revenue", "EBITDA", "Tax", "Profit After Tax (PAT)"])
    .format("{:.1f}%", subset=["EBITDA Margin (%)", "Tax Rate (%)"])
    .set_properties(**{"text-align": "center"})
)

# =================================================
# TABLE 2: CASH FLOW STATEMENT (EXPLAINED)
# =================================================
st.subheader("💵 Cash Flow Statement (Explained)")

cashflow_vertical = pd.DataFrame({
    "Profit After Tax": model_df["PAT"],
    "Reinvestment Rate (%)": 10.0,
    "Less: Reinvestment": -0.10 * model_df["PAT"],
    "Free Cash Flow": model_df["FCF"]
}).T

st.dataframe(
    cashflow_vertical.style
    .format("{:,.0f}", subset=["Profit After Tax", "Less: Reinvestment", "Free Cash Flow"])
    .format("{:.1f}%", subset=["Reinvestment Rate (%)"])
    .set_properties(**{"text-align": "center"})
)

# =================================================
# DCF CALCULATION
# =================================================
discount_factors = [
    1 / (1 + discount_rate) ** (i + 1)
    for i in range(len(model_df))
]

pv_fcf = model_df["FCF"].values * discount_factors

terminal_value = (
    model_df["FCF"].iloc[-1] * (1 + terminal_growth)
    / (discount_rate - terminal_growth)
)

pv_terminal_value = terminal_value * discount_factors[-1]
enterprise_value = pv_fcf.sum() + pv_terminal_value

# =================================================
# TABLE 3: VALUATION SUMMARY (EXPLAINED)
# =================================================
st.subheader("💰 Valuation Summary (Explained)")

valuation_vertical = pd.DataFrame(
    {
        "Amount": [
            discount_rate * 100,
            terminal_growth * 100,
            pv_fcf.sum(),
            terminal_value,
            pv_terminal_value,
            enterprise_value
        ]
    },
    index=[
        "Discount Rate (WACC %)",
        "Terminal Growth Rate (%)",
        "PV of Free Cash Flows",
        "Terminal Value",
        "PV of Terminal Value",
        "Enterprise Value"
    ]
)

st.dataframe(
    valuation_vertical.style
    .format("{:,.0f}", subset=["Amount"])
    .format("{:.1f}%", subset=["Amount"], na_rep="")
    .set_properties(**{"text-align": "center"})
)

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Explanation Panel
# -------------------------------------------------
with st.expander("🧠 How each statement is calculated"):
    st.markdown("""
**Profit & Loss**
- Revenue grows annually using the selected growth rate
- EBITDA = Revenue × EBITDA Margin
- Tax = EBITDA × Tax Rate
- PAT = EBITDA − Tax

**Cash Flow**
- Reinvestment assumed at 10% of PAT
- Free Cash Flow = PAT − Reinvestment

**Valuation**
- FCFs discounted using WACC
- Terminal Value calculated using Gordon Growth Model
- Enterprise Value = PV of FCFs + PV of Terminal Value
""")
