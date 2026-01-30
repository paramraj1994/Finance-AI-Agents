import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Scenario Valuation", layout="wide")
st.title("📊 Financial Model – Scenario-Based Valuation")
st.caption("Vertical Financial Statements | Scenario Toggle | DCF")

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
st.dataframe(assumptions_df.style.format("{:.1%}"))

# -------------------------------------------------
# Scenario Toggle
# -------------------------------------------------
selected_scenario = st.radio(
    "🎯 Select Scenario",
    options=list(scenarios.keys()),
    horizontal=True
)

growth = scenarios[selected_scenario]["Revenue Growth"]
margin = scenarios[selected_scenario]["EBITDA Margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# -------------------------------------------------
# Financial Model
# -------------------------------------------------
revenue = base_revenue
data = {
    "Revenue": [],
    "EBITDA": [],
    "Tax": [],
    "PAT": [],
    "FCF": []
}

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * 0.9

    data["Revenue"].append(revenue)
    data["EBITDA"].append(ebitda)
    data["Tax"].append(tax)
    data["PAT"].append(pat)
    data["FCF"].append(fcf)

model_df = pd.DataFrame(data, index=years)

# -------------------------------------------------
# Trend Chart
# -------------------------------------------------
st.subheader(f"📈 {selected_scenario} Case – Financial Trends")

fig, ax = plt.subplots()
ax.plot(model_df.index, model_df["Revenue"], marker="o", label="Revenue")
ax.plot(model_df.index, model_df["EBITDA"], marker="o", label="EBITDA")
ax.plot(model_df.index, model_df["PAT"], marker="o", label="PAT")
ax.set_ylabel("Amount")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# =================================================
# TABLE 1: PROFIT & LOSS (VERTICAL)
# =================================================
st.subheader("📑 Profit & Loss Statement (Vertical)")

pnl_vertical = pd.DataFrame({
    "Revenue": model_df["Revenue"],
    "EBITDA": model_df["EBITDA"],
    "Tax": model_df["Tax"],
    "Profit After Tax (PAT)": model_df["PAT"]
}).T

st.dataframe(
    pnl_vertical.style.format("{:,.0f}")
)

# =================================================
# TABLE 2: CASH FLOW STATEMENT (VERTICAL)
# =================================================
st.subheader("💵 Cash Flow Statement (Vertical)")

cashflow_vertical = pd.DataFrame({
    "Profit After Tax": model_df["PAT"],
    "Less: Reinvestment (10%)": -0.10 * model_df["PAT"],
    "Free Cash Flow": model_df["FCF"]
}).T

st.dataframe(
    cashflow_vertical.style.format("{:,.0f}")
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
# TABLE 3: VALUATION SUMMARY (VERTICAL)
# =================================================
st.subheader("💰 Valuation Summary (Vertical)")

valuation_vertical = pd.DataFrame(
    {
        "Amount": [
            pv_fcf.sum(),
            terminal_value,
            pv_terminal_value,
            enterprise_value
        ]
    },
    index=[
        "PV of Free Cash Flows",
        "Terminal Value",
        "PV of Terminal Value",
        "Enterprise Value"
    ]
)

st.dataframe(
    valuation_vertical.style.format("{:,.0f}")
)

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Explanation Block
# -------------------------------------------------
with st.expander("🧠 How EBITDA of 280 is calculated from Revenue 1120"):
    st.write("""
**EBITDA = Revenue × EBITDA Margin**

Example (Base Case):
- Revenue = 1,120
- EBITDA Margin = 25%

**EBITDA = 1,120 × 25% = 280**
""")
