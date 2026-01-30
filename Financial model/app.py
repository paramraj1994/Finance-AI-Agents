import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def fmt_num(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def fmt_pct(x):
    try:
        if pd.isna(x):
            return ""
        return f"{float(x):.1f}%"
    except Exception:
        return str(x)

def style_vertical_statement(df: pd.DataFrame, row_formatters: dict):
    """
    df: vertical statement (rows = line items, cols = years)
    row_formatters: mapping {row_label: formatter_function}
    """
    def _apply_row_format(row):
        f = row_formatters.get(row.name, fmt_num)
        return row.map(f)

    return (
        df.style
        .apply(_apply_row_format, axis=1)
        .set_properties(**{"text-align": "center"})
        .set_table_styles([
            {"selector": "th", "props": [("text-align", "center")]},
            {"selector": "td", "props": [("text-align", "center")]}
        ])
    )

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Explained Statements", layout="wide")
st.title("📊 Financial Model – Explained Financial Statements")
st.caption("Scenario-based | Vertical Statements | Explained Calculations | Center aligned")

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

reinvestment_rate = st.sidebar.slider("Reinvestment Rate (% of PAT)", 0.0, 0.50, 0.10, 0.01)

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
    .set_table_styles([{"selector": "th", "props": [("text-align", "center")]}])
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
# Build Model
# -------------------------------------------------
revenue = base_revenue
rows = {"Revenue": [], "EBITDA": [], "Tax": [], "PAT": [], "FCF": []}

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * ebitda_margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * (1 - reinvestment_rate)

    rows["Revenue"].append(revenue)
    rows["EBITDA"].append(ebitda)
    rows["Tax"].append(tax)
    rows["PAT"].append(pat)
    rows["FCF"].append(fcf)

model_df = pd.DataFrame(rows, index=years)

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
ax.set_ylabel("Amount")
st.pyplot(fig)

# =================================================
# 1) PROFIT & LOSS (Vertical + Explained)
# =================================================
st.subheader("📑 Profit & Loss Statement (Vertical, Explained)")

pnl_vertical = pd.DataFrame(index=[
    "Revenue",
    "EBITDA Margin (%)",
    "EBITDA",
    "Tax Rate (%)",
    "Tax",
    "Profit After Tax (PAT)"
], columns=years, dtype=float)

pnl_vertical.loc["Revenue"] = model_df["Revenue"].values
pnl_vertical.loc["EBITDA Margin (%)"] = (ebitda_margin * 100.0)
pnl_vertical.loc["EBITDA"] = model_df["EBITDA"].values
pnl_vertical.loc["Tax Rate (%)"] = (tax_rate * 100.0)
pnl_vertical.loc["Tax"] = model_df["Tax"].values
pnl_vertical.loc["Profit After Tax (PAT)"] = model_df["PAT"].values

pnl_formatters = {
    "EBITDA Margin (%)": fmt_pct,
    "Tax Rate (%)": fmt_pct,
}

st.dataframe(style_vertical_statement(pnl_vertical, pnl_formatters))

# =================================================
# 2) CASH FLOW (Vertical + Explained)
# =================================================
st.subheader("💵 Cash Flow Statement (Vertical, Explained)")

cashflow_vertical = pd.DataFrame(index=[
    "Profit After Tax (PAT)",
    "Reinvestment Rate (% of PAT)",
    "Less: Reinvestment",
    "Free Cash Flow (FCF)"
], columns=years, dtype=float)

cashflow_vertical.loc["Profit After Tax (PAT)"] = model_df["PAT"].values
cashflow_vertical.loc["Reinvestment Rate (% of PAT)"] = (reinvestment_rate * 100.0)
cashflow_vertical.loc["Less: Reinvestment"] = -(reinvestment_rate * model_df["PAT"]).values
cashflow_vertical.loc["Free Cash Flow (FCF)"] = model_df["FCF"].values

cf_formatters = {
    "Reinvestment Rate (% of PAT)": fmt_pct,
}

st.dataframe(style_vertical_statement(cashflow_vertical, cf_formatters))

# =================================================
# DCF CALCULATION
# =================================================
discount_factors = np.array([1 / (1 + discount_rate) ** (i + 1) for i in range(projection_years)], dtype=float)
pv_fcf = model_df["FCF"].values * discount_factors

terminal_value = (model_df["FCF"].iloc[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
pv_terminal_value = terminal_value * discount_factors[-1]
enterprise_value = pv_fcf.sum() + pv_terminal_value

# =================================================
# 3) VALUATION SUMMARY (Vertical + Explained)
# =================================================
st.subheader("💰 Valuation Summary (Vertical, Explained)")

valuation_vertical = pd.DataFrame(index=[
    "Discount Rate (WACC %)",
    "Terminal Growth Rate (%)",
    "PV of FCF (Projection Period)",
    "Terminal Value (Gordon Growth)",
    "PV of Terminal Value",
    "Enterprise Value"
], columns=["Amount"], dtype=float)

valuation_vertical.loc["Discount Rate (WACC %)","Amount"] = discount_rate * 100.0
valuation_vertical.loc["Terminal Growth Rate (%)","Amount"] = terminal_growth * 100.0
valuation_vertical.loc["PV of FCF (Projection Period)","Amount"] = pv_fcf.sum()
valuation_vertical.loc["Terminal Value (Gordon Growth)","Amount"] = terminal_value
valuation_vertical.loc["PV of Terminal Value","Amount"] = pv_terminal_value
valuation_vertical.loc["Enterprise Value","Amount"] = enterprise_value

val_formatters = {
    "Discount Rate (WACC %)": fmt_pct,
    "Terminal Growth Rate (%)": fmt_pct,
}

st.dataframe(style_vertical_statement(valuation_vertical, val_formatters))

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Explanation Panel
# -------------------------------------------------
with st.expander("🧠 How the model calculates each line item"):
    st.markdown(f"""
### Profit & Loss
- **Revenue** grows annually: Revenueₜ = Revenueₜ₋₁ × (1 + **{growth:.0%}**)
- **EBITDA** = Revenue × **EBITDA Margin ({ebitda_margin:.0%})**
- **Tax** = EBITDA × **Tax Rate ({tax_rate:.0%})**
- **PAT** = EBITDA − Tax

### Cash Flow
- **Reinvestment** = PAT × **Reinvestment Rate ({reinvestment_rate:.0%})**
- **FCF** = PAT − Reinvestment

### Valuation (DCF)
- **PV of FCF** = FCFₜ × Discount Factorₜ, where Discount Factorₜ = 1/(1+WACC)ᵗ
- **Terminal Value** (Gordon growth) = FCF_last × (1+g) / (WACC − g)
- **Enterprise Value** = Sum(PV of FCFs) + PV(Terminal Value)
""")
