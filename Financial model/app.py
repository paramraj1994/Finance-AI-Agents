import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Financial Model – Final Investor View", layout="wide")
st.title("📊 Financial Model – Investor-Ready Valuation")
st.caption("P&L • Cash Flow • DCF • Clean Formatting")

# =================================================
# CSS – CENTER ALIGN ALL TABLE VALUES
# =================================================
st.markdown("""
<style>
div[data-testid="stDataFrame"] table th,
div[data-testid="stDataFrame"] table td {
    text-align: center !important;
    vertical-align: middle !important;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# FORMATTERS
# =================================================
def fmt_num(x):
    try:
        return f"{int(round(float(x))):,}"
    except:
        return ""

def fmt_pct(x):
    try:
        return f"{float(x):.1f}%"
    except:
        return ""

def format_table(df, percent_rows=set()):
    out = df.copy()
    for r in out.index:
        if r in percent_rows:
            out.loc[r] = out.loc[r].apply(fmt_pct)
        else:
            out.loc[r] = out.loc[r].apply(fmt_num)
    return out

# =================================================
# SIDEBAR ASSUMPTIONS
# =================================================
st.sidebar.header("🔧 Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
base_revenue = st.sidebar.number_input("Last Historical Revenue", value=1120.0, step=100.0)

tax_rate = st.sidebar.slider("Tax Rate", 0.15, 0.40, 0.25, 0.01)
reinvestment_rate = st.sidebar.slider("Reinvestment (% of PAT)", 0.0, 0.50, 0.10, 0.01)

discount_rate = st.sidebar.slider("WACC", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth", 0.02, 0.06, 0.04, 0.005)

# =================================================
# SCENARIOS
# =================================================
scenarios = {
    "Bear": {"growth": 0.07, "margin": 0.20},
    "Base": {"growth": 0.12, "margin": 0.25},
    "Bull": {"growth": 0.18, "margin": 0.30},
}

st.subheader("📌 Scenario Assumptions")
assump = pd.DataFrame(scenarios).T
assump["growth"] = assump["growth"].map(lambda x: f"{x:.1%}")
assump["margin"] = assump["margin"].map(lambda x: f"{x:.1%}")
st.dataframe(assump)

selected = st.radio("Select Scenario", scenarios.keys(), horizontal=True)
growth = scenarios[selected]["growth"]
margin = scenarios[selected]["margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# =================================================
# BUILD MODEL
# =================================================
revenue = base_revenue
model = {"Revenue": [], "EBITDA": [], "Tax": [], "PAT": [], "FCF": []}

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * (1 - reinvestment_rate)

    model["Revenue"].append(revenue)
    model["EBITDA"].append(ebitda)
    model["Tax"].append(tax)
    model["PAT"].append(pat)
    model["FCF"].append(fcf)

df = pd.DataFrame(model, index=years)

# =================================================
# PROFIT & LOSS (VERTICAL)
# =================================================
st.subheader("📑 Profit & Loss Statement")

pnl = pd.DataFrame(index=[
    "Revenue",
    "EBITDA Margin (%)",
    "EBITDA",
    "Tax Rate (%)",
    "Tax",
    "Profit After Tax (PAT)"
], columns=years)

pnl.loc["Revenue"] = df["Revenue"].values
pnl.loc["EBITDA Margin (%)"] = margin * 100
pnl.loc["EBITDA"] = df["EBITDA"].values
pnl.loc["Tax Rate (%)"] = tax_rate * 100
pnl.loc["Tax"] = df["Tax"].values
pnl.loc["Profit After Tax (PAT)"] = df["PAT"].values

st.dataframe(format_table(
    pnl,
    percent_rows={"EBITDA Margin (%)", "Tax Rate (%)"}
))

# =================================================
# CASH FLOW STATEMENT
# =================================================
st.subheader("💵 Cash Flow Statement")

cf = pd.DataFrame(index=[
    "Profit After Tax (PAT)",
    "Reinvestment Rate (%)",
    "Less: Reinvestment",
    "Free Cash Flow (FCF)"
], columns=years)

cf.loc["Profit After Tax (PAT)"] = df["PAT"].values
cf.loc["Reinvestment Rate (%)"] = reinvestment_rate * 100
cf.loc["Less: Reinvestment"] = -(reinvestment_rate * df["PAT"]).values
cf.loc["Free Cash Flow (FCF)"] = df["FCF"].values

st.dataframe(format_table(
    cf,
    percent_rows={"Reinvestment Rate (%)"}
))

# =================================================
# DCF CALCULATION
# =================================================
discount_factors = [(1 / (1 + discount_rate) ** (i + 1)) for i in range(projection_years)]
pv_fcf = [df["FCF"].iloc[i] * discount_factors[i] for i in range(projection_years)]

dcf = pd.DataFrame(index=years)
dcf["Free Cash Flow"] = df["FCF"].values
dcf["Discount Factor"] = discount_factors
dcf["PV of FCF"] = pv_fcf

st.subheader("💰 Discounted Cash Flow")

st.dataframe(format_table(dcf))

# =================================================
# TERMINAL VALUE
# =================================================
last_fcf = df["FCF"].iloc[-1]
terminal_value = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
pv_terminal = terminal_value * discount_factors[-1]

tv = pd.DataFrame({
    "Amount": [
        last_fcf,
        terminal_growth * 100,
        discount_rate * 100,
        terminal_value,
        pv_terminal
    ]
}, index=[
    "Final Year FCF",
    "Terminal Growth Rate (%)",
    "Discount Rate (WACC %)",
    "Terminal Value",
    "PV of Terminal Value"
])

st.subheader("📘 Terminal Value Calculation")
st.dataframe(format_table(
    tv,
    percent_rows={"Terminal Growth Rate (%)", "Discount Rate (WACC %)"}
))

# =================================================
# ENTERPRISE VALUE
# =================================================
enterprise_value = sum(pv_fcf) + pv_terminal

ev = pd.DataFrame({
    "Amount": [
        sum(pv_fcf),
        pv_terminal,
        enterprise_value
    ]
}, index=[
    "PV of Explicit FCFs",
    "PV of Terminal Value",
    "Enterprise Value"
])

st.subheader("🏁 Enterprise Value")
st.dataframe(format_table(ev))
st.metric("Enterprise Value", fmt_num(enterprise_value))
