import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Financial Model – Investor View", layout="wide")
st.title("📊 Financial Model – Investor-Ready Valuation Model")
st.caption("Proper layout • readable headings • correct number formatting")

# =================================================
# CSS – CENTER ALIGN ALL TABLE CELLS
# =================================================
st.markdown("""
<style>
div[data-testid="stDataFrame"] table th,
div[data-testid="stDataFrame"] table td {
    text-align: center !important;
    vertical-align: middle !important;
    white-space: nowrap;
}
</style>
""", unsafe_allow_html=True)

# =================================================
# FORMATTERS
# =================================================
def fmt_currency(x):
    try:
        return f"{float(x):,.0f}"
    except:
        return ""

def fmt_pct_2(x):
    try:
        return f"{float(x):.2f}%"
    except:
        return ""

def fmt_float_2(x):
    try:
        return f"{float(x):.2f}"
    except:
        return ""

# =================================================
# SIDEBAR ASSUMPTIONS
# =================================================
st.sidebar.header("🔧 Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
base_revenue = st.sidebar.number_input("Last Historical Revenue", value=1120.0, step=100.0)

tax_rate = st.sidebar.slider("Tax Rate", 0.15, 0.40, 0.25, 0.01)
reinvestment_rate = st.sidebar.slider("Reinvestment (% of PAT)", 0.0, 0.50, 0.10, 0.01)

discount_rate = st.sidebar.slider("WACC", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

# =================================================
# SCENARIOS
# =================================================
scenarios = {
    "Bear": {"growth": 0.07, "margin": 0.20},
    "Base": {"growth": 0.12, "margin": 0.25},
    "Bull": {"growth": 0.18, "margin": 0.30},
}

st.subheader("📌 Scenario Assumptions")
assump = pd.DataFrame(scenarios).T.reset_index().rename(columns={"index": "Scenario"})
assump["Revenue Growth"] = assump["growth"].apply(lambda x: fmt_pct_2(x * 100))
assump["EBITDA Margin"] = assump["margin"].apply(lambda x: fmt_pct_2(x * 100))
assump = assump[["Scenario", "Revenue Growth", "EBITDA Margin"]]
st.dataframe(assump, use_container_width=True)

selected = st.radio("Select Scenario", scenarios.keys(), horizontal=True)
growth = scenarios[selected]["growth"]
margin = scenarios[selected]["margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# =================================================
# BUILD MODEL
# =================================================
revenue = base_revenue
rows = []

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * (1 - reinvestment_rate)
    rows.append([revenue, ebitda, tax, pat, fcf])

df = pd.DataFrame(rows, columns=["Revenue", "EBITDA", "Tax", "PAT", "FCF"], index=years)

# =================================================
# PROFIT & LOSS STATEMENT
# =================================================
st.subheader("📑 Profit & Loss Statement")

pnl = pd.DataFrame({
    "Line Item": [
        "Revenue",
        "EBITDA Margin (%)",
        "EBITDA",
        "Tax Rate (%)",
        "Tax",
        "Profit After Tax (PAT)"
    ]
})

for y in years:
    pnl[y] = [
        fmt_currency(df.loc[y, "Revenue"]),
        fmt_pct_2(margin * 100),
        fmt_currency(df.loc[y, "EBITDA"]),
        fmt_pct_2(tax_rate * 100),
        fmt_currency(df.loc[y, "Tax"]),
        fmt_currency(df.loc[y, "PAT"])
    ]

st.dataframe(pnl, use_container_width=True)

# =================================================
# CASH FLOW STATEMENT
# =================================================
st.subheader("💵 Cash Flow Statement")

cf = pd.DataFrame({
    "Line Item": [
        "Profit After Tax (PAT)",
        "Reinvestment Rate (%)",
        "Less: Reinvestment",
        "Free Cash Flow (FCF)"
    ]
})

for y in years:
    cf[y] = [
        fmt_currency(df.loc[y, "PAT"]),
        fmt_pct_2(reinvestment_rate * 100),
        fmt_currency(-reinvestment_rate * df.loc[y, "PAT"]),
        fmt_currency(df.loc[y, "FCF"])
    ]

st.dataframe(cf, use_container_width=True)

# =================================================
# DISCOUNTED CASH FLOW
# =================================================
st.subheader("💰 Discounted Cash Flow")

discount_factors = []
pv_fcf = []

for i, y in enumerate(years):
    dfactor = 1 / ((1 + discount_rate) ** (i + 1))
    discount_factors.append(dfactor)
    pv_fcf.append(df.loc[y, "FCF"] * dfactor)

dcf = pd.DataFrame({
    "Line Item": [
        "Free Cash Flow",
        "Discount Factor",
        "Present Value of FCF"
    ]
})

for i, y in enumerate(years):
    dcf[y] = [
        fmt_currency(df.loc[y, "FCF"]),
        fmt_float_2(discount_factors[i]),
        fmt_currency(pv_fcf[i])
    ]

st.dataframe(dcf, use_container_width=True)

# =================================================
# TERMINAL VALUE
# =================================================
st.subheader("📘 Terminal Value Calculation")

last_fcf = df["FCF"].iloc[-1]
terminal_value = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
pv_terminal = terminal_value * discount_factors[-1]

tv = pd.DataFrame({
    "Line Item": [
        "Final Year Free Cash Flow",
        "Terminal Growth Rate (%)",
        "Discount Rate (WACC %)",
        "Terminal Value",
        "PV of Terminal Value"
    ],
    "Amount": [
        fmt_currency(last_fcf),
        fmt_pct_2(terminal_growth * 100),
        fmt_pct_2(discount_rate * 100),
        fmt_currency(terminal_value),
        fmt_currency(pv_terminal)
    ]
})

st.dataframe(tv, use_container_width=True)

# =================================================
# ENTERPRISE VALUE
# =================================================
enterprise_value = sum(pv_fcf) + pv_terminal

ev = pd.DataFrame({
    "Line Item": [
        "PV of Explicit Cash Flows",
        "PV of Terminal Value",
        "Enterprise Value"
    ],
    "Amount": [
        fmt_currency(sum(pv_fcf)),
        fmt_currency(pv_terminal),
        fmt_currency(enterprise_value)
    ]
})

st.subheader("🏁 Enterprise Value Summary")
st.dataframe(ev, use_container_width=True)
st.metric("Enterprise Value", fmt_currency(enterprise_value))
