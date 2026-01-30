import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Scenario Valuation", layout="wide")
st.title("📊 Financial Model – Scenario-Based Valuation")
st.caption("Profit & Loss • Cash Flow • Valuation Summary")

# -------------------------------------------------
# File Upload (Optional)
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Historical Financials (Excel – optional)",
    type=["xlsx"]
)

if uploaded_file is not None:
    hist_df = pd.read_excel(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(hist_df)
else:
    st.info("No file uploaded. Model will use manual assumptions.")

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
    value=1000.0,
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
def build_model(growth, margin):
    revenue = base_revenue
    rows = []

    for _ in years:
        revenue *= (1 + growth)
        ebitda = revenue * margin
        tax = ebitda * tax_rate
        pat = ebitda - tax
        fcf = pat * 0.9  # simple, explainable proxy

        rows.append([revenue, ebitda, tax, pat, fcf])

    return pd.DataFrame(
        rows,
        columns=["Revenue", "EBITDA", "Tax", "PAT", "FCF"],
        index=years
    )

model_df = build_model(growth, margin)

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
# TABLE 1: PROFIT & LOSS STATEMENT
# =================================================
st.subheader("📑 Profit & Loss Statement")

pnl_table = model_df[[
    "Revenue",
    "EBITDA",
    "Tax",
    "PAT"
]]

st.dataframe(
    pnl_table.style.format({
        "Revenue": "{:,.0f}",
        "EBITDA": "{:,.0f}",
        "Tax": "{:,.0f}",
        "PAT": "{:,.0f}",
    })
)

# =================================================
# TABLE 2: CASH FLOW STATEMENT
# =================================================
st.subheader("💵 Cash Flow Statement")

cashflow_df = pd.DataFrame(index=model_df.index)

cashflow_df["Profit After Tax"] = model_df["PAT"]
cashflow_df["Less: Reinvestment (10%)"] = -0.10 * model_df["PAT"]
cashflow_df["Free Cash Flow"] = model_df["FCF"]

st.dataframe(
    cashflow_df.style.format("{:,.0f}")
)

# =================================================
# DCF CALCULATION
# =================================================
model_df = model_df.copy()

model_df["Discount Factor"] = [
    1 / (1 + discount_rate) ** (i + 1)
    for i in range(len(model_df))
]

model_df["PV of FCF"] = model_df["FCF"] * model_df["Discount Factor"]

terminal_value = (
    model_df["FCF"].iloc[-1] * (1 + terminal_growth)
    / (discount_rate - terminal_growth)
)

pv_terminal_value = terminal_value * model_df["Discount Factor"].iloc[-1]
enterprise_value = model_df["PV of FCF"].sum() + pv_terminal_value

# =================================================
# TABLE 3: VALUATION SUMMARY
# =================================================
st.subheader("💰 Valuation Summary")

valuation_df = pd.DataFrame({
    "Item": [
        "PV of Free Cash Flows (Projection Period)",
        "Terminal Value",
        "PV of Terminal Value",
        "Enterprise Value"
    ],
    "Amount": [
        model_df["PV of FCF"].sum(),
        terminal_value,
        pv_terminal_value,
        enterprise_value
    ]
})

st.dataframe(
    valuation_df.style.format({"Amount": "{:,.0f}"})
)

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Download
# -------------------------------------------------
st.subheader("⬇️ Download Selected Scenario Model")

download_df = pd.concat(
    [pnl_table, cashflow_df, model_df[["Discount Factor", "PV of FCF"]]],
    axis=1
)

csv = download_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False)

st.download_button(
    f"Download {selected_scenario} Case CSV",
    csv,
    f"{selected_scenario}_Financial_Model.csv",
    "text/csv"
)
