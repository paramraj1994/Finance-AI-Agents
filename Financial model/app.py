import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Explained Statements", layout="wide")
st.title("📊 Financial Model – Explained Financial Statements")
st.caption("Vertical Statements • Scenario Toggle • Explained Calculations • Center aligned")

# -------------------------------------------------
# Center-align ALL Streamlit dataframes (CSS)
# -------------------------------------------------
st.markdown(
    """
    <style>
      /* Center align Streamlit dataframe cells + headers */
      div[data-testid="stDataFrame"] table td, 
      div[data-testid="stDataFrame"] table th {
        text-align: center !important;
        vertical-align: middle !important;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Formatting helpers (safe: no pandas Styler)
# -------------------------------------------------
def fmt_num(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return str(x)

def fmt_pct(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    try:
        return f"{float(x):.1f}%"
    except Exception:
        return str(x)

def format_statement_vertical(df_numeric: pd.DataFrame, percent_rows: set, formula_col: str = "Formula") -> pd.DataFrame:
    """
    Takes a vertical statement where index = line items and columns = years plus optional 'Formula'.
    Returns a string-formatted dataframe (numbers + % rows) safe for st.dataframe.
    """
    df = df_numeric.copy()

    # Make sure formula column exists (if passed)
    cols = list(df.columns)
    year_cols = [c for c in cols if c != formula_col]

    # Convert all year columns to strings with proper formatting per row
    for r in df.index:
        if r in percent_rows:
            df.loc[r, year_cols] = [fmt_pct(v) for v in df.loc[r, year_cols].tolist()]
        else:
            df.loc[r, year_cols] = [fmt_num(v) for v in df.loc[r, year_cols].tolist()]

    # Keep formula as-is (string)
    if formula_col in df.columns:
        df[formula_col] = df[formula_col].fillna("").astype(str)

    # Reorder columns: Formula first, then years (looks like "explanation + numbers")
    if formula_col in df.columns:
        df = df[[formula_col] + year_cols]

    return df

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
# Scenario Assumptions Table (simple, safe formatting)
# -------------------------------------------------
st.subheader("📌 Scenario Assumptions")
assumptions_df = pd.DataFrame(scenarios).T.copy()
assumptions_df["Revenue Growth"] = assumptions_df["Revenue Growth"].map(lambda v: f"{v:.1%}")
assumptions_df["EBITDA Margin"] = assumptions_df["EBITDA Margin"].map(lambda v: f"{v:.1%}")
st.dataframe(assumptions_df, use_container_width=True)

# -------------------------------------------------
# Scenario Toggle
# -------------------------------------------------
selected_scenario = st.radio("🎯 Select Scenario", options=list(scenarios.keys()), horizontal=True)
growth = scenarios[selected_scenario]["Revenue Growth"]
ebitda_margin = scenarios[selected_scenario]["EBITDA Margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# -------------------------------------------------
# Build Model (numeric base)
# -------------------------------------------------
revenue = base_revenue
model = {"Revenue": [], "EBITDA": [], "Tax": [], "PAT": [], "FCF": []}

for _ in years:
    revenue *= (1 + growth)
    ebitda = revenue * ebitda_margin
    tax = ebitda * tax_rate
    pat = ebitda - tax
    fcf = pat * (1 - reinvestment_rate)  # explained proxy
    model["Revenue"].append(revenue)
    model["EBITDA"].append(ebitda)
    model["Tax"].append(tax)
    model["PAT"].append(pat)
    model["FCF"].append(fcf)

model_df = pd.DataFrame(model, index=years)

# -------------------------------------------------
# Trends Chart
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
# TABLE 1: PROFIT & LOSS (Vertical + Explained + Formula)
# =================================================
st.subheader("📑 Profit & Loss Statement (Vertical, Explained)")

pnl = pd.DataFrame(index=[
    "Revenue",
    "Revenue Growth (%)",
    "EBITDA Margin (%)",
    "EBITDA",
    "Tax Rate (%)",
    "Tax",
    "Profit After Tax (PAT)"
], columns=["Formula"] + years, dtype=float)

pnl.loc["Revenue", years] = model_df["Revenue"].values
pnl.loc["Revenue Growth (%)", years] = growth * 100.0
pnl.loc["EBITDA Margin (%)", years] = ebitda_margin * 100.0
pnl.loc["EBITDA", years] = model_df["EBITDA"].values
pnl.loc["Tax Rate (%)", years] = tax_rate * 100.0
pnl.loc["Tax", years] = model_df["Tax"].values
pnl.loc["Profit After Tax (PAT)", years] = model_df["PAT"].values

pnl.loc["Revenue", "Formula"] = "Revenueₜ = Revenueₜ₋₁ × (1 + Growth)"
pnl.loc["Revenue Growth (%)", "Formula"] = "Selected scenario growth"
pnl.loc["EBITDA Margin (%)", "Formula"] = "Selected scenario margin"
pnl.loc["EBITDA", "Formula"] = "EBITDA = Revenue × EBITDA Margin"
pnl.loc["Tax Rate (%)", "Formula"] = "Global tax assumption"
pnl.loc["Tax", "Formula"] = "Tax = EBITDA × Tax Rate"
pnl.loc["Profit After Tax (PAT)", "Formula"] = "PAT = EBITDA − Tax"

pnl_display = format_statement_vertical(
    pnl,
    percent_rows={"Revenue Growth (%)", "EBITDA Margin (%)", "Tax Rate (%)"},
    formula_col="Formula"
)
st.dataframe(pnl_display, use_container_width=True)

# =================================================
# TABLE 2: CASH FLOW (Vertical + Explained + Formula)
# =================================================
st.subheader("💵 Cash Flow Statement (Vertical, Explained)")

cf = pd.DataFrame(index=[
    "Profit After Tax (PAT)",
    "Reinvestment Rate (% of PAT)",
    "Less: Reinvestment",
    "Free Cash Flow (FCF)"
], columns=["Formula"] + years, dtype=float)

cf.loc["Profit After Tax (PAT)", years] = model_df["PAT"].values
cf.loc["Reinvestment Rate (% of PAT)", years] = reinvestment_rate * 100.0
cf.loc["Less: Reinvestment", years] = -(reinvestment_rate * model_df["PAT"]).values
cf.loc["Free Cash Flow (FCF)", years] = model_df["FCF"].values

cf.loc["Profit After Tax (PAT)", "Formula"] = "From P&L"
cf.loc["Reinvestment Rate (% of PAT)", "Formula"] = "Global reinvestment assumption"
cf.loc["Less: Reinvestment", "Formula"] = "Reinvestment = PAT × Reinvestment Rate"
cf.loc["Free Cash Flow (FCF)", "Formula"] = "FCF = PAT − Reinvestment"

cf_display = format_statement_vertical(
    cf,
    percent_rows={"Reinvestment Rate (% of PAT)"},
    formula_col="Formula"
)
st.dataframe(cf_display, use_container_width=True)

# =================================================
# DCF CALCULATION
# =================================================
discount_factors = np.array([1 / (1 + discount_rate) ** (i + 1) for i in range(projection_years)], dtype=float)
pv_fcf = model_df["FCF"].values * discount_factors

terminal_value = (model_df["FCF"].iloc[-1] * (1 + terminal_growth)) / (discount_rate - terminal_growth)
pv_terminal_value = terminal_value * discount_factors[-1]
enterprise_value = pv_fcf.sum() + pv_terminal_value

# =================================================
# TABLE 3: VALUATION SUMMARY (Vertical + Explained)
# =================================================
st.subheader("💰 Valuation Summary (Vertical, Explained)")

val = pd.DataFrame(index=[
    "Discount Rate (WACC %)",
    "Terminal Growth Rate (%)",
    "PV of FCF (Projection Period)",
    "Terminal Value",
    "PV of Terminal Value",
    "Enterprise Value"
], columns=["Formula", "Amount"], dtype=float)

val.loc["Discount Rate (WACC %)", "Amount"] = discount_rate * 100.0
val.loc["Terminal Growth Rate (%)", "Amount"] = terminal_growth * 100.0
val.loc["PV of FCF (Projection Period)", "Amount"] = pv_fcf.sum()
val.loc["Terminal Value", "Amount"] = terminal_value
val.loc["PV of Terminal Value", "Amount"] = pv_terminal_value
val.loc["Enterprise Value", "Amount"] = enterprise_value

val.loc["Discount Rate (WACC %)", "Formula"] = "Global WACC assumption"
val.loc["Terminal Growth Rate (%)", "Formula"] = "Global terminal growth assumption"
val.loc["PV of FCF (Projection Period)", "Formula"] = "Σ(FCFₜ × DFₜ)"
val.loc["Terminal Value", "Formula"] = "FCF_last × (1+g) / (WACC − g)"
val.loc["PV of Terminal Value", "Formula"] = "Terminal Value × DF_last"
val.loc["Enterprise Value", "Formula"] = "PV(FCFs) + PV(Terminal)"

# Format valuation table: percent rows + numeric rows
val_display = val.copy()
val_display["Amount"] = val_display["Amount"].astype(float)

# Convert to string safely
val_out = pd.DataFrame(index=val_display.index, columns=val_display.columns)
val_out["Formula"] = val_display["Formula"].fillna("").astype(str)
for r in val_display.index:
    if r in {"Discount Rate (WACC %)", "Terminal Growth Rate (%)"}:
        val_out.loc[r, "Amount"] = fmt_pct(val_display.loc[r, "Amount"])
    else:
        val_out.loc[r, "Amount"] = fmt_num(val_display.loc[r, "Amount"])

st.dataframe(val_out, use_container_width=True)

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Optional: Download a combined model (numeric)
# -------------------------------------------------
st.subheader("⬇️ Download Selected Scenario (Numeric Model)")

download_df = model_df.copy()
download_df["Discount Factor"] = discount_factors
download_df["PV of FCF"] = pv_fcf

csv = download_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False)
st.download_button(
    f"Download {selected_scenario} Case CSV",
    csv,
    f"{selected_scenario}_Financial_Model.csv",
    "text/csv"
)

# -------------------------------------------------
# Explanation Panel
# -------------------------------------------------
with st.expander("🧠 Explanation (end-to-end)"):
    st.markdown(f"""
### Profit & Loss
- **Revenue** grows annually using scenario growth (**{growth:.0%}**).
- **EBITDA = Revenue × EBITDA Margin** (**{ebitda_margin:.0%}**)
- **Tax = EBITDA × Tax Rate** (**{tax_rate:.0%}**)
- **PAT = EBITDA − Tax**

### Cash Flow
- **Reinvestment = PAT × Reinvestment Rate** (**{reinvestment_rate:.0%}**)
- **FCF = PAT − Reinvestment**

### Valuation (DCF)
- **Discount Factorₜ = 1/(1+WACC)ᵗ** where WACC = **{discount_rate:.0%}**
- **PV of FCF = Σ(FCFₜ × DFₜ)**
- **Terminal Value = FCF_last × (1+g)/(WACC − g)** where g = **{terminal_growth:.0%}**
- **Enterprise Value = PV of FCF + PV of Terminal Value**
""")
