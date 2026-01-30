import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Financial Model – Investor View", layout="wide")
st.title("📊 Financial Model – Investor-Ready Valuation Model")
st.caption("Clean tables • no index column • proper formatting • detailed DCF • scenario editing")

# =================================================
# CSS – CENTER ALIGN ALL TABLE CELLS
# =================================================
st.markdown("""
<style>
div[data-testid="stDataEditor"] table th,
div[data-testid="stDataEditor"] table td {
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
    except Exception:
        return ""

def fmt_pct_2(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return ""

def fmt_float_2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""

def show_table(df: pd.DataFrame, title: str | None = None):
    """Display tables without the 0,1,2 index column."""
    if title:
        st.subheader(title)
    st.data_editor(
        df,
        hide_index=True,
        disabled=True,
        use_container_width=True
    )

# =================================================
# DEFAULT SCENARIOS (stored in session state)
# =================================================
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {
        "Bear": {"growth": 0.07, "margin": 0.20},
        "Base": {"growth": 0.12, "margin": 0.25},
        "Bull": {"growth": 0.18, "margin": 0.30},
    }

scenarios = st.session_state["scenarios"]

# =================================================
# SIDEBAR ASSUMPTIONS
# =================================================
st.sidebar.header("🔧 Global Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
base_revenue = st.sidebar.number_input("Last Historical Revenue", value=1120.0, step=100.0)

tax_rate = st.sidebar.slider("Tax Rate", 0.15, 0.40, 0.25, 0.01)
reinvestment_rate = st.sidebar.slider("Reinvestment (% of PAT)", 0.0, 0.50, 0.10, 0.01)

discount_rate = st.sidebar.slider("WACC", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

# =================================================
# SIDEBAR: EDIT ONE SCENARIO AT A TIME
# =================================================
st.sidebar.header("📌 Scenario Inputs (Editable)")

edit_scenario = st.sidebar.selectbox(
    "Which scenario do you want to edit?",
    options=["Bear", "Base", "Bull"]
)

# Pull current values
current_growth = scenarios[edit_scenario]["growth"] * 100
current_margin = scenarios[edit_scenario]["margin"] * 100

new_growth_pct = st.sidebar.number_input(
    f"{edit_scenario}: Revenue Growth (%)",
    value=float(current_growth),
    step=0.50
)

new_margin_pct = st.sidebar.number_input(
    f"{edit_scenario}: EBITDA Margin (%)",
    value=float(current_margin),
    step=0.50
)

if st.sidebar.button("✅ Save Scenario Changes"):
    scenarios[edit_scenario]["growth"] = new_growth_pct / 100
    scenarios[edit_scenario]["margin"] = new_margin_pct / 100
    st.sidebar.success(f"{edit_scenario} scenario updated!")

# =================================================
# SCENARIO ASSUMPTIONS TABLE
# =================================================
st.subheader("📌 Scenario Assumptions (Current)")

assump = pd.DataFrame({
    "Scenario": list(scenarios.keys()),
    "Revenue Growth": [fmt_pct_2(v["growth"] * 100) for v in scenarios.values()],
    "EBITDA Margin": [fmt_pct_2(v["margin"] * 100) for v in scenarios.values()],
})
show_table(assump)

# =================================================
# TOGGLE SCENARIO FOR MODEL OUTPUT
# =================================================
selected = st.radio("🎯 Select Scenario to Run Model", list(scenarios.keys()), horizontal=True)
growth = scenarios[selected]["growth"]
margin = scenarios[selected]["margin"]

years = [f"Year {i+1}" for i in range(projection_years)]

# =================================================
# BUILD MODEL (numeric)
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
# TREND CHART
# =================================================
st.subheader(f"📈 {selected} Case – Financial Trends")
fig, ax = plt.subplots()
ax.plot(df.index, df["Revenue"], marker="o", label="Revenue")
ax.plot(df.index, df["EBITDA"], marker="o", label="EBITDA")
ax.plot(df.index, df["PAT"], marker="o", label="PAT")
ax.legend()
ax.grid(True)
ax.set_ylabel("Amount")
st.pyplot(fig)

# =================================================
# PROFIT & LOSS STATEMENT
# =================================================
pnl = pd.DataFrame({"Line Item": [
    "Revenue",
    "Revenue Growth (%)",
    "EBITDA Margin (%)",
    "EBITDA",
    "Tax Rate (%)",
    "Tax",
    "Profit After Tax (PAT)"
]})

for y in years:
    pnl[y] = [
        fmt_currency(df.loc[y, "Revenue"]),
        fmt_pct_2(growth * 100),
        fmt_pct_2(margin * 100),
        fmt_currency(df.loc[y, "EBITDA"]),
        fmt_pct_2(tax_rate * 100),
        fmt_currency(df.loc[y, "Tax"]),
        fmt_currency(df.loc[y, "PAT"]),
    ]

show_table(pnl, "📑 Profit & Loss Statement")

# =================================================
# CASH FLOW STATEMENT
# =================================================
cf = pd.DataFrame({"Line Item": [
    "Profit After Tax (PAT)",
    "Reinvestment Rate (% of PAT)",
    "Less: Reinvestment",
    "Free Cash Flow (FCF)"
]})

for y in years:
    cf[y] = [
        fmt_currency(df.loc[y, "PAT"]),
        fmt_pct_2(reinvestment_rate * 100),
        fmt_currency(-reinvestment_rate * df.loc[y, "PAT"]),
        fmt_currency(df.loc[y, "FCF"]),
    ]

show_table(cf, "💵 Cash Flow Statement")

# =================================================
# DCF – DETAILED (per-year)
# =================================================
discount_factors = []
pv_fcf = []

for i, y in enumerate(years):
    dfactor = 1 / ((1 + discount_rate) ** (i + 1))
    discount_factors.append(dfactor)
    pv_fcf.append(df.loc[y, "FCF"] * dfactor)

dcf_detail = pd.DataFrame({"Line Item": [
    "Free Cash Flow (FCF)",
    "Discount Factor",
    "PV of FCF"
]})

for i, y in enumerate(years):
    dcf_detail[y] = [
        fmt_currency(df.loc[y, "FCF"]),
        fmt_float_2(discount_factors[i]),
        fmt_currency(pv_fcf[i]),
    ]

show_table(dcf_detail, "💰 Discounted Cash Flow (Detailed)")

# =================================================
# TERMINAL VALUE – FULL BREAKDOWN
# =================================================
last_fcf = df["FCF"].iloc[-1]
terminal_value = (last_fcf * (1 + terminal_growth)) / (discount_rate - terminal_growth)
pv_terminal = terminal_value * discount_factors[-1]

tv = pd.DataFrame({
    "Line Item": [
        "Final Year FCF",
        "Terminal Growth Rate (g) (%)",
        "Discount Rate (WACC) (%)",
        "Terminal Value = FCF×(1+g)/(WACC−g)",
        "PV of Terminal Value = TV×DF_last"
    ],
    "Amount": [
        fmt_currency(last_fcf),
        fmt_pct_2(terminal_growth * 100),
        fmt_pct_2(discount_rate * 100),
        fmt_currency(terminal_value),
        fmt_currency(pv_terminal),
    ]
})
show_table(tv, "📘 Terminal Value Calculation (Detailed)")

# =================================================
# ENTERPRISE VALUE SUMMARY
# =================================================
enterprise_value = sum(pv_fcf) + pv_terminal

ev = pd.DataFrame({
    "Line Item": [
        "PV of Explicit FCFs (Sum of PV of FCF)",
        "PV of Terminal Value",
        "Enterprise Value"
    ],
    "Amount": [
        fmt_currency(sum(pv_fcf)),
        fmt_currency(pv_terminal),
        fmt_currency(enterprise_value),
    ]
})
show_table(ev, "🏁 Enterprise Value Summary")
st.metric("Enterprise Value", fmt_currency(enterprise_value))

# =================================================
# DOWNLOAD (numeric model)
# =================================================
st.subheader("⬇️ Download Selected Scenario (Numeric Model)")
download_df = df.copy()
download_df["Discount Factor"] = discount_factors
download_df["PV of FCF"] = pv_fcf

csv = download_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False)
st.download_button(
    f"Download {selected} Case CSV",
    csv,
    f"{selected}_Financial_Model.csv",
    "text/csv"
)

with st.expander("🧠 Explanation (End-to-End)"):
    st.markdown(f"""
### Profit & Loss
- Revenue grows annually using scenario growth (**{growth:.2%}**)
- EBITDA = Revenue × EBITDA Margin (**{margin:.2%}**)
- Tax = EBITDA × Tax Rate (**{tax_rate:.2%}**)
- PAT = EBITDA − Tax

### Cash Flow
- Reinvestment = PAT × Reinvestment Rate (**{reinvestment_rate:.2%}**)
- FCF = PAT − Reinvestment

### DCF Valuation
- Discount Factorₜ = 1/(1+WACC)ᵗ where WACC = **{discount_rate:.2%}**
- PV of FCF = FCFₜ × Discount Factorₜ
- Terminal Value = FCF_last × (1+g) / (WACC − g) where g = **{terminal_growth:.2%}**
- Enterprise Value = Sum(PV of FCF) + PV of Terminal Value
""")
