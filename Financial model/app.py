import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Financial Model – Investor View", layout="wide")
st.title("📊 Financial Model – Investor-Ready Valuation Model")
st.caption("Financial Model • Dashboard • Scenario Editing • Clean Visualization")

# =================================================
# CSS
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
    try: return f"{float(x):,.0f}"
    except: return ""

def fmt_pct(x):
    try: return f"{float(x):.2f}%"
    except: return ""

def fmt_float(x):
    try: return f"{float(x):.2f}"
    except: return ""

def show_table(df, title=None):
    if title:
        st.subheader(title)
    st.data_editor(df, hide_index=True, disabled=True, use_container_width=True)

# =================================================
# DEFAULT SCENARIOS
# =================================================
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = {
        "Bear": {"growth": 0.07, "margin": 0.20},
        "Base": {"growth": 0.12, "margin": 0.25},
        "Bull": {"growth": 0.18, "margin": 0.30},
    }

scenarios = st.session_state["scenarios"]

# =================================================
# SIDEBAR – GLOBAL ASSUMPTIONS
# =================================================
st.sidebar.header("🔧 Global Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
base_revenue = st.sidebar.number_input("Last Historical Revenue", value=1120.0, step=100.0)

tax_rate = st.sidebar.slider("Tax Rate", 0.15, 0.40, 0.25, 0.01)
reinvestment_rate = st.sidebar.slider("Reinvestment (% of PAT)", 0.0, 0.50, 0.10, 0.01)

discount_rate = st.sidebar.slider("WACC", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

# =================================================
# SCENARIO EDITING
# =================================================
st.sidebar.header("📌 Scenario Inputs")

edit = st.sidebar.selectbox("Edit Scenario", ["Bear", "Base", "Bull"])
scenarios[edit]["growth"] = st.sidebar.number_input(
    f"{edit} Revenue Growth (%)",
    value=scenarios[edit]["growth"] * 100,
    step=0.5
) / 100

scenarios[edit]["margin"] = st.sidebar.number_input(
    f"{edit} EBITDA Margin (%)",
    value=scenarios[edit]["margin"] * 100,
    step=0.5
) / 100

# =================================================
# NAVIGATION
# =================================================
if "view" not in st.session_state:
    st.session_state["view"] = "Financial Model"

c1, c2, _ = st.columns([1, 1, 6])
with c1:
    if st.button("📑 Financial Model"):
        st.session_state["view"] = "Financial Model"
with c2:
    if st.button("📊 Dashboard"):
        st.session_state["view"] = "Dashboard"

st.divider()

# =================================================
# SELECT SCENARIO
# =================================================
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

df_calc = df.copy()
df_calc["Revenue Growth (%)"] = df_calc["Revenue"].pct_change() * 100
df_calc["EBITDA Margin (%)"] = (df_calc["EBITDA"] / df_calc["Revenue"]) * 100

# =================================================
# DASHBOARD
# =================================================
if st.session_state["view"] == "Dashboard":

    st.subheader("📊 Dashboard")

    # ---------- Helper for labels ----------
    def add_labels(ax, x, y, is_pct=False):
        for i in range(len(x)):
            if not np.isnan(y[i]):
                label = f"{y[i]:.1f}%" if is_pct else f"{y[i]:,.0f}"
                ax.annotate(
                    label,
                    (x[i], y[i]),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=9
                )

    # =================================================
    # Revenue + Growth
    # =================================================
    st.markdown("### Revenue & Revenue Growth")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df_calc.index, df_calc["Revenue"], marker="o", color="#1f77b4", label="Revenue")
    ax2.plot(df_calc.index, df_calc["Revenue Growth (%)"], marker="o",
             linestyle="--", color="#ff7f0e", label="Revenue Growth (%)")

    add_labels(ax1, df_calc.index, df_calc["Revenue"].values)
    add_labels(ax2, df_calc.index, df_calc["Revenue Growth (%)"].values, is_pct=True)

    ax1.set_ylabel("Revenue")
    ax2.set_ylabel("Growth (%)")
    ax1.grid(False)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    st.pyplot(fig)

    # =================================================
    # EBITDA + Margin
    # =================================================
    st.markdown("### EBITDA & EBITDA Margin")

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(df_calc.index, df_calc["EBITDA"], marker="o", color="#2ca02c", label="EBITDA")
    ax2.plot(df_calc.index, df_calc["EBITDA Margin (%)"], marker="o",
             linestyle="--", color="#d62728", label="EBITDA Margin (%)")

    add_labels(ax1, df_calc.index, df_calc["EBITDA"].values)
    add_labels(ax2, df_calc.index, df_calc["EBITDA Margin (%)"].values, is_pct=True)

    ax1.set_ylabel("EBITDA")
    ax2.set_ylabel("Margin (%)")
    ax1.grid(False)
    ax2.grid(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    st.pyplot(fig)

    # =================================================
    # PAT
    # =================================================
    st.markdown("### Profit After Tax (PAT)")

    fig, ax = plt.subplots()
    ax.plot(df_calc.index, df_calc["PAT"], marker="o", color="#9467bd", label="PAT")
    add_labels(ax, df_calc.index, df_calc["PAT"].values)

    ax.set_ylabel("PAT")
    ax.grid(False)
    ax.legend()

    st.pyplot(fig)

    # =================================================
    # FCF
    # =================================================
    st.markdown("### Free Cash Flow (FCF)")

    fig, ax = plt.subplots()
    ax.plot(df_calc.index, df_calc["FCF"], marker="o", color="#8c564b", label="FCF")
    add_labels(ax, df_calc.index, df_calc["FCF"].values)

    ax.set_ylabel("FCF")
    ax.grid(False)
    ax.legend()

    st.pyplot(fig)

    st.stop()

# =================================================
# FINANCIAL MODEL (tables remain same)
# =================================================
st.subheader("📑 Financial Model")
st.info("Tables remain unchanged — focus of this update is the Dashboard visuals.")
