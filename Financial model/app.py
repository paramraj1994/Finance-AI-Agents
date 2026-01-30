import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Financial Model – Investor View", layout="wide")
st.title("📊 Financial Model – Investor-Ready Valuation Model")
st.caption("Financial Model • Dashboard • Scenario Editing • Detailed DCF")

# =================================================
# CSS – CENTER ALIGN TABLE CELLS
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
# SIDEBAR: EDIT ONE SCENARIO AT A TIME
# =================================================
st.sidebar.header("📌 Scenario Inputs (Editable)")

edit_scenario = st.sidebar.selectbox(
    "Which scenario do you want to edit?",
    options=["Bear", "Base", "Bull"]
)

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
# TOP NAV BUTTONS: Financial Model vs Dashboard
# =================================================
if "view" not in st.session_state:
    st.session_state["view"] = "Financial Model"

nav1, nav2, _ = st.columns([1, 1, 6])
with nav1:
    if st.button("📑 Financial Model"):
        st.session_state["view"] = "Financial Model"
with nav2:
    if st.button("📊 Dashboard"):
        st.session_state["view"] = "Dashboard"

st.divider()

# =================================================
# TOGGLE SCENARIO FOR OUTPUT
# =================================================
selected = st.radio("🎯 Select Scenario to Run", list(scenarios.keys()), horizontal=True)
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

# -------------------------------------------------
# For charts: show growth/margin correctly for ALL years
# (Because your model assumes constant growth and margin)
# -------------------------------------------------
growth_series_pct = pd.Series([growth * 100] * projection_years, index=years)  # 12.00, 12.00, ...
margin_series_pct = pd.Series([margin * 100] * projection_years, index=years)  # 25.00, 25.00, ...

# =================================================
# DASHBOARD VIEW
# =================================================
def add_point_labels(ax, x_labels, y_vals, is_pct=False):
    """Add labels that try to stay inside chart bounds."""
    # Convert x_labels into positions 0..n-1 for annotate
    x_pos = np.arange(len(x_labels))

    # Add some padding to axis so labels don’t clip
    ymin, ymax = ax.get_ylim()
    rng = (ymax - ymin) if ymax != ymin else 1.0
    pad = rng * 0.18
    ax.set_ylim(ymin, ymax + pad)

    # Re-read after expanding
    ymin, ymax = ax.get_ylim()

    for i, v in enumerate(y_vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue

        label = f"{v:.1f}%" if is_pct else f"{v:,.0f}"

        # If point is near the top, push label downward
        if v > (ymax - pad * 0.55):
            dy = -14
            va = "top"
        else:
            dy = 8
            va = "bottom"

        ax.annotate(
            label,
            (x_pos[i], v),
            textcoords="offset points",
            xytext=(0, dy),
            ha="center",
            va=va,
            fontsize=9,
            clip_on=True,
        )

def dual_axis_chart(title, x_labels, y_left, y_right, left_label, right_label,
                    left_color, right_color, right_is_pct=True):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    x_pos = np.arange(len(x_labels))

    ax1.plot(x_pos, y_left, marker="o", color=left_color, label=left_label)
    ax2.plot(x_pos, y_right, marker="o", linestyle="--", color=right_color, label=right_label)

    # X axis labels
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)

    # Labels
    ax1.set_ylabel(left_label)
    ax2.set_ylabel(right_label)

    # Remove gridlines
    ax1.grid(False)
    ax2.grid(False)

    # Add data labels
    add_point_labels(ax1, x_labels, np.array(y_left), is_pct=False)
    add_point_labels(ax2, x_labels, np.array(y_right), is_pct=right_is_pct)

    # Make percent axis look nice
    if right_is_pct:
        rmin = np.nanmin(y_right)
        rmax = np.nanmax(y_right)
        if np.isfinite(rmin) and np.isfinite(rmax):
            # Add a small band even if flat, so it doesn't look "stuck"
            if abs(rmax - rmin) < 0.001:
                ax2.set_ylim(rmin - 1.0, rmax + 1.0)
            else:
                ax2.set_ylim(rmin - 0.15 * (rmax - rmin), rmax + 0.15 * (rmax - rmin))

    # Legend combined
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ax1.set_title(title)
    return fig

if st.session_state["view"] == "Dashboard":
    st.subheader("📊 Dashboard")

    # 1) Revenue + Growth
    st.markdown("### Revenue & Revenue Growth (%)")
    fig = dual_axis_chart(
        title="Revenue vs Revenue Growth",
        x_labels=years,
        y_left=df["Revenue"].values,
        y_right=growth_series_pct.values,
        left_label="Revenue",
        right_label="Revenue Growth (%)",
        left_color="#1f77b4",
        right_color="#ff7f0e",
        right_is_pct=True,
    )
    st.pyplot(fig)

    # 2) EBITDA + Margin
    st.markdown("### EBITDA & EBITDA Margin (%)")
    fig = dual_axis_chart(
        title="EBITDA vs EBITDA Margin",
        x_labels=years,
        y_left=df["EBITDA"].values,
        y_right=margin_series_pct.values,
        left_label="EBITDA",
        right_label="EBITDA Margin (%)",
        left_color="#2ca02c",
        right_color="#d62728",
        right_is_pct=True,
    )
    st.pyplot(fig)

    # 3) PAT
    st.markdown("### Profit After Tax (PAT)")
    fig, ax = plt.subplots()
    x_pos = np.arange(len(years))
    ax.plot(x_pos, df["PAT"].values, marker="o", color="#9467bd", label="PAT")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("PAT")
    ax.grid(False)
    ax.legend(loc="best")
    add_point_labels(ax, years, df["PAT"].values, is_pct=False)
    st.pyplot(fig)

    # 4) FCF
    st.markdown("### Free Cash Flow (FCF)")
    fig, ax = plt.subplots()
    x_pos = np.arange(len(years))
    ax.plot(x_pos, df["FCF"].values, marker="o", color="#8c564b", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("FCF")
    ax.grid(False)
    ax.legend(loc="best")
    add_point_labels(ax, years, df["FCF"].values, is_pct=False)
    st.pyplot(fig)

    st.stop()

# =================================================
# FINANCIAL MODEL VIEW
# =================================================
st.subheader("📑 Financial Model")

# -------------------------------------------------
# Profit & Loss Statement
# -------------------------------------------------
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

# -------------------------------------------------
# Cash Flow Statement
# -------------------------------------------------
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

# -------------------------------------------------
# DCF – Detailed
# -------------------------------------------------
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

# -------------------------------------------------
# Terminal Value – Detailed
# -------------------------------------------------
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

# -------------------------------------------------
# Enterprise Value Summary
# -------------------------------------------------
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

# -------------------------------------------------
# Download (numeric model)
# -------------------------------------------------
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
