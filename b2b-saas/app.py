import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="B2B SaaS Financial Model – Investor Suite", layout="wide")
st.title("🧩 B2B SaaS Financial Model – Investor Suite")
st.caption("Driver-based SaaS model • year-wise scenarios • Dashboard • DCF valuation • export • save/load assumptions")

# =================================================
# CSS – center align tables (st.data_editor)
# =================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataEditor"] table th,
    div[data-testid="stDataEditor"] table td {
        text-align: center !important;
        vertical-align: middle !important;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =================================================
# Helpers
# =================================================
def fmt_currency(x):
    try:
        return f"{float(x):,.0f}"
    except Exception:
        return ""

def fmt_pct2(x):
    try:
        return f"{float(x):.2f}%"
    except Exception:
        return ""

def fmt_float2(x):
    try:
        return f"{float(x):.2f}"
    except Exception:
        return ""

def show_table(df: pd.DataFrame, title: str | None = None):
    if title:
        st.subheader(title)
    st.data_editor(df, hide_index=True, disabled=True, use_container_width=True)

def ensure_len(arr, n, fill=None):
    arr = list(arr) if isinstance(arr, (list, tuple)) else []
    if len(arr) >= n:
        return arr[:n]
    if fill is None:
        fill = arr[-1] if arr else 0.0
    return arr + [fill] * (n - len(arr))

def safe_minmax(series):
    s = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return 0.0, 1.0
    return float(s.min()), float(s.max())

def add_point_labels(ax, x_pos, y_vals, is_pct=False):
    ymin, ymax = ax.get_ylim()
    rng = (ymax - ymin) if ymax != ymin else 1.0
    pad = rng * 0.18
    ax.set_ylim(ymin, ymax + pad)
    ymin, ymax = ax.get_ylim()

    for i, v in enumerate(y_vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        label = f"{v:.1f}%" if is_pct else f"{v:,.0f}"
        if v > (ymax - pad * 0.55):
            dy, va = -14, "top"
        else:
            dy, va = 8, "bottom"
        ax.annotate(
            label, (x_pos[i], v),
            textcoords="offset points", xytext=(0, dy),
            ha="center", va=va, fontsize=9, clip_on=True
        )

def dual_axis_line_chart(title, x_labels, y_left, y_right,
                         left_label, right_label,
                         left_color, right_color,
                         right_is_pct=True):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x_pos = np.arange(len(x_labels))

    ax1.plot(x_pos, y_left, marker="o", color=left_color, label=left_label)
    ax2.plot(x_pos, y_right, marker="o", linestyle="--", color=right_color, label=right_label)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    ax1.set_ylabel(left_label)
    ax2.set_ylabel(right_label)

    ax1.grid(False)
    ax2.grid(False)

    add_point_labels(ax1, x_pos, np.array(y_left, dtype=float), is_pct=False)
    add_point_labels(ax2, x_pos, np.array(y_right, dtype=float), is_pct=right_is_pct)

    if right_is_pct:
        rmin, rmax = safe_minmax(y_right)
        if abs(rmax - rmin) < 1e-6:
            ax2.set_ylim(rmin - 1.0, rmax + 1.0)
        else:
            band = 0.15 * (rmax - rmin)
            ax2.set_ylim(rmin - band, rmax + band)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax1.set_title(title)
    fig.tight_layout()
    return fig

# =================================================
# Defaults
# =================================================
DEFAULT_GLOBALS = {
    "projection_years": 5,
    "tax_rate_pct": 25.0,
    "wacc_pct": 12.0,
    "terminal_growth_pct": 4.0,
    "revenue_recognition_pct_of_arr": 100.0,  # 100% of ARR recognized as revenue annually
    "da_pct_rev": 3.0,
    "capex_pct_rev": 4.0,
    "nwc_pct_rev": 3.0,  # SaaS usually low WC
    "net_debt": 0.0,
    "cash": 0.0,
    "exit_multiple_ev_rev": 6.0,  # if Exit Multiple method: EV / Revenue
}

# Year-wise scenario inputs for SaaS (edit per year):
# - New customer growth (%)
# - Gross churn (% of beginning customers)
# - Net Revenue Retention (NRR %) - drives expansion on ARR
# - ARPA growth (%)
# - Gross margin (%)
# - CAC (currency per new customer)
# - S&M % of revenue
# - R&D % of revenue
# - G&A % of revenue
DEFAULT_SCENARIOS = {
    "Bear": {
        "new_cust_growth_pct": [20, 18, 16, 14, 12],
        "gross_churn_pct":    [12, 12, 11, 11, 10],
        "nrr_pct":            [100, 102, 103, 104, 105],
        "arpa_growth_pct":    [3, 3, 3, 2, 2],
        "gross_margin_pct":   [75, 76, 76, 77, 78],
        "cac_per_new":        [2000, 1900, 1800, 1750, 1700],
        "sm_pct_rev":         [45, 42, 40, 38, 36],
        "rnd_pct_rev":        [18, 17, 16, 15, 14],
        "ga_pct_rev":         [12, 12, 11, 11, 10],
    },
    "Base": {
        "new_cust_growth_pct": [35, 30, 26, 22, 18],
        "gross_churn_pct":     [10, 9.5, 9, 8.5, 8],
        "nrr_pct":             [105, 108, 110, 112, 113],
        "arpa_growth_pct":     [5, 5, 4, 4, 3],
        "gross_margin_pct":    [78, 79, 80, 81, 82],
        "cac_per_new":         [1800, 1700, 1600, 1550, 1500],
        "sm_pct_rev":          [48, 45, 42, 40, 38],
        "rnd_pct_rev":         [20, 19, 18, 17, 16],
        "ga_pct_rev":          [12, 12, 11.5, 11, 10.5],
    },
    "Bull": {
        "new_cust_growth_pct": [55, 45, 35, 28, 22],
        "gross_churn_pct":     [8, 7.5, 7, 6.5, 6],
        "nrr_pct":             [112, 116, 120, 122, 124],
        "arpa_growth_pct":     [7, 7, 6, 5, 5],
        "gross_margin_pct":    [80, 82, 83, 84, 85],
        "cac_per_new":         [1600, 1500, 1400, 1350, 1300],
        "sm_pct_rev":          [50, 46, 42, 38, 35],
        "rnd_pct_rev":         [22, 21, 20, 19, 18],
        "ga_pct_rev":          [12, 11.5, 11, 10.5, 10],
    },
}

# =================================================
# Session state init
# =================================================
if "globals" not in st.session_state:
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = json.loads(json.dumps(DEFAULT_SCENARIOS))  # deep-ish copy
if "view" not in st.session_state:
    st.session_state["view"] = "Financial Model"
if "terminal_method" not in st.session_state:
    st.session_state["terminal_method"] = "Gordon Growth"

# =================================================
# Sidebar: Global assumptions
# =================================================
st.sidebar.header("🔧 Global Assumptions")
g = st.session_state["globals"]

projection_years = st.sidebar.slider("Projection Years", 3, 10, int(g["projection_years"]))
tax_rate_pct = st.sidebar.number_input("Tax Rate (%)", value=float(g["tax_rate_pct"]), step=0.5)
wacc_pct = st.sidebar.number_input("WACC (%)", value=float(g["wacc_pct"]), step=0.5)
terminal_growth_pct = st.sidebar.number_input("Terminal Growth (%)", value=float(g["terminal_growth_pct"]), step=0.25)

rev_rec_pct = st.sidebar.number_input(
    "Revenue Recognition (% of ARR)",
    value=float(g["revenue_recognition_pct_of_arr"]),
    step=5.0,
    help="If revenue recognized differs from ARR (e.g., multi-year contracts / timing). Usually ~100% annualized."
)

da_pct_rev = st.sidebar.number_input("D&A (% of Revenue)", value=float(g["da_pct_rev"]), step=0.25)
capex_pct_rev = st.sidebar.number_input("Capex (% of Revenue)", value=float(g["capex_pct_rev"]), step=0.25)
nwc_pct_rev = st.sidebar.number_input("NWC (% of Revenue)", value=float(g["nwc_pct_rev"]), step=0.25)

net_debt = st.sidebar.number_input("Net Debt", value=float(g["net_debt"]), step=100.0)
cash = st.sidebar.number_input("Cash", value=float(g["cash"]), step=100.0)
exit_multiple_ev_rev = st.sidebar.number_input("Exit Multiple (EV/Revenue)", value=float(g["exit_multiple_ev_rev"]), step=0.5)

st.sidebar.header("📌 Starting SaaS Base")
base_customers = st.sidebar.number_input("Starting Customers", value=200.0, step=10.0)
base_arpa = st.sidebar.number_input("Starting ARPA (Annual $/Customer)", value=6000.0, step=250.0)

# persist globals
st.session_state["globals"] = {
    "projection_years": projection_years,
    "tax_rate_pct": tax_rate_pct,
    "wacc_pct": wacc_pct,
    "terminal_growth_pct": terminal_growth_pct,
    "revenue_recognition_pct_of_arr": rev_rec_pct,
    "da_pct_rev": da_pct_rev,
    "capex_pct_rev": capex_pct_rev,
    "nwc_pct_rev": nwc_pct_rev,
    "net_debt": net_debt,
    "cash": cash,
    "exit_multiple_ev_rev": exit_multiple_ev_rev,
}

# =================================================
# Ensure scenario arrays match projection_years
# =================================================
SCN_KEYS = [
    "new_cust_growth_pct", "gross_churn_pct", "nrr_pct", "arpa_growth_pct",
    "gross_margin_pct", "cac_per_new", "sm_pct_rev", "rnd_pct_rev", "ga_pct_rev"
]
for scn_name in ["Bear", "Base", "Bull"]:
    for k in SCN_KEYS:
        st.session_state["scenarios"][scn_name][k] = ensure_len(
            st.session_state["scenarios"][scn_name].get(k, []),
            projection_years,
            fill=None
        )

# =================================================
# Save/Load assumptions + Reset
# =================================================
st.sidebar.header("💾 Save / Load Assumptions")

assump_payload = {
    "globals": st.session_state["globals"],
    "scenarios": st.session_state["scenarios"],
    "terminal_method": st.session_state["terminal_method"],
    "base": {"customers": base_customers, "arpa": base_arpa},
}
assump_json_str = json.dumps(assump_payload, indent=2)

st.sidebar.download_button(
    "Download Assumptions (JSON)",
    data=assump_json_str,
    file_name="saas_assumptions.json",
    mime="application/json"
)

assump_upload = st.sidebar.file_uploader("Load Assumptions (JSON)", type=["json"], key="assump_json_uploader")
if assump_upload is not None:
    try:
        loaded = json.load(assump_upload)
        if isinstance(loaded, dict) and "globals" in loaded and "scenarios" in loaded:
            st.session_state["globals"] = loaded["globals"]
            st.session_state["scenarios"] = loaded["scenarios"]
            st.session_state["terminal_method"] = loaded.get("terminal_method", "Gordon Growth")
            base_customers = float(loaded.get("base", {}).get("customers", base_customers))
            base_arpa = float(loaded.get("base", {}).get("arpa", base_arpa))
            st.sidebar.success("Assumptions loaded. App will refresh.")
            st.rerun()
        else:
            st.sidebar.error("Invalid assumptions JSON format.")
    except Exception:
        st.sidebar.error("Could not read JSON.")

if st.sidebar.button("🔁 Reset to Defaults"):
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
    st.session_state["scenarios"] = json.loads(json.dumps(DEFAULT_SCENARIOS))
    st.session_state["terminal_method"] = "Gordon Growth"
    st.sidebar.success("Reset done.")
    st.rerun()

# =================================================
# Sidebar: Edit one scenario year-by-year
# =================================================
st.sidebar.header("📌 Scenario Inputs (Year-wise)")
edit_scn = st.sidebar.selectbox("Select Scenario to Edit", ["Bear", "Base", "Bull"], key="edit_scn")

years_labels = [f"Year {i+1}" for i in range(projection_years)]
edit_df = pd.DataFrame({
    "Year": years_labels,
    "New Cust Growth (%)": st.session_state["scenarios"][edit_scn]["new_cust_growth_pct"],
    "Gross Churn (%)": st.session_state["scenarios"][edit_scn]["gross_churn_pct"],
    "NRR (%)": st.session_state["scenarios"][edit_scn]["nrr_pct"],
    "ARPA Growth (%)": st.session_state["scenarios"][edit_scn]["arpa_growth_pct"],
    "Gross Margin (%)": st.session_state["scenarios"][edit_scn]["gross_margin_pct"],
    "CAC per New Cust": st.session_state["scenarios"][edit_scn]["cac_per_new"],
    "S&M (% Rev)": st.session_state["scenarios"][edit_scn]["sm_pct_rev"],
    "R&D (% Rev)": st.session_state["scenarios"][edit_scn]["rnd_pct_rev"],
    "G&A (% Rev)": st.session_state["scenarios"][edit_scn]["ga_pct_rev"],
})

st.sidebar.caption("Edit values below. % columns are in percent (e.g., 10 = 10%).")
edited = st.sidebar.data_editor(
    edit_df,
    hide_index=True,
    disabled=["Year"],
    use_container_width=True,
    key=f"editor_{edit_scn}_{projection_years}"
)

if st.sidebar.button("✅ Save Scenario Year-wise"):
    try:
        def col_to_list(col):
            return pd.to_numeric(edited[col], errors="coerce").fillna(0.0).tolist()

        st.session_state["scenarios"][edit_scn]["new_cust_growth_pct"] = ensure_len(col_to_list("New Cust Growth (%)"), projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["gross_churn_pct"] = ensure_len(col_to_list("Gross Churn (%)"), projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["nrr_pct"] = ensure_len(col_to_list("NRR (%)"), projection_years, fill=100.0)
        st.session_state["scenarios"][edit_scn]["arpa_growth_pct"] = ensure_len(col_to_list("ARPA Growth (%)"), projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["gross_margin_pct"] = ensure_len(col_to_list("Gross Margin (%)"), projection_years, fill=75.0)
        st.session_state["scenarios"][edit_scn]["cac_per_new"] = ensure_len(col_to_list("CAC per New Cust"), projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["sm_pct_rev"] = ensure_len(col_to_list("S&M (% Rev)"), projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["rnd_pct_rev"] = ensure_len(col_to_list("R&D (% Rev)"), projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["ga_pct_rev"] = ensure_len(col_to_list("G&A (% Rev)"), projection_years, fill=0.0)

        st.sidebar.success(f"{edit_scn} updated.")
        st.rerun()
    except Exception:
        st.sidebar.error("Could not save. Please check inputs.")

# =================================================
# Top navigation buttons
# =================================================
nav1, nav2, _ = st.columns([1, 1, 6])
with nav1:
    if st.button("📑 Financial Model"):
        st.session_state["view"] = "Financial Model"
with nav2:
    if st.button("📊 Dashboard"):
        st.session_state["view"] = "Dashboard"
st.divider()

# =================================================
# Scenario selection + terminal method
# =================================================
selected = st.radio("🎯 Select Scenario to Run", ["Bear", "Base", "Bull"], horizontal=True)

terminal_method = st.radio(
    "Terminal Value Method",
    ["Gordon Growth", "Exit Multiple (EV/Revenue)"],
    horizontal=True,
    index=0 if st.session_state["terminal_method"] == "Gordon Growth" else 1
)
st.session_state["terminal_method"] = terminal_method

# Validation
if terminal_method == "Gordon Growth" and (wacc_pct / 100.0) <= (terminal_growth_pct / 100.0):
    st.error("❗ WACC must be greater than Terminal Growth for Gordon Growth terminal value. Please adjust assumptions.")

# =================================================
# SaaS model builder (driver-based)
# =================================================
def build_saas_model(base_customers, base_arpa, scenario: dict, globals_dict: dict):
    n = int(globals_dict["projection_years"])
    years = [f"Year {i+1}" for i in range(n)]

    # inputs
    new_cust_growth = np.array(ensure_len(scenario["new_cust_growth_pct"], n, fill=0.0), dtype=float) / 100.0
    gross_churn = np.array(ensure_len(scenario["gross_churn_pct"], n, fill=0.0), dtype=float) / 100.0
    nrr = np.array(ensure_len(scenario["nrr_pct"], n, fill=100.0), dtype=float) / 100.0
    arpa_growth = np.array(ensure_len(scenario["arpa_growth_pct"], n, fill=0.0), dtype=float) / 100.0
    gm = np.array(ensure_len(scenario["gross_margin_pct"], n, fill=75.0), dtype=float) / 100.0
    cac = np.array(ensure_len(scenario["cac_per_new"], n, fill=0.0), dtype=float)
    sm_pct = np.array(ensure_len(scenario["sm_pct_rev"], n, fill=0.0), dtype=float) / 100.0
    rnd_pct = np.array(ensure_len(scenario["rnd_pct_rev"], n, fill=0.0), dtype=float) / 100.0
    ga_pct = np.array(ensure_len(scenario["ga_pct_rev"], n, fill=0.0), dtype=float) / 100.0

    # globals
    tax = globals_dict["tax_rate_pct"] / 100.0
    wacc = globals_dict["wacc_pct"] / 100.0
    tg = globals_dict["terminal_growth_pct"] / 100.0
    rev_rec = globals_dict["revenue_recognition_pct_of_arr"] / 100.0
    da_pct_rev = globals_dict["da_pct_rev"] / 100.0
    capex_pct_rev = globals_dict["capex_pct_rev"] / 100.0
    nwc_pct_rev = globals_dict["nwc_pct_rev"] / 100.0

    # state vars
    customers_beg = float(base_customers)
    arpa = float(base_arpa)

    # For ΔNWC
    prev_nwc = 0.0

    rows = []
    for i in range(n):
        # Customers
        new_customers = customers_beg * new_cust_growth[i]
        churned_customers = customers_beg * gross_churn[i]
        customers_end = max(customers_beg + new_customers - churned_customers, 0.0)
        customers_avg = (customers_beg + customers_end) / 2.0

        # ARPA
        arpa = arpa * (1 + arpa_growth[i])

        # ARR (approx)
        # Base ARR from average customers * ARPA, then apply NRR to capture net expansion (incl. churn & upsell)
        arr = customers_avg * arpa * nrr[i]

        # Revenue recognition
        revenue = arr * rev_rec

        # COGS / Gross profit
        gross_profit = revenue * gm[i]
        cogs = revenue - gross_profit

        # Opex (SaaS style)
        sm = revenue * sm_pct[i]
        rnd = revenue * rnd_pct[i]
        ga = revenue * ga_pct[i]

        # CAC spend (often sits inside S&M, but we show separately as an explain line)
        cac_spend = new_customers * cac[i]

        # EBITDA (treat CAC as part of S&M; we do NOT double-count, but we show CAC separately for explainability)
        ebitda = gross_profit - sm - rnd - ga

        # EBIT and taxes
        da = revenue * da_pct_rev
        ebit = ebitda - da
        nopat = ebit * (1 - tax)

        # FCF
        capex_val = revenue * capex_pct_rev
        nwc = revenue * nwc_pct_rev
        delta_nwc = nwc - prev_nwc
        prev_nwc = nwc

        fcf = nopat + da - capex_val - delta_nwc

        rows.append([
            customers_beg, new_customers, churned_customers, customers_end,
            arpa, arr, revenue,
            cogs, gross_profit,
            sm, rnd, ga, ebitda,
            da, ebit, nopat,
            capex_val, delta_nwc, fcf,
            cac_spend
        ])

        customers_beg = customers_end

    df = pd.DataFrame(
        rows,
        columns=[
            "Customers (Beg)", "New Customers", "Churned Customers", "Customers (End)",
            "ARPA", "ARR", "Revenue",
            "COGS", "Gross Profit",
            "Sales & Marketing", "R&D", "G&A", "EBITDA",
            "D&A", "EBIT", "NOPAT",
            "Capex", "ΔNWC", "FCF",
            "CAC Spend (info)"
        ],
        index=years
    )

    # Helpful ratios
    df["Gross Margin (%)"] = np.where(df["Revenue"] > 0, (df["Gross Profit"] / df["Revenue"]) * 100, np.nan)
    df["EBITDA Margin (%)"] = np.where(df["Revenue"] > 0, (df["EBITDA"] / df["Revenue"]) * 100, np.nan)

    # Discounting
    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    # Terminal value
    if terminal_method == "Gordon Growth":
        tv = np.nan if wacc <= tg else (df["FCF"].iloc[-1] * (1 + tg) / (wacc - tg))
    else:
        tv = df["Revenue"].iloc[-1] * float(globals_dict["exit_multiple_ev_rev"])

    pv_tv = tv * discount_factors[-1] if np.isfinite(tv) else np.nan
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
    ev = pv_explicit + (float(pv_tv) if np.isfinite(pv_tv) else 0.0)

    equity_value = ev - float(globals_dict["net_debt"]) + float(globals_dict["cash"])
    tv_share = (pv_tv / ev) if ev != 0 and np.isfinite(pv_tv) else np.nan

    return {
        "years": years,
        "df": df,
        "pv_explicit": pv_explicit,
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "enterprise_value": ev,
        "equity_value": equity_value,
        "terminal_share": tv_share,
    }

globals_dict = st.session_state["globals"]
scenario_dict = st.session_state["scenarios"][selected]
result = build_saas_model(base_customers, base_arpa, scenario_dict, globals_dict)
df = result["df"]
years = result["years"]

# Build all scenarios for comparison cards/table
all_results = {
    name: build_saas_model(base_customers, base_arpa, st.session_state["scenarios"][name], globals_dict)
    for name in ["Bear", "Base", "Bull"]
}

# =================================================
# Dashboard View
# =================================================
if st.session_state["view"] == "Dashboard":
    st.subheader("📊 Dashboard")

    # KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)

    rev0 = float(df["Revenue"].iloc[0])
    revn = float(df["Revenue"].iloc[-1])
    n = len(years)
    rev_cagr = ((revn / rev0) ** (1 / max(n - 1, 1)) - 1) * 100 if rev0 > 0 else np.nan

    with k1: st.metric("Revenue CAGR", fmt_pct2(rev_cagr))
    with k2: st.metric("Avg Gross Margin", fmt_pct2(float(df["Gross Margin (%)"].mean())))
    with k3: st.metric("Avg EBITDA Margin", fmt_pct2(float(df["EBITDA Margin (%)"].mean())))
    with k4: st.metric("Enterprise Value", fmt_currency(result["enterprise_value"]))
    with k5: st.metric("PV(TV) % of EV", fmt_pct2(float(result["terminal_share"] * 100) if np.isfinite(result["terminal_share"]) else np.nan))

    st.markdown("### Scenario Comparison (EV / Equity)")
    comp = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull"],
        "Enterprise Value": [fmt_currency(all_results[s]["enterprise_value"]) for s in ["Bear", "Base", "Bull"]],
        "Equity Value": [fmt_currency(all_results[s]["equity_value"]) for s in ["Bear", "Base", "Bull"]],
    })
    show_table(comp)

    # Revenue + (New customer growth as proxy growth driver) on secondary axis
    st.markdown("### Revenue & New Customer Growth (%)")
    new_growth = np.array(ensure_len(scenario_dict["new_cust_growth_pct"], projection_years, fill=0.0), dtype=float)
    st.pyplot(dual_axis_line_chart(
        title="Revenue vs New Customer Growth",
        x_labels=years,
        y_left=df["Revenue"].values,
        y_right=new_growth,
        left_label="Revenue",
        right_label="New Cust Growth (%)",
        left_color="#1f77b4",
        right_color="#ff7f0e",
        right_is_pct=True,
    ))

    # EBITDA + EBITDA margin
    st.markdown("### EBITDA & EBITDA Margin (%)")
    st.pyplot(dual_axis_line_chart(
        title="EBITDA vs EBITDA Margin",
        x_labels=years,
        y_left=df["EBITDA"].values,
        y_right=df["EBITDA Margin (%)"].values,
        left_label="EBITDA",
        right_label="EBITDA Margin (%)",
        left_color="#2ca02c",
        right_color="#d62728",
        right_is_pct=True,
    ))

    # NOPAT
    st.markdown("### NOPAT")
    fig, ax = plt.subplots()
    x_pos = np.arange(len(years))
    ax.plot(x_pos, df["NOPAT"].values, marker="o", color="#9467bd", label="NOPAT")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("NOPAT")
    ax.grid(False)
    add_point_labels(ax, x_pos, df["NOPAT"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    # FCF
    st.markdown("### Free Cash Flow (FCF)")
    fig, ax = plt.subplots()
    ax.plot(x_pos, df["FCF"].values, marker="o", color="#8c564b", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("FCF")
    ax.grid(False)
    add_point_labels(ax, x_pos, df["FCF"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.stop()

# =================================================
# Financial Model View (Statements)
# =================================================
st.subheader("📑 SaaS Financial Model")

# 1) SaaS Operating Drivers
drivers = pd.DataFrame({"Line Item": [
    "Customers (Beg)",
    "New Customers",
    "Churned Customers",
    "Customers (End)",
    "ARPA",
    "NRR (%)",
    "ARR",
    "Revenue"
]})
for y in years:
    # NRR from scenario (year-wise)
    idx = years.index(y)
    nrr_val = ensure_len(scenario_dict["nrr_pct"], projection_years, fill=100.0)[idx]
    drivers[y] = [
        fmt_float2(df.loc[y, "Customers (Beg)"]),
        fmt_float2(df.loc[y, "New Customers"]),
        fmt_float2(df.loc[y, "Churned Customers"]),
        fmt_float2(df.loc[y, "Customers (End)"]),
        fmt_currency(df.loc[y, "ARPA"]),
        fmt_pct2(nrr_val),
        fmt_currency(df.loc[y, "ARR"]),
        fmt_currency(df.loc[y, "Revenue"]),
    ]
show_table(drivers, "🧭 Operating Drivers")

# 2) Profit & Loss (SaaS-style)
pnl = pd.DataFrame({"Line Item": [
    "Revenue",
    "COGS",
    "Gross Profit",
    "Gross Margin (%)",
    "Sales & Marketing",
    "R&D",
    "G&A",
    "EBITDA",
    "EBITDA Margin (%)",
    "D&A",
    "EBIT",
    "Tax Rate (%)",
    "NOPAT"
]})
for y in years:
    pnl[y] = [
        fmt_currency(df.loc[y, "Revenue"]),
        fmt_currency(df.loc[y, "COGS"]),
        fmt_currency(df.loc[y, "Gross Profit"]),
        fmt_pct2(df.loc[y, "Gross Margin (%)"]),
        fmt_currency(df.loc[y, "Sales & Marketing"]),
        fmt_currency(df.loc[y, "R&D"]),
        fmt_currency(df.loc[y, "G&A"]),
        fmt_currency(df.loc[y, "EBITDA"]),
        fmt_pct2(df.loc[y, "EBITDA Margin (%)"]),
        fmt_currency(df.loc[y, "D&A"]),
        fmt_currency(df.loc[y, "EBIT"]),
        fmt_pct2(globals_dict["tax_rate_pct"]),
        fmt_currency(df.loc[y, "NOPAT"]),
    ]
show_table(pnl, "📑 Profit & Loss Statement (SaaS)")

# 3) Cash Flow + DCF bridge
cf = pd.DataFrame({"Line Item": [
    "NOPAT",
    "Add: D&A",
    "Less: Capex",
    "Less: ΔNWC",
    "Free Cash Flow (FCF)"
]})
for y in years:
    cf[y] = [
        fmt_currency(df.loc[y, "NOPAT"]),
        fmt_currency(df.loc[y, "D&A"]),
        fmt_currency(-df.loc[y, "Capex"]),
        fmt_currency(-df.loc[y, "ΔNWC"]),
        fmt_currency(df.loc[y, "FCF"]),
    ]
show_table(cf, "💵 Cash Flow Statement (FCF Build)")

dcf = pd.DataFrame({"Line Item": ["FCF", "Discount Factor", "PV of FCF"]})
for y in years:
    dcf[y] = [
        fmt_currency(df.loc[y, "FCF"]),
        fmt_float2(df.loc[y, "Discount Factor"]),
        fmt_currency(df.loc[y, "PV of FCF"]),
    ]
show_table(dcf, "💰 Discounted Cash Flow (Detailed)")

# 4) Valuation summary
tv_rows = [
    ("Terminal Method", terminal_method),
    ("Final Year Revenue", fmt_currency(df["Revenue"].iloc[-1])),
    ("Final Year FCF", fmt_currency(df["FCF"].iloc[-1])),
    ("WACC (%)", fmt_pct2(globals_dict["wacc_pct"])),
    ("Terminal Growth (%)", fmt_pct2(globals_dict["terminal_growth_pct"])),
    ("Exit Multiple (EV/Revenue)", fmt_float2(globals_dict["exit_multiple_ev_rev"])),
    ("Terminal Value", fmt_currency(result["terminal_value"])),
    ("PV of Terminal Value", fmt_currency(result["pv_terminal_value"])),
    ("PV of Explicit FCFs", fmt_currency(result["pv_explicit"])),
    ("Enterprise Value", fmt_currency(result["enterprise_value"])),
    ("Less: Net Debt", fmt_currency(globals_dict["net_debt"])),
    ("Add: Cash", fmt_currency(globals_dict["cash"])),
    ("Equity Value", fmt_currency(result["equity_value"])),
]
show_table(pd.DataFrame(tv_rows, columns=["Line Item", "Amount"]), "📘 Valuation Summary")

# =================================================
# Export Excel (multi-sheet)
# =================================================
st.subheader("⬇️ Export Excel (multi-sheet)")

def to_excel_bytes():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Assumptions
        pd.DataFrame([st.session_state["globals"]]).to_excel(writer, sheet_name="Assumptions_Global", index=False)

        sc_rows = []
        for name in ["Bear", "Base", "Bull"]:
            d = st.session_state["scenarios"][name]
            sc_rows.append({
                "Scenario": name,
                "New Cust Growth (%)": ", ".join([f"{v:.2f}" for v in d["new_cust_growth_pct"]]),
                "Gross Churn (%)": ", ".join([f"{v:.2f}" for v in d["gross_churn_pct"]]),
                "NRR (%)": ", ".join([f"{v:.2f}" for v in d["nrr_pct"]]),
                "ARPA Growth (%)": ", ".join([f"{v:.2f}" for v in d["arpa_growth_pct"]]),
                "Gross Margin (%)": ", ".join([f"{v:.2f}" for v in d["gross_margin_pct"]]),
                "CAC per New": ", ".join([f"{v:.2f}" for v in d["cac_per_new"]]),
                "S&M (% Rev)": ", ".join([f"{v:.2f}" for v in d["sm_pct_rev"]]),
                "R&D (% Rev)": ", ".join([f"{v:.2f}" for v in d["rnd_pct_rev"]]),
                "G&A (% Rev)": ", ".join([f"{v:.2f}" for v in d["ga_pct_rev"]]),
            })
        pd.DataFrame(sc_rows).to_excel(writer, sheet_name="Assumptions_Scenarios", index=False)

        # Model
        df.to_excel(writer, sheet_name="Model", index=True)

        # Drivers
        drivers_num = df[[
            "Customers (Beg)", "New Customers", "Churned Customers", "Customers (End)",
            "ARPA", "ARR", "Revenue"
        ]].copy()
        drivers_num.to_excel(writer, sheet_name="Drivers", index=True)

        # P&L
        pnl_num = df[[
            "Revenue", "COGS", "Gross Profit", "Sales & Marketing", "R&D", "G&A", "EBITDA", "D&A", "EBIT", "NOPAT"
        ]].copy()
        pnl_num.to_excel(writer, sheet_name="PnL", index=True)

        # CF
        cf_num = df[["NOPAT", "D&A", "Capex", "ΔNWC", "FCF", "Discount Factor", "PV of FCF"]].copy()
        cf_num.to_excel(writer, sheet_name="CashFlow_DCF", index=True)

        # Valuation
        pd.DataFrame(tv_rows, columns=["Item", "Value"]).to_excel(writer, sheet_name="Valuation", index=False)

    output.seek(0)
    return output.getvalue()

xlsx_bytes = to_excel_bytes()
st.download_button(
    "Download Excel Workbook",
    data=xlsx_bytes,
    file_name=f"b2b_saas_model_{selected.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("🧠 Model Notes / Explanation"):
    st.markdown(
        """
**Core SaaS engine:**
- Customers: Begin → New → Churn → End (year-wise)
- ARPA grows year-wise
- ARR ≈ Avg Customers × ARPA × NRR (NRR captures expansion / net retention)
- Revenue = ARR × Revenue Recognition %
- Gross Profit = Revenue × Gross Margin %
- EBITDA = Gross Profit − (S&M + R&D + G&A)

**FCF build:**
- EBIT = EBITDA − D&A
- NOPAT = EBIT × (1 − Tax)
- FCF = NOPAT + D&A − Capex − ΔNWC

**DCF:**
- PV(FCF) using WACC
- Terminal Value via Gordon Growth or Exit Multiple (EV/Revenue)
- Equity Value = EV − Net Debt + Cash
"""
    )
