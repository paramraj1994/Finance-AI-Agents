import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="Financial Model – Investor Suite", layout="wide")
st.title("📊 Financial Model – Investor Suite")
st.caption("Scenario modeling • year-wise assumptions • detailed DCF • dashboard • export • save/load assumptions")

# =================================================
# CSS – Center align tables (st.data_editor)
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
# Helpers: formatting + display
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

def safe_minmax(series):
    s = pd.Series(series).astype(float)
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
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
# Defaults + session init
# =================================================
DEFAULT_GLOBALS = {
    "projection_years": 5,
    "tax_rate_pct": 25.0,
    "wacc_pct": 12.0,
    "terminal_growth_pct": 4.0,
    "da_pct_rev": 3.0,
    "capex_pct_rev": 4.0,
    "nwc_pct_rev": 10.0,
    "net_debt": 0.0,
    "cash": 0.0,
    "exit_multiple": 10.0,
}
DEFAULT_SCENARIOS = {
    "Bear": {"growth_by_year_pct": [7, 7, 6, 6, 5], "margin_by_year_pct": [20, 20, 21, 21, 22]},
    "Base": {"growth_by_year_pct": [12, 12, 11, 11, 10], "margin_by_year_pct": [25, 25, 26, 26, 27]},
    "Bull": {"growth_by_year_pct": [18, 18, 16, 15, 14], "margin_by_year_pct": [30, 30, 31, 31, 32]},
}

if "globals" not in st.session_state:
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = DEFAULT_SCENARIOS.copy()
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
da_pct_rev = st.sidebar.number_input("D&A (% of Revenue)", value=float(g["da_pct_rev"]), step=0.25)
capex_pct_rev = st.sidebar.number_input("Capex (% of Revenue)", value=float(g["capex_pct_rev"]), step=0.25)
nwc_pct_rev = st.sidebar.number_input("NWC (% of Revenue)", value=float(g["nwc_pct_rev"]), step=0.5)
net_debt = st.sidebar.number_input("Net Debt", value=float(g["net_debt"]), step=100.0)
cash = st.sidebar.number_input("Cash", value=float(g["cash"]), step=100.0)
exit_multiple = st.sidebar.number_input("Exit Multiple (EV/EBITDA)", value=float(g["exit_multiple"]), step=0.5)

base_revenue = st.sidebar.number_input("Base Revenue (Last Historical)", value=1120.0, step=100.0)

st.session_state["globals"] = {
    "projection_years": projection_years,
    "tax_rate_pct": tax_rate_pct,
    "wacc_pct": wacc_pct,
    "terminal_growth_pct": terminal_growth_pct,
    "da_pct_rev": da_pct_rev,
    "capex_pct_rev": capex_pct_rev,
    "nwc_pct_rev": nwc_pct_rev,
    "net_debt": net_debt,
    "cash": cash,
    "exit_multiple": exit_multiple,
}

# =================================================
# Ensure scenarios arrays match projection_years
# =================================================
def ensure_len(arr, n, fill=None):
    arr = list(arr) if isinstance(arr, (list, tuple)) else []
    if len(arr) >= n:
        return arr[:n]
    if fill is None:
        fill = arr[-1] if arr else 0.0
    return arr + [fill] * (n - len(arr))

for name in ["Bear", "Base", "Bull"]:
    st.session_state["scenarios"][name]["growth_by_year_pct"] = ensure_len(
        st.session_state["scenarios"][name].get("growth_by_year_pct", []),
        projection_years,
        fill=None
    )
    st.session_state["scenarios"][name]["margin_by_year_pct"] = ensure_len(
        st.session_state["scenarios"][name].get("margin_by_year_pct", []),
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
}
assump_json_str = json.dumps(assump_payload, indent=2)

st.sidebar.download_button(
    "Download Assumptions (JSON)",
    data=assump_json_str,
    file_name="assumptions.json",
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
            st.sidebar.success("Assumptions loaded.")
            st.rerun()
        else:
            st.sidebar.error("Invalid assumptions JSON format.")
    except Exception:
        st.sidebar.error("Could not read JSON.")

if st.sidebar.button("🔁 Reset to Defaults"):
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
    st.session_state["scenarios"] = DEFAULT_SCENARIOS.copy()
    st.session_state["terminal_method"] = "Gordon Growth"
    st.sidebar.success("Reset done.")
    st.rerun()

# =================================================
# Sidebar: Edit one scenario year-by-year (Growth + Margin)
# =================================================
st.sidebar.header("📌 Scenario Inputs (Year-wise)")

edit_scn = st.sidebar.selectbox("Select Scenario to Edit", ["Bear", "Base", "Bull"], key="edit_scenario_select_yearwise")

# Build editable table (Year, Growth%, Margin%)
years_labels = [f"Year {i+1}" for i in range(projection_years)]
edit_df = pd.DataFrame({
    "Year": years_labels,
    "Revenue Growth (%)": st.session_state["scenarios"][edit_scn]["growth_by_year_pct"],
    "EBITDA Margin (%)": st.session_state["scenarios"][edit_scn]["margin_by_year_pct"],
})

st.sidebar.caption("Edit the table below. Values are in %.")
edited = st.sidebar.data_editor(
    edit_df,
    hide_index=True,
    disabled=["Year"],
    use_container_width=True,
    key=f"editor_{edit_scn}_{projection_years}"
)

if st.sidebar.button("✅ Save Year-wise Assumptions"):
    try:
        g_list = pd.to_numeric(edited["Revenue Growth (%)"], errors="coerce").fillna(0.0).tolist()
        m_list = pd.to_numeric(edited["EBITDA Margin (%)"], errors="coerce").fillna(0.0).tolist()
        st.session_state["scenarios"][edit_scn]["growth_by_year_pct"] = ensure_len(g_list, projection_years, fill=0.0)
        st.session_state["scenarios"][edit_scn]["margin_by_year_pct"] = ensure_len(m_list, projection_years, fill=0.0)
        st.sidebar.success(f"{edit_scn} year-wise assumptions saved.")
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
# Scenario selection to run + Terminal method
# =================================================
selected = st.radio("🎯 Select Scenario to Run", ["Bear", "Base", "Bull"], horizontal=True)

terminal_method = st.radio(
    "Terminal Value Method",
    ["Gordon Growth", "Exit Multiple"],
    horizontal=True,
    index=0 if st.session_state["terminal_method"] == "Gordon Growth" else 1
)
st.session_state["terminal_method"] = terminal_method

# Validate WACC > g when Gordon Growth
if terminal_method == "Gordon Growth" and (wacc_pct / 100.0) <= (terminal_growth_pct / 100.0):
    st.error("❗ WACC must be greater than Terminal Growth for Gordon Growth terminal value. Please adjust assumptions.")

# =================================================
# Build model (year-wise series)
# =================================================
def build_model(base_rev, growth_by_year_pct, margin_by_year_pct, globals_dict):
    n = int(globals_dict["projection_years"])
    years = [f"Year {i+1}" for i in range(n)]

    growth_path = np.array(ensure_len(growth_by_year_pct, n, fill=0.0), dtype=float) / 100.0
    margin_path = np.array(ensure_len(margin_by_year_pct, n, fill=0.0), dtype=float) / 100.0

    tax = globals_dict["tax_rate_pct"] / 100.0
    da_pct = globals_dict["da_pct_rev"] / 100.0
    capex_pct = globals_dict["capex_pct_rev"] / 100.0
    nwc_pct = globals_dict["nwc_pct_rev"] / 100.0
    wacc = globals_dict["wacc_pct"] / 100.0
    tg = globals_dict["terminal_growth_pct"] / 100.0

    revenue = float(base_rev)
    prev_nwc = revenue * nwc_pct

    rows = []
    for i in range(n):
        revenue *= (1 + growth_path[i])
        ebitda = revenue * margin_path[i]
        da = revenue * da_pct
        ebit = ebitda - da
        nopat = ebit * (1 - tax)
        capex = revenue * capex_pct
        nwc = revenue * nwc_pct
        delta_nwc = nwc - prev_nwc
        prev_nwc = nwc
        fcf = nopat + da - capex - delta_nwc

        rows.append([
            revenue, growth_path[i]*100, margin_path[i]*100,
            ebitda, da, ebit, nopat, capex, delta_nwc, fcf
        ])

    df = pd.DataFrame(
        rows,
        columns=[
            "Revenue", "Revenue Growth (%)", "EBITDA Margin (%)", "EBITDA",
            "D&A", "EBIT", "NOPAT", "Capex", "ΔNWC", "FCF"
        ],
        index=years
    )

    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    # Terminal value
    if terminal_method == "Gordon Growth":
        tv = np.nan if wacc <= tg else (df["FCF"].iloc[-1] * (1 + tg) / (wacc - tg))
    else:
        tv = df["EBITDA"].iloc[-1] * float(globals_dict["exit_multiple"])

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
run_growth = st.session_state["scenarios"][selected]["growth_by_year_pct"]
run_margin = st.session_state["scenarios"][selected]["margin_by_year_pct"]

result = build_model(base_revenue, run_growth, run_margin, globals_dict)
df = result["df"]
years = result["years"]

all_results = {
    name: build_model(
        base_revenue,
        st.session_state["scenarios"][name]["growth_by_year_pct"],
        st.session_state["scenarios"][name]["margin_by_year_pct"],
        globals_dict,
    )
    for name in ["Bear", "Base", "Bull"]
}

# =================================================
# Scenario assumptions summary table
# =================================================
st.subheader("📌 Scenario Assumptions (Year-wise Summary)")
assump_summary = []
for name in ["Bear", "Base", "Bull"]:
    g_list = st.session_state["scenarios"][name]["growth_by_year_pct"]
    m_list = st.session_state["scenarios"][name]["margin_by_year_pct"]
    assump_summary.append({
        "Scenario": name,
        "Growth (Y1 → Yn)": f"{g_list[0]:.2f}% → {g_list[-1]:.2f}%",
        "Margin (Y1 → Yn)": f"{m_list[0]:.2f}% → {m_list[-1]:.2f}%",
    })
show_table(pd.DataFrame(assump_summary))

# =================================================
# DASHBOARD VIEW
# =================================================
if st.session_state["view"] == "Dashboard":
    st.subheader("📊 Dashboard")

    # KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    n = len(years)
    rev0 = float(base_revenue)
    revn = float(df["Revenue"].iloc[-1])
    rev_cagr = ((revn / rev0) ** (1 / n) - 1) * 100 if rev0 > 0 else np.nan

    with k1: st.metric("Revenue CAGR", fmt_pct2(rev_cagr))
    with k2: st.metric("Avg EBITDA Margin", fmt_pct2(float(df["EBITDA Margin (%)"].mean())))
    with k3: st.metric("Enterprise Value", fmt_currency(result["enterprise_value"]))
    with k4: st.metric("Equity Value", fmt_currency(result["equity_value"]))
    with k5: st.metric("PV(TV) as % of EV", fmt_pct2(float(result["terminal_share"] * 100) if np.isfinite(result["terminal_share"]) else np.nan))

    # Revenue vs Growth
    st.markdown("### Revenue & Revenue Growth (%) — Selected Scenario")
    st.pyplot(dual_axis_line_chart(
        title="Revenue vs Revenue Growth",
        x_labels=years,
        y_left=df["Revenue"].values,
        y_right=df["Revenue Growth (%)"].values,
        left_label="Revenue",
        right_label="Revenue Growth (%)",
        left_color="#1f77b4",
        right_color="#ff7f0e",
        right_is_pct=True,
    ))

    # EBITDA vs Margin
    st.markdown("### EBITDA & EBITDA Margin (%) — Selected Scenario")
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
    ax.plot(x_pos, df["NOPAT"].values, marker="o", label="NOPAT")
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
    ax.plot(x_pos, df["FCF"].values, marker="o", label="FCF")
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
# FINANCIAL MODEL VIEW
# =================================================
st.subheader("📑 Financial Model")

# P&L
pnl = pd.DataFrame({"Line Item": [
    "Revenue",
    "Revenue Growth (%)",
    "EBITDA Margin (%)",
    "EBITDA",
    "D&A",
    "EBIT",
    "Tax Rate (%)",
    "NOPAT"
]})
for y in years:
    pnl[y] = [
        fmt_currency(df.loc[y, "Revenue"]),
        fmt_pct2(df.loc[y, "Revenue Growth (%)"]),
        fmt_pct2(df.loc[y, "EBITDA Margin (%)"]),
        fmt_currency(df.loc[y, "EBITDA"]),
        fmt_currency(df.loc[y, "D&A"]),
        fmt_currency(df.loc[y, "EBIT"]),
        fmt_pct2(globals_dict["tax_rate_pct"]),
        fmt_currency(df.loc[y, "NOPAT"]),
    ]
show_table(pnl, "📑 Profit & Loss (Expanded)")

# Cash Flow
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
show_table(cf, "💵 Cash Flow (Detailed)")

# DCF
dcf = pd.DataFrame({"Line Item": ["FCF", "Discount Factor", "PV of FCF"]})
for y in years:
    dcf[y] = [
        fmt_currency(df.loc[y, "FCF"]),
        fmt_float2(df.loc[y, "Discount Factor"]),
        fmt_currency(df.loc[y, "PV of FCF"]),
    ]
show_table(dcf, "💰 Discounted Cash Flow (Detailed)")

# Valuation summary
tv_rows = [
    ("Terminal Method", terminal_method),
    ("Final Year FCF", fmt_currency(df["FCF"].iloc[-1])),
    ("Final Year EBITDA", fmt_currency(df["EBITDA"].iloc[-1])),
    ("WACC (%)", fmt_pct2(globals_dict["wacc_pct"])),
    ("Terminal Growth (%)", fmt_pct2(globals_dict["terminal_growth_pct"])),
    ("Exit Multiple (EV/EBITDA)", fmt_float2(globals_dict["exit_multiple"])),
    ("Terminal Value", fmt_currency(result["terminal_value"])),
    ("PV of Terminal Value", fmt_currency(result["pv_terminal_value"])),
    ("PV of Explicit FCFs", fmt_currency(result["pv_explicit"])),
    ("Enterprise Value", fmt_currency(result["enterprise_value"])),
    ("Less: Net Debt", fmt_currency(globals_dict["net_debt"])),
    ("Add: Cash", fmt_currency(globals_dict["cash"])),
    ("Equity Value", fmt_currency(result["equity_value"])),
]
show_table(pd.DataFrame(tv_rows, columns=["Line Item", "Amount"]), "📘 Valuation Summary (Detailed)")

# =================================================
# Export Excel (multi-sheet)
# =================================================
st.subheader("⬇️ Export Excel (multi-sheet)")

def to_excel_bytes():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([st.session_state["globals"]]).to_excel(writer, sheet_name="Assumptions_Global", index=False)

        sc_rows = []
        for name in ["Bear", "Base", "Bull"]:
            d = {
                "Scenario": name,
                "Growth By Year (%)": ", ".join([f"{v:.2f}" for v in st.session_state["scenarios"][name]["growth_by_year_pct"]]),
                "Margin By Year (%)": ", ".join([f"{v:.2f}" for v in st.session_state["scenarios"][name]["margin_by_year_pct"]]),
            }
            sc_rows.append(d)
        pd.DataFrame(sc_rows).to_excel(writer, sheet_name="Assumptions_Scenarios", index=False)

        df.to_excel(writer, sheet_name="Model_Statement", index=True)

        pnl_num = df[["Revenue", "Revenue Growth (%)", "EBITDA Margin (%)", "EBITDA", "D&A", "EBIT", "NOPAT"]].copy()
        pnl_num.to_excel(writer, sheet_name="PnL", index=True)

        cf_num = df[["NOPAT", "D&A", "Capex", "ΔNWC", "FCF"]].copy()
        cf_num.to_excel(writer, sheet_name="CashFlow", index=True)

        val_df = pd.DataFrame({"Item": [r[0] for r in tv_rows], "Value": [r[1] for r in tv_rows]})
        val_df.to_excel(writer, sheet_name="Valuation", index=False)

    output.seek(0)
    return output.getvalue()

xlsx_bytes = to_excel_bytes()
st.download_button(
    "Download Excel Workbook",
    data=xlsx_bytes,
    file_name=f"financial_model_{selected.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

with st.expander("🧠 Quick model explanation"):
    st.markdown(
        """
**Revenue projection:** Uses *year-wise* revenue growth rates from the selected scenario.  
**EBITDA:** Revenue × *year-wise* EBITDA margin.  
**EBIT:** EBITDA − D&A (D&A = % of revenue).  
**NOPAT:** EBIT × (1 − tax rate).  
**FCF:** NOPAT + D&A − Capex − ΔNWC  
**DCF:** PV = FCF × Discount Factor (WACC).  
**Equity Value:** EV − Net Debt + Cash  
"""
    )
Share this email in Google Chat
Discuss in real time with an individual, group, space, or even someone outside 
