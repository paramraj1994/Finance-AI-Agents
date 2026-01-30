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
st.caption("Scenario modeling • detailed DCF • sensitivity • dashboard • export • save/load assumptions")

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
# Helpers: formatting + table display
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
    # Expand y-limits so labels don’t clip
    ymin, ymax = ax.get_ylim()
    rng = (ymax - ymin) if ymax != ymin else 1.0
    pad = rng * 0.18
    ax.set_ylim(ymin, ymax + pad)
    ymin, ymax = ax.get_ylim()

    for i, v in enumerate(y_vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        label = f"{v:.1f}%" if is_pct else f"{v:,.0f}"
        # If near top, place below
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

    # remove gridlines
    ax1.grid(False)
    ax2.grid(False)

    add_point_labels(ax1, x_pos, np.array(y_left, dtype=float), is_pct=False)
    add_point_labels(ax2, x_pos, np.array(y_right, dtype=float), is_pct=right_is_pct)

    # Percent axis should not look “stuck” if flat
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
# Session state defaults
# =================================================
DEFAULT_SCENARIOS = {
    "Bear": {"growth_start": 7.0, "growth_end": 5.0, "margin_start": 20.0, "margin_end": 22.0},
    "Base": {"growth_start": 12.0, "growth_end": 10.0, "margin_start": 25.0, "margin_end": 27.0},
    "Bull": {"growth_start": 18.0, "growth_end": 14.0, "margin_start": 30.0, "margin_end": 32.0},
}
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
    "currency_label": "Amount",
}

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = DEFAULT_SCENARIOS.copy()
if "globals" not in st.session_state:
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
if "view" not in st.session_state:
    st.session_state["view"] = "Financial Model"
if "terminal_method" not in st.session_state:
    st.session_state["terminal_method"] = "Gordon Growth"

# =================================================
# Upload historical Excel (auto-detect base revenue)
# =================================================
st.sidebar.header("📂 Historical File (Optional)")
uploaded = st.sidebar.file_uploader("Upload Excel", type=["xlsx"])

hist_df = None
auto_base_revenue = None
auto_hist_years = None

def detect_column(cols, keywords):
    cols_lower = {c: str(c).strip().lower() for c in cols}
    for c, cl in cols_lower.items():
        if any(k in cl for k in keywords):
            return c
    return None

if uploaded is not None:
    try:
        hist_df = pd.read_excel(uploaded)
        if not hist_df.empty:
            # Try to detect year + revenue columns
            year_col = detect_column(hist_df.columns, ["year", "fy", "fiscal"])
            rev_col = detect_column(hist_df.columns, ["revenue", "sales", "turnover"])
            if rev_col is not None:
                # Take last non-null
                s = pd.to_numeric(hist_df[rev_col], errors="coerce").dropna()
                if not s.empty:
                    auto_base_revenue = float(s.iloc[-1])
            if year_col is not None:
                y = pd.to_numeric(hist_df[year_col], errors="coerce").dropna()
                if not y.empty:
                    auto_hist_years = y.astype(int).tolist()
    except Exception:
        hist_df = None

if hist_df is not None:
    st.subheader("📄 Historical Data Preview")
    st.data_editor(hist_df.head(50), hide_index=True, disabled=True, use_container_width=True)

# =================================================
# Sidebar: Global assumptions (with tooltips)
# =================================================
st.sidebar.header("🔧 Global Assumptions")

g = st.session_state["globals"]
projection_years = st.sidebar.slider("Projection Years", 3, 10, int(g["projection_years"]))
tax_rate_pct = st.sidebar.number_input("Tax Rate (%)", value=float(g["tax_rate_pct"]), step=0.5, help="Tax applied to EBIT (NOPAT).")
wacc_pct = st.sidebar.number_input("WACC (%)", value=float(g["wacc_pct"]), step=0.5, help="Discount rate used for PV of FCFs.")
terminal_growth_pct = st.sidebar.number_input("Terminal Growth (%)", value=float(g["terminal_growth_pct"]), step=0.25, help="Used in Gordon Growth terminal value.")
da_pct_rev = st.sidebar.number_input("Depreciation & Amortization (% of Revenue)", value=float(g["da_pct_rev"]), step=0.25, help="D&A proxy used to move from EBITDA to EBIT and cash flow.")
capex_pct_rev = st.sidebar.number_input("Capex (% of Revenue)", value=float(g["capex_pct_rev"]), step=0.25, help="Capex proxy used in FCF build.")
nwc_pct_rev = st.sidebar.number_input("Net Working Capital (% of Revenue)", value=float(g["nwc_pct_rev"]), step=0.5, help="NWC proxy; ΔNWC impacts FCF.")
net_debt = st.sidebar.number_input("Net Debt", value=float(g["net_debt"]), step=100.0, help="Enterprise Value → Equity Value adjustment.")
cash = st.sidebar.number_input("Cash", value=float(g["cash"]), step=100.0, help="Enterprise Value → Equity Value adjustment.")
exit_multiple = st.sidebar.number_input("Exit Multiple (EV/EBITDA)", value=float(g["exit_multiple"]), step=0.5, help="Used if Terminal Value method is Exit Multiple.")

# Base revenue: use auto if available; allow override
st.sidebar.header("📌 Starting Point")
default_base = auto_base_revenue if auto_base_revenue is not None else 1120.0
base_revenue = st.sidebar.number_input(
    "Base Revenue (Last Historical)",
    value=float(default_base),
    step=100.0,
    help="Starting revenue used for Year 1 projection."
)

# Validate DCF feasibility
if (wacc_pct / 100.0) <= (terminal_growth_pct / 100.0) and st.session_state["terminal_method"] == "Gordon Growth":
    st.sidebar.error("WACC must be greater than Terminal Growth for Gordon Growth terminal value.")

# Persist globals
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
    "currency_label": "Amount",
}

# =================================================
# Sidebar: Save/Load assumptions + Reset
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
# Sidebar: Edit one scenario at a time (start/end = year-wise curve)
# =================================================
st.sidebar.header("📌 Scenario Inputs (Edit One)")

edit_scn = st.sidebar.selectbox("Select Scenario to Edit", ["Bear", "Base", "Bull"], key="edit_scenario_select")
scn = st.session_state["scenarios"][edit_scn]

growth_start = st.sidebar.number_input(f"{edit_scn}: Growth Start (%)", value=float(scn["growth_start"]), step=0.5, help="Growth in Year 1.")
growth_end = st.sidebar.number_input(f"{edit_scn}: Growth End (%)", value=float(scn["growth_end"]), step=0.5, help="Growth in final projection year.")
margin_start = st.sidebar.number_input(f"{edit_scn}: Margin Start (%)", value=float(scn["margin_start"]), step=0.5, help="EBITDA margin in Year 1.")
margin_end = st.sidebar.number_input(f"{edit_scn}: Margin End (%)", value=float(scn["margin_end"]), step=0.5, help="EBITDA margin in final projection year.")

if st.sidebar.button("✅ Save Scenario"):
    st.session_state["scenarios"][edit_scn] = {
        "growth_start": float(growth_start),
        "growth_end": float(growth_end),
        "margin_start": float(margin_start),
        "margin_end": float(margin_end),
    }
    st.sidebar.success(f"{edit_scn} updated.")

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
# Scenario selection for running the model
# =================================================
selected = st.radio("🎯 Select Scenario to Run", ["Bear", "Base", "Bull"], horizontal=True)

# Terminal Value method toggle
terminal_method = st.radio(
    "Terminal Value Method",
    ["Gordon Growth", "Exit Multiple"],
    horizontal=True,
    index=0 if st.session_state["terminal_method"] == "Gordon Growth" else 1
)
st.session_state["terminal_method"] = terminal_method

# =================================================
# Model builder
# =================================================
def build_yearwise_series(start_pct, end_pct, n_years):
    if n_years <= 1:
        return np.array([start_pct], dtype=float)
    return np.linspace(float(start_pct), float(end_pct), n_years)

def build_model_for_scenario(base_rev, scn_dict, globals_dict, label_prefix="Year "):
    n = int(globals_dict["projection_years"])
    years = [f"{label_prefix}{i+1}" for i in range(n)]

    growth_path = build_yearwise_series(scn_dict["growth_start"], scn_dict["growth_end"], n) / 100.0
    margin_path = build_yearwise_series(scn_dict["margin_start"], scn_dict["margin_end"], n) / 100.0

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
        revenue = revenue * (1 + growth_path[i])
        ebitda = revenue * margin_path[i]
        da = revenue * da_pct
        ebit = ebitda - da
        nopat = ebit * (1 - tax)
        capex = revenue * capex_pct
        nwc = revenue * nwc_pct
        delta_nwc = nwc - prev_nwc
        prev_nwc = nwc
        fcf = nopat + da - capex - delta_nwc

        rows.append([revenue, growth_path[i]*100, margin_path[i]*100, ebitda, da, ebit, nopat, capex, delta_nwc, fcf])

    df = pd.DataFrame(
        rows,
        columns=[
            "Revenue", "Revenue Growth (%)", "EBITDA Margin (%)", "EBITDA",
            "D&A", "EBIT", "NOPAT", "Capex", "ΔNWC", "FCF"
        ],
        index=years
    )

    # Discounting
    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    # Terminal value
    if terminal_method == "Gordon Growth":
        if wacc <= tg:
            tv = np.nan
        else:
            tv = df["FCF"].iloc[-1] * (1 + tg) / (wacc - tg)
    else:
        tv = df["EBITDA"].iloc[-1] * float(globals_dict["exit_multiple"])

    pv_tv = tv * discount_factors[-1] if np.isfinite(tv) else np.nan
    ev = float(np.nansum(df["PV of FCF"].values)) + float(pv_tv if np.isfinite(pv_tv) else 0.0)

    # Equity value
    net_debt = float(globals_dict["net_debt"])
    cash = float(globals_dict["cash"])
    equity_value = ev - net_debt + cash

    # Diagnostics
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
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
        "wacc": wacc,
        "tg": tg,
    }

globals_dict = st.session_state["globals"]
scenario_dict = st.session_state["scenarios"][selected]
result = build_model_for_scenario(base_revenue, scenario_dict, globals_dict)
df = result["df"]
years = result["years"]

# Also build all scenarios for comparison (dashboard)
all_results = {
    name: build_model_for_scenario(base_revenue, st.session_state["scenarios"][name], globals_dict)
    for name in ["Bear", "Base", "Bull"]
}

# =================================================
# Scenario assumptions table (current, start/end)
# =================================================
st.subheader("📌 Scenario Assumptions (Start → End)")
assump_tbl = pd.DataFrame([
    {
        "Scenario": name,
        "Growth Start": fmt_pct2(st.session_state["scenarios"][name]["growth_start"]),
        "Growth End": fmt_pct2(st.session_state["scenarios"][name]["growth_end"]),
        "Margin Start": fmt_pct2(st.session_state["scenarios"][name]["margin_start"]),
        "Margin End": fmt_pct2(st.session_state["scenarios"][name]["margin_end"]),
    }
    for name in ["Bear", "Base", "Bull"]
])
show_table(assump_tbl)

# =================================================
# DASHBOARD VIEW
# =================================================
if st.session_state["view"] == "Dashboard":
    st.subheader("📊 Dashboard")

    # KPI cards
    k1, k2, k3, k4, k5 = st.columns(5)
    # Revenue CAGR
    n = len(years)
    rev0 = float(base_revenue)
    revn = float(df["Revenue"].iloc[-1])
    rev_cagr = ((revn / rev0) ** (1 / n) - 1) * 100 if rev0 > 0 else np.nan

    with k1: st.metric("Revenue CAGR", fmt_pct2(rev_cagr))
    with k2: st.metric("Avg EBITDA Margin", fmt_pct2(float(df["EBITDA Margin (%)"].mean())))
    with k3: st.metric("Enterprise Value", fmt_currency(result["enterprise_value"]))
    with k4: st.metric("Equity Value", fmt_currency(result["equity_value"]))
    with k5: st.metric("PV(TV) as % of EV", fmt_pct2(float(result["terminal_share"] * 100) if np.isfinite(result["terminal_share"]) else np.nan))

    st.markdown("### Scenario Comparison (EV / Equity)")
    comp = pd.DataFrame({
        "Scenario": ["Bear", "Base", "Bull"],
        "Enterprise Value": [fmt_currency(all_results[s]["enterprise_value"]) for s in ["Bear", "Base", "Bull"]],
        "Equity Value": [fmt_currency(all_results[s]["equity_value"]) for s in ["Bear", "Base", "Bull"]],
    })
    show_table(comp)

    # Revenue vs Growth (selected)
    st.markdown("### Revenue & Revenue Growth (%) — Selected Scenario")
    fig = dual_axis_line_chart(
        title="Revenue vs Revenue Growth",
        x_labels=years,
        y_left=df["Revenue"].values,
        y_right=df["Revenue Growth (%)"].values,
        left_label="Revenue",
        right_label="Revenue Growth (%)",
        left_color="#1f77b4",
        right_color="#ff7f0e",
        right_is_pct=True,
    )
    st.pyplot(fig)

    # EBITDA vs Margin (selected)
    st.markdown("### EBITDA & EBITDA Margin (%) — Selected Scenario")
    fig = dual_axis_line_chart(
        title="EBITDA vs EBITDA Margin",
        x_labels=years,
        y_left=df["EBITDA"].values,
        y_right=df["EBITDA Margin (%)"].values,
        left_label="EBITDA",
        right_label="EBITDA Margin (%)",
        left_color="#2ca02c",
        right_color="#d62728",
        right_is_pct=True,
    )
    st.pyplot(fig)

    # PAT proxy (NOPAT) chart
    st.markdown("### NOPAT (proxy for Profit After Tax)")
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

    # FCF chart
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

    # Waterfall: PV explicit + PV TV = EV
    st.markdown("### EV Waterfall")
    pv_exp = result["pv_explicit"]
    pv_tv = result["pv_terminal_value"]
    ev = result["enterprise_value"]

    fig, ax = plt.subplots()
    labels = ["PV of Explicit FCFs", "PV of Terminal Value", "Enterprise Value"]
    vals = [pv_exp, pv_tv, ev]
    xw = np.arange(len(labels))
    ax.bar(xw, vals)
    ax.set_xticks(xw)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(False)
    add_point_labels(ax, xw, np.array(vals), is_pct=False)
    fig.tight_layout()
    st.pyplot(fig)

    # Cumulative PV of FCF
    st.markdown("### Cumulative PV of FCF (plus PV of Terminal Value)")
    cum_pv = np.cumsum(df["PV of FCF"].values)
    fig, ax = plt.subplots()
    ax.plot(x_pos, cum_pv, marker="o", label="Cumulative PV of FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.grid(False)
    add_point_labels(ax, x_pos, cum_pv, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.stop()

# =================================================
# FINANCIAL MODEL VIEW
# =================================================
st.subheader("📑 Financial Model")

# P&L (expanded)
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
tax_rate = globals_dict["tax_rate_pct"]
for y in years:
    pnl[y] = [
        fmt_currency(df.loc[y, "Revenue"]),
        fmt_pct2(df.loc[y, "Revenue Growth (%)"]),
        fmt_pct2(df.loc[y, "EBITDA Margin (%)"]),
        fmt_currency(df.loc[y, "EBITDA"]),
        fmt_currency(df.loc[y, "D&A"]),
        fmt_currency(df.loc[y, "EBIT"]),
        fmt_pct2(tax_rate),
        fmt_currency(df.loc[y, "NOPAT"]),
    ]
show_table(pnl, "📑 Profit & Loss (Expanded)")

# Cash flow (detailed)
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

# DCF (detailed)
dcf = pd.DataFrame({"Line Item": [
    "FCF",
    "Discount Factor",
    "PV of FCF"
]})
for y in years:
    dcf[y] = [
        fmt_currency(df.loc[y, "FCF"]),
        fmt_float2(df.loc[y, "Discount Factor"]),
        fmt_currency(df.loc[y, "PV of FCF"]),
    ]
show_table(dcf, "💰 Discounted Cash Flow (Detailed)")

# Terminal value detail + Equity bridge
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
tv = pd.DataFrame(tv_rows, columns=["Line Item", "Amount"])
show_table(tv, "📘 Valuation Summary (Detailed)")

# =================================================
# Sensitivities
# =================================================
st.subheader("📉 Sensitivity Analysis")

wacc_base = globals_dict["wacc_pct"]
tg_base = globals_dict["terminal_growth_pct"]

# WACC x Terminal Growth (only meaningful for Gordon Growth)
st.markdown("### EV Sensitivity: WACC × Terminal Growth (Gordon Growth)")
wacc_grid = np.array([wacc_base - 2, wacc_base - 1, wacc_base, wacc_base + 1, wacc_base + 2], dtype=float)
tg_grid = np.array([tg_base - 1, tg_base - 0.5, tg_base, tg_base + 0.5, tg_base + 1], dtype=float)

def ev_with_wacc_tg(wacc_pct_local, tg_pct_local):
    tmp_globals = globals_dict.copy()
    tmp_globals["wacc_pct"] = float(wacc_pct_local)
    tmp_globals["terminal_growth_pct"] = float(tg_pct_local)
    # force Gordon Growth for this grid
    old_method = st.session_state["terminal_method"]
    st.session_state["terminal_method"] = "Gordon Growth"
    out = build_model_for_scenario(base_revenue, scenario_dict, tmp_globals)
    st.session_state["terminal_method"] = old_method
    return out["enterprise_value"]

sens = pd.DataFrame(index=[f"{tg:.2f}%" for tg in tg_grid], columns=[f"{w:.2f}%" for w in wacc_grid])
for tg in tg_grid:
    for w in wacc_grid:
        # invalid if WACC <= g
        if (w / 100.0) <= (tg / 100.0):
            sens.loc[f"{tg:.2f}%", f"{w:.2f}%"] = "—"
        else:
            sens.loc[f"{tg:.2f}%", f"{w:.2f}%"] = fmt_currency(ev_with_wacc_tg(w, tg))

sens_out = sens.reset_index().rename(columns={"index": "Terminal Growth"})
show_table(sens_out)

# Exit multiple sensitivity
st.markdown("### EV Sensitivity: Exit Multiple (EV/EBITDA)")
mult_grid = np.array([exit_multiple - 2, exit_multiple - 1, exit_multiple, exit_multiple + 1, exit_multiple + 2], dtype=float)
mult_tbl = []
old_method = st.session_state["terminal_method"]
st.session_state["terminal_method"] = "Exit Multiple"
for m in mult_grid:
    tmp_globals = globals_dict.copy()
    tmp_globals["exit_multiple"] = float(m)
    out = build_model_for_scenario(base_revenue, scenario_dict, tmp_globals)
    mult_tbl.append({"Exit Multiple": f"{m:.2f}x", "Enterprise Value": fmt_currency(out["enterprise_value"])})
st.session_state["terminal_method"] = old_method
show_table(pd.DataFrame(mult_tbl), title=None)

# =================================================
# Export Excel (multi-sheet)
# =================================================
st.subheader("⬇️ Export Excel (multi-sheet)")

def to_excel_bytes():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        # Assumptions
        globals_sheet = pd.DataFrame([st.session_state["globals"]])
        globals_sheet.to_excel(writer, sheet_name="Assumptions_Global", index=False)

        sc_rows = []
        for name in ["Bear", "Base", "Bull"]:
            d = st.session_state["scenarios"][name].copy()
            d["Scenario"] = name
            sc_rows.append(d)
        pd.DataFrame(sc_rows).to_excel(writer, sheet_name="Assumptions_Scenarios", index=False)

        # Statements (numeric)
        df.to_excel(writer, sheet_name="Model_Statement", index=True)

        # P&L view (numeric)
        pnl_num = df[["Revenue", "Revenue Growth (%)", "EBITDA Margin (%)", "EBITDA", "D&A", "EBIT", "NOPAT"]].copy()
        pnl_num.to_excel(writer, sheet_name="PnL", index=True)

        # Cash flow view (numeric)
        cf_num = df[["NOPAT", "D&A", "Capex", "ΔNWC", "FCF"]].copy()
        cf_num.to_excel(writer, sheet_name="CashFlow", index=True)

        # Valuation
        val_df = pd.DataFrame({
            "Item": [r[0] for r in tv_rows],
            "Value": [r[1] for r in tv_rows],
        })
        val_df.to_excel(writer, sheet_name="Valuation", index=False)

        # Sensitivities
        sens.to_excel(writer, sheet_name="Sensitivity_WACC_g", index=True)
        pd.DataFrame(mult_tbl).to_excel(writer, sheet_name="Sensitivity_ExitMultiple", index=False)

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
        f"""
**Revenue projection:** Year-wise growth path (start → end).  
**EBITDA:** Revenue × year-wise EBITDA margin (start → end).  
**EBIT:** EBITDA − D&A (D&A = % of revenue).  
**NOPAT:** EBIT × (1 − tax rate).  
**FCF:** NOPAT + D&A − Capex − ΔNWC  
**DCF:** PV = FCF × Discount Factor (WACC).  
**Terminal Value:** {terminal_method}  
**Equity Value:** EV − Net Debt + Cash  
"""
    )
