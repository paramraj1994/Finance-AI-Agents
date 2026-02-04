import json
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# =================================================
# PAGE SETUP
# =================================================
st.set_page_config(page_title="B2B SaaS Model – Investor Suite", layout="wide")
st.title("🧩 B2B SaaS Financial Model – Investor Suite")
st.caption("Assumptions • Financial Model • Dashboard • Marketing Spend • Subscriber Acquisition • DCF • Export")

# =================================================
# CSS (center align tables)
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

def ensure_len(arr, n, fill=None):
    arr = list(arr) if isinstance(arr, (list, tuple)) else []
    if len(arr) >= n:
        return arr[:n]
    if fill is None:
        fill = arr[-1] if arr else 0.0
    return arr + [fill] * (n - len(arr))

def show_table(df: pd.DataFrame, title: str | None = None):
    if title:
        st.subheader(title)
    st.data_editor(df, hide_index=True, disabled=True, use_container_width=True)

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
    "revenue_recognition_pct_of_arr": 100.0,
    "da_pct_rev": 3.0,
    "capex_pct_rev": 4.0,
    "nwc_pct_rev": 3.0,
    "net_debt": 0.0,
    "cash": 0.0,
    "exit_multiple_ev_rev": 6.0,
    "base_customers": 200.0,
    "base_arpa": 6000.0,
}

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

SCN_KEYS = [
    "new_cust_growth_pct", "gross_churn_pct", "nrr_pct", "arpa_growth_pct",
    "gross_margin_pct", "cac_per_new", "sm_pct_rev", "rnd_pct_rev", "ga_pct_rev"
]

# =================================================
# Safe merge / normalize (prevents KeyError across reruns + old JSON)
# =================================================
def merge_defaults(defaults: dict, incoming: dict | None) -> dict:
    out = defaults.copy()
    if isinstance(incoming, dict):
        out.update(incoming)
    return out

def normalize_scenarios(scenarios_obj: dict | None, years: int) -> dict:
    out = {}
    for name in ["Bear", "Base", "Bull"]:
        base = json.loads(json.dumps(DEFAULT_SCENARIOS[name]))
        incoming = scenarios_obj.get(name, {}) if isinstance(scenarios_obj, dict) else {}
        if isinstance(incoming, dict):
            for k in SCN_KEYS:
                if k in incoming:
                    base[k] = incoming[k]
        for k in SCN_KEYS:
            base[k] = ensure_len(base.get(k, []), years, fill=None)
        out[name] = base
    return out

# =================================================
# Session State Init
# =================================================
if "globals" not in st.session_state:
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
else:
    st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, st.session_state["globals"])

if "scenarios" not in st.session_state:
    st.session_state["scenarios"] = json.loads(json.dumps(DEFAULT_SCENARIOS))

if "terminal_method" not in st.session_state:
    st.session_state["terminal_method"] = "Gordon Growth"
if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "Base"

# =================================================
# Tabs
# =================================================
tab_assump, tab_model, tab_dash, tab_mkt, tab_acq = st.tabs(
    ["🧾 Assumptions", "📑 Financial Model", "📊 Dashboard", "📣 Marketing Spend", "🧲 Subscriber Acquisition"]
)

# =================================================
# Model Builder
# =================================================
def build_saas_model(globals_dict: dict, scenario: dict, terminal_method: str):
    n = int(globals_dict["projection_years"])
    years = [f"Year {i+1}" for i in range(n)]

    customers_beg = float(globals_dict["base_customers"])
    arpa = float(globals_dict["base_arpa"])

    new_cust_growth = np.array(ensure_len(scenario["new_cust_growth_pct"], n, fill=0.0), dtype=float) / 100.0
    gross_churn = np.array(ensure_len(scenario["gross_churn_pct"], n, fill=0.0), dtype=float) / 100.0
    nrr = np.array(ensure_len(scenario["nrr_pct"], n, fill=100.0), dtype=float) / 100.0
    arpa_growth = np.array(ensure_len(scenario["arpa_growth_pct"], n, fill=0.0), dtype=float) / 100.0
    gm = np.array(ensure_len(scenario["gross_margin_pct"], n, fill=75.0), dtype=float) / 100.0
    cac = np.array(ensure_len(scenario["cac_per_new"], n, fill=0.0), dtype=float)
    sm_pct = np.array(ensure_len(scenario["sm_pct_rev"], n, fill=0.0), dtype=float) / 100.0
    rnd_pct = np.array(ensure_len(scenario["rnd_pct_rev"], n, fill=0.0), dtype=float) / 100.0
    ga_pct = np.array(ensure_len(scenario["ga_pct_rev"], n, fill=0.0), dtype=float) / 100.0

    tax = float(globals_dict["tax_rate_pct"]) / 100.0
    wacc = float(globals_dict["wacc_pct"]) / 100.0
    tg = float(globals_dict["terminal_growth_pct"]) / 100.0
    rev_rec = float(globals_dict["revenue_recognition_pct_of_arr"]) / 100.0
    da_pct_rev = float(globals_dict["da_pct_rev"]) / 100.0
    capex_pct_rev = float(globals_dict["capex_pct_rev"]) / 100.0
    nwc_pct_rev = float(globals_dict["nwc_pct_rev"]) / 100.0

    prev_nwc = 0.0
    rows = []

    for i in range(n):
        new_customers = customers_beg * new_cust_growth[i]
        churned_customers = customers_beg * gross_churn[i]
        customers_end = max(customers_beg + new_customers - churned_customers, 0.0)
        customers_avg = (customers_beg + customers_end) / 2.0

        arpa = arpa * (1 + arpa_growth[i])

        # ARR approximation (avg customers * ARPA * NRR)
        arr = customers_avg * arpa * nrr[i]
        revenue = arr * rev_rec

        gross_profit = revenue * gm[i]
        cogs = revenue - gross_profit

        # Total Sales & Marketing (from % revenue)
        sm_total = revenue * sm_pct[i]

        # CAC spend (computed)
        cac_spend = new_customers * cac[i]

        # Non-CAC marketing spend = residual of S&M after CAC (never below 0 in display)
        non_cac_sm = sm_total - cac_spend

        rnd = revenue * rnd_pct[i]
        ga = revenue * ga_pct[i]

        # EBITDA uses total S&M (CAC is part of it), no double count
        ebitda = gross_profit - sm_total - rnd - ga

        da = revenue * da_pct_rev
        ebit = ebitda - da
        nopat = ebit * (1 - tax)

        capex_val = revenue * capex_pct_rev
        nwc = revenue * nwc_pct_rev
        delta_nwc = nwc - prev_nwc
        prev_nwc = nwc

        fcf = nopat + da - capex_val - delta_nwc

        rows.append([
            customers_beg, new_customers, churned_customers, customers_end,
            arpa, arr, revenue,
            cogs, gross_profit,
            sm_total, cac_spend, non_cac_sm, rnd, ga, ebitda,
            da, ebit, nopat,
            capex_val, delta_nwc, fcf
        ])

        customers_beg = customers_end

    df = pd.DataFrame(
        rows,
        columns=[
            "Customers (Beg)", "New Customers", "Churned Customers", "Customers (End)",
            "ARPA", "ARR", "Revenue",
            "COGS", "Gross Profit",
            "Sales & Marketing (Total)", "CAC Spend", "Non-CAC S&M",
            "R&D", "G&A", "EBITDA",
            "D&A", "EBIT", "NOPAT",
            "Capex", "ΔNWC", "FCF",
        ],
        index=years
    )

    df["Gross Margin (%)"] = np.where(df["Revenue"] > 0, (df["Gross Profit"] / df["Revenue"]) * 100, np.nan)
    df["EBITDA Margin (%)"] = np.where(df["Revenue"] > 0, (df["EBITDA"] / df["Revenue"]) * 100, np.nan)
    df["S&M % of Revenue"] = np.where(df["Revenue"] > 0, (df["Sales & Marketing (Total)"] / df["Revenue"]) * 100, np.nan)
    df["CAC % of S&M"] = np.where(df["Sales & Marketing (Total)"] > 0, (df["CAC Spend"] / df["Sales & Marketing (Total)"]) * 100, np.nan)

    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

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

# =================================================
# ASSUMPTIONS TAB
# =================================================
with tab_assump:
    st.subheader("🧾 Assumptions (All inputs are here)")

    st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
    g = st.session_state["globals"]

    st.markdown("### Global Assumptions")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        g["projection_years"] = int(st.number_input("Projection Years", min_value=3, max_value=10, value=int(g.get("projection_years", 5)), step=1))
        g["tax_rate_pct"] = float(st.number_input("Tax Rate (%)", value=float(g.get("tax_rate_pct", 25.0)), step=0.5))
        g["wacc_pct"] = float(st.number_input("WACC (%)", value=float(g.get("wacc_pct", 12.0)), step=0.5))
    with c2:
        g["terminal_growth_pct"] = float(st.number_input("Terminal Growth (%)", value=float(g.get("terminal_growth_pct", 4.0)), step=0.25))
        g["revenue_recognition_pct_of_arr"] = float(st.number_input("Revenue Recognition (% of ARR)", value=float(g.get("revenue_recognition_pct_of_arr", 100.0)), step=5.0))
        g["exit_multiple_ev_rev"] = float(st.number_input("Exit Multiple (EV/Revenue)", value=float(g.get("exit_multiple_ev_rev", 6.0)), step=0.5))
    with c3:
        g["da_pct_rev"] = float(st.number_input("D&A (% of Revenue)", value=float(g.get("da_pct_rev", 3.0)), step=0.25))
        g["capex_pct_rev"] = float(st.number_input("Capex (% of Revenue)", value=float(g.get("capex_pct_rev", 4.0)), step=0.25))
        g["nwc_pct_rev"] = float(st.number_input("NWC (% of Revenue)", value=float(g.get("nwc_pct_rev", 3.0)), step=0.25))
    with c4:
        g["net_debt"] = float(st.number_input("Net Debt", value=float(g.get("net_debt", 0.0)), step=100.0))
        g["cash"] = float(st.number_input("Cash", value=float(g.get("cash", 0.0)), step=100.0))
        g["base_customers"] = float(st.number_input("Starting Customers", value=float(g.get("base_customers", 200.0)), step=10.0))
        g["base_arpa"] = float(st.number_input("Starting ARPA (Annual $/Customer)", value=float(g.get("base_arpa", 6000.0)), step=250.0))

    st.session_state["globals"] = g

    st.markdown("### Terminal Value Method")
    st.session_state["terminal_method"] = st.radio(
        "Choose terminal method",
        ["Gordon Growth", "Exit Multiple (EV/Revenue)"],
        horizontal=True,
        index=0 if st.session_state.get("terminal_method", "Gordon Growth") == "Gordon Growth" else 1
    )

    if st.session_state["terminal_method"] == "Gordon Growth" and (g["wacc_pct"] / 100.0) <= (g["terminal_growth_pct"] / 100.0):
        st.error("❗ WACC must be greater than Terminal Growth for Gordon Growth terminal value.")

    st.divider()

    st.markdown("### Scenario Assumptions (Year-wise)")
    st.session_state["selected_scenario"] = st.radio(
        "Select scenario to edit / run",
        ["Bear", "Base", "Bull"],
        horizontal=True,
        index=["Bear", "Base", "Bull"].index(st.session_state.get("selected_scenario", "Base")),
    )
    selected_scn = st.session_state["selected_scenario"]

    n = int(st.session_state["globals"]["projection_years"])
    st.session_state["scenarios"] = normalize_scenarios(st.session_state.get("scenarios", {}), n)

    years_labels = [f"Year {i+1}" for i in range(n)]
    edit_df = pd.DataFrame({
        "Year": years_labels,
        "New Cust Growth (%)": st.session_state["scenarios"][selected_scn]["new_cust_growth_pct"],
        "Gross Churn (%)": st.session_state["scenarios"][selected_scn]["gross_churn_pct"],
        "NRR (%)": st.session_state["scenarios"][selected_scn]["nrr_pct"],
        "ARPA Growth (%)": st.session_state["scenarios"][selected_scn]["arpa_growth_pct"],
        "Gross Margin (%)": st.session_state["scenarios"][selected_scn]["gross_margin_pct"],
        "CAC per New Cust": st.session_state["scenarios"][selected_scn]["cac_per_new"],
        "S&M (% Rev)": st.session_state["scenarios"][selected_scn]["sm_pct_rev"],
        "R&D (% Rev)": st.session_state["scenarios"][selected_scn]["rnd_pct_rev"],
        "G&A (% Rev)": st.session_state["scenarios"][selected_scn]["ga_pct_rev"],
    })

    st.caption("Edit values in the grid. All % columns use percent units (10 = 10%).")
    edited = st.data_editor(edit_df, hide_index=True, disabled=["Year"], use_container_width=True)

    col_map = {
        "New Cust Growth (%)": "new_cust_growth_pct",
        "Gross Churn (%)": "gross_churn_pct",
        "NRR (%)": "nrr_pct",
        "ARPA Growth (%)": "arpa_growth_pct",
        "Gross Margin (%)": "gross_margin_pct",
        "CAC per New Cust": "cac_per_new",
        "S&M (% Rev)": "sm_pct_rev",
        "R&D (% Rev)": "rnd_pct_rev",
        "G&A (% Rev)": "ga_pct_rev",
    }

    if st.button("✅ Save Scenario Assumptions"):
        for ui_col, key in col_map.items():
            values = pd.to_numeric(edited[ui_col], errors="coerce").fillna(0.0).tolist()
            st.session_state["scenarios"][selected_scn][key] = ensure_len(values, n, fill=0.0)
        st.success(f"{selected_scn} scenario saved.")

    # Warn if CAC spend > total S&M (means S&M% too low for CAC assumptions)
    tmp_globals = merge_defaults(DEFAULT_GLOBALS, st.session_state["globals"])
    tmp_res = build_saas_model(tmp_globals, st.session_state["scenarios"][selected_scn], st.session_state["terminal_method"])
    tmp_df = tmp_res["df"]
    if (tmp_df["Non-CAC S&M"] < 0).any():
        st.warning(
            "In some years, CAC Spend is greater than total S&M (from S&M % of revenue). "
            "This implies you should increase S&M % or reduce CAC / acquisition growth."
        )

    st.divider()

    st.markdown("### Save / Load Assumptions (JSON)")
    payload = {
        "globals": st.session_state["globals"],
        "scenarios": st.session_state["scenarios"],
        "terminal_method": st.session_state["terminal_method"],
        "selected_scenario": st.session_state["selected_scenario"],
    }
    json_str = json.dumps(payload, indent=2)
    st.download_button("Download Assumptions JSON", data=json_str, file_name="saas_assumptions.json", mime="application/json")

    uploaded_json = st.file_uploader("Load Assumptions JSON", type=["json"])
    if uploaded_json is not None:
        try:
            loaded = json.load(uploaded_json)
            if isinstance(loaded, dict) and "globals" in loaded and "scenarios" in loaded:
                st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, loaded["globals"])
                n_loaded = int(st.session_state["globals"]["projection_years"])
                st.session_state["scenarios"] = normalize_scenarios(loaded["scenarios"], n_loaded)
                st.session_state["terminal_method"] = loaded.get("terminal_method", "Gordon Growth")
                st.session_state["selected_scenario"] = loaded.get("selected_scenario", "Base")
                st.success("Loaded assumptions. Refreshing…")
                st.rerun()
            else:
                st.error("Invalid JSON format.")
        except Exception:
            st.error("Could not read JSON.")

    c_reset, _ = st.columns([1, 6])
    with c_reset:
        if st.button("🔁 Reset to Defaults"):
            st.session_state["globals"] = DEFAULT_GLOBALS.copy()
            st.session_state["scenarios"] = json.loads(json.dumps(DEFAULT_SCENARIOS))
            st.session_state["terminal_method"] = "Gordon Growth"
            st.session_state["selected_scenario"] = "Base"
            st.success("Reset complete.")
            st.rerun()

# =================================================
# Compute model for selected scenario
# =================================================
globals_dict = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
terminal_method = st.session_state.get("terminal_method", "Gordon Growth")
selected = st.session_state.get("selected_scenario", "Base")

n = int(globals_dict["projection_years"])
st.session_state["scenarios"] = normalize_scenarios(st.session_state.get("scenarios", {}), n)

result = build_saas_model(globals_dict, st.session_state["scenarios"][selected], terminal_method)
df = result["df"]
years = result["years"]

all_results = {
    name: build_saas_model(globals_dict, st.session_state["scenarios"][name], terminal_method)
    for name in ["Bear", "Base", "Bull"]
}

# =================================================
# FINANCIAL MODEL TAB
# =================================================
with tab_model:
    st.subheader(f"📑 Financial Model – {selected} Scenario")

    drivers = pd.DataFrame({"Line Item": [
        "Customers (Beg)",
        "New Customers",
        "Churned Customers",
        "Customers (End)",
        "ARPA",
        "ARR",
        "Revenue",
        "Gross Margin (%)",
        "EBITDA Margin (%)"
    ]})
    for y in years:
        drivers[y] = [
            fmt_float2(df.loc[y, "Customers (Beg)"]),
            fmt_float2(df.loc[y, "New Customers"]),
            fmt_float2(df.loc[y, "Churned Customers"]),
            fmt_float2(df.loc[y, "Customers (End)"]),
            fmt_currency(df.loc[y, "ARPA"]),
            fmt_currency(df.loc[y, "ARR"]),
            fmt_currency(df.loc[y, "Revenue"]),
            fmt_pct2(df.loc[y, "Gross Margin (%)"]),
            fmt_pct2(df.loc[y, "EBITDA Margin (%)"]),
        ]
    show_table(drivers, "🧭 SaaS Operating Drivers")

    pnl = pd.DataFrame({"Line Item": [
        "Revenue",
        "COGS",
        "Gross Profit",
        "Sales & Marketing (Total)",
        "R&D",
        "G&A",
        "EBITDA",
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
            fmt_currency(df.loc[y, "Sales & Marketing (Total)"]),
            fmt_currency(df.loc[y, "R&D"]),
            fmt_currency(df.loc[y, "G&A"]),
            fmt_currency(df.loc[y, "EBITDA"]),
            fmt_currency(df.loc[y, "D&A"]),
            fmt_currency(df.loc[y, "EBIT"]),
            fmt_pct2(globals_dict["tax_rate_pct"]),
            fmt_currency(df.loc[y, "NOPAT"]),
        ]
    show_table(pnl, "📑 Profit & Loss Statement")

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
    show_table(cf, "💵 Cash Flow (FCF Build)")

    dcf = pd.DataFrame({"Line Item": ["FCF", "Discount Factor", "PV of FCF"]})
    for y in years:
        dcf[y] = [
            fmt_currency(df.loc[y, "FCF"]),
            fmt_float2(df.loc[y, "Discount Factor"]),
            fmt_currency(df.loc[y, "PV of FCF"]),
        ]
    show_table(dcf, "💰 DCF (Detailed)")

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

    st.subheader("⬇️ Export Excel")

    def to_excel_bytes():
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            pd.DataFrame([globals_dict]).to_excel(writer, sheet_name="Assumptions_Global", index=False)

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

            df.to_excel(writer, sheet_name="Model", index=True)

            pnl_num = df[["Revenue", "COGS", "Gross Profit", "Sales & Marketing (Total)", "R&D", "G&A", "EBITDA", "EBIT", "NOPAT"]].copy()
            pnl_num.to_excel(writer, sheet_name="PnL", index=True)

            cf_num = df[["NOPAT", "D&A", "Capex", "ΔNWC", "FCF", "Discount Factor", "PV of FCF"]].copy()
            cf_num.to_excel(writer, sheet_name="CashFlow_DCF", index=True)

            mkt_num = df[["Sales & Marketing (Total)", "CAC Spend", "Non-CAC S&M", "CAC % of S&M", "S&M % of Revenue"]].copy()
            mkt_num.to_excel(writer, sheet_name="Marketing", index=True)

        output.seek(0)
        return output.getvalue()

    st.download_button(
        "Download Excel Workbook",
        data=to_excel_bytes(),
        file_name=f"b2b_saas_model_{selected.lower()}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# =================================================
# DASHBOARD TAB
# =================================================
with tab_dash:
    st.subheader(f"📊 Dashboard – {selected} Scenario")

    k1, k2, k3, k4, k5 = st.columns(5)
    rev0 = float(df["Revenue"].iloc[0])
    revn = float(df["Revenue"].iloc[-1])
    nyrs = max(len(years) - 1, 1)
    rev_cagr = ((revn / rev0) ** (1 / nyrs) - 1) * 100 if rev0 > 0 else np.nan

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

    st.markdown("### Revenue & New Customer Growth (%)")
    scn = st.session_state["scenarios"][selected]
    new_growth = np.array(ensure_len(scn["new_cust_growth_pct"], n, fill=0.0), dtype=float)

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

    st.markdown("### Free Cash Flow (FCF)")
    fig, ax = plt.subplots()
    x_pos = np.arange(len(years))
    ax.plot(x_pos, df["FCF"].values, marker="o", color="#8c564b", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.set_ylabel("FCF")
    ax.grid(False)
    add_point_labels(ax, x_pos, df["FCF"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

# =================================================
# MARKETING SPEND TAB (new)
# =================================================
with tab_mkt:
    st.subheader(f"📣 Marketing Spend – {selected} Scenario")
    st.caption("Breakdown of total Sales & Marketing into CAC-driven acquisition spend vs. other/brand/field marketing.")

    # Table
    mkt_tbl = pd.DataFrame({"Line Item": [
        "Revenue",
        "Sales & Marketing (Total)",
        "S&M % of Revenue",
        "CAC Spend",
        "Non-CAC S&M",
        "CAC % of S&M",
        "New Customers",
        "CAC per New Customer",
    ]})
    scn = st.session_state["scenarios"][selected]
    cac_per_new = np.array(ensure_len(scn["cac_per_new"], n, fill=0.0), dtype=float)

    for i, y in enumerate(years):
        mkt_tbl[y] = [
            fmt_currency(df.loc[y, "Revenue"]),
            fmt_currency(df.loc[y, "Sales & Marketing (Total)"]),
            fmt_pct2(df.loc[y, "S&M % of Revenue"]),
            fmt_currency(df.loc[y, "CAC Spend"]),
            fmt_currency(df.loc[y, "Non-CAC S&M"]),
            fmt_pct2(df.loc[y, "CAC % of S&M"]),
            fmt_float2(df.loc[y, "New Customers"]),
            fmt_currency(cac_per_new[i]),
        ]
    show_table(mkt_tbl, "Marketing Spend Detail")

    # Chart: Stacked bar S&M split
    st.markdown("### S&M split: CAC vs Non-CAC (stacked)")
    x = np.arange(len(years))
    cac_vals = df["CAC Spend"].values
    non_cac_vals = np.maximum(df["Non-CAC S&M"].values, 0.0)

    fig, ax = plt.subplots()
    ax.bar(x, cac_vals, label="CAC Spend")
    ax.bar(x, non_cac_vals, bottom=cac_vals, label="Non-CAC S&M")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Spend")
    ax.grid(False)
    # labels on top of stacks
    totals = cac_vals + non_cac_vals
    for i, t in enumerate(totals):
        ax.annotate(f"{t:,.0f}", (x[i], t), textcoords="offset points", xytext=(0, 8),
                    ha="center", va="bottom", fontsize=9, clip_on=True)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    # Chart: CAC % of S&M
    st.markdown("### CAC % of S&M")
    fig, ax = plt.subplots()
    ax.plot(x, df["CAC % of S&M"].values, marker="o", label="CAC % of S&M")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Percent")
    ax.grid(False)
    add_point_labels(ax, x, df["CAC % of S&M"].values, is_pct=True)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    # Diagnostics
    if (df["Non-CAC S&M"] < 0).any():
        st.warning(
            "In some years, CAC Spend > Total S&M (S&M% of Revenue). "
            "Non-CAC S&M becomes negative, which typically indicates inconsistent assumptions."
        )

# =================================================
# SUBSCRIBER ACQUISITION TAB (new)
# =================================================
with tab_acq:
    st.subheader(f"🧲 Subscriber Acquisition – {selected} Scenario")
    st.caption("Customer waterfall + acquisition efficiency metrics (CAC, LTV, payback proxy).")

    # Build acquisition metrics
    scn = st.session_state["scenarios"][selected]
    churn_pct = np.array(ensure_len(scn["gross_churn_pct"], n, fill=0.0), dtype=float) / 100.0
    gm_pct = np.array(ensure_len(scn["gross_margin_pct"], n, fill=75.0), dtype=float) / 100.0
    cac_per_new = np.array(ensure_len(scn["cac_per_new"], n, fill=0.0), dtype=float)

    # Contribution margin per customer per year (gross profit per customer proxy)
    # = ARPA * GM
    contrib_per_cust = df["ARPA"].values * gm_pct

    # LTV (gross) approx = (ARPA*GM) / churn_rate
    ltv = np.where(churn_pct > 0, contrib_per_cust / churn_pct, np.nan)

    # LTV/CAC
    ltv_cac = np.where(cac_per_new > 0, ltv / cac_per_new, np.nan)

    # CAC payback months (proxy) = CAC / (monthly gross profit per customer)
    monthly_contrib = np.where(contrib_per_cust > 0, contrib_per_cust / 12.0, np.nan)
    payback_months = np.where(monthly_contrib > 0, cac_per_new / monthly_contrib, np.nan)

    # Table
    acq_tbl = pd.DataFrame({"Line Item": [
        "Customers (Beg)",
        "New Customers",
        "Churned Customers",
        "Customers (End)",
        "Net Adds",
        "CAC per New Customer",
        "CAC Spend",
        "ARPA",
        "Gross Margin (%)",
        "Gross Profit / Customer (annual)",
        "LTV (gross) (approx)",
        "LTV / CAC",
        "CAC Payback (months) (proxy)"
    ]})

    for i, y in enumerate(years):
        net_adds = df.loc[y, "Customers (End)"] - df.loc[y, "Customers (Beg)"]
        acq_tbl[y] = [
            fmt_float2(df.loc[y, "Customers (Beg)"]),
            fmt_float2(df.loc[y, "New Customers"]),
            fmt_float2(df.loc[y, "Churned Customers"]),
            fmt_float2(df.loc[y, "Customers (End)"]),
            fmt_float2(net_adds),
            fmt_currency(cac_per_new[i]),
            fmt_currency(df.loc[y, "CAC Spend"]),
            fmt_currency(df.loc[y, "ARPA"]),
            fmt_pct2(df.loc[y, "Gross Margin (%)"]),
            fmt_currency(contrib_per_cust[i]),
            fmt_currency(ltv[i]) if np.isfinite(ltv[i]) else "",
            fmt_float2(ltv_cac[i]) if np.isfinite(ltv_cac[i]) else "",
            fmt_float2(payback_months[i]) if np.isfinite(payback_months[i]) else "",
        ]
    show_table(acq_tbl, "Subscriber Acquisition Detail")

    # Charts
    x = np.arange(len(years))

    st.markdown("### Customer Waterfall (Begin vs End) + New / Churn")
    fig, ax = plt.subplots()
    ax.plot(x, df["Customers (Beg)"].values, marker="o", label="Customers (Beg)")
    ax.plot(x, df["Customers (End)"].values, marker="o", label="Customers (End)")
    ax.bar(x - 0.2, df["New Customers"].values, width=0.4, label="New Customers")
    ax.bar(x + 0.2, df["Churned Customers"].values, width=0.4, label="Churned Customers")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Customers")
    ax.grid(False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### CAC per New vs LTV (gross) (approx)")
    fig, ax = plt.subplots()
    ax.plot(x, cac_per_new, marker="o", label="CAC per New")
    ax.plot(x, ltv, marker="o", label="LTV (gross) approx")
    ax.set_xticks(x)
    ax.set_xticklabels(years)
    ax.set_ylabel("Currency")
    ax.grid(False)
    add_point_labels(ax, x, cac_per_new, is_pct=False)
    # labels for LTV can be huge; label only if finite and not insane
    for i, val in enumerate(ltv):
        if np.isfinite(val) and val < 10_000_000:
            ax.annotate(f"{val:,.0f}", (x[i], val), textcoords="offset points", xytext=(0, 8),
                        ha="center", va="bottom", fontsize=9, clip_on=True)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### LTV/CAC and Payback (months)")
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(x, ltv_cac, marker="o", label="LTV/CAC")
    ax2.plot(x, payback_months, marker="o", linestyle="--", label="Payback (months)")

    ax1.set_xticks(x)
    ax1.set_xticklabels(years)
    ax1.set_ylabel("LTV/CAC (x)")
    ax2.set_ylabel("Payback (months)")
    ax1.grid(False)
    ax2.grid(False)

    # label points
    for i, v in enumerate(ltv_cac):
        if np.isfinite(v):
            ax1.annotate(f"{v:.2f}x", (x[i], v), textcoords="offset points", xytext=(0, 8),
                         ha="center", va="bottom", fontsize=9, clip_on=True)
    for i, v in enumerate(payback_months):
        if np.isfinite(v):
            ax2.annotate(f"{v:.1f}", (x[i], v), textcoords="offset points", xytext=(0, -14),
                         ha="center", va="top", fontsize=9, clip_on=True)

    # combined legend
    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="best")

    fig.tight_layout()
    st.pyplot(fig)
