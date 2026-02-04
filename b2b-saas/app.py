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
st.caption("Assumptions • Financial Model • Dashboard • DCF • Export")

# =================================================
# CSS (center align tables)
# =================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td,
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

    ymin2, ymax2 = ax.get_ylim()
    for i, v in enumerate(y_vals):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        label = f"{v:.1f}%" if is_pct else f"{v:,.0f}"
        # keep labels in bounds
        dy = -14 if v > (ymax2 - (ymax2 - ymin2) * 0.12) else 8
        va = "top" if dy < 0 else "bottom"
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

def merge_defaults(defaults: dict, incoming: dict | None) -> dict:
    out = defaults.copy()
    if isinstance(incoming, dict):
        out.update(incoming)
    return out

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
    "brand_marketing_share_of_sm_pct": 30.0,
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
if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "Base"

# =================================================
# Tabs
# =================================================
tab_assump, tab_model, tab_dash = st.tabs(["🧾 Assumptions", "📑 Financial Model", "📊 Dashboard"])

# =================================================
# Model Builder
# =================================================
def build_saas_model(globals_dict: dict, scenario: dict):
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
    rev_rec = float(globals_dict["revenue_recognition_pct_of_arr"]) / 100.0
    da_pct_rev = float(globals_dict["da_pct_rev"]) / 100.0
    capex_pct_rev = float(globals_dict["capex_pct_rev"]) / 100.0
    nwc_pct_rev = float(globals_dict["nwc_pct_rev"]) / 100.0

    brand_share = float(globals_dict.get("brand_marketing_share_of_sm_pct", 30.0)) / 100.0
    brand_share = min(max(brand_share, 0.0), 1.0)

    prev_nwc = 0.0
    rows = []

    for i in range(n):
        new_customers = customers_beg * new_cust_growth[i]
        churned_customers = customers_beg * gross_churn[i]
        customers_end = max(customers_beg + new_customers - churned_customers, 0.0)
        customers_avg = (customers_beg + customers_end) / 2.0

        arpa = arpa * (1 + arpa_growth[i])
        arr = customers_avg * arpa * nrr[i]
        revenue = arr * rev_rec

        gross_profit = revenue * gm[i]
        cogs = revenue - gross_profit

        sm_total = revenue * sm_pct[i]
        brand_marketing = sm_total * brand_share
        cac_spend = new_customers * cac[i]
        acquisition_other = max(sm_total - brand_marketing - cac_spend, 0.0)
        acquisition_total = cac_spend + acquisition_other

        rnd = revenue * rnd_pct[i]
        ga = revenue * ga_pct[i]

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
            sm_total, brand_marketing, acquisition_total, cac_spend, acquisition_other,
            rnd, ga, ebitda,
            da, ebit, nopat,
            capex_val, delta_nwc, fcf,
        ])

        customers_beg = customers_end

    df = pd.DataFrame(
        rows,
        columns=[
            "Customers (Beg)", "New Customers", "Churned Customers", "Customers (End)",
            "ARPA", "ARR", "Revenue",
            "COGS", "Gross Profit",
            "Sales & Marketing (Total)", "Brand Marketing", "Acquisition (Total)", "CAC Spend", "Acquisition Other",
            "R&D", "G&A", "EBITDA",
            "D&A", "EBIT", "NOPAT",
            "Capex", "ΔNWC", "FCF",
        ],
        index=years
    )

    df["Gross Margin (%)"] = np.where(df["Revenue"] > 0, (df["Gross Profit"] / df["Revenue"]) * 100, np.nan)
    df["EBITDA Margin (%)"] = np.where(df["Revenue"] > 0, (df["EBITDA"] / df["Revenue"]) * 100, np.nan)

    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    return {"years": years, "df": df}

# =================================================
# Valuation helpers (both methods always computed)
# =================================================
def valuation_gordon(df: pd.DataFrame, globals_dict: dict, wacc_override=None, g_override=None):
    wacc = (float(globals_dict["wacc_pct"]) / 100.0) if wacc_override is None else float(wacc_override)
    g = (float(globals_dict["terminal_growth_pct"]) / 100.0) if g_override is None else float(g_override)
    if wacc <= g:
        return {"tv": np.nan, "pv_tv": np.nan, "pv_explicit": float(np.nansum(df["PV of FCF"])), "ev": np.nan, "equity": np.nan}

    fcf_last = float(df["FCF"].iloc[-1])
    tv = fcf_last * (1 + g) / (wacc - g)
    pv_tv = tv * float(df["Discount Factor"].iloc[-1])
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
    ev = pv_explicit + pv_tv
    equity = ev - float(globals_dict["net_debt"]) + float(globals_dict["cash"])
    return {"tv": tv, "pv_tv": pv_tv, "pv_explicit": pv_explicit, "ev": ev, "equity": equity, "wacc": wacc, "g": g}

def valuation_exit_multiple(df: pd.DataFrame, globals_dict: dict, multiple_override=None):
    multiple = float(globals_dict["exit_multiple_ev_rev"]) if multiple_override is None else float(multiple_override)
    revenue_last = float(df["Revenue"].iloc[-1])
    tv = revenue_last * multiple
    pv_tv = tv * float(df["Discount Factor"].iloc[-1])
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
    ev = pv_explicit + pv_tv
    equity = ev - float(globals_dict["net_debt"]) + float(globals_dict["cash"])
    return {"tv": tv, "pv_tv": pv_tv, "pv_explicit": pv_explicit, "ev": ev, "equity": equity, "multiple": multiple}

def football_field_chart(method_rows):
    """
    method_rows: list of dicts with keys: name, low, mid, high
    """
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.set_facecolor("#0b6623")  # green field

    y_positions = np.arange(len(method_rows))[::-1]  # top to bottom
    for idx, row in enumerate(method_rows):
        y = y_positions[idx]
        low, mid, high = row["low"], row["mid"], row["high"]

        # floating bar
        ax.barh(y, max(high - low, 0.0), left=low, height=0.35, alpha=0.85, color="#d7f5d7")
        # midpoint marker
        ax.plot([mid], [y], marker="o", markersize=7, color="#ffcc00")
        # labels
        ax.text(low, y + 0.18, fmt_currency(low), va="bottom", ha="left", fontsize=9, color="white")
        ax.text(high, y + 0.18, fmt_currency(high), va="bottom", ha="right", fontsize=9, color="white")
        ax.text(mid, y - 0.22, fmt_currency(mid), va="top", ha="center", fontsize=9, color="white")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r["name"] for r in method_rows], color="white", fontsize=10)

    # yard lines
    xmin = min(r["low"] for r in method_rows if np.isfinite(r["low"]))
    xmax = max(r["high"] for r in method_rows if np.isfinite(r["high"]))
    if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin:
        for frac in np.linspace(0, 1, 6):
            x = xmin + frac * (xmax - xmin)
            ax.axvline(x, color="white", alpha=0.18, linewidth=1)

    ax.set_xlabel("Enterprise Value (EV)", color="white")
    ax.tick_params(axis="x", colors="white")
    ax.grid(False)
    ax.set_title("Football Field: EV Range by Method", color="white")
    fig.tight_layout()
    return fig

# =================================================
# ASSUMPTIONS TAB
# =================================================
with tab_assump:
    st.subheader("🧾 Assumptions")

    g = st.session_state["globals"]

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

    st.markdown("### Marketing Split")
    g["brand_marketing_share_of_sm_pct"] = float(
        st.number_input("Brand/Programs share of S&M (%)", value=float(g.get("brand_marketing_share_of_sm_pct", 30.0)), step=5.0)
    )

    st.divider()
    st.markdown("### Scenario (edit & run)")
    st.session_state["selected_scenario"] = st.radio(
        "Select scenario",
        ["Bear", "Base", "Bull"],
        horizontal=True,
        index=["Bear", "Base", "Bull"].index(st.session_state.get("selected_scenario", "Base")),
    )
    selected_scn = st.session_state["selected_scenario"]

    n = int(g["projection_years"])
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

    st.caption("Edit values. % columns use percent units (10 = 10%).")
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

# =================================================
# Compute model
# =================================================
globals_dict = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
selected = st.session_state.get("selected_scenario", "Base")
n = int(globals_dict["projection_years"])
st.session_state["scenarios"] = normalize_scenarios(st.session_state.get("scenarios", {}), n)

model_out = build_saas_model(globals_dict, st.session_state["scenarios"][selected])
df = model_out["df"]
years = model_out["years"]

# =================================================
# FINANCIAL MODEL TAB (P&L / CF / Valuation separate)
# =================================================
with tab_model:
    st.subheader(f"📑 Financial Model – {selected} Scenario")

    tab_pl, tab_cf, tab_val = st.tabs(["📑 P&L", "💵 Cash Flow", "📘 Valuation"])

    with tab_pl:
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
            "NOPAT",
            "Gross Margin (%)",
            "EBITDA Margin (%)",
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
                fmt_currency(df.loc[y, "NOPAT"]),
                fmt_pct2(df.loc[y, "Gross Margin (%)"]),
                fmt_pct2(df.loc[y, "EBITDA Margin (%)"]),
            ]
        st.dataframe(pnl, hide_index=True, use_container_width=True)

    with tab_cf:
        cf = pd.DataFrame({"Line Item": [
            "NOPAT",
            "Add: D&A",
            "Less: Capex",
            "Less: ΔNWC",
            "Free Cash Flow (FCF)",
            "Discount Factor",
            "PV of FCF",
        ]})
        for y in years:
            cf[y] = [
                fmt_currency(df.loc[y, "NOPAT"]),
                fmt_currency(df.loc[y, "D&A"]),
                fmt_currency(-df.loc[y, "Capex"]),
                fmt_currency(-df.loc[y, "ΔNWC"]),
                fmt_currency(df.loc[y, "FCF"]),
                fmt_float2(df.loc[y, "Discount Factor"]),
                fmt_currency(df.loc[y, "PV of FCF"]),
            ]
        st.dataframe(cf, hide_index=True, use_container_width=True)

    with tab_val:
        st.markdown("### Detailed DCF Valuation (Both Methods)")

        g = globals_dict
        # Base valuations (mid)
        vg_mid = valuation_gordon(df, g)
        ve_mid = valuation_exit_multiple(df, g)

        # Build valuation ranges for football field:
        # Gordon range uses a simple +/- 0.50% band on WACC and g (bounded to keep WACC > g)
        wacc = float(g["wacc_pct"]) / 100.0
        tg = float(g["terminal_growth_pct"]) / 100.0
        band = 0.005  # 0.50%

        # "low" = higher discount & lower growth
        wacc_low = wacc + band
        g_low = max(tg - band, 0.0)

        # "high" = lower discount & higher growth (but keep WACC > g)
        wacc_high = max(wacc - band, 0.0001)
        g_high = tg + band
        if wacc_high <= g_high:
            g_high = max(wacc_high - 0.0005, 0.0)

        vg_low = valuation_gordon(df, g, wacc_override=wacc_low, g_override=g_low)
        vg_high = valuation_gordon(df, g, wacc_override=wacc_high, g_override=g_high)

        # Exit multiple range uses +/- 1.0x around the multiple (floored at 0)
        mult = float(g["exit_multiple_ev_rev"])
        ve_low = valuation_exit_multiple(df, g, multiple_override=max(mult - 1.0, 0.0))
        ve_high = valuation_exit_multiple(df, g, multiple_override=mult + 1.0)

        # Summary table (EV + Equity)
        yearN = years[-1]
        summary = pd.DataFrame({
            "Method": ["Gordon Growth (DCF)", "Exit Multiple (EV/Revenue)"],
            "EV (Low)": [fmt_currency(vg_low["ev"]), fmt_currency(ve_low["ev"])],
            "EV (Mid)": [fmt_currency(vg_mid["ev"]), fmt_currency(ve_mid["ev"])],
            "EV (High)": [fmt_currency(vg_high["ev"]), fmt_currency(ve_high["ev"])],
            "Equity (Mid)": [fmt_currency(vg_mid["equity"]), fmt_currency(ve_mid["equity"])],
        })
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown("### Terminal Value + PV Walk (Mid-case)")
        tv_df = pd.DataFrame({
            "Line Item": [
                "Final Year FCF",
                "Final Year Revenue",
                "WACC (%)",
                "Terminal Growth (%)",
                "Exit Multiple (EV/Revenue)",
                "Terminal Value (Gordon)",
                "PV of Terminal Value (Gordon)",
                "Terminal Value (Exit Multiple)",
                "PV of Terminal Value (Exit Multiple)",
                "PV of Explicit FCFs",
                "EV (Gordon)",
                "EV (Exit Multiple)",
            ],
            yearN: [
                fmt_currency(df["FCF"].iloc[-1]),
                fmt_currency(df["Revenue"].iloc[-1]),
                fmt_pct2(g["wacc_pct"]),
                fmt_pct2(g["terminal_growth_pct"]),
                fmt_float2(g["exit_multiple_ev_rev"]),
                fmt_currency(vg_mid["tv"]),
                fmt_currency(vg_mid["pv_tv"]),
                fmt_currency(ve_mid["tv"]),
                fmt_currency(ve_mid["pv_tv"]),
                fmt_currency(vg_mid["pv_explicit"]),
                fmt_currency(vg_mid["ev"]),
                fmt_currency(ve_mid["ev"]),
            ]
        })
        st.dataframe(tv_df, hide_index=True, use_container_width=True)

        st.markdown("### Football Field Chart (EV Range by Method)")
        method_rows = [
            {"name": "Gordon Growth (DCF)", "low": vg_low["ev"], "mid": vg_mid["ev"], "high": vg_high["ev"]},
            {"name": "Exit Multiple", "low": ve_low["ev"], "mid": ve_mid["ev"], "high": ve_high["ev"]},
        ]
        st.pyplot(football_field_chart(method_rows))

        st.caption(
            "Football field charts summarize valuation ranges side-by-side across methods. "
            "Here, ranges are driven by small sensitivity bands (±0.50% WACC/g; ±1.0x multiple)."
        )

# =================================================
# DASHBOARD TAB
# =================================================
with tab_dash:
    st.subheader(f"📊 Dashboard – {selected} Scenario")

    k1, k2, k3, k4 = st.columns(4)
    rev0 = float(df["Revenue"].iloc[0])
    revn = float(df["Revenue"].iloc[-1])
    nyrs = max(len(years) - 1, 1)
    rev_cagr = ((revn / rev0) ** (1 / nyrs) - 1) * 100 if rev0 > 0 else np.nan
    with k1: st.metric("Revenue CAGR", fmt_pct2(rev_cagr))
    with k2: st.metric("Avg Gross Margin", fmt_pct2(float(df["Gross Margin (%)"].mean())))
    with k3: st.metric("Avg EBITDA Margin", fmt_pct2(float(df["EBITDA Margin (%)"].mean())))
    with k4: st.metric("PV(FCF) Sum", fmt_currency(float(np.nansum(df["PV of FCF"].values))))

    # Revenue & growth proxy: use new customer growth series (your model driver)
    scn = st.session_state["scenarios"][selected]
    new_growth = np.array(ensure_len(scn["new_cust_growth_pct"], n, fill=0.0), dtype=float)

    st.markdown("### Revenue & New Customer Growth (%)")
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
    ax.plot(x_pos, df["FCF"].values, marker="o", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.grid(False)
    add_point_labels(ax, x_pos, df["FCF"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

# =================================================
# Export Excel
# =================================================
st.divider()
st.subheader("⬇️ Export Excel Workbook")

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

    output.seek(0)
    return output.getvalue()

st.download_button(
    "Download Excel Workbook",
    data=to_excel_bytes(),
    file_name=f"b2b_saas_model_{selected.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
