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
st.caption("Assumptions • Financial Model • Dashboards • Sensitivity • DCF • Export • Audit")

# =================================================
# CSS (center align tables)
# =================================================
st.markdown(
    """
    <style>
    div[data-testid="stDataFrame"] table th,
    div[data-testid="stDataFrame"] table td {
        text-align: center !important;
        vertical-align: middle !important;
        white-space: nowrap;
    }
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

def dual_axis_line_chart(
    title, x_labels, y_left, y_right,
    left_label, right_label,
    left_color, right_color,
    right_is_pct=True,
    rotate=0
):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    x_pos = np.arange(len(x_labels))

    ax1.plot(x_pos, y_left, marker="o", color=left_color, label=left_label)
    ax2.plot(x_pos, y_right, marker="o", linestyle="--", color=right_color, label=right_label)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=rotate, ha="right" if rotate else "center")
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
# AUDIT (selector-based, no clicking required)
# =================================================
def audit_key(table: str, line_item: str, period: str) -> str:
    return f"{table}||{line_item}||{period}"

def audit_pack(formula: str, components: list[tuple[str, float | str]], notes: str = "") -> dict:
    return {"formula": formula, "components": components, "notes": notes}

def show_audit_panel(audit_store: dict, table: str, line_item: str, period: str):
    k = audit_key(table, line_item, period)
    payload = audit_store.get(k)

    st.markdown("### 🔍 Audit Trace")
    st.write(f"**Table:** {table}  |  **Line Item:** {line_item}  |  **Period:** {period}")

    if not payload:
        st.info("No audit trace available for this selection.")
        return

    st.write("**Formula**")
    st.code(payload.get("formula", ""), language="text")

    notes = payload.get("notes", "")
    if notes:
        st.caption(notes)

    st.write("**Inputs / Components**")
    rows = []
    for name, val in payload.get("components", []):
        if isinstance(val, str):
            vdisp = val
        else:
            vdisp = fmt_currency(val)
        rows.append({"Component": name, "Value": vdisp})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

def audit_selector_ui(audit_store: dict, default_table="PnL", default_item="EBITDA", default_period=None, key_prefix="audit"):
    keys = list(audit_store.keys())
    if not keys:
        st.info("No audit traces found.")
        return

    parts = [k.split("||") for k in keys]
    tables = sorted(list({p[0] for p in parts}))
    table = st.selectbox("Table", tables, index=tables.index(default_table) if default_table in tables else 0, key=f"{key_prefix}_table")

    items = sorted(list({p[1] for p in parts if p[0] == table}))
    item = st.selectbox("Line Item", items, index=items.index(default_item) if default_item in items else 0, key=f"{key_prefix}_item")

    periods = sorted(list({p[2] for p in parts if p[0] == table and p[1] == item}))
    if default_period is None:
        default_period = periods[0] if periods else ""
    period = st.selectbox("Period", periods, index=periods.index(default_period) if default_period in periods else 0, key=f"{key_prefix}_period")

    show_audit_panel(audit_store, table, item, period)

# =================================================
# Monthly aggregation utilities (fix messy dashboard)
# =================================================
def _month_num(idx_label: str) -> int:
    # "Month 12" -> 12 ; "Year 2" -> 2*12 fallback
    try:
        n = int(str(idx_label).split()[-1])
        return n
    except Exception:
        return 1

def aggregate_monthly_to_quarterly(df_m: pd.DataFrame) -> pd.DataFrame:
    df = df_m.copy()
    df["__m"] = [ _month_num(i) for i in df.index ]
    df["Quarter"] = ((df["__m"] - 1) // 3) + 1
    df["Year"] = ((df["__m"] - 1) // 12) + 1
    df["Period"] = df["Year"].astype(int).astype(str).radd("Y") + " Q" + df["Quarter"].astype(int).astype(str)
    return _aggregate(df, group_col="Period")

def aggregate_monthly_to_annual(df_m: pd.DataFrame) -> pd.DataFrame:
    df = df_m.copy()
    df["__m"] = [ _month_num(i) for i in df.index ]
    df["Year"] = ((df["__m"] - 1) // 12) + 1
    df["Period"] = df["Year"].astype(int).astype(str).radd("Year ")
    return _aggregate(df, group_col="Period")

def _aggregate(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Sums for flow items, end-of-period for stock where appropriate,
    averages for % and some per-customer metrics.
    """
    flow_sum = [
        "New Customers","Churned Customers","Net Adds",
        "Revenue","COGS","Gross Profit",
        "Sales & Marketing (Total)","Brand Marketing","Acquisition (Total)","CAC Spend","Acquisition Other",
        "R&D","G&A","EBITDA","D&A","EBIT","NOPAT","Capex","ΔNWC","FCF","PV of FCF"
    ]
    stock_last = ["Customers (End)","ARR"]
    stock_first = ["Customers (Beg)"]
    avg_cols = ["ARPA","CAC Payback (months)","Discount Factor","Gross Margin (%)","EBITDA Margin (%)"]

    existing = set(df.columns)

    agg_dict = {}
    for c in flow_sum:
        if c in existing: agg_dict[c] = "sum"
    for c in stock_last:
        if c in existing: agg_dict[c] = "last"
    for c in stock_first:
        if c in existing: agg_dict[c] = "first"
    for c in avg_cols:
        if c in existing: agg_dict[c] = "mean"

    out = df.groupby(group_col).agg(agg_dict)
    out.index.name = None
    return out

def compute_growth(series: pd.Series) -> pd.Series:
    s = pd.Series(series).astype(float)
    g = s.pct_change() * 100.0
    return g

# =================================================
# Defaults
# =================================================
DEFAULT_GLOBALS = {
    "projection_years": 5,
    "reporting_period": "Annual",  # Annual | Monthly
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

def normalize_scenarios_annual(scenarios_obj: dict | None, years: int) -> dict:
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

def normalize_scenarios_monthly(scenarios_obj: dict | None, months: int) -> dict:
    out = {}
    for name in ["Bear", "Base", "Bull"]:
        incoming = scenarios_obj.get(name, {}) if isinstance(scenarios_obj, dict) else {}
        base = {}
        for k in SCN_KEYS:
            arr = incoming.get(k, None)
            if isinstance(arr, list) and len(arr) > 0:
                base[k] = ensure_len(arr, months, fill=arr[-1])
            else:
                annual_default = DEFAULT_SCENARIOS[name].get(k, [0.0])
                seed = float(annual_default[0]) if annual_default else 0.0
                base[k] = [seed] * months
        out[name] = base
    return out

# =================================================
# Session State Init
# =================================================
if "globals" not in st.session_state:
    st.session_state["globals"] = DEFAULT_GLOBALS.copy()
else:
    st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, st.session_state["globals"])

if "scenarios_annual" not in st.session_state:
    st.session_state["scenarios_annual"] = json.loads(json.dumps(DEFAULT_SCENARIOS))

if "scenarios_monthly" not in st.session_state:
    st.session_state["scenarios_monthly"] = {}

if "terminal_method" not in st.session_state:
    st.session_state["terminal_method"] = "Gordon Growth"

if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "Base"

# =================================================
# Tabs
# =================================================
tab_assump, tab_model, tab_dash, tab_metrics, tab_sens, tab_audit = st.tabs(
    ["🧾 Assumptions", "📑 Financial Model", "📊 Dashboard", "📈 Metrics Dashboard", "📌 Sensitivity", "🔍 Audit"]
)

# =================================================
# Model Builder
# =================================================
def build_saas_model(globals_dict: dict, scenario_annual: dict, scenario_monthly: dict | None, terminal_method: str):
    n_years = int(globals_dict["projection_years"])
    reporting = globals_dict.get("reporting_period", "Annual")

    if reporting == "Monthly":
        n = n_years * 12
        periods = [f"Month {i+1}" for i in range(n)]
        scn = scenario_monthly

        new_cust_growth = np.array(ensure_len(scn["new_cust_growth_pct"], n, fill=0.0), dtype=float) / 100.0
        gross_churn     = np.array(ensure_len(scn["gross_churn_pct"], n, fill=0.0), dtype=float) / 100.0
        nrr_mult        = np.array(ensure_len(scn["nrr_pct"], n, fill=100.0), dtype=float) / 100.0
        arpa_growth     = np.array(ensure_len(scn["arpa_growth_pct"], n, fill=0.0), dtype=float) / 100.0

        gm              = np.array(ensure_len(scn["gross_margin_pct"], n, fill=75.0), dtype=float) / 100.0
        sm_pct          = np.array(ensure_len(scn["sm_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        rnd_pct         = np.array(ensure_len(scn["rnd_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        ga_pct          = np.array(ensure_len(scn["ga_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        cac_per_new     = np.array(ensure_len(scn["cac_per_new"], n, fill=0.0), dtype=float)
    else:
        n = n_years
        periods = [f"Year {i+1}" for i in range(n)]
        scn = scenario_annual

        new_cust_growth = np.array(ensure_len(scn["new_cust_growth_pct"], n, fill=0.0), dtype=float) / 100.0
        gross_churn     = np.array(ensure_len(scn["gross_churn_pct"], n, fill=0.0), dtype=float) / 100.0
        nrr_mult        = np.array(ensure_len(scn["nrr_pct"], n, fill=100.0), dtype=float) / 100.0
        arpa_growth     = np.array(ensure_len(scn["arpa_growth_pct"], n, fill=0.0), dtype=float) / 100.0

        gm              = np.array(ensure_len(scn["gross_margin_pct"], n, fill=75.0), dtype=float) / 100.0
        sm_pct          = np.array(ensure_len(scn["sm_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        rnd_pct         = np.array(ensure_len(scn["rnd_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        ga_pct          = np.array(ensure_len(scn["ga_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        cac_per_new     = np.array(ensure_len(scn["cac_per_new"], n, fill=0.0), dtype=float)

    tax_annual = float(globals_dict["tax_rate_pct"]) / 100.0
    wacc_annual = float(globals_dict["wacc_pct"]) / 100.0
    tg_annual = float(globals_dict["terminal_growth_pct"]) / 100.0

    rev_rec = float(globals_dict["revenue_recognition_pct_of_arr"]) / 100.0
    da_pct_rev = float(globals_dict["da_pct_rev"]) / 100.0
    capex_pct_rev = float(globals_dict["capex_pct_rev"]) / 100.0
    nwc_pct_rev = float(globals_dict["nwc_pct_rev"]) / 100.0

    brand_share = float(globals_dict.get("brand_marketing_share_of_sm_pct", 30.0)) / 100.0
    brand_share = min(max(brand_share, 0.0), 1.0)

    customers_beg = float(globals_dict["base_customers"])
    arpa = float(globals_dict["base_arpa"])  # annual per customer

    if reporting == "Monthly":
        wacc = (1 + wacc_annual) ** (1/12) - 1
        tg = (1 + tg_annual) ** (1/12) - 1
        tax = tax_annual / 12.0
    else:
        wacc = wacc_annual
        tg = tg_annual
        tax = tax_annual

    prev_nwc = 0.0
    rows = []
    audit = {}

    for i in range(n):
        period = periods[i]

        new_customers = customers_beg * new_cust_growth[i]
        churned_customers = customers_beg * gross_churn[i]
        customers_end = max(customers_beg + new_customers - churned_customers, 0.0)
        customers_avg = (customers_beg + customers_end) / 2.0
        net_adds = customers_end - customers_beg

        arpa = arpa * (1 + arpa_growth[i])

        arr = customers_avg * arpa * nrr_mult[i]
        revenue = (arr * rev_rec) if reporting == "Annual" else (arr * rev_rec / 12.0)

        gross_profit = revenue * gm[i]
        cogs = revenue - gross_profit

        sm_total = revenue * sm_pct[i]
        brand_marketing = sm_total * brand_share

        cac_spend = new_customers * cac_per_new[i]
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
        prev_nwc_prev = prev_nwc
        prev_nwc = nwc

        fcf = nopat + da - capex_val - delta_nwc

        gp_per_cust_per_month = (arpa * gm[i]) / 12.0 if (arpa * gm[i]) > 0 else np.nan
        cac_payback_months = (cac_per_new[i] / gp_per_cust_per_month) if (gp_per_cust_per_month and np.isfinite(gp_per_cust_per_month)) else np.nan

        rows.append([
            customers_beg, new_customers, churned_customers, net_adds, customers_end,
            arpa, arr, revenue,
            cogs, gross_profit,
            sm_total, brand_marketing, acquisition_total, cac_spend, acquisition_other,
            cac_payback_months,
            rnd, ga, ebitda,
            da, ebit, nopat,
            capex_val, delta_nwc, fcf,
        ])

        # audit (core)
        audit[audit_key("Drivers", "Revenue", period)] = audit_pack(
            "Revenue = ARR × Revenue Recognition" + (" / 12 (monthly)" if reporting == "Monthly" else ""),
            [("ARR", arr), ("Revenue Recognition", f"{rev_rec*100:.2f}%"), ("Divide by 12?", "Yes" if reporting == "Monthly" else "No")],
        )
        audit[audit_key("PnL", "EBITDA", period)] = audit_pack(
            "EBITDA = Gross Profit − S&M − R&D − G&A",
            [("Gross Profit", gross_profit), ("S&M", sm_total), ("R&D", rnd), ("G&A", ga)],
        )
        audit[audit_key("Cash Flow", "FCF", period)] = audit_pack(
            "FCF = NOPAT + D&A − Capex − ΔNWC",
            [("NOPAT", nopat), ("D&A", da), ("Capex", capex_val), ("ΔNWC", delta_nwc)],
        )

        customers_beg = customers_end

    df = pd.DataFrame(
        rows,
        columns=[
            "Customers (Beg)", "New Customers", "Churned Customers", "Net Adds", "Customers (End)",
            "ARPA", "ARR", "Revenue",
            "COGS", "Gross Profit",
            "Sales & Marketing (Total)", "Brand Marketing", "Acquisition (Total)", "CAC Spend", "Acquisition Other",
            "CAC Payback (months)",
            "R&D", "G&A", "EBITDA",
            "D&A", "EBIT", "NOPAT",
            "Capex", "ΔNWC", "FCF",
        ],
        index=periods
    )

    df["Gross Margin (%)"] = np.where(df["Revenue"] > 0, (df["Gross Profit"] / df["Revenue"]) * 100, np.nan)
    df["EBITDA Margin (%)"] = np.where(df["Revenue"] > 0, (df["EBITDA"] / df["Revenue"]) * 100, np.nan)

    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    last_period = periods[-1]

    if terminal_method == "Gordon Growth":
        if wacc <= tg:
            tv = np.nan
        else:
            fcf_last = float(df["FCF"].iloc[-1])
            tv = fcf_last * (1 + tg) / (wacc - tg)
    else:
        revenue_last = float(df["Revenue"].iloc[-1])
        revenue_for_tv = revenue_last * (12.0 if reporting == "Monthly" else 1.0)
        tv = revenue_for_tv * float(globals_dict["exit_multiple_ev_rev"])

    pv_tv = tv * discount_factors[-1] if np.isfinite(tv) else np.nan
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
    ev = pv_explicit + (float(pv_tv) if np.isfinite(pv_tv) else 0.0)
    equity_value = ev - float(globals_dict["net_debt"]) + float(globals_dict["cash"])
    tv_share = (pv_tv / ev) if ev != 0 and np.isfinite(pv_tv) else np.nan

    return {
        "periods": periods,
        "df": df,
        "audit": audit,
        "pv_explicit": pv_explicit,
        "terminal_value": tv,
        "pv_terminal_value": pv_tv,
        "enterprise_value": ev,
        "equity_value": equity_value,
        "terminal_share": tv_share,
        "reporting_period": reporting,
        "wacc_period": wacc,
        "tg_period": tg,
        "last_period": last_period,
    }

# =================================================
# ASSUMPTIONS TAB
# =================================================
with tab_assump:
    st.subheader("🧾 Assumptions")

    st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
    g = st.session_state["globals"]

    g["reporting_period"] = st.radio(
        "Reporting Period",
        ["Annual", "Monthly"],
        horizontal=True,
        index=0 if g.get("reporting_period", "Annual") == "Annual" else 1,
        help="Annual shows Year 1..N. Monthly shows Month 1..(N×12) and lets you edit month-wise assumptions."
    )

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
    st.session_state["globals"] = g

    st.markdown("### Terminal Method (used in main valuation summary)")
    st.session_state["terminal_method"] = st.radio(
        "Choose terminal method",
        ["Gordon Growth", "Exit Multiple (EV/Revenue)"],
        horizontal=True,
        index=0 if st.session_state.get("terminal_method", "Gordon Growth") == "Gordon Growth" else 1
    )

    st.divider()
    st.markdown("### Scenario Assumptions")
    st.session_state["selected_scenario"] = st.radio(
        "Select scenario to edit / run",
        ["Bear", "Base", "Bull"],
        horizontal=True,
        index=["Bear", "Base", "Bull"].index(st.session_state.get("selected_scenario", "Base")),
    )
    selected_scn = st.session_state["selected_scenario"]

    n_years = int(g["projection_years"])
    reporting = g["reporting_period"]

    st.session_state["scenarios_annual"] = normalize_scenarios_annual(st.session_state.get("scenarios_annual", {}), n_years)

    if reporting == "Monthly":
        n_months = n_years * 12
        st.session_state["scenarios_monthly"] = normalize_scenarios_monthly(st.session_state.get("scenarios_monthly", {}), n_months)

        months_labels = [f"Month {i+1}" for i in range(n_months)]
        edit_df = pd.DataFrame({
            "Month": months_labels,
            "New Cust Growth (%)": st.session_state["scenarios_monthly"][selected_scn]["new_cust_growth_pct"],
            "Gross Churn (%)": st.session_state["scenarios_monthly"][selected_scn]["gross_churn_pct"],
            "NRR (%)": st.session_state["scenarios_monthly"][selected_scn]["nrr_pct"],
            "ARPA Growth (%)": st.session_state["scenarios_monthly"][selected_scn]["arpa_growth_pct"],
            "Gross Margin (%)": st.session_state["scenarios_monthly"][selected_scn]["gross_margin_pct"],
            "CAC per New Cust": st.session_state["scenarios_monthly"][selected_scn]["cac_per_new"],
            "S&M (% Rev)": st.session_state["scenarios_monthly"][selected_scn]["sm_pct_rev"],
            "R&D (% Rev)": st.session_state["scenarios_monthly"][selected_scn]["rnd_pct_rev"],
            "G&A (% Rev)": st.session_state["scenarios_monthly"][selected_scn]["ga_pct_rev"],
        })

        st.caption("Monthly editor (per month). NRR is a monthly multiplier in % (e.g., 100.50 = 1.005).")
        edited = st.data_editor(edit_df, hide_index=True, disabled=["Month"], use_container_width=True)

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

        if st.button("✅ Save Monthly Scenario Assumptions"):
            for ui_col, key in col_map.items():
                values = pd.to_numeric(edited[ui_col], errors="coerce").fillna(0.0).tolist()
                st.session_state["scenarios_monthly"][selected_scn][key] = ensure_len(values, n_months, fill=0.0)
            st.success(f"{selected_scn} monthly scenario saved.")

    else:
        years_labels = [f"Year {i+1}" for i in range(n_years)]
        edit_df = pd.DataFrame({
            "Year": years_labels,
            "New Cust Growth (%)": st.session_state["scenarios_annual"][selected_scn]["new_cust_growth_pct"],
            "Gross Churn (%)": st.session_state["scenarios_annual"][selected_scn]["gross_churn_pct"],
            "NRR (%)": st.session_state["scenarios_annual"][selected_scn]["nrr_pct"],
            "ARPA Growth (%)": st.session_state["scenarios_annual"][selected_scn]["arpa_growth_pct"],
            "Gross Margin (%)": st.session_state["scenarios_annual"][selected_scn]["gross_margin_pct"],
            "CAC per New Cust": st.session_state["scenarios_annual"][selected_scn]["cac_per_new"],
            "S&M (% Rev)": st.session_state["scenarios_annual"][selected_scn]["sm_pct_rev"],
            "R&D (% Rev)": st.session_state["scenarios_annual"][selected_scn]["rnd_pct_rev"],
            "G&A (% Rev)": st.session_state["scenarios_annual"][selected_scn]["ga_pct_rev"],
        })

        st.caption("Annual editor (per year).")
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

        if st.button("✅ Save Annual Scenario Assumptions"):
            for ui_col, key in col_map.items():
                values = pd.to_numeric(edited[ui_col], errors="coerce").fillna(0.0).tolist()
                st.session_state["scenarios_annual"][selected_scn][key] = ensure_len(values, n_years, fill=0.0)
            st.success(f"{selected_scn} annual scenario saved.")

# =================================================
# Compute model
# =================================================
globals_dict = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
terminal_method = st.session_state.get("terminal_method", "Gordon Growth")
selected = st.session_state.get("selected_scenario", "Base")
n_years = int(globals_dict["projection_years"])
reporting = globals_dict.get("reporting_period", "Annual")

scenarios_annual = normalize_scenarios_annual(st.session_state.get("scenarios_annual", {}), n_years)

scenario_monthly = None
if reporting == "Monthly":
    n_months = n_years * 12
    st.session_state["scenarios_monthly"] = normalize_scenarios_monthly(st.session_state.get("scenarios_monthly", {}), n_months)
    scenario_monthly = st.session_state["scenarios_monthly"][selected]

result = build_saas_model(globals_dict, scenarios_annual[selected], scenario_monthly, terminal_method)
df = result["df"]
periods = result["periods"]
audit_store = result["audit"]

# =================================================
# FINANCIAL MODEL TAB
# =================================================
with tab_model:
    st.subheader(f"📑 Financial Model – {selected} Scenario ({reporting})")

    with st.expander("🔍 Audit Mode (works without clicking)"):
        default_period = periods[0] if periods else None
        audit_selector_ui(audit_store, default_table="PnL", default_item="EBITDA", default_period=default_period, key_prefix="fm_audit")

    sub_pnl, sub_cf, sub_val = st.tabs(["📑 P&L", "💵 Cash Flow", "📘 Valuation"])

    with sub_pnl:
        pnl = pd.DataFrame({"Line Item": [
            "Revenue", "COGS", "Gross Profit",
            "Sales & Marketing (Total)", "R&D", "G&A",
            "EBITDA", "D&A", "EBIT", "NOPAT"
        ]})
        for p in periods:
            pnl[p] = [
                fmt_currency(df.loc[p, "Revenue"]),
                fmt_currency(df.loc[p, "COGS"]),
                fmt_currency(df.loc[p, "Gross Profit"]),
                fmt_currency(df.loc[p, "Sales & Marketing (Total)"]),
                fmt_currency(df.loc[p, "R&D"]),
                fmt_currency(df.loc[p, "G&A"]),
                fmt_currency(df.loc[p, "EBITDA"]),
                fmt_currency(df.loc[p, "D&A"]),
                fmt_currency(df.loc[p, "EBIT"]),
                fmt_currency(df.loc[p, "NOPAT"]),
            ]
        st.dataframe(pnl, hide_index=True, use_container_width=True)

    with sub_cf:
        cf = pd.DataFrame({"Line Item": ["NOPAT", "D&A", "Capex", "ΔNWC", "FCF"]})
        for p in periods:
            cf[p] = [
                fmt_currency(df.loc[p, "NOPAT"]),
                fmt_currency(df.loc[p, "D&A"]),
                fmt_currency(df.loc[p, "Capex"]),
                fmt_currency(df.loc[p, "ΔNWC"]),
                fmt_currency(df.loc[p, "FCF"]),
            ]
        st.dataframe(cf, hide_index=True, use_container_width=True)

    with sub_val:
        last_p = periods[-1]
        val = pd.DataFrame({
            "Line Item": ["Terminal Value", "PV of Terminal Value", "PV of Explicit FCFs", "Enterprise Value", "Equity Value"],
            last_p: [
                fmt_currency(result["terminal_value"]),
                fmt_currency(result["pv_terminal_value"]),
                fmt_currency(result["pv_explicit"]),
                fmt_currency(result["enterprise_value"]),
                fmt_currency(result["equity_value"]),
            ]
        })
        st.dataframe(val, hide_index=True, use_container_width=True)

# =================================================
# DASHBOARD (clean view selector for Monthly)
# =================================================
with tab_dash:
    st.subheader(f"📊 Dashboard – {selected} Scenario ({reporting})")

    if reporting == "Monthly":
        view = st.radio("Dashboard View", ["Monthly (Range)", "Quarterly (Aggregated)", "Annual (Aggregated)"], horizontal=True)
        if view == "Monthly (Range)":
            max_m = len(df)
            start, end = st.slider("Select month range", 1, max_m, (1, min(max_m, 24)))
            chart_df = df.iloc[start-1:end].copy()
            x_labels = list(chart_df.index)
            rotate = 45
        elif view == "Quarterly (Aggregated)":
            chart_df = aggregate_monthly_to_quarterly(df)
            x_labels = list(chart_df.index)
            rotate = 0
        else:
            chart_df = aggregate_monthly_to_annual(df)
            x_labels = list(chart_df.index)
            rotate = 0
    else:
        chart_df = df.copy()
        x_labels = list(chart_df.index)
        rotate = 0

    # KPIs (based on chart_df last)
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Revenue (Last)", fmt_currency(chart_df["Revenue"].iloc[-1]))
    with c2:
        st.metric("EBITDA (Last)", fmt_currency(chart_df["EBITDA"].iloc[-1]))
    with c3:
        st.metric("EBITDA Margin (Avg)", fmt_pct2(float(pd.Series(chart_df["EBITDA Margin (%)"]).dropna().mean())))
    with c4:
        st.metric("FCF (Last)", fmt_currency(chart_df["FCF"].iloc[-1]))
    with c5:
        st.metric("Customers (End)", fmt_float2(chart_df["Customers (End)"].iloc[-1]))

    st.divider()

    # Compact summary table (not 60 columns)
    summary = pd.DataFrame({
        "Period": x_labels,
        "Revenue": [fmt_currency(v) for v in chart_df["Revenue"].values],
        "EBITDA": [fmt_currency(v) for v in chart_df["EBITDA"].values],
        "EBITDA Margin (%)": [fmt_pct2(v) for v in chart_df["EBITDA Margin (%)"].values],
        "FCF": [fmt_currency(v) for v in chart_df["FCF"].values],
        "Customers (End)": [fmt_float2(v) for v in chart_df["Customers (End)"].values],
    })
    st.dataframe(summary, hide_index=True, use_container_width=True)

# =================================================
# METRICS DASHBOARD (the graphs you asked for)
# =================================================
with tab_metrics:
    st.subheader(f"📈 Metrics Dashboard – {selected} Scenario ({reporting})")

    # For charts: if Monthly, default to Quarterly aggregated to avoid mess
    if reporting == "Monthly":
        mode = st.radio("Chart Granularity", ["Monthly (Range)", "Quarterly", "Annual"], horizontal=True)
        if mode == "Monthly (Range)":
            max_m = len(df)
            start, end = st.slider("Select month range (charts)", 1, max_m, (1, min(max_m, 24)), key="mrange_charts")
            dff = df.iloc[start-1:end].copy()
            x = list(dff.index)
            rotate = 45
        elif mode == "Quarterly":
            dff = aggregate_monthly_to_quarterly(df)
            x = list(dff.index)
            rotate = 0
        else:
            dff = aggregate_monthly_to_annual(df)
            x = list(dff.index)
            rotate = 0
    else:
        dff = df.copy()
        x = list(dff.index)
        rotate = 0

    # Growth rates (period over period)
    rev_growth = compute_growth(dff["Revenue"])
    ebitda_margin = dff["EBITDA Margin (%)"]
    customers_growth = compute_growth(dff["Customers (End)"])

    st.markdown("### Revenue & Revenue Growth (%)")
    st.pyplot(dual_axis_line_chart(
        title="Revenue vs Revenue Growth",
        x_labels=x,
        y_left=dff["Revenue"].values,
        y_right=rev_growth.fillna(0.0).values,
        left_label="Revenue",
        right_label="Revenue Growth (%)",
        left_color="#1f77b4",
        right_color="#ff7f0e",
        right_is_pct=True,
        rotate=rotate
    ))

    st.markdown("### EBITDA & EBITDA Margin (%)")
    st.pyplot(dual_axis_line_chart(
        title="EBITDA vs EBITDA Margin",
        x_labels=x,
        y_left=dff["EBITDA"].values,
        y_right=ebitda_margin.fillna(0.0).values,
        left_label="EBITDA",
        right_label="EBITDA Margin (%)",
        left_color="#2ca02c",
        right_color="#d62728",
        right_is_pct=True,
        rotate=rotate
    ))

    st.markdown("### NOPAT (Profit After Tax proxy)")
    fig, ax = plt.subplots()
    x_pos = np.arange(len(x))
    ax.plot(x_pos, dff["NOPAT"].values, marker="o", label="NOPAT")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(False)
    add_point_labels(ax, x_pos, dff["NOPAT"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Free Cash Flow (FCF)")
    fig, ax = plt.subplots()
    ax.plot(x_pos, dff["FCF"].values, marker="o", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x, rotation=rotate, ha="right" if rotate else "center")
    ax.grid(False)
    add_point_labels(ax, x_pos, dff["FCF"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Customers (End) & Customer Growth (%)")
    st.pyplot(dual_axis_line_chart(
        title="Customers vs Customer Growth",
        x_labels=x,
        y_left=dff["Customers (End)"].values,
        y_right=customers_growth.fillna(0.0).values,
        left_label="Customers (End)",
        right_label="Customer Growth (%)",
        left_color="#9467bd",
        right_color="#8c564b",
        right_is_pct=True,
        rotate=rotate
    ))

# =================================================
# SENSITIVITY ANALYSIS TAB
# =================================================
def run_model_for_sensitivity(globals_base: dict, scenario_annual: dict, scenario_monthly: dict | None, terminal_method: str,
                              wacc_pct: float, tg_pct: float, exit_mult: float):
    g2 = globals_base.copy()
    g2["wacc_pct"] = float(wacc_pct)
    g2["terminal_growth_pct"] = float(tg_pct)
    g2["exit_multiple_ev_rev"] = float(exit_mult)
    return build_saas_model(g2, scenario_annual, scenario_monthly, terminal_method)

def football_field_chart(values_dict: dict, title: str):
    """
    values_dict: {label: (low, mid, high)} shown as horizontal ranges
    """
    labels = list(values_dict.keys())
    lows = np.array([values_dict[k][0] for k in labels], dtype=float)
    meds = np.array([values_dict[k][1] for k in labels], dtype=float)
    highs = np.array([values_dict[k][2] for k in labels], dtype=float)

    y = np.arange(len(labels))[::-1]  # top-to-bottom
    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * len(labels))))

    for i, lab in enumerate(labels):
        lo, md, hi = lows[i], meds[i], highs[i]
        ax.plot([lo, hi], [y[i], y[i]], linewidth=6, solid_capstyle="butt")
        ax.scatter([md], [y[i]], s=80)
        ax.text(lo, y[i]+0.12, fmt_currency(lo), ha="left", va="bottom", fontsize=9)
        ax.text(hi, y[i]+0.12, fmt_currency(hi), ha="right", va="bottom", fontsize=9)
        ax.text(md, y[i]-0.18, fmt_currency(md), ha="center", va="top", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.grid(False)
    ax.set_title(title)
    ax.set_xlabel("Enterprise Value")
    fig.tight_layout()
    return fig

with tab_sens:
    st.subheader(f"📌 Sensitivity Analysis – {selected} Scenario ({reporting})")
    st.caption("All sensitivities are on Enterprise Value (EV). Use grids + football-field chart for summary.")

    colA, colB, colC = st.columns(3)
    with colA:
        wacc_min = st.number_input("WACC min (%)", value=max(1.0, float(globals_dict["wacc_pct"]) - 3.0), step=0.5)
        wacc_max = st.number_input("WACC max (%)", value=float(globals_dict["wacc_pct"]) + 3.0, step=0.5)
    with colB:
        tg_min = st.number_input("Terminal Growth min (%)", value=max(0.0, float(globals_dict["terminal_growth_pct"]) - 2.0), step=0.25)
        tg_max = st.number_input("Terminal Growth max (%)", value=float(globals_dict["terminal_growth_pct"]) + 2.0, step=0.25)
    with colC:
        mult_min = st.number_input("Exit Multiple min (EV/Rev)", value=max(0.5, float(globals_dict["exit_multiple_ev_rev"]) - 2.0), step=0.5)
        mult_max = st.number_input("Exit Multiple max (EV/Rev)", value=float(globals_dict["exit_multiple_ev_rev"]) + 2.0, step=0.5)

    steps = st.slider("Grid steps", 3, 9, 5)

    wacc_grid = np.linspace(wacc_min, wacc_max, steps)
    tg_grid = np.linspace(tg_min, tg_max, steps)
    mult_grid = np.linspace(mult_min, mult_max, steps)

    scenario_monthly_sel = scenario_monthly if reporting == "Monthly" else None
    scn_ann_sel = scenarios_annual[selected]

    st.divider()
    st.markdown("### Gordon Growth Sensitivity (EV) — WACC × Terminal Growth")
    gordon_tbl = pd.DataFrame(index=[f"{w:.2f}%" for w in wacc_grid], columns=[f"{t:.2f}%" for t in tg_grid])
    for w in wacc_grid:
        for t in tg_grid:
            r = run_model_for_sensitivity(globals_dict, scn_ann_sel, scenario_monthly_sel, "Gordon Growth", w, t, float(globals_dict["exit_multiple_ev_rev"]))
            gordon_tbl.loc[f"{w:.2f}%", f"{t:.2f}%"] = fmt_currency(r["enterprise_value"])
    st.dataframe(gordon_tbl, use_container_width=True)

    st.markdown("### Exit Multiple Sensitivity (EV) — WACC × EV/Revenue Multiple")
    mult_tbl = pd.DataFrame(index=[f"{w:.2f}%" for w in wacc_grid], columns=[f"{m:.2f}x" for m in mult_grid])
    for w in wacc_grid:
        for m in mult_grid:
            r = run_model_for_sensitivity(globals_dict, scn_ann_sel, scenario_monthly_sel, "Exit Multiple (EV/Revenue)", w, float(globals_dict["terminal_growth_pct"]), m)
            mult_tbl.loc[f"{w:.2f}%", f"{m:.2f}x"] = fmt_currency(r["enterprise_value"])
    st.dataframe(mult_tbl, use_container_width=True)

    st.divider()
    st.markdown("### Football Field (EV) — Gordon vs Exit Multiple + Scenarios")
    # Build EV ranges (low/mid/high) from the grids
    # Gordon range for selected scenario
    g_vals = []
    for w in wacc_grid:
        for t in tg_grid:
            rr = run_model_for_sensitivity(globals_dict, scn_ann_sel, scenario_monthly_sel, "Gordon Growth", w, t, float(globals_dict["exit_multiple_ev_rev"]))
            g_vals.append(rr["enterprise_value"])
    g_low, g_mid, g_high = float(np.nanmin(g_vals)), float(np.nanmedian(g_vals)), float(np.nanmax(g_vals))

    m_vals = []
    for w in wacc_grid:
        for m in mult_grid:
            rr = run_model_for_sensitivity(globals_dict, scn_ann_sel, scenario_monthly_sel, "Exit Multiple (EV/Revenue)", w, float(globals_dict["terminal_growth_pct"]), m)
            m_vals.append(rr["enterprise_value"])
    m_low, m_mid, m_high = float(np.nanmin(m_vals)), float(np.nanmedian(m_vals)), float(np.nanmax(m_vals))

    # Scenario EV midpoints under current globals + each method
    ff = {
        f"{selected} – Gordon": (g_low, g_mid, g_high),
        f"{selected} – Exit Multiple": (m_low, m_mid, m_high),
    }

    # Add scenario comparison (midpoint EV using current method inputs)
    for scn_name in ["Bear", "Base", "Bull"]:
        scn_month = None
        if reporting == "Monthly":
            # if missing, use selected monthly as best effort
            scn_month = st.session_state.get("scenarios_monthly", {}).get(scn_name, scenario_monthly_sel)
        rG = build_saas_model(globals_dict, scenarios_annual[scn_name], scn_month, "Gordon Growth")
        rM = build_saas_model(globals_dict, scenarios_annual[scn_name], scn_month, "Exit Multiple (EV/Revenue)")
        ff[f"{scn_name} – Gordon (Base inputs)"] = (rG["enterprise_value"], rG["enterprise_value"], rG["enterprise_value"])
        ff[f"{scn_name} – Exit Mult (Base inputs)"] = (rM["enterprise_value"], rM["enterprise_value"], rM["enterprise_value"])

    st.pyplot(football_field_chart(ff, "Enterprise Value – Football Field"))

# =================================================
# AUDIT TAB
# =================================================
with tab_audit:
    st.subheader("🔍 Audit")
    st.caption("Select any table/metric/period to see exactly how the number is calculated (no clicking required).")
    default_period = periods[0] if periods else None
    audit_selector_ui(audit_store, default_table="PnL", default_item="EBITDA", default_period=default_period, key_prefix="audit_tab")

# =================================================
# Export Excel
# =================================================
st.divider()
st.subheader("⬇️ Export Excel Workbook")

def to_excel_bytes():
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame([globals_dict]).to_excel(writer, sheet_name="Assumptions_Global", index=False)
        df.to_excel(writer, sheet_name="Model", index=True)

        audit_rows = []
        for k, v in audit_store.items():
            t, li, per = k.split("||")
            comps = "; ".join([f"{nm}={val}" for nm, val in v.get("components", [])])
            audit_rows.append({
                "Table": t,
                "Line Item": li,
                "Period": per,
                "Formula": v.get("formula", ""),
                "Components": comps,
                "Notes": v.get("notes", "")
            })
        pd.DataFrame(audit_rows).to_excel(writer, sheet_name="Audit", index=False)

    output.seek(0)
    return output.getvalue()

st.download_button(
    "Download Excel Workbook",
    data=to_excel_bytes(),
    file_name=f"b2b_saas_model_{selected.lower()}_{reporting.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
