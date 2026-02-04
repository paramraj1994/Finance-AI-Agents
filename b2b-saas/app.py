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
st.caption("Assumptions • Financial Model • Dashboard • DCF • Export • Audit")

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
# AUDIT (robust, no clicking required)
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
# Period conversion helpers
# =================================================
def annual_pct_to_monthly_decimal(pct_annual: float) -> float:
    """Converts annual % (e.g., 12 for 12%) to monthly decimal compounding."""
    a = float(pct_annual) / 100.0
    return (1.0 + a) ** (1.0 / 12.0) - 1.0

def annual_multiplier_to_monthly_multiplier(mult_annual: float) -> float:
    """Converts annual multiplier (e.g., NRR 1.10) to monthly multiplier."""
    return float(mult_annual) ** (1.0 / 12.0)

def repeat_yearly_to_monthly(values_yearly: list, n_years: int, mode: str):
    """
    Expand year-wise assumptions into month-wise arrays of length n_years*12.
    mode:
      - "pct_growth": annual % that should compound to monthly rate (growth, churn, arpa growth)
      - "pct_level": annual % that stays same each month (GM%, cost %)
      - "nrr_pct": NRR in % as annual multiplier then take 12th root
      - "value": numeric repeated each month (CAC per new)
    """
    vals = ensure_len(values_yearly, n_years, fill=None)
    out = []
    for y in range(n_years):
        v = 0.0 if vals[y] is None else float(vals[y])
        if mode == "pct_growth":
            m = annual_pct_to_monthly_decimal(v)
            out.extend([m] * 12)
        elif mode == "pct_level":
            out.extend([v / 100.0] * 12)
        elif mode == "nrr_pct":
            annual_mult = (v / 100.0) if v > 0 else 1.0
            m_mult = annual_multiplier_to_monthly_multiplier(annual_mult)
            out.extend([m_mult] * 12)
        elif mode == "value":
            out.extend([v] * 12)
        else:
            out.extend([v] * 12)
    return np.array(out, dtype=float)

# =================================================
# Defaults
# =================================================
DEFAULT_GLOBALS = {
    "projection_years": 5,
    "reporting_period": "Annual",  # NEW
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
if "terminal_method" not in st.session_state:
    st.session_state["terminal_method"] = "Gordon Growth"
if "selected_scenario" not in st.session_state:
    st.session_state["selected_scenario"] = "Base"

# =================================================
# Tabs
# =================================================
tab_assump, tab_model, tab_dash, tab_audit = st.tabs(
    ["🧾 Assumptions", "📑 Financial Model", "📊 Dashboard", "🔍 Audit"]
)

# =================================================
# Model Builder + Audit Trace (supports Annual / Monthly)
# =================================================
def build_saas_model(globals_dict: dict, scenario: dict, terminal_method: str):
    n_years = int(globals_dict["projection_years"])
    reporting = globals_dict.get("reporting_period", "Annual")

    # Period setup
    if reporting == "Monthly":
        n = n_years * 12
        periods = [f"Month {i+1}" for i in range(n)]
        periods_per_year = 12
    else:
        n = n_years
        periods = [f"Year {i+1}" for i in range(n)]
        periods_per_year = 1

    # Base inputs
    customers_beg = float(globals_dict["base_customers"])
    arpa_annual = float(globals_dict["base_arpa"])  # annual $ per customer

    # Scenario inputs
    if reporting == "Monthly":
        new_cust_growth = repeat_yearly_to_monthly(scenario["new_cust_growth_pct"], n_years, "pct_growth")
        gross_churn     = repeat_yearly_to_monthly(scenario["gross_churn_pct"], n_years, "pct_growth")
        nrr_mult        = repeat_yearly_to_monthly(scenario["nrr_pct"], n_years, "nrr_pct")
        arpa_growth     = repeat_yearly_to_monthly(scenario["arpa_growth_pct"], n_years, "pct_growth")

        gm              = repeat_yearly_to_monthly(scenario["gross_margin_pct"], n_years, "pct_level")
        sm_pct          = repeat_yearly_to_monthly(scenario["sm_pct_rev"], n_years, "pct_level")
        rnd_pct         = repeat_yearly_to_monthly(scenario["rnd_pct_rev"], n_years, "pct_level")
        ga_pct          = repeat_yearly_to_monthly(scenario["ga_pct_rev"], n_years, "pct_level")
        cac_per_new     = repeat_yearly_to_monthly(scenario["cac_per_new"], n_years, "value")
    else:
        new_cust_growth = np.array(ensure_len(scenario["new_cust_growth_pct"], n, fill=0.0), dtype=float) / 100.0
        gross_churn     = np.array(ensure_len(scenario["gross_churn_pct"], n, fill=0.0), dtype=float) / 100.0
        nrr_mult        = np.array(ensure_len(scenario["nrr_pct"], n, fill=100.0), dtype=float) / 100.0
        arpa_growth     = np.array(ensure_len(scenario["arpa_growth_pct"], n, fill=0.0), dtype=float) / 100.0

        gm              = np.array(ensure_len(scenario["gross_margin_pct"], n, fill=75.0), dtype=float) / 100.0
        sm_pct          = np.array(ensure_len(scenario["sm_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        rnd_pct         = np.array(ensure_len(scenario["rnd_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        ga_pct          = np.array(ensure_len(scenario["ga_pct_rev"], n, fill=0.0), dtype=float) / 100.0
        cac_per_new     = np.array(ensure_len(scenario["cac_per_new"], n, fill=0.0), dtype=float)

    # Global assumptions
    tax_rate_annual = float(globals_dict["tax_rate_pct"]) / 100.0
    wacc_annual = float(globals_dict["wacc_pct"]) / 100.0
    tg_annual = float(globals_dict["terminal_growth_pct"]) / 100.0
    rev_rec = float(globals_dict["revenue_recognition_pct_of_arr"]) / 100.0
    da_pct_rev = float(globals_dict["da_pct_rev"]) / 100.0
    capex_pct_rev = float(globals_dict["capex_pct_rev"]) / 100.0
    nwc_pct_rev = float(globals_dict["nwc_pct_rev"]) / 100.0

    brand_share = float(globals_dict.get("brand_marketing_share_of_sm_pct", 30.0)) / 100.0
    brand_share = min(max(brand_share, 0.0), 1.0)

    # Periodic versions
    if reporting == "Monthly":
        wacc = (1 + wacc_annual) ** (1/12) - 1
        tg = (1 + tg_annual) ** (1/12) - 1
        tax_rate = tax_rate_annual / 12.0  # simple monthly allocation
        arpa = arpa_annual
    else:
        wacc = wacc_annual
        tg = tg_annual
        tax_rate = tax_rate_annual
        arpa = arpa_annual

    prev_nwc = 0.0
    rows = []
    audit = {}

    for i in range(n):
        period = periods[i]

        # Customer mechanics
        new_customers = customers_beg * new_cust_growth[i]
        churned_customers = customers_beg * gross_churn[i]
        customers_end = max(customers_beg + new_customers - churned_customers, 0.0)
        customers_avg = (customers_beg + customers_end) / 2.0
        net_adds = customers_end - customers_beg

        # ARPA growth
        arpa = arpa * (1 + arpa_growth[i])

        # ARR approximation (annualized)
        arr = customers_avg * arpa * nrr_mult[i]

        # Revenue recognition
        # Annual: Revenue = ARR × rev_rec
        # Monthly: Revenue = (ARR / 12) × rev_rec
        revenue = (arr * rev_rec) if reporting == "Annual" else (arr * rev_rec / 12.0)

        # Gross profit / COGS
        gross_profit = revenue * gm[i]
        cogs = revenue - gross_profit

        # Opex
        sm_total = revenue * sm_pct[i]
        brand_marketing = sm_total * brand_share
        cac_spend = new_customers * cac_per_new[i]
        acquisition_other = max(sm_total - brand_marketing - cac_spend, 0.0)
        acquisition_total = cac_spend + acquisition_other

        rnd = revenue * rnd_pct[i]
        ga = revenue * ga_pct[i]

        # EBITDA
        ebitda = gross_profit - sm_total - rnd - ga

        # D&A, EBIT, NOPAT (periodic tax)
        da = revenue * da_pct_rev
        ebit = ebitda - da
        nopat = ebit * (1 - tax_rate)

        # FCF build
        capex_val = revenue * capex_pct_rev
        nwc = revenue * nwc_pct_rev
        delta_nwc = nwc - prev_nwc
        prev_nwc_prev = prev_nwc
        prev_nwc = nwc

        fcf = nopat + da - capex_val - delta_nwc

        # CAC Payback months (works in both modes, because ARPA is annual)
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

        # ------- AUDIT TRACES -------
        audit[audit_key("Drivers", "ARR", period)] = audit_pack(
            "ARR = Avg Customers × ARPA × NRR",
            [
                ("Customers Beg", customers_beg),
                ("Customers End", customers_end),
                ("Avg Customers", customers_avg),
                ("ARPA (annual)", arpa),
                ("NRR", f"{(nrr_mult[i]*100):.2f}%"),
            ],
            notes="ARR is an annualized run-rate. Monthly revenue uses ARR/12."
        )
        audit[audit_key("Drivers", "Revenue", period)] = audit_pack(
            "Revenue = ARR × Revenue Recognition" + (" / 12 (monthly)" if reporting == "Monthly" else ""),
            [("ARR", arr), ("Revenue Recognition", f"{(rev_rec*100):.2f}%"), ("Divide by 12?", "Yes" if reporting == "Monthly" else "No")],
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

    # Discount factors
    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    # Terminal value (computed from last period)
    last_period = periods[-1]

    if terminal_method == "Gordon Growth":
        if wacc <= tg:
            tv = np.nan
            tv_formula = "TV invalid: discount rate must be greater than terminal growth"
            tv_components = [("r", f"{wacc*100:.2f}%"), ("g", f"{tg*100:.2f}%")]
        else:
            fcf_last = float(df["FCF"].iloc[-1])
            tv = fcf_last * (1 + tg) / (wacc - tg)
            tv_formula = "TV (Gordon) = FCF_last × (1 + g) / (r − g)"
            tv_components = [("FCF_last", fcf_last), ("g", f"{tg*100:.2f}%"), ("r", f"{wacc*100:.2f}%")]
    else:
        # Exit multiple uses annualized revenue if monthly
        revenue_last = float(df["Revenue"].iloc[-1])
        revenue_for_tv = revenue_last * (12.0 if reporting == "Monthly" else 1.0)
        tv = revenue_for_tv * float(globals_dict["exit_multiple_ev_rev"])
        tv_formula = "TV (Exit Multiple) = Revenue_last(annualized) × Exit Multiple"
        tv_components = [("Revenue_last (annualized)", revenue_for_tv), ("Exit Multiple", float(globals_dict["exit_multiple_ev_rev"]))]

    pv_tv = tv * discount_factors[-1] if np.isfinite(tv) else np.nan
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
    ev = pv_explicit + (float(pv_tv) if np.isfinite(pv_tv) else 0.0)

    equity_value = ev - float(globals_dict["net_debt"]) + float(globals_dict["cash"])
    tv_share = (pv_tv / ev) if ev != 0 and np.isfinite(pv_tv) else np.nan

    audit[audit_key("Valuation", "Terminal Value", last_period)] = audit_pack(tv_formula, tv_components)
    audit[audit_key("Valuation", "PV of Terminal Value", last_period)] = audit_pack(
        "PV(TV) = TV × Discount Factor (last period)",
        [("TV", float(tv) if np.isfinite(tv) else "NA"), ("Discount Factor", float(discount_factors[-1]))],
    )
    audit[audit_key("Valuation", "Enterprise Value", last_period)] = audit_pack(
        "EV = PV Explicit FCFs + PV(TV)",
        [("PV Explicit", pv_explicit), ("PV(TV)", float(pv_tv) if np.isfinite(pv_tv) else "NA")],
    )
    audit[audit_key("Valuation", "Equity Value", last_period)] = audit_pack(
        "Equity Value = EV − Net Debt + Cash",
        [("EV", ev), ("Net Debt", float(globals_dict["net_debt"])), ("Cash", float(globals_dict["cash"]))],
    )

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
    }

# =================================================
# ASSUMPTIONS TAB
# =================================================
with tab_assump:
    st.subheader("🧾 Assumptions")

    st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
    g = st.session_state["globals"]

    # NEW: Reporting toggle
    g["reporting_period"] = st.radio(
        "Reporting Period",
        ["Annual", "Monthly"],
        horizontal=True,
        index=0 if g.get("reporting_period", "Annual") == "Annual" else 1,
        help="Annual shows Year 1..N. Monthly runs 12× periods and converts year-wise assumptions into monthly equivalents."
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

    st.markdown("### Terminal Method")
    st.session_state["terminal_method"] = st.radio(
        "Choose terminal method",
        ["Gordon Growth", "Exit Multiple (EV/Revenue)"],
        horizontal=True,
        index=0 if st.session_state.get("terminal_method", "Gordon Growth") == "Gordon Growth" else 1
    )

    # Validation note for Gordon
    wacc_a = float(g["wacc_pct"]) / 100.0
    tg_a = float(g["terminal_growth_pct"]) / 100.0
    if st.session_state["terminal_method"] == "Gordon Growth" and wacc_a <= tg_a:
        st.error("❗ For Gordon Growth terminal value: WACC must be greater than Terminal Growth (annual inputs).")

    st.divider()

    st.markdown("### Scenario Assumptions (Year-wise)")
    st.session_state["selected_scenario"] = st.radio(
        "Select scenario to edit / run",
        ["Bear", "Base", "Bull"],
        horizontal=True,
        index=["Bear", "Base", "Bull"].index(st.session_state.get("selected_scenario", "Base")),
    )
    selected_scn = st.session_state["selected_scenario"]

    n_years = int(st.session_state["globals"]["projection_years"])
    st.session_state["scenarios"] = normalize_scenarios(st.session_state.get("scenarios", {}), n_years)

    years_labels = [f"Year {i+1}" for i in range(n_years)]
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

    st.caption("Edit year-wise values. Monthly mode converts these into monthly equivalents automatically.")
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
            st.session_state["scenarios"][selected_scn][key] = ensure_len(values, n_years, fill=0.0)
        st.success(f"{selected_scn} scenario saved.")

# =================================================
# Compute model
# =================================================
globals_dict = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
terminal_method = st.session_state.get("terminal_method", "Gordon Growth")
selected = st.session_state.get("selected_scenario", "Base")

n_years = int(globals_dict["projection_years"])
st.session_state["scenarios"] = normalize_scenarios(st.session_state.get("scenarios", {}), n_years)

result = build_saas_model(globals_dict, st.session_state["scenarios"][selected], terminal_method)
df = result["df"]
periods = result["periods"]
audit_store = result["audit"]
reporting = result["reporting_period"]

# =================================================
# FINANCIAL MODEL TAB
# =================================================
with tab_model:
    st.subheader(f"📑 Financial Model – {selected} Scenario ({reporting})")

    with st.expander("🔍 Audit Mode (works without clicking)"):
        default_period = periods[0] if periods else None
        audit_selector_ui(audit_store, default_table="PnL", default_item="EBITDA", default_period=default_period, key_prefix="fm_audit")

    sub_core, sub_acq, sub_mkt = st.tabs(["📑 Core Statements", "👥 Subscriber Acquisition", "📣 Marketing Spend"])

    with sub_core:
        st.subheader("📑 Profit & Loss")
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

        st.subheader("💵 Cash Flow")
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

        st.subheader("📘 Valuation Summary")
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

    with sub_acq:
        st.subheader("👥 Subscriber Acquisition")
        acq = pd.DataFrame({"Line Item": [
            "Customers (Beg)", "New Customers", "Churned Customers", "Net Adds", "Customers (End)",
            "CAC Spend", "CAC Payback (months)"
        ]})
        for p in periods:
            acq[p] = [
                fmt_float2(df.loc[p, "Customers (Beg)"]),
                fmt_float2(df.loc[p, "New Customers"]),
                fmt_float2(df.loc[p, "Churned Customers"]),
                fmt_float2(df.loc[p, "Net Adds"]),
                fmt_float2(df.loc[p, "Customers (End)"]),
                fmt_currency(df.loc[p, "CAC Spend"]),
                fmt_float2(df.loc[p, "CAC Payback (months)"]) if np.isfinite(df.loc[p, "CAC Payback (months)"]) else "",
            ]
        st.dataframe(acq, hide_index=True, use_container_width=True)

    with sub_mkt:
        st.subheader("📣 Marketing Spend")
        mkt = pd.DataFrame({"Line Item": [
            "Sales & Marketing (Total)", "Brand Marketing", "CAC Spend", "Acquisition Other", "Acquisition (Total)"
        ]})
        for p in periods:
            mkt[p] = [
                fmt_currency(df.loc[p, "Sales & Marketing (Total)"]),
                fmt_currency(df.loc[p, "Brand Marketing"]),
                fmt_currency(df.loc[p, "CAC Spend"]),
                fmt_currency(df.loc[p, "Acquisition Other"]),
                fmt_currency(df.loc[p, "Acquisition (Total)"]),
            ]
        st.dataframe(mkt, hide_index=True, use_container_width=True)

# =================================================
# DASHBOARD TAB
# =================================================
with tab_dash:
    st.subheader(f"📊 Dashboard – {selected} Scenario ({reporting})")

    # Keep dashboard readable: only chart first up to 24 months in monthly mode
    if reporting == "Monthly" and len(periods) > 24:
        chart_periods = periods[:24]
        dff = df.loc[chart_periods].copy()
        x_labels = chart_periods
        st.caption("Showing first 24 months for charts (to keep charts readable).")
    else:
        dff = df.copy()
        x_labels = periods

    st.markdown("### Revenue")
    fig, ax = plt.subplots()
    x_pos = np.arange(len(x_labels))
    ax.plot(x_pos, dff["Revenue"].values, marker="o", label="Revenue")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(False)
    add_point_labels(ax, x_pos, dff["Revenue"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### EBITDA")
    fig, ax = plt.subplots()
    ax.plot(x_pos, dff["EBITDA"].values, marker="o", label="EBITDA")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(False)
    add_point_labels(ax, x_pos, dff["EBITDA"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("### Free Cash Flow (FCF)")
    fig, ax = plt.subplots()
    ax.plot(x_pos, dff["FCF"].values, marker="o", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.grid(False)
    add_point_labels(ax, x_pos, dff["FCF"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

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
