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
def audit_key(table: str, line_item: str, year: str) -> str:
    return f"{table}||{line_item}||{year}"

def audit_pack(formula: str, components: list[tuple[str, float | str]], notes: str = "") -> dict:
    return {"formula": formula, "components": components, "notes": notes}

def show_audit_panel(audit_store: dict, table: str, line_item: str, year: str):
    k = audit_key(table, line_item, year)
    payload = audit_store.get(k)

    st.markdown("### 🔍 Audit Trace")
    st.write(f"**Table:** {table}  |  **Line Item:** {line_item}  |  **Year:** {year}")

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

def audit_selector_ui(audit_store: dict, default_table="PnL", default_item="EBITDA", default_year="Year 1", key_prefix="audit"):
    keys = list(audit_store.keys())
    if not keys:
        st.info("No audit traces found.")
        return

    parts = [k.split("||") for k in keys]
    tables = sorted(list({p[0] for p in parts}))
    table = st.selectbox("Table", tables, index=tables.index(default_table) if default_table in tables else 0, key=f"{key_prefix}_table")

    items = sorted(list({p[1] for p in parts if p[0] == table}))
    item = st.selectbox("Line Item", items, index=items.index(default_item) if default_item in items else 0, key=f"{key_prefix}_item")

    years = sorted(list({p[2] for p in parts if p[0] == table and p[1] == item}))
    year = st.selectbox("Year", years, index=years.index(default_year) if default_year in years else 0, key=f"{key_prefix}_year")

    show_audit_panel(audit_store, table, item, year)

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
# Session State Init (with migration)
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
# Model Builder + Audit Trace
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

    brand_share = float(globals_dict.get("brand_marketing_share_of_sm_pct", 30.0)) / 100.0
    brand_share = min(max(brand_share, 0.0), 1.0)

    prev_nwc = 0.0
    rows = []
    audit = {}

    for i in range(n):
        year = years[i]

        new_customers = customers_beg * new_cust_growth[i]
        churned_customers = customers_beg * gross_churn[i]
        customers_end = max(customers_beg + new_customers - churned_customers, 0.0)
        customers_avg = (customers_beg + customers_end) / 2.0
        net_adds = customers_end - customers_beg

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
        prev_nwc_prev = prev_nwc
        prev_nwc = nwc

        fcf = nopat + da - capex_val - delta_nwc

        gp_per_cust_per_month = (arpa * gm[i]) / 12.0 if (arpa * gm[i]) > 0 else np.nan
        cac_payback_months = (cac[i] / gp_per_cust_per_month) if (gp_per_cust_per_month and np.isfinite(gp_per_cust_per_month)) else np.nan

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
        audit[audit_key("Drivers", "ARR", year)] = audit_pack(
            "ARR = Avg Customers × ARPA × NRR",
            [
                ("Customers Beg", customers_beg),
                ("Customers End", customers_end),
                ("Avg Customers", customers_avg),
                ("ARPA", arpa),
                ("NRR", f"{(nrr[i]*100):.2f}%"),
            ],
        )
        audit[audit_key("Drivers", "Revenue", year)] = audit_pack(
            "Revenue = ARR × Revenue Recognition",
            [("ARR", arr), ("Revenue Recognition", f"{(rev_rec*100):.2f}%")],
        )
        audit[audit_key("PnL", "Gross Profit", year)] = audit_pack(
            "Gross Profit = Revenue × Gross Margin",
            [("Revenue", revenue), ("Gross Margin", f"{(gm[i]*100):.2f}%")],
        )
        audit[audit_key("PnL", "COGS", year)] = audit_pack(
            "COGS = Revenue − Gross Profit",
            [("Revenue", revenue), ("Gross Profit", gross_profit)],
        )
        audit[audit_key("PnL", "Sales & Marketing (Total)", year)] = audit_pack(
            "S&M (Total) = Revenue × S&M % of Revenue",
            [("Revenue", revenue), ("S&M %", f"{(sm_pct[i]*100):.2f}%")],
        )
        audit[audit_key("Marketing", "Brand Marketing", year)] = audit_pack(
            "Brand Marketing = S&M (Total) × Brand Share",
            [("S&M (Total)", sm_total), ("Brand Share", f"{(brand_share*100):.2f}%")],
        )
        audit[audit_key("Marketing", "CAC Spend", year)] = audit_pack(
            "CAC Spend = New Customers × CAC per New Customer",
            [("New Customers", new_customers), ("CAC per New", cac[i])],
        )
        audit[audit_key("Marketing", "Acquisition Other", year)] = audit_pack(
            "Acquisition Other = max(S&M − Brand Marketing − CAC Spend, 0)",
            [("S&M (Total)", sm_total), ("Brand Marketing", brand_marketing), ("CAC Spend", cac_spend), ("Floor", "0")],
        )
        audit[audit_key("Marketing", "Acquisition (Total)", year)] = audit_pack(
            "Acquisition (Total) = CAC Spend + Acquisition Other",
            [("CAC Spend", cac_spend), ("Acquisition Other", acquisition_other)],
        )
        audit[audit_key("PnL", "R&D", year)] = audit_pack(
            "R&D = Revenue × R&D % of Revenue",
            [("Revenue", revenue), ("R&D %", f"{(rnd_pct[i]*100):.2f}%")],
        )
        audit[audit_key("PnL", "G&A", year)] = audit_pack(
            "G&A = Revenue × G&A % of Revenue",
            [("Revenue", revenue), ("G&A %", f"{(ga_pct[i]*100):.2f}%")],
        )
        audit[audit_key("PnL", "EBITDA", year)] = audit_pack(
            "EBITDA = Gross Profit − S&M − R&D − G&A",
            [("Gross Profit", gross_profit), ("S&M", sm_total), ("R&D", rnd), ("G&A", ga)],
        )
        audit[audit_key("PnL", "D&A", year)] = audit_pack(
            "D&A = Revenue × D&A % of Revenue",
            [("Revenue", revenue), ("D&A %", f"{(da_pct_rev*100):.2f}%")],
        )
        audit[audit_key("PnL", "EBIT", year)] = audit_pack(
            "EBIT = EBITDA − D&A",
            [("EBITDA", ebitda), ("D&A", da)],
        )
        audit[audit_key("PnL", "NOPAT", year)] = audit_pack(
            "NOPAT = EBIT × (1 − Tax Rate)",
            [("EBIT", ebit), ("Tax Rate", f"{(tax*100):.2f}%")],
        )
        audit[audit_key("Cash Flow", "Capex", year)] = audit_pack(
            "Capex = Revenue × Capex % of Revenue",
            [("Revenue", revenue), ("Capex %", f"{(capex_pct_rev*100):.2f}%")],
        )
        audit[audit_key("Cash Flow", "ΔNWC", year)] = audit_pack(
            "ΔNWC = NWC(t) − NWC(t−1), NWC = Revenue × NWC % of Revenue",
            [("NWC(t)", nwc), ("NWC(t−1)", prev_nwc_prev), ("NWC %", f"{(nwc_pct_rev*100):.2f}%")],
        )
        audit[audit_key("Cash Flow", "FCF", year)] = audit_pack(
            "FCF = NOPAT + D&A − Capex − ΔNWC",
            [("NOPAT", nopat), ("D&A", da), ("Capex", capex_val), ("ΔNWC", delta_nwc)],
        )
        audit[audit_key("Acquisition", "CAC Payback (months)", year)] = audit_pack(
            "CAC Payback (months) = CAC per New / ((ARPA × GM)/12)",
            [("CAC per New", cac[i]), ("ARPA", arpa), ("GM", f"{(gm[i]*100):.2f}%"), ("Monthly GP/customer", gp_per_cust_per_month if np.isfinite(gp_per_cust_per_month) else "NA")],
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
        index=years
    )

    df["Gross Margin (%)"] = np.where(df["Revenue"] > 0, (df["Gross Profit"] / df["Revenue"]) * 100, np.nan)
    df["EBITDA Margin (%)"] = np.where(df["Revenue"] > 0, (df["EBITDA"] / df["Revenue"]) * 100, np.nan)

    discount_factors = np.array([1 / ((1 + wacc) ** (i + 1)) for i in range(n)], dtype=float)
    df["Discount Factor"] = discount_factors
    df["PV of FCF"] = df["FCF"].values * discount_factors

    if terminal_method == "Gordon Growth":
        tv = np.nan if wacc <= tg else (df["FCF"].iloc[-1] * (1 + tg) / (wacc - tg))
        tv_formula = "TV (Gordon) = FCF_last × (1 + g) / (WACC − g)"
        tv_components = [("FCF_last", float(df["FCF"].iloc[-1])), ("g", f"{tg*100:.2f}%"), ("WACC", f"{wacc*100:.2f}%")]
    else:
        tv = df["Revenue"].iloc[-1] * float(globals_dict["exit_multiple_ev_rev"])
        tv_formula = "TV (Exit Multiple) = Revenue_last × Exit Multiple"
        tv_components = [("Revenue_last", float(df["Revenue"].iloc[-1])), ("Exit Multiple", float(globals_dict["exit_multiple_ev_rev"]))]

    pv_tv = tv * discount_factors[-1] if np.isfinite(tv) else np.nan
    pv_explicit = float(np.nansum(df["PV of FCF"].values))
    ev = pv_explicit + (float(pv_tv) if np.isfinite(pv_tv) else 0.0)

    equity_value = ev - float(globals_dict["net_debt"]) + float(globals_dict["cash"])
    tv_share = (pv_tv / ev) if ev != 0 and np.isfinite(pv_tv) else np.nan

    yearN = years[-1]
    audit[audit_key("Valuation", "Terminal Value", yearN)] = audit_pack(tv_formula, tv_components)
    audit[audit_key("Valuation", "PV of Terminal Value", yearN)] = audit_pack(
        "PV(TV) = TV × Discount Factor (Year N)",
        [("TV", float(tv) if np.isfinite(tv) else "NA"), ("Discount Factor (Year N)", float(discount_factors[-1]))],
    )
    audit[audit_key("Valuation", "PV of Explicit FCFs", yearN)] = audit_pack(
        "PV of Explicit FCFs = Σ PV(FCF_t)",
        [("Sum PV of FCF", pv_explicit)],
    )
    audit[audit_key("Valuation", "Enterprise Value", yearN)] = audit_pack(
        "EV = PV Explicit FCFs + PV(TV)",
        [("PV Explicit", pv_explicit), ("PV(TV)", float(pv_tv) if np.isfinite(pv_tv) else "NA")],
    )
    audit[audit_key("Valuation", "Equity Value", yearN)] = audit_pack(
        "Equity Value = EV − Net Debt + Cash",
        [("EV", ev), ("Net Debt", float(globals_dict["net_debt"])), ("Cash", float(globals_dict["cash"]))],
    )

    return {
        "years": years,
        "df": df,
        "audit": audit,
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
    st.subheader("🧾 Assumptions")

    st.session_state["globals"] = merge_defaults(DEFAULT_GLOBALS, st.session_state.get("globals", {}))
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
    st.session_state["globals"] = g

    st.markdown("### Terminal Method")
    st.session_state["terminal_method"] = st.radio(
        "Choose terminal method",
        ["Gordon Growth", "Exit Multiple (EV/Revenue)"],
        horizontal=True,
        index=0 if st.session_state.get("terminal_method", "Gordon Growth") == "Gordon Growth" else 1
    )

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
terminal_method = st.session_state.get("terminal_method", "Gordon Growth")
selected = st.session_state.get("selected_scenario", "Base")

n = int(globals_dict["projection_years"])
st.session_state["scenarios"] = normalize_scenarios(st.session_state.get("scenarios", {}), n)

result = build_saas_model(globals_dict, st.session_state["scenarios"][selected], terminal_method)
df = result["df"]
years = result["years"]
audit_store = result["audit"]

all_results = {
    name: build_saas_model(globals_dict, st.session_state["scenarios"][name], terminal_method)
    for name in ["Bear", "Base", "Bull"]
}

# =================================================
# FINANCIAL MODEL TAB
# =================================================
with tab_model:
    st.subheader(f"📑 Financial Model – {selected} Scenario")

    # Audit mode panel (works even when click doesn't)
    with st.expander("🔍 Audit Mode (works without clicking)"):
        audit_selector_ui(audit_store, default_table="PnL", default_item="EBITDA", default_year="Year 1", key_prefix="fm_audit")

    sub_core, sub_acq, sub_mkt = st.tabs(["📑 Core Statements", "👥 Subscriber Acquisition", "📣 Marketing Spend"])

    with sub_core:
        st.subheader("📑 Profit & Loss")
        pnl = pd.DataFrame({"Line Item": [
            "Revenue", "COGS", "Gross Profit",
            "Sales & Marketing (Total)", "R&D", "G&A",
            "EBITDA", "D&A", "EBIT", "NOPAT"
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
            ]
        st.dataframe(pnl, hide_index=True, use_container_width=True)

        st.subheader("💵 Cash Flow")
        cf = pd.DataFrame({"Line Item": ["NOPAT", "D&A", "Capex", "ΔNWC", "FCF"]})
        for y in years:
            cf[y] = [
                fmt_currency(df.loc[y, "NOPAT"]),
                fmt_currency(df.loc[y, "D&A"]),
                fmt_currency(df.loc[y, "Capex"]),
                fmt_currency(df.loc[y, "ΔNWC"]),
                fmt_currency(df.loc[y, "FCF"]),
            ]
        st.dataframe(cf, hide_index=True, use_container_width=True)

        st.subheader("📘 Valuation Summary")
        yearN = years[-1]
        val = pd.DataFrame({
            "Line Item": ["Terminal Value", "PV of Terminal Value", "PV of Explicit FCFs", "Enterprise Value", "Equity Value"],
            yearN: [
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
        for y in years:
            acq[y] = [
                fmt_float2(df.loc[y, "Customers (Beg)"]),
                fmt_float2(df.loc[y, "New Customers"]),
                fmt_float2(df.loc[y, "Churned Customers"]),
                fmt_float2(df.loc[y, "Net Adds"]),
                fmt_float2(df.loc[y, "Customers (End)"]),
                fmt_currency(df.loc[y, "CAC Spend"]),
                fmt_float2(df.loc[y, "CAC Payback (months)"]) if np.isfinite(df.loc[y, "CAC Payback (months)"]) else "",
            ]
        st.dataframe(acq, hide_index=True, use_container_width=True)

    with sub_mkt:
        st.subheader("📣 Marketing Spend")
        mkt = pd.DataFrame({"Line Item": [
            "Sales & Marketing (Total)", "Brand Marketing", "CAC Spend", "Acquisition Other", "Acquisition (Total)"
        ]})
        for y in years:
            mkt[y] = [
                fmt_currency(df.loc[y, "Sales & Marketing (Total)"]),
                fmt_currency(df.loc[y, "Brand Marketing"]),
                fmt_currency(df.loc[y, "CAC Spend"]),
                fmt_currency(df.loc[y, "Acquisition Other"]),
                fmt_currency(df.loc[y, "Acquisition (Total)"]),
            ]
        st.dataframe(mkt, hide_index=True, use_container_width=True)

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
    ax.plot(x_pos, df["FCF"].values, marker="o", label="FCF")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(years)
    ax.grid(False)
    add_point_labels(ax, x_pos, df["FCF"].values, is_pct=False)
    ax.legend(loc="best")
    fig.tight_layout()
    st.pyplot(fig)

# =================================================
# AUDIT TAB (Works everywhere)
# =================================================
with tab_audit:
    st.subheader("🔍 Audit")
    st.caption("Select any table/metric/year to see exactly how the number is calculated (no clicking required).")
    audit_selector_ui(audit_store, default_table="PnL", default_item="EBITDA", default_year="Year 1", key_prefix="audit_tab")

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
            t, li, yr = k.split("||")
            comps = "; ".join([f"{nm}={val}" for nm, val in v.get("components", [])])
            audit_rows.append({
                "Table": t,
                "Line Item": li,
                "Year": yr,
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
    file_name=f"b2b_saas_model_{selected.lower()}.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
