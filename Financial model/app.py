import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Investor Dashboard", layout="wide")
st.title("📊 Financial Model – Investor Dashboard")
st.caption("Scenario modeling, sensitivity analysis & valuation")

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Historical Financials (Excel)",
    type=["xlsx"]
)

if uploaded_file is not None:
    hist_df = pd.read_excel(uploaded_file)
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(hist_df)
else:
    st.info("Upload an Excel file (optional). You can still run projections manually.")

# -------------------------------------------------
# Sidebar Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.15, 0.40, 0.25, 0.01)
discount_rate = st.sidebar.slider("Discount Rate (WACC)", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

base_revenue = st.sidebar.number_input(
    "Last Historical Revenue",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

# -------------------------------------------------
# Scenario Definitions
# -------------------------------------------------
scenarios = {
    "Bear": {"growth": 0.07, "margin": 0.20},
    "Base": {"growth": 0.12, "margin": 0.25},
    "Bull": {"growth": 0.18, "margin": 0.30},
}

years = [f"Y{i+1}" for i in range(projection_years)]

# -------------------------------------------------
# Financial Model Function
# -------------------------------------------------
def build_model(growth, margin):
    revenue = base_revenue
    rows = []

    for _ in years:
        revenue *= (1 + growth)
        ebitda = revenue * margin
        tax = ebitda * tax_rate
        pat = ebitda - tax
        fcf = pat * 0.9  # proxy

        rows.append([revenue, ebitda, tax, pat, fcf])

    return pd.DataFrame(
        rows,
        columns=["Revenue", "EBITDA", "Tax", "PAT", "FCF"],
        index=years
    )

models = {name: build_model(**params) for name, params in scenarios.items()}

# -------------------------------------------------
# Scenario Comparison Chart
# -------------------------------------------------
st.subheader("📈 Scenario Comparison")

fig, ax = plt.subplots()
for name, df in models.items():
    ax.plot(df.index, df["Revenue"], marker="o", label=f"{name} Revenue")
    ax.plot(df.index, df["PAT"], linestyle="--", marker="o", label=f"{name} PAT")

ax.set_title("Revenue & PAT by Scenario")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# -------------------------------------------------
# Scenario Tables
# -------------------------------------------------
st.subheader("📑 Scenario Financials")

tabs = st.tabs(models.keys())
for tab, (name, df) in zip(tabs, models.items()):
    with tab:
        st.dataframe(df.style.format("{:,.0f}"))

# -------------------------------------------------
# Sensitivity Analysis
# -------------------------------------------------
st.subheader("🔥 Sensitivity Analysis – Cumulative PAT")

growth_range = np.arange(0.08, 0.20, 0.02)
margin_range = np.arange(0.20, 0.35, 0.03)

heatmap = pd.DataFrame(index=margin_range, columns=growth_range)

for g in growth_range:
    for m in margin_range:
        heatmap.loc[m, g] = build_model(g, m)["PAT"].sum()

heatmap = heatmap.astype(float)

fig, ax = plt.subplots()
im = ax.imshow(heatmap.values)
ax.set_xticks(range(len(growth_range)))
ax.set_yticks(range(len(margin_range)))
ax.set_xticklabels([f"{g:.0%}" for g in growth_range])
ax.set_yticklabels([f"{m:.0%}" for m in margin_range])
ax.set_title("PAT Sensitivity")
plt.colorbar(im)
st.pyplot(fig)

# -------------------------------------------------
# DCF Valuation
# -------------------------------------------------
st.subheader("💰 DCF Valuation (Base Case)")

base_df = models["Base"]
discount_factors = [(1 / (1 + discount_rate) ** (i + 1)) for i in range(projection_years)]

base_df["PV of FCF"] = base_df["FCF"] * discount_factors

terminal_value = (
    base_df["FCF"].iloc[-1] * (1 + terminal_growth)
    / (discount_rate - terminal_growth)
)

enterprise_value = base_df["PV of FCF"].sum() + terminal_value * discount_factors[-1]

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Download
# -------------------------------------------------
st.subheader("⬇️ Download Base Case")

csv = base_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False)

st.download_button(
    "Download Base Case CSV",
    csv,
    "Base_Case_Model.csv",
    "text/csv"
)
