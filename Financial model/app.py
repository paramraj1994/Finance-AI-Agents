import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -------------------------------------------------
# Page Setup
# -------------------------------------------------
st.set_page_config(page_title="Financial Model – Investor Dashboard", layout="wide")
st.title("📊 Financial Model – Investor Dashboard")
st.caption("Scenario-based P&L, Sensitivity, Cash Flow & Valuation")

# -------------------------------------------------
# Sidebar – Global Assumptions
# -------------------------------------------------
st.sidebar.header("🔧 Core Assumptions")

projection_years = st.sidebar.slider("Projection Years", 3, 10, 5)
tax_rate = st.sidebar.slider("Tax Rate (%)", 0.15, 0.40, 0.25, 0.01)
discount_rate = st.sidebar.slider("Discount Rate (WACC)", 0.08, 0.18, 0.12, 0.01)
terminal_growth = st.sidebar.slider("Terminal Growth Rate", 0.02, 0.06, 0.04, 0.005)

base_revenue = st.sidebar.number_input("Last Historical Revenue", 100.0, value=1000.0)

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
# Model Function
# -------------------------------------------------
def build_model(growth, margin):
    revenue = base_revenue
    rows = []

    for _ in years:
        revenue *= (1 + growth)
        ebitda = revenue * margin
        tax = ebitda * tax_rate
        pat = ebitda - tax
        fcf = pat * 0.9  # simple proxy

        rows.append([revenue, ebitda, tax, pat, fcf])

    df = pd.DataFrame(
        rows,
        columns=["Revenue", "EBITDA", "Tax", "PAT", "FCF"],
        index=years
    )
    return df

# -------------------------------------------------
# Build Scenario Models
# -------------------------------------------------
models = {name: build_model(**params) for name, params in scenarios.items()}

# -------------------------------------------------
# Scenario Comparison – Interactive Chart
# -------------------------------------------------
st.subheader("📈 Scenario Comparison – Revenue & PAT")

fig = go.Figure()
for name, df in models.items():
    fig.add_trace(go.Scatter(x=df.index, y=df["Revenue"], mode="lines+markers", name=f"{name} Revenue"))
    fig.add_trace(go.Scatter(x=df.index, y=df["PAT"], mode="lines+markers", name=f"{name} PAT"))

fig.update_layout(title="Scenario Comparison", xaxis_title="Year", yaxis_title="Amount")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Scenario Tables
# -------------------------------------------------
st.subheader("📑 Scenario Financials")

tabs = st.tabs(models.keys())
for tab, (name, df) in zip(tabs, models.items()):
    with tab:
        st.dataframe(df.style.format("{:,.0f}"))

# -------------------------------------------------
# Sensitivity Analysis – Heatmap
# -------------------------------------------------
st.subheader("🔥 Sensitivity Analysis – PAT")

growth_range = np.arange(0.08, 0.20, 0.02)
margin_range = np.arange(0.20, 0.35, 0.03)

heatmap = pd.DataFrame(index=[f"{m:.0%}" for m in margin_range],
                       columns=[f"{g:.0%}" for g in growth_range])

for g in growth_range:
    for m in margin_range:
        df = build_model(g, m)
        heatmap.loc[f"{m:.0%}", f"{g:.0%}"] = df["PAT"].sum()

heatmap = heatmap.astype(float)

fig, ax = plt.subplots()
im = ax.imshow(heatmap.values)
ax.set_xticks(range(len(heatmap.columns)))
ax.set_yticks(range(len(heatmap.index)))
ax.set_xticklabels(heatmap.columns)
ax.set_yticklabels(heatmap.index)
ax.set_title("Cumulative PAT Sensitivity")
plt.colorbar(im)
st.pyplot(fig)

# -------------------------------------------------
# DCF Valuation
# -------------------------------------------------
st.subheader("💰 Valuation – DCF (Base Case)")

base_df = models["Base"]

discount_factors = [(1 / (1 + discount_rate) ** (i + 1)) for i in range(projection_years)]
base_df["Discount Factor"] = discount_factors
base_df["PV of FCF"] = base_df["FCF"] * base_df["Discount Factor"]

terminal_value = (
    base_df["FCF"].iloc[-1] * (1 + terminal_growth)
    / (discount_rate - terminal_growth)
)

pv_terminal = terminal_value * discount_factors[-1]
enterprise_value = base_df["PV of FCF"].sum() + pv_terminal

st.metric("Enterprise Value", f"{enterprise_value:,.0f}")

# -------------------------------------------------
# Valuation Bridge Chart
# -------------------------------------------------
fig = go.Figure()
fig.add_bar(x=base_df.index, y=base_df["PV of FCF"], name="PV of FCF")
fig.add_bar(x=["Terminal"], y=[pv_terminal], name="PV of Terminal Value")
fig.update_layout(title="DCF Value Bridge")
st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------
# Download Model
# -------------------------------------------------
st.subheader("⬇️ Download Base Case Model")

csv = base_df.reset_index().rename(columns={"index": "Year"}).to_csv(index=False)

st.download_button(
    "Download Base Case CSV",
    csv,
    "Base_Case_Financial_Model.csv",
    "text/csv"
)
