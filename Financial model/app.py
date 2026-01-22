import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Interactive Financial Model",
    layout="wide"
)

st.title("📊 Interactive Financial Model")
st.write("Upload your financial Excel file and adjust assumptions to see projections.")

# -------------------------------------------------
# File Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload your Excel file",
    type=["xlsx"]
)

if uploaded_file is None:
    st.info("Please upload an Excel file to continue.")
    st.stop()

# -------------------------------------------------
# Load Excel
# -------------------------------------------------
try:
    df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error("Unable to read the Excel file.")
    st.stop()

st.subheader("📄 Uploaded Excel Data")
st.dataframe(df)

# -------------------------------------------------
# Assumptions Sidebar
# -------------------------------------------------
st.sidebar.header("🔧 Model Assumptions")

revenue_growth = st.sidebar.slider(
    "Revenue Growth (%)",
    min_value=0.0,
    max_value=0.5,
    value=0.10,
    step=0.01
)

ebitda_margin = st.sidebar.slider(
    "EBITDA Margin (%)",
    min_value=0.05,
    max_value=0.6,
    value=0.25,
    step=0.01
)

tax_rate = st.sidebar.slider(
    "Tax Rate (%)",
    min_value=0.1,
    max_value=0.4,
    value=0.25,
    step=0.01
)

# -------------------------------------------------
# Base Revenue Input
# -------------------------------------------------
st.subheader("📌 Base Inputs")

base_revenue = st.number_input(
    "Base Year Revenue",
    min_value=0.0,
    value=1000.0,
    step=100.0
)

forecast_years = st.multiselect(
    "Select Forecast Years",
    options=["2026", "2027", "2028", "2029", "2030"],
    default=["2026", "2027", "2028"]
)

if not forecast_years:
    st.warning("Please select at least one forecast year.")
    st.stop()

# -------------------------------------------------
# Financial Model Logic
# -------------------------------------------------
projections = []
revenue = base_revenue

for year in forecast_years:
    revenue = revenue * (1 + revenue_growth)
    ebitda = revenue * ebitda_margin
    tax = ebitda * tax_rate
    pat = ebitda - tax

    projections.append({
        "Year": year,
        "Revenue": revenue,
        "EBITDA": ebitda,
        "PAT": pat
    })

model_df = pd.DataFrame(projections)

# -------------------------------------------------
# Output Table
# -------------------------------------------------
st.subheader("📈 Financial Projections")

st.dataframe(
    model_df.style.format({
        "Revenue": "{:,.0f}",
        "EBITDA": "{:,.0f}",
        "PAT": "{:,.0f}"
    })
)

# -------------------------------------------------
# Charts
# -------------------------------------------------
st.subheader("📉 Financial Performance Chart")

fig, ax = plt.subplots()
ax.plot(model_df["Year"], model_df["Revenue"], marker="o", label="Revenue")
ax.plot(model_df["Year"], model_df["EBITDA"], marker="o", label="EBITDA")
ax.plot(model_df["Year"], model_df["PAT"], marker="o", label="PAT")

ax.set_xlabel("Year")
ax.set_ylabel("Amount")
ax.legend()
ax.grid(True)

st.pyplot(fig)

# -------------------------------------------------
# Download Output
# -------------------------------------------------
st.subheader("⬇️ Download Projections")

csv = model_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Projections as CSV",
    data=csv,
    file_name="financial_projections.csv",
    mime="text/csv"
)
