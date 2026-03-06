import streamlit as st
import pandas as pd

st.set_page_config(page_title="FP&A Monthly Reporting", layout="wide")

st.title("📊 FP&A Monthly Reporting Dashboard")

st.markdown("""
Upload your financial dataset to analyze:
- Monthly Performance
- Budget vs Actual Variance
- KPI trends
- Department performance
- Sensitivity analysis
""")

uploaded_file = st.file_uploader("Upload Financial Data (CSV)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.session_state["data"] = df

    st.success("File uploaded successfully!")

    st.subheader("Preview Data")
    st.dataframe(df)

else:
    st.info("Please upload a dataset to begin analysis.")
