import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="FP&A Monthly Reporting Dashboard", layout="wide")

st.title("📊 FP&A Monthly Reporting Analysis")

uploaded_file = st.file_uploader("Upload Financial Data (CSV or Excel)", type=["csv","xlsx"])

if uploaded_file is not None:

    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Raw Data")
    st.dataframe(df)

    if "Month" in df.columns:
        month = st.selectbox("Select Month", df["Month"].unique())
        filtered_df = df[df["Month"] == month]
    else:
        filtered_df = df

    st.subheader("Filtered Data")
    st.dataframe(filtered_df)

    if "Revenue" in df.columns and "Month" in df.columns:
        fig = px.line(df, x="Month", y="Revenue", title="Revenue Trend")
        st.plotly_chart(fig, use_container_width=True)

    if "Expense" in df.columns and "Month" in df.columns:
        fig2 = px.bar(df, x="Month", y="Expense", title="Expense Trend")
        st.plotly_chart(fig2, use_container_width=True)

    col1, col2, col3 = st.columns(3)

    if "Revenue" in df.columns:
        col1.metric("Total Revenue", f"{df['Revenue'].sum():,.0f}")

    if "Expense" in df.columns:
        col2.metric("Total Expense", f"{df['Expense'].sum():,.0f}")

    if "Revenue" in df.columns and "Expense" in df.columns:
        profit = df["Revenue"].sum() - df["Expense"].sum()
        col3.metric("Profit", f"{profit:,.0f}")

else:
    st.info("Upload a financial dataset to start analysis.")
