import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📈 KPI Dashboard")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload data from the main page.")
    st.stop()

revenue = df["Revenue"].sum()
cogs = df["COGS"].sum()
profit = revenue - cogs
margin = profit / revenue * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Revenue", f"${revenue:,.0f}")
col2.metric("Total COGS", f"${cogs:,.0f}")
col3.metric("Total Profit", f"${profit:,.0f}")
col4.metric("Gross Margin", f"{margin:.2f}%")

fig = px.line(df, x="Month", y="Revenue", title="Revenue Trend")

st.plotly_chart(fig, use_container_width=True)
