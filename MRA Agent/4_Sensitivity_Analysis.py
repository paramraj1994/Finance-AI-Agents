import streamlit as st
import pandas as pd

st.title("⚙️ Sensitivity Analysis")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload data first.")
    st.stop()

revenue_change = st.slider("Revenue Change (%)", -50, 50, 0)
cost_change = st.slider("Cost Change (%)", -50, 50, 0)

revenue = df["Revenue"].sum() * (1 + revenue_change/100)
cost = df["COGS"].sum() * (1 + cost_change/100)

profit = revenue - cost

st.metric("Adjusted Revenue", f"${revenue:,.0f}")
st.metric("Adjusted Cost", f"${cost:,.0f}")
st.metric("Adjusted Profit", f"${profit:,.0f}")
