import streamlit as st
import pandas as pd
import plotly.express as px

st.title("📊 Variance Analysis")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload data first.")
    st.stop()

df["Variance"] = df["Actual"] - df["Budget"]

st.dataframe(df)

fig = px.bar(
    df,
    x="Month",
    y="Variance",
    title="Budget vs Actual Variance"
)

st.plotly_chart(fig, use_container_width=True)
