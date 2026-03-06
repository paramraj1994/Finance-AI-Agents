import streamlit as st
import plotly.express as px

st.title("🏢 Department Analysis")

df = st.session_state.get("data")

if df is None:
    st.warning("Upload data first.")
    st.stop()

dept = st.selectbox("Select Department", df["Department"].unique())

filtered = df[df["Department"] == dept]

fig = px.line(
    filtered,
    x="Month",
    y="Actual",
    title=f"{dept} Performance"
)

st.plotly_chart(fig, use_container_width=True)
