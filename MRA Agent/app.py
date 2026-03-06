import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="FP&A Monthly Reporting Dashboard",
    layout="wide"
)

st.title("📊 FP&A Monthly Reporting & Analysis Dashboard")

# ---------------------------------------------------
# FILE UPLOAD
# ---------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload Financial Data (CSV)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.success("File uploaded successfully!")

    # ---------------------------------------------------
    # DATA PREPARATION
    # ---------------------------------------------------

    if "Month" in df.columns:
        df["Month"] = pd.to_datetime(df["Month"])

    # ---------------------------------------------------
    # SIDEBAR FILTERS
    # ---------------------------------------------------

    st.sidebar.header("Filters")

    if "Department" in df.columns:
        departments = st.sidebar.multiselect(
            "Select Department",
            df["Department"].unique(),
            default=df["Department"].unique()
        )

        df = df[df["Department"].isin(departments)]

    # ---------------------------------------------------
    # KPI SECTION
    # ---------------------------------------------------

    st.subheader("📌 Key Financial KPIs")

    revenue = df["Revenue"].sum()
    cogs = df["COGS"].sum()
    profit = revenue - cogs
    margin = (profit / revenue) * 100 if revenue != 0 else 0

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Revenue", f"${revenue:,.0f}")
    col2.metric("Total COGS", f"${cogs:,.0f}")
    col3.metric("Total Profit", f"${profit:,.0f}")
    col4.metric("Gross Margin", f"{margin:.2f}%")

    st.divider()

    # ---------------------------------------------------
    # MONTHLY TREND
    # ---------------------------------------------------

    st.subheader("📈 Revenue Trend")

    fig_revenue = px.line(
        df,
        x="Month",
        y="Revenue",
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(fig_revenue, use_container_width=True)

    # ---------------------------------------------------
    # COST TREND
    # ---------------------------------------------------

    st.subheader("📉 Cost Trend")

    fig_cost = px.line(
        df,
        x="Month",
        y="COGS",
        title="Monthly Cost Trend"
    )

    st.plotly_chart(fig_cost, use_container_width=True)

    # ---------------------------------------------------
    # VARIANCE ANALYSIS
    # ---------------------------------------------------

    if "Budget" in df.columns and "Actual" in df.columns:

        st.subheader("📊 Budget vs Actual Variance")

        df["Variance"] = df["Actual"] - df["Budget"]

        fig_variance = px.bar(
            df,
            x="Month",
            y="Variance",
            title="Monthly Variance"
        )

        st.plotly_chart(fig_variance, use_container_width=True)

        st.dataframe(df[["Month","Actual","Budget","Variance"]])

    # ---------------------------------------------------
    # DEPARTMENT ANALYSIS
    # ---------------------------------------------------

    if "Department" in df.columns:

        st.subheader("🏢 Department Performance")

        dept_summary = df.groupby("Department")[["Revenue","COGS"]].sum().reset_index()

        fig_dept = px.bar(
            dept_summary,
            x="Department",
            y="Revenue",
            title="Revenue by Department"
        )

        st.plotly_chart(fig_dept, use_container_width=True)

    # ---------------------------------------------------
    # SENSITIVITY ANALYSIS
    # ---------------------------------------------------

    st.subheader("⚙️ Sensitivity Analysis")

    col1, col2 = st.columns(2)

    revenue_change = col1.slider(
        "Revenue Change (%)",
        -50,
        50,
        0
    )

    cost_change = col2.slider(
        "Cost Change (%)",
        -50,
        50,
        0
    )

    adj_revenue = revenue * (1 + revenue_change/100)
    adj_cost = cogs * (1 + cost_change/100)
    adj_profit = adj_revenue - adj_cost

    col1, col2, col3 = st.columns(3)

    col1.metric("Adjusted Revenue", f"${adj_revenue:,.0f}")
    col2.metric("Adjusted Cost", f"${adj_cost:,.0f}")
    col3.metric("Adjusted Profit", f"${adj_profit:,.0f}")

    # ---------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------

    st.subheader("📄 Data Preview")

    st.dataframe(df)

else:
    st.info("Upload a CSV file to start analysis.")
