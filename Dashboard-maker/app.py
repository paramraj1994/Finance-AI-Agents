
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv
import plotly.express as px

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# Streamlit App UI
st.title("📊 Dashboard Maker – AI-Powered Interactive Dashboards")
st.write("Upload an Excel file and instantly generate a dashboard with AI-driven insights!")

# File uploader
uploaded_file = st.file_uploader("📂 Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    # Read the Excel file
    df = pd.read_excel(uploaded_file)

    # Display data preview
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Detect numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

    # Create a Dashboard Layout
    st.subheader("📈 Dashboard - Interactive Data Visualization")

    # Line Chart
    if len(numerical_columns) >= 2:
        fig_line = px.line(df, x=df.index, y=numerical_columns[:2], title="📉 Line Chart (First Two Numerical Columns)")
        st.plotly_chart(fig_line)

    # Bar Chart
    if categorical_columns and numerical_columns:
        fig_bar = px.bar(df, x=categorical_columns[0], y=numerical_columns[0], title=f"📊 Bar Chart: {categorical_columns[0]} vs {numerical_columns[0]}")
        st.plotly_chart(fig_bar)

    # Scatter Plot
    if len(numerical_columns) > 1:
        fig_scatter = px.scatter(df, x=numerical_columns[0], y=numerical_columns[1], title=f"📌 Scatter Plot: {numerical_columns[0]} vs {numerical_columns[1]}")
        st.plotly_chart(fig_scatter)

    # Heatmap
    if len(numerical_columns) > 1:
        plt.figure(figsize=(10, 6))
        sns.heatmap(df[numerical_columns].corr(), annot=True, cmap="coolwarm", linewidths=0.5)
        plt.title("🔥 Heatmap of Correlations")
        st.pyplot(plt)

    # AI Section
    st.subheader("🤖 AI-Powered Insights")

    # AI Summary of Data
    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are an AI-powered data analyst providing insights on uploaded datasets."},
            {"role": "user", "content": f"Here is a summary of the dataset:\n{df.describe(include='all').to_string()}\nWhat are the key insights?"}
        ],
        model="llama3-8b-8192",
    )

    ai_summary = response.choices[0].message.content
    st.write(ai_summary)

    # AI Chat - Users Can Ask Questions
    st.subheader("🗣️ Chat with AI About Your Data")

    user_query = st.text_input("🔍 Ask the AI about your dataset:")
    if user_query:
        chat_response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are an AI data scientist helping users understand their datasets."},
                {"role": "user", "content": f"Dataset Summary:\n{df.describe(include='all').to_string()}\n{user_query}"}
            ],
            model="llama3-8b-8192",
        )
        st.write(chat_response.choices[0].message.content)
