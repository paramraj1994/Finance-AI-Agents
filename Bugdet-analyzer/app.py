import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from groq import Groq
from dotenv import load_dotenv

# Load API key securely
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("🚨 API Key is missing! Set it in Streamlit Secrets or a .env file.")
    st.stop()

# **🎨 Streamlit UI Styling**
st.set_page_config(page_title="FP&A Budget Variance Analyzer", page_icon="📊", layout="wide")

st.markdown("""
    <style>
        .title { text-align: center; font-size: 36px; font-weight: bold; color: #2E0249; }
        .subtitle { text-align: center; font-size: 20px; color: #4A0072; }
        .stButton>button { width: 100%; background-color: #2E0249; color: white; font-size: 16px; font-weight: bold; }
        .stFileUploader { text-align: center; }
        .analysis-container { padding: 15px; border-radius: 10px; margin: 10px 0; background-color: #EDE7F6; }
        .analysis-title { font-size: 20px; font-weight: bold; color: #4A0072; }
        .analysis-desc { font-size: 16px; color: #2E0249; }
    </style>
""", unsafe_allow_html=True)

# **📢 Title & Description**
st.markdown('<h1 class="title">📊 FP&A Budget Variance Analyzer</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload your budget vs. actuals data, select a sheet, and AI will generate variance insights like a Head of FP&A.</p>', unsafe_allow_html=True)

# **📂 Upload Financial Data**
st.subheader("📥 Upload Your Budget Variance Data (Excel)")
uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    # Load sheet names
    xls = pd.ExcelFile(uploaded_file)
    sheet_names = xls.sheet_names

    # Let user select sheet
    selected_sheet = st.selectbox("📑 Select a Sheet to Analyze:", sheet_names)

    if selected_sheet:
        # Load selected sheet
        df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)  

        # **Data Processing**
        df["Actuals vs Budget"] = df["Actual"] - df["FY25 Budget"]

        # **Calculate Totals**
        totals = df.select_dtypes(include=[np.number]).sum()
        totals["Category"] = "TOTAL"
        totals["Account"] = ""  
        df_totals = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)

        # **Format Numbers**
        def currency_format(val):
            if pd.notna(val):
                return f"${val:,.0f}" if val >= 0 else f"(${abs(val):,.0f})"
            return ""

        # **Apply Color Coding for Variance**
        def highlight_variance(val):
            color = "lightcoral" if val > 0 else "lightgreen"
            return f"background-color: {color}"

        # **Apply Formatting & Color-Coding to Dataframe**
        styled_df = df_totals.style.applymap(highlight_variance, subset=["Actuals vs Budget"]).format({
            "Actual": currency_format,
            "FY25 Budget": currency_format,
            "Actuals vs Budget": currency_format
        })

        # **Display Data Table with Color-Coding**
        st.subheader("📊 Variance Analysis Table (Color-Coded)")
        st.dataframe(df_totals)

        # **Visualization - Budget Variance Chart**
        df_sorted = df.sort_values(by="Actuals vs Budget")

        plt.figure(figsize=(10, 5))
        sns.barplot(
            x="Account", y="Actuals vs Budget", data=df_sorted, 
            palette=["red" if x > 0 else "green" for x in df_sorted["Actuals vs Budget"]]
        )
        plt.xticks(rotation=90)
        plt.axhline(0, color='black', linewidth=1)
        plt.title("Actuals vs Budget Variance")
        plt.ylabel("Variance")
        st.pyplot(plt)

        # **Convert Full Variance Table to JSON for AI Processing**
        data_for_ai = df_totals[["Account", "Category", "Actual", "FY25 Budget", "Actuals vs Budget"]].to_json(orient="records")

        # **Generate FP&A Commentary**
        st.subheader("🤖 AI-Generated FP&A Commentary")

        client = Groq(api_key=GROQ_API_KEY)

        prompt = f"""
        You are the Head of FP&A at a SaaS company.
        Your task is to analyze the full budget variance table and provide:
        - Key insights from the data.
        - Areas of concern and key drivers for variance.
        - A CFO-ready summary using the Pyramid Principle.
        - Actionable recommendations to improve financial performance.

        Here is the full dataset in JSON format:
        {data_for_ai}
        """

        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "You are a financial planning and analysis (FP&A) expert, specializing in SaaS companies."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
        )

        ai_commentary = response.choices[0].message.content

        # **Display AI Commentary**
        st.markdown('<div class="analysis-container">', unsafe_allow_html=True)
        st.subheader("📖 AI-Generated FP&A Commentary")
        st.write(ai_commentary)
        st.markdown('</div>', unsafe_allow_html=True)
