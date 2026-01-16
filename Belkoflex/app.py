import streamlit as st
from agent.financial_model import load_xlsb, apply_assumptions
from agent.assumptions import default_assumptions
from agent.ai_agent import explain_changes

st.set_page_config(page_title="AI Financial Agent", layout="wide")

st.title("📊 AI Cash Flow Analysis Agent")

# Load data
path = "data/Belkofleks_Weekly_Cash_Flows.xlsb"
sheets = load_xlsb(path)

sheet_name = st.selectbox("Select Sheet", sheets.keys())
df = sheets[sheet_name]

st.subheader("📄 Original Data")
st.dataframe(df)

# Assumptions
st.sidebar.header("🔧 Assumptions")
assumptions = default_assumptions()

assumptions["revenue_growth"] = st.sidebar.slider(
    "Revenue Growth %",
    -20.0, 50.0, assumptions["revenue_growth"] * 100
) / 100

assumptions["cost_inflation"] = st.sidebar.slider(
    "Cost Inflation %",
    0.0, 30.0, assumptions["cost_inflation"] * 100
) / 100

# Apply assumptions
adjusted_df = apply_assumptions(df, assumptions)

st.subheader("📈 Adjusted Cash Flow")
st.dataframe(adjusted_df)

# AI Explanation
if st.button("🤖 Explain Impact"):
    with st.spinner("Analyzing..."):
        explanation = explain_changes(df, adjusted_df)
    st.markdown("### AI Insights")
    st.write(explanation)
