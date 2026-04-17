import streamlit as st
import json
import plotly.express as px
from agents.cost_analyzer import analyze_cost

st.set_page_config(page_title="AI Cloud Cost Optimization", layout="wide")

st.title("AI Cloud Cost Optimization Dashboard")

st.write("Upload cloud usage data and let AI analyze cost optimization opportunities.")

uploaded_file = st.file_uploader("Upload Cloud Cost JSON", type="json")

if uploaded_file is not None:

    data = json.load(uploaded_file)

    st.subheader("Uploaded Cloud Data")
    st.json(data)

    # Convert JSON to chart data
    metrics = []
    values = []

    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics.append(key)
            values.append(value)

    # Create chart
    chart_data = {"Metric": metrics, "Value": values}

    fig = px.bar(chart_data, x="Metric", y="Value", title="Cloud System Metrics")

    st.subheader("Cloud Metrics Overview")
    st.plotly_chart(fig, use_container_width=True)

    # Show cost separately
    if "cost" in data:
        st.metric("Current Cloud Cost", f"${data['cost']}")

    if st.button("Analyze Cost with AI"):

        with st.spinner("AI analyzing cloud metrics..."):

            result = analyze_cost(data)

        st.subheader("AI Optimization Suggestions")
        st.write(result)