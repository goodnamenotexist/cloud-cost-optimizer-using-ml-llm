import streamlit as st
import json
import plotly.express as px
import boto3
import time

# AWS Clients
s3 = boto3.client("s3", region_name="us-east-1")
lambda_client = boto3.client("lambda", region_name="us-east-1")

BUCKET_NAME = "cloud-cost-optimizer-bucket"
INPUT_FILE = "cost_data.json"
OUTPUT_FILE = "cloud_cost_report.txt"

st.set_page_config(page_title="AI Cloud Cost Optimization", layout="wide")

st.title("AI Cloud Cost Optimization Dashboard")

st.write("Upload cloud usage data and let AI analyze cost optimization opportunities.")

uploaded_file = st.file_uploader("Upload Cloud Cost JSON", type="json")

if uploaded_file is not None:

    data = json.load(uploaded_file)

    st.subheader("Uploaded Cloud Data")
    st.json(data)


    metrics = []
    values = []

    for key, value in data.items():
        if isinstance(value, (int, float)):
            metrics.append(key)
            values.append(value)

    chart_data = {"Metric": metrics, "Value": values}
    fig = px.bar(chart_data, x="Metric", y="Value", title="Cloud System Metrics")

    st.subheader("Cloud Metrics Overview")
    st.plotly_chart(fig, use_container_width=True)

    if "cost" in data:
        st.metric("Current Cloud Cost", f"${data['cost']}")


    if st.button("Analyze Cost with AI"):

        with st.spinner("Uploading data to AWS S3..."):

            # Upload JSON to S3
            s3.put_object(
                Bucket=BUCKET_NAME,
                Key=INPUT_FILE,
                Body=json.dumps(data)
            )

        with st.spinner("Triggering Lambda function..."):

            # Trigger Lambda
            lambda_client.invoke(
                FunctionName="arn:aws:lambda:us-east-1:406271521365:function:cloud-project-3y",
                InvocationType="Event"
            )

        with st.spinner("Waiting for analysis..."):

            time.sleep(5)

        # Fetch result from S3
        try:
            response = s3.get_object(Bucket=BUCKET_NAME, Key=OUTPUT_FILE)
            result = response["Body"].read().decode("utf-8")

            st.subheader("AI Optimization Suggestions")
            st.text(result)

        except Exception as e:
            st.error("Error fetching result from S3")
            st.write(e)