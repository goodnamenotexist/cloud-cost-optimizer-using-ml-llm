import json
import time
from agents.cost_analyzer import analyze_cost
from report import save_report
from aws_s3_fetcher import get_data_from_s3
from aws_s3_uploader import upload_report_to_s3


print("Fetching cloud data from AWS S3...\n")

# Load data from S3
try:
    all_data = get_data_from_s3()
except Exception as e:
    print("Error fetching data from AWS S3:")
    print(e)
    exit()

print("Analyzing cloud cost using AI...\n")


chunks = all_data if isinstance(all_data, list) else [all_data]

for i, cost_chunk in enumerate(chunks, start=1):

    print(f"----- Analysis {i} -----")

    try:
        # ML + Mini LLM
        result = analyze_cost(cost_chunk)

        print("\nAI Optimization Suggestions:\n")
        print(result)
        print("\n")

        save_report(result)
        upload_report_to_s3("cloud_cost_report.txt")

    except Exception as e:
        print("Error during AI analysis:")
        print(e)

    # small delay between requests
    if i < len(chunks):
        print("Waiting before next analysis...\n")
        time.sleep(2)

print("Analysis Complete.")