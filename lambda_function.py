import json
import boto3
from agent.mini_llm import generate_suggestion

s3 = boto3.client("s3")

BUCKET_NAME = "cloud-cost-optimizer-bucket"
INPUT_FILE = "cost_data.json"
OUTPUT_FILE = "cloud_cost_report.txt"

def lambda_handler(event, context):

    response = s3.get_object(Bucket=BUCKET_NAME, Key=INPUT_FILE)
    data = json.loads(response["Body"].read())

    cpu = data.get("cpu_usage", 0)

    if cpu < 30:
        prediction = 1
    else:
        prediction = 0

    result = generate_suggestion(prediction, data)

    with open("/tmp/report.txt", "w") as f:
        f.write(result)

    s3.upload_file("/tmp/report.txt", BUCKET_NAME, OUTPUT_FILE)

    return {
        "statusCode": 200,
        "body": "LLM analysis completed"
    }