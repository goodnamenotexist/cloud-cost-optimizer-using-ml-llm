import boto3
import json

def get_data_from_s3():

    bucket_name = "cloud-cost-optimizer-bucket"
    file_key = "cost_data.json"

    s3 = boto3.client("s3")

    response = s3.get_object(Bucket=bucket_name, Key=file_key)

    data = json.loads(response["Body"].read().decode("utf-8"))

    return data