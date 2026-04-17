import boto3

def upload_report_to_s3(file_name):

    bucket_name = "cloud-cost-optimizer-bucket"   # your bucket name
    s3 = boto3.client("s3")

    try:
        s3.upload_file(file_name, bucket_name, file_name)
        print(f"Report uploaded to S3: {file_name}")
    except Exception as e:
        print("Error uploading report to S3:")
        print(e)