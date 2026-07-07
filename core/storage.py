import boto3
import os
import uuid
import logging

logger = logging.getLogger(__name__)

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_REGION
)

def upload_dataset_to_s3(file_bytes: bytes, filename: str, user_id: str) -> str:
    if not S3_BUCKET_NAME:
        logger.warning("S3_BUCKET_NAME not set. Simulating S3 upload.")
        return f"simulated-s3-key-{uuid.uuid4()}-{filename}"
        
    s3_key = f"datasets/{user_id}/{uuid.uuid4()}-{filename}"
    try:
        s3_client.put_object(
            Bucket=S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes
        )
        return s3_key
    except Exception as e:
        logger.error(f"Failed to upload to S3: {e}")
        raise e

def download_dataset_from_s3(s3_key: str) -> bytes:
    if not S3_BUCKET_NAME or s3_key.startswith("simulated"):
        logger.warning("Simulated S3 download.")
        return b""
        
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        raise e
