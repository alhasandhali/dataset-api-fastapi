import boto3
import os
import uuid
import logging
import tempfile

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
) if AWS_ACCESS_KEY_ID else None

LOCAL_STORAGE_DIR = os.path.join(tempfile.gettempdir(), "dataset_api_storage")
if not S3_BUCKET_NAME:
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)

def upload_dataset_to_s3(file_bytes: bytes, filename: str, user_id: str) -> str:
    s3_key = f"datasets/{user_id}/{uuid.uuid4()}-{filename}"
    
    if not S3_BUCKET_NAME:
        logger.warning("S3_BUCKET_NAME not set. Saving file locally instead.")
        local_path = os.path.join(LOCAL_STORAGE_DIR, s3_key.replace("/", "_"))
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        return f"local:{local_path}"
        
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
    if s3_key.startswith("local:"):
        local_path = s3_key.replace("local:", "")
        logger.info(f"Reading file locally from {local_path}")
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()
        return b""
        
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
        return response['Body'].read()
    except Exception as e:
        logger.error(f"Failed to download from S3: {e}")
        raise e
