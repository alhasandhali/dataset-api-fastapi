"""S3 and local storage operations."""

import boto3
import os
import uuid
import logging
import tempfile

from app.config import settings

logger = logging.getLogger(__name__)

s3_client = boto3.client(
    "s3",
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_REGION,
) if settings.AWS_ACCESS_KEY_ID else None

LOCAL_STORAGE_DIR = os.path.join(tempfile.gettempdir(), "dataset_api_storage")
if not settings.S3_BUCKET_NAME:
    os.makedirs(LOCAL_STORAGE_DIR, exist_ok=True)


def upload_to_s3(file_bytes: bytes, filename: str, user_id: str) -> str:
    """Upload file bytes to S3 or local fallback. Returns the storage key."""
    s3_key = f"datasets/{user_id}/{uuid.uuid4()}-{filename}"

    if not settings.S3_BUCKET_NAME:
        logger.warning("S3_BUCKET_NAME not set. Saving file locally instead.")
        local_path = os.path.join(LOCAL_STORAGE_DIR, s3_key.replace("/", "_"))
        with open(local_path, "wb") as f:
            f.write(file_bytes)
        return f"local:{local_path}"

    try:
        s3_client.put_object(
            Bucket=settings.S3_BUCKET_NAME,
            Key=s3_key,
            Body=file_bytes,
        )
        return s3_key
    except Exception as e:
        logger.error("Failed to upload to S3: %s", e)
        raise


def download_from_s3(s3_key: str) -> bytes:
    """Download file bytes from S3 or local fallback."""
    if s3_key.startswith("local:"):
        local_path = s3_key.replace("local:", "")
        logger.info("Reading file locally from %s", local_path)
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()
        return b""

    try:
        response = s3_client.get_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        return response["Body"].read()
    except Exception as e:
        logger.error("Failed to download from S3: %s", e)
        raise


def delete_from_s3(s3_key: str) -> bool:
    """Delete a file from S3 or local storage."""
    if s3_key.startswith("local:"):
        local_path = s3_key.replace("local:", "")
        logger.info("Deleting local file: %s", local_path)
        if os.path.exists(local_path):
            os.remove(local_path)
            return True
        return False

    if not s3_client or not settings.S3_BUCKET_NAME:
        logger.warning("S3 not configured, cannot delete file")
        return False

    try:
        s3_client.delete_object(Bucket=settings.S3_BUCKET_NAME, Key=s3_key)
        logger.info("Deleted S3 object: %s", s3_key)
        return True
    except Exception as e:
        logger.error("Failed to delete from S3: %s", e)
        return False
