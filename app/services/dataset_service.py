"""Dataset service — orchestrates the fetch → transform → save pipeline.

This is the single place where the shared boilerplate lives. Every
transformation endpoint calls these methods instead of repeating the
same 60 lines of S3 download / DataFrame parse / S3 upload / MongoDB insert.
"""

import logging
from io import BytesIO
from datetime import datetime, timezone
from typing import Callable

import pandas as pd
import numpy as np
from bson import ObjectId
from starlette.concurrency import run_in_threadpool
from fastapi import BackgroundTasks

from app.config import settings
from app.core.storage import upload_to_s3, download_from_s3
from app.core.exceptions import (
    InvalidIdError,
    DatasetNotFoundError,
    MissingS3KeyError,
    StorageFileNotFoundError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    EmptyDatasetError,
)
from app.schemas.datasets import DatasetMetadata, TransformationResponse
from app.repositories import dataset_repo, analysis_repo
from app.workers.ml_worker import run_ml_pipeline

logger = logging.getLogger(__name__)


# ─── Shared Helpers ───────────────────────────────────────


async def get_validated_doc(dataset_id: str, user_id: str) -> dict:
    """Validate ID, fetch document from MongoDB, verify S3 key exists.

    Raises domain-specific exceptions on failure.
    """
    if not ObjectId.is_valid(dataset_id):
        raise InvalidIdError("dataset", dataset_id)

    doc = await dataset_repo.find_by_id(dataset_id, user_id)
    if not doc:
        raise DatasetNotFoundError(dataset_id)

    if not doc.get("s3_key"):
        raise MissingS3KeyError(dataset_id)

    return doc


def _parse_bytes(file_bytes: bytes, file_type: str) -> pd.DataFrame:
    """Parse raw bytes into a DataFrame. Runs in threadpool."""
    if file_type == "csv":
        return pd.read_csv(BytesIO(file_bytes))
    else:
        return pd.read_excel(BytesIO(file_bytes))


async def load_dataframe(doc: dict) -> tuple[pd.DataFrame, dict]:
    """Download from S3 and parse into a DataFrame.

    Uses run_in_threadpool to avoid blocking the event loop.
    Returns (DataFrame, metadata_dict).
    """
    s3_key = doc["s3_key"]
    file_bytes = await run_in_threadpool(download_from_s3, s3_key)
    if not file_bytes:
        raise StorageFileNotFoundError()

    metadata = doc.get("metadata", {})
    file_type = metadata.get("file_type", "csv")
    df = await run_in_threadpool(_parse_bytes, file_bytes, file_type)
    return df, metadata


def _serialize_df(df: pd.DataFrame, file_type: str) -> bytes:
    """Serialize DataFrame to bytes. Runs in threadpool."""
    out_buffer = BytesIO()
    if file_type == "csv":
        df.to_csv(out_buffer, index=False)
    else:
        df.to_excel(out_buffer, index=False)
    out_buffer.seek(0)
    return out_buffer.read()


async def save_transformed(
    df: pd.DataFrame,
    metadata: dict,
    name_suffix: str,
    user_id: str,
    background_tasks: BackgroundTasks,
) -> TransformationResponse:
    """Serialize → S3 upload → MongoDB insert → trigger ML pipeline.

    This is the shared "save" step for ALL transformation endpoints.
    """
    original_name = metadata.get("name", "dataset")
    file_type = metadata.get("file_type", "csv")

    # Avoid double-suffixing
    new_name = original_name if original_name.endswith(name_suffix) else f"{original_name}{name_suffix}"
    filename = f"{new_name}.{file_type}"

    # Serialize (non-blocking)
    new_file_bytes = await run_in_threadpool(_serialize_df, df, file_type)

    # Upload to S3 (non-blocking)
    new_s3_key = await run_in_threadpool(upload_to_s3, new_file_bytes, filename, user_id)

    # Create pending analysis task
    task_id = await analysis_repo.insert({
        "user_id": user_id,
        "filename": filename,
        "status": "processing",
        "created_at": datetime.now(timezone.utc),
    })

    # Create new dataset document
    new_metadata = DatasetMetadata(
        name=new_name,
        description=metadata.get("description"),
        file_type=file_type,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        uploaded_at=datetime.now(timezone.utc),
    )

    new_dataset_id = await dataset_repo.insert({
        "user_id": user_id,
        "metadata": new_metadata.model_dump(),
        "s3_key": new_s3_key,
        "analysis_id": task_id,
    })

    # Trigger background ML pipeline
    background_tasks.add_task(run_ml_pipeline, task_id, new_s3_key, user_id, filename)

    return TransformationResponse(
        message=f"Transformation '{name_suffix.strip('_')}' applied successfully.",
        dataset_id=new_dataset_id,
        task_id=task_id,
        s3_key=new_s3_key,
    )


# ─── Upload / Parse Helpers ──────────────────────────────


async def parse_upload_bytes(file_bytes: bytes, filename: str) -> pd.DataFrame:
    """Parse uploaded file bytes into a DataFrame with validation."""
    if len(file_bytes) > settings.MAX_FILE_SIZE:
        raise FileTooLargeError(settings.MAX_FILE_SIZE // (1024 * 1024))

    if filename.endswith(".csv"):
        file_type = "csv"
    elif filename.endswith(".xlsx"):
        file_type = "xlsx"
    else:
        raise UnsupportedFileTypeError()

    df = await run_in_threadpool(_parse_bytes, file_bytes, file_type)

    if df.empty:
        raise EmptyDatasetError()

    logger.info("Parsed file '%s': %d rows x %d columns", filename, len(df), len(df.columns))
    return df
