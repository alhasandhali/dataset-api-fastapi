"""Dataset CRUD router — list, get, save, delete, download, preview."""

import logging
from io import BytesIO
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from fastapi import APIRouter, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.responses import Response
from starlette.concurrency import run_in_threadpool

from app.core.auth import get_current_user
from app.core.storage import upload_to_s3, download_from_s3, delete_from_s3
from app.core.exceptions import (
    InvalidIdError,
    DatasetNotFoundError,
    StorageFileNotFoundError,
)
from app.schemas.datasets import DatasetMetadata, DatasetResponse
from app.repositories import dataset_repo
from app.services import dataset_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Datasets"])


# ─── List Datasets ────────────────────────────────────────


@router.get("")
async def list_datasets(
    skip: int = 0,
    limit: int = 20,
    sort_by: str = "uploaded_at",
    user_id: str = Depends(get_current_user),
) -> dict:
    """List saved datasets (metadata only) with pagination."""
    datasets, total = await dataset_repo.find_many(user_id, skip, limit, sort_by)
    return {"total": total, "skip": skip, "limit": limit, "datasets": datasets}


# ─── Get Single Dataset ──────────────────────────────────


@router.get("/{dataset_id}")
async def get_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    """Retrieve a single dataset with full metadata."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)
    doc["_id"] = str(doc["_id"])
    return doc


# ─── Save (Upload) Dataset ───────────────────────────────


@router.post("", response_model=DatasetResponse, status_code=201)
async def save_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    analysis_id: str = Form(None),
    s3_key: str = Form(None),
    user_id: str = Depends(get_current_user),
) -> DatasetResponse:
    """Upload a CSV/XLSX file and save it to MongoDB."""
    file_bytes = await file.read()
    df = await dataset_service.parse_upload_bytes(file_bytes, file.filename)

    dataset_name = name or file.filename
    file_type = "csv" if file.filename.endswith(".csv") else "xlsx"

    if not s3_key:
        s3_key = await run_in_threadpool(upload_to_s3, file_bytes, file.filename, user_id)

    metadata = DatasetMetadata(
        name=dataset_name,
        description=description,
        file_type=file_type,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        uploaded_at=datetime.now(timezone.utc),
    )

    # Extract preview data (first 100 rows)
    df_preview = df.head(100).replace({np.nan: None})
    preview_data = df_preview.to_dict(orient="records")

    document = {
        "user_id": user_id,
        "metadata": metadata.model_dump(),
        "s3_key": s3_key,
        "preview_data": preview_data,
    }
    if analysis_id:
        document["analysis_id"] = analysis_id

    new_id = await dataset_repo.insert(document)
    logger.info("Dataset '%s' saved for user '%s', id=%s", dataset_name, user_id, new_id)

    return DatasetResponse(id=new_id, message="Dataset saved successfully", metadata=metadata)


# ─── Dataset Preview ──────────────────────────────────────


@router.get("/{dataset_id}/preview")
async def get_dataset_preview(
    dataset_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    """Retrieve a preview of the first 100 rows."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)

    # Check if preview_data is already saved in the document
    if "preview_data" in doc:
        return {"rows": doc["preview_data"]}

    # Fallback for old datasets without preview_data
    try:
        df, metadata = await dataset_service.load_dataframe(doc)
        df_preview = df.head(100)
        df_preview = df_preview.replace({np.nan: None})
        rows = df_preview.to_dict(orient="records")
        
        # Optionally, save it back to the database for future requests
        # (skipping for now to keep it simple and safe)
        
        return {"rows": rows}
    except Exception as e:
        logger.error("Failed to load preview for %s: %s", dataset_id, e)
        return {"rows": [], "error": str(e)}


# ─── Download Dataset ─────────────────────────────────────


@router.get("/{dataset_id}/download")
async def download_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user),
) -> Response:
    """Download the raw dataset file."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)

    file_bytes = await run_in_threadpool(download_from_s3, doc["s3_key"])
    if not file_bytes:
        raise StorageFileNotFoundError()

    metadata = doc.get("metadata", {})
    filename = metadata.get("name", "dataset")
    file_type = metadata.get("file_type", "csv")
    full_filename = f"{filename}.{file_type}" if not filename.endswith(f".{file_type}") else filename

    return Response(
        content=file_bytes,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{full_filename}"'},
    )


# ─── Delete Dataset ───────────────────────────────────────


@router.delete("/{dataset_id}")
async def delete_dataset(
    dataset_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    """Delete a saved dataset and its S3 object."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)

    deleted = await dataset_repo.delete_by_id(dataset_id, user_id)
    if deleted == 0:
        raise DatasetNotFoundError(dataset_id)

    if "s3_key" in doc:
        await run_in_threadpool(delete_from_s3, doc["s3_key"])

    logger.info("Dataset %s deleted", dataset_id)
    return {"message": "Dataset deleted successfully", "id": dataset_id}
