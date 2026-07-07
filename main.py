import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from io import BytesIO

from database import datasets_collection, analyses_collection, create_indexes, close_connection
from models import DatasetMetadata, DatasetResponse

from core.storage import upload_dataset_to_s3
from core.analysis import analyze_dataset, make_json_safe
from workers.ml_worker import run_ml_pipeline
from core.celery_app import celery_app

# ─── Logging ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle events."""
    await create_indexes()
    logger.info("Dataset Analyzer API started")
    yield
    await close_connection()
    logger.info("Dataset Analyzer API shut down")


app = FastAPI(
    title="Dataset Analyzer API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Utility Functions ────────────────────────────────────


async def parse_upload(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX file into a DataFrame with size validation.

    Reads the file in chunks to avoid loading oversized files into memory.
    """
    # Read in chunks to enforce size limit without loading entire file first
    chunks: list[bytes] = []
    total_size = 0

    while True:
        chunk = await file.read(1024 * 1024)  # Read 1MB at a time
        if not chunk:
            break
        total_size += len(chunk)
        if total_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size is "
                       f"{MAX_FILE_SIZE // (1024 * 1024)}MB.",
            )
        chunks.append(chunk)

    contents = b"".join(chunks)
    filename = file.filename

    if filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(contents))
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(contents))
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only CSV or XLSX file supported.",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file contains no data rows.",
        )

    logger.info(
        "Parsed file '%s': %d rows x %d columns",
        filename, len(df), len(df.columns),
    )

    return df

# ─── Endpoints ────────────────────────────────────────────


@app.get("/")
def home() -> dict:
    """Root endpoint confirming the API is running."""
    return {
        "message": "Dataset Analyzer API Running"
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: str = Form(...),
) -> dict:
    """Upload a file and trigger asynchronous ML analysis."""
    # Read the file bytes directly for S3
    file_bytes = await file.read()
    
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)}MB.",
        )
        
    s3_key = upload_dataset_to_s3(file_bytes, file.filename, user_id)
    
    # Trigger celery task
    task = run_ml_pipeline.delay(s3_key, user_id, file.filename)
    
    return {"task_id": task.id, "status": "processing"}

@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    task_result = celery_app.AsyncResult(task_id)
    if task_result.ready():
        return task_result.result # Contains status and result/error
    return {"status": "processing"}


# ─── MongoDB Dataset CRUD Endpoints ───────────────────────


@app.post("/save-dataset", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def save_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    user_id: str = Form(...),
) -> DatasetResponse:
    """Upload a CSV/XLSX file and save it to MongoDB with user association."""

    file_bytes = await file.read()
    
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large."
        )

    # Parse briefly just to get metadata
    df = pd.read_csv(BytesIO(file_bytes)) if file.filename.endswith(".csv") else pd.read_excel(BytesIO(file_bytes))

    # Use filename as name if not provided
    dataset_name = name or file.filename
    file_type = "csv" if file.filename.endswith(".csv") else "xlsx"
    
    s3_key = upload_dataset_to_s3(file_bytes, file.filename, user_id)

    # Build metadata
    metadata = DatasetMetadata(
        name=dataset_name,
        description=description,
        file_type=file_type,
        row_count=len(df),
        column_count=len(df.columns),
        columns=list(df.columns),
        uploaded_at=datetime.now(timezone.utc),
    )

    # Build the document to insert
    document = {
        "user_id": user_id,
        "metadata": metadata.model_dump(),
        "s3_key": s3_key,
    }

    # Insert into MongoDB
    result = await datasets_collection.insert_one(document)

    logger.info(
        "Dataset '%s' saved for user '%s', id=%s",
        dataset_name, user_id, result.inserted_id,
    )

    return DatasetResponse(
        id=str(result.inserted_id),
        message="Dataset saved successfully",
        metadata=metadata,
    )


@app.get("/datasets")
async def list_datasets(
    user_id: str | None = None,
    skip: int = 0,
    limit: int = 20,
) -> dict:
    """List saved datasets (metadata only) with pagination.

    Optionally filter by user_id for user-scoped queries.
    """
    query_filter = {}
    if user_id:
        query_filter["user_id"] = user_id

    datasets = []

    cursor = datasets_collection.find(
        query_filter, {"data": 0}  # Exclude row data for performance
    ).skip(skip).limit(limit)

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        datasets.append(doc)

    total = await datasets_collection.count_documents(query_filter)

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "datasets": datasets,
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str) -> dict:
    """Retrieve a single dataset with full row data."""

    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format."
        )

    doc = await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id)}
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )

    doc["_id"] = str(doc["_id"])
    return doc


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str) -> dict:
    """Delete a saved dataset."""

    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format."
        )

    result = await datasets_collection.delete_one(
        {"_id": ObjectId(dataset_id)}
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )

    logger.info("Dataset %s deleted", dataset_id)

    return {
        "message": "Dataset deleted successfully",
        "id": dataset_id,
    }


# ─── Analysis History Endpoints ───────────────────────────


@app.get("/analyses")
async def list_analyses(
    user_id: str | None = None,
    skip: int = 0,
    limit: int = 20,
) -> dict:
    """List saved analyses with pagination.

    Optionally filter by user_id for user-scoped queries.
    """
    query_filter = {}
    if user_id:
        query_filter["user_id"] = user_id

    analyses = []

    cursor = analyses_collection.find(
        query_filter, {"analysis": 0}  # Exclude full analysis for performance
    ).skip(skip).limit(limit)

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        analyses.append(doc)

    total = await analyses_collection.count_documents(query_filter)

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "analyses": analyses,
    }


@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str) -> dict:
    """Retrieve a specific analysis result."""

    if not ObjectId.is_valid(analysis_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis ID format."
        )

    doc = await analyses_collection.find_one(
        {"_id": ObjectId(analysis_id)}
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found."
        )

    doc["_id"] = str(doc["_id"])
    return doc