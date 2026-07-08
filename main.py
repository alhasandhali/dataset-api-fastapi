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

from core.storage import upload_dataset_to_s3, delete_dataset_from_s3, download_dataset_from_s3
from core.analysis import analyze_dataset, make_json_safe
from workers.ml_worker import run_ml_pipeline
from fastapi import BackgroundTasks, Depends
from core.deps import get_current_user
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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
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
    
    # Create a pending document in MongoDB to get a task ID
    task_doc = {
        "user_id": user_id,
        "filename": file.filename,
        "status": "processing",
        "created_at": datetime.now(timezone.utc),
    }
    db_result = await analyses_collection.insert_one(task_doc)
    task_id = str(db_result.inserted_id)
    
    # Trigger background task
    background_tasks.add_task(run_ml_pipeline, task_id, s3_key, user_id, file.filename)
    
    return {"task_id": task_id, "status": "processing", "s3_key": s3_key}

@app.get("/tasks/{task_id}")
async def get_task_status(task_id: str, current_user: str = Depends(get_current_user)):
    if not ObjectId.is_valid(task_id):
        return {"status": "failed", "error": "Invalid task ID"}
        
    doc = await analyses_collection.find_one({"_id": ObjectId(task_id), "user_id": current_user})
    if not doc:
        return {"status": "failed", "error": "Task not found"}
        
    if doc.get("status") == "processing":
        return {"status": "processing"}
    elif doc.get("status") == "completed":
        # The frontend expects {"status": "completed", "result": <analysis>}
        doc["analysis"]["id"] = str(doc["_id"])
        return {"status": "completed", "result": doc.get("analysis")}
    else:
        return {"status": "failed", "error": doc.get("error", "Unknown error")}


# ─── MongoDB Dataset CRUD Endpoints ───────────────────────


@app.post("/save-dataset", response_model=DatasetResponse, status_code=status.HTTP_201_CREATED)
async def save_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    analysis_id: str = Form(None),
    s3_key: str = Form(None),
    user_id: str = Depends(get_current_user),
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
    
    if not s3_key:
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
    
    if analysis_id:
        document["analysis_id"] = analysis_id

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
    skip: int = 0,
    limit: int = 20,
    sort_by: str = "uploaded_at",
    user_id: str = Depends(get_current_user),
) -> dict:
    """List saved datasets (metadata only) with pagination.

    Optionally filter by user_id for user-scoped queries.
    """
    query_filter = {}
    if user_id:
        query_filter["user_id"] = user_id

    datasets = []

    sort_order = [("metadata.uploaded_at", -1)]
    if sort_by == "name":
        sort_order = [("metadata.name", 1)]

    cursor = datasets_collection.find(
        query_filter, {"data": 0}  # Exclude row data for performance
    ).sort(sort_order).skip(skip).limit(limit)

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
async def get_dataset(dataset_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """Retrieve a single dataset with full row data."""

    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format."
        )

    doc = await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id), "user_id": user_id}
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )

    doc["_id"] = str(doc["_id"])
    return doc


@app.get("/datasets/{dataset_id}/preview")
async def get_dataset_preview(dataset_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """Retrieve a preview of the dataset rows."""
    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format."
        )

    doc = await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id), "user_id": user_id}
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )
        
    s3_key = doc.get("s3_key")
    if not s3_key:
        return {"rows": []}
        
    try:
        file_bytes = download_dataset_from_s3(s3_key)
        # Parse briefly just to get rows
        metadata = doc.get("metadata", {})
        filename = metadata.get("name", "dataset") + "." + metadata.get("file_type", "csv")
        df = pd.read_csv(BytesIO(file_bytes)) if filename.endswith(".csv") else pd.read_excel(BytesIO(file_bytes))
        df_preview = df.head(100)
        # Replace NaN with None
        df_preview = df_preview.replace({np.nan: None})
        rows = df_preview.to_dict(orient="records")
        return {"rows": rows}
    except Exception as e:
        logger.error(f"Failed to load preview for {dataset_id}: {e}")
        return {"rows": [], "error": str(e)}

@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """Delete a saved dataset and its S3 object."""

    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid dataset ID format."
        )

    # Fetch document first to get s3_key
    doc = await datasets_collection.find_one({"_id": ObjectId(dataset_id), "user_id": user_id})
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )

    result = await datasets_collection.delete_one(
        {"_id": ObjectId(dataset_id), "user_id": user_id}
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dataset not found."
        )

    if "s3_key" in doc:
        delete_dataset_from_s3(doc["s3_key"])

    logger.info("Dataset %s deleted", dataset_id)

    return {
        "message": "Dataset deleted successfully",
        "id": dataset_id,
    }


# ─── Analysis History Endpoints ───────────────────────────


@app.get("/analyses")
async def list_analyses(
    skip: int = 0,
    limit: int = 20,
    user_id: str = Depends(get_current_user),
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
async def get_analysis(analysis_id: str, user_id: str = Depends(get_current_user)) -> dict:
    """Retrieve a specific analysis result."""

    if not ObjectId.is_valid(analysis_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis ID format."
        )

    doc = await analyses_collection.find_one(
        {"_id": ObjectId(analysis_id), "user_id": user_id}
    )

    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found."
        )

    doc["_id"] = str(doc["_id"])
    return doc