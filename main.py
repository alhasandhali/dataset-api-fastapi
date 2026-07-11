"""Dataset Analyzer API — Application entry point.

This file is the thin app factory. All business logic lives in:
  - app/routers/       → HTTP route handlers
  - app/services/      → Business logic orchestration
  - app/repositories/  → Database queries
  - app/core/          → Auth, storage, analysis, exceptions
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import create_indexes, close_connection
from app.core.exceptions import AppException, app_exception_handler

from app.routers import datasets, transformations, tasks, analyses

# ─── Logging ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ─── Lifespan ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle events."""
    await create_indexes()
    logger.info("Dataset Analyzer API started")
    yield
    await close_connection()
    logger.info("Dataset Analyzer API shut down")


# ─── App Factory ──────────────────────────────────────────

app = FastAPI(
    title="Dataset Analyzer API",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global Exception Handler ────────────────────────────

app.add_exception_handler(AppException, app_exception_handler)

# ─── Routers ─────────────────────────────────────────────

app.include_router(datasets.router)
app.include_router(transformations.router)
app.include_router(tasks.router)
app.include_router(analyses.router)


# ─── Health Check ────────────────────────────────────────


@app.get("/")
def home() -> dict:
    """Root endpoint confirming the API is running."""
    return {"message": "Dataset Analyzer API Running"}


# ─── Backward Compatibility ─────────────────────────────
# The frontend currently calls POST /save-dataset.
# The new RESTful route is POST /datasets.
# This alias ensures zero frontend breakage.

from fastapi import UploadFile, File, Form, Depends
from starlette.concurrency import run_in_threadpool
from app.core.auth import get_current_user
from app.core.storage import upload_to_s3
from app.schemas.datasets import DatasetMetadata, DatasetResponse
from app.repositories import dataset_repo
from app.services import dataset_service
from datetime import datetime, timezone


@app.post("/save-dataset", response_model=DatasetResponse, status_code=201, tags=["Datasets"])
async def save_dataset_compat(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    analysis_id: str = Form(None),
    s3_key: str = Form(None),
    user_id: str = Depends(get_current_user),
) -> DatasetResponse:
    """Backward-compatible alias for POST /datasets."""
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

    document = {
        "user_id": user_id,
        "metadata": metadata.model_dump(),
        "s3_key": s3_key,
    }
    if analysis_id:
        document["analysis_id"] = analysis_id

    new_id = await dataset_repo.insert(document)
    return DatasetResponse(id=new_id, message="Dataset saved successfully", metadata=metadata)