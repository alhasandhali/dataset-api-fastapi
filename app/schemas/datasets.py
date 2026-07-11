"""Pydantic schemas for request/response models."""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime, timezone


# ─── Dataset Schemas ──────────────────────────────────────


class DatasetMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    file_type: str
    row_count: int
    column_count: int
    columns: List[str]
    uploaded_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class DatasetResponse(BaseModel):
    id: str
    message: str
    metadata: DatasetMetadata


class TransformationResponse(BaseModel):
    """Standardized response for all dataset transformation endpoints."""
    message: str
    dataset_id: str
    task_id: str
    s3_key: str


class NoChangeResponse(BaseModel):
    """Response when a transformation finds nothing to change."""
    message: str
    dataset_id: str


# ─── Task Schemas ─────────────────────────────────────────


class TaskStatusResponse(BaseModel):
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


# ─── Pagination ───────────────────────────────────────────


class PaginatedResponse(BaseModel):
    total: int
    skip: int
    limit: int
    datasets: Optional[List[dict]] = None
    analyses: Optional[List[dict]] = None
