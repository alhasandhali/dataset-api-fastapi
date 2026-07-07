from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class DatasetMetadata(BaseModel):
    name: str
    description: Optional[str] = None
    file_type: str
    row_count: int
    column_count: int
    columns: List[str]
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)


class DatasetRecord(BaseModel):
    metadata: DatasetMetadata
    data: List[dict[str, Any]]


class DatasetResponse(BaseModel):
    id: str
    message: str
    metadata: DatasetMetadata
