import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from bson import ObjectId
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from io import BytesIO

from database import datasets_collection, analyses_collection
from models import DatasetMetadata, DatasetResponse

# ─── Logging ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ─── App Setup ────────────────────────────────────────────

app = FastAPI(title="Dataset Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


# ─── Utility Functions ────────────────────────────────────


async def parse_upload(file: UploadFile) -> pd.DataFrame:
    """Parse an uploaded CSV/XLSX file into a DataFrame with size validation."""

    contents = await file.read()

    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is "
                   f"{MAX_FILE_SIZE // (1024 * 1024)}MB.",
        )

    filename = file.filename

    if filename.endswith(".csv"):
        df = pd.read_csv(BytesIO(contents))
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(BytesIO(contents))
    else:
        raise HTTPException(
            status_code=400,
            detail="Only CSV or XLSX file supported.",
        )

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file contains no data rows.",
        )

    logger.info(
        "Parsed file '%s': %d rows x %d columns",
        filename, len(df), len(df.columns),
    )

    return df


def analyze_dataset(df):

    result = {}

    row_count = len(df)

    # Shape
    result["rows"] = row_count
    result["columns"] = len(df.columns)

    # Column Information
    column_info = []

    for col in df.columns:
        missing = int(df[col].isnull().sum())
        column_info.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "missing_values": missing,
            "missing_percentage": round(
                (missing / row_count) * 100, 2
            ),
            "unique_values": int(df[col].nunique())
        })

    result["column_info"] = column_info

    # Duplicate Rows
    result["duplicate_rows"] = int(df.duplicated().sum())

    # Memory Usage
    result["memory_usage_MB"] = round(
        df.memory_usage(deep=True).sum() / 1024 / 1024,
        2
    )

    # Numeric Statistics
    numeric_cols = df.select_dtypes(include=np.number)

    if not numeric_cols.empty:
        result["numeric_summary"] = numeric_cols.describe().to_dict()

    # Categorical Statistics
    categorical_cols = df.select_dtypes(include="object")

    cat_summary = {}

    for col in categorical_cols.columns:
        cat_summary[col] = {
            "top_value":
                str(df[col].mode().iloc[0])
                if not df[col].mode().empty else None,

            "top_frequency":
                int(df[col].value_counts().iloc[0])
                if not df[col].value_counts().empty else 0
        }

    result["categorical_summary"] = cat_summary

    # Outlier Detection (IQR method)
    outliers = {}

    for col in numeric_cols.columns:

        q1 = numeric_cols[col].quantile(0.25)
        q3 = numeric_cols[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        count = numeric_cols[
            (numeric_cols[col] < lower) |
            (numeric_cols[col] > upper)
        ][col].count()

        outliers[col] = int(count)

    result["outliers"] = outliers

    return result


def make_json_safe(data):
    """Convert NaN, Infinity values to None for JSON serialization."""
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(item) for item in data]
    elif isinstance(data, float) and (
        np.isnan(data) or np.isinf(data)
    ):
        return None
    return data


# ─── Endpoints ────────────────────────────────────────────


@app.get("/")
def home():
    return {
        "message": "Dataset Analyzer API Running"
    }


@app.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    user_id: str = Form(...),
):

    df = await parse_upload(file)

    analysis_result = analyze_dataset(df)
    safe_result = make_json_safe(analysis_result)

    # Save analysis output to MongoDB
    analysis_doc = {
        "user_id": user_id,
        "filename": file.filename,
        "analysis": safe_result,
        "analyzed_at": datetime.now(timezone.utc),
    }
    db_result = await analyses_collection.insert_one(analysis_doc)

    logger.info(
        "Analysis saved for user '%s', file '%s', id=%s",
        user_id, file.filename, db_result.inserted_id,
    )

    safe_result["id"] = str(db_result.inserted_id)

    return safe_result


# ─── MongoDB Dataset CRUD Endpoints ───────────────────────


@app.post("/save-dataset", response_model=DatasetResponse)
async def save_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
    user_id: str = Form(...),
):
    """Upload a CSV/XLSX file and save it to MongoDB with user association."""

    df = await parse_upload(file)

    filename = file.filename
    file_type = "csv" if filename.endswith(".csv") else "xlsx"

    # Use filename as name if not provided
    dataset_name = name or filename

    # Convert DataFrame to JSON-safe list of dicts
    records = make_json_safe(
        df.to_dict(orient="records")
    )

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
        "data": records,
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
async def list_datasets(skip: int = 0, limit: int = 20):
    """List all saved datasets (metadata only) with pagination."""

    datasets = []

    cursor = datasets_collection.find(
        {}, {"data": 0}  # Exclude row data for performance
    ).skip(skip).limit(limit)

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        datasets.append(doc)

    total = await datasets_collection.count_documents({})

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "datasets": datasets,
    }


@app.get("/datasets/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Retrieve a single dataset with full row data."""

    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset ID format."
        )

    doc = await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id)}
    )

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    doc["_id"] = str(doc["_id"])
    return doc


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a saved dataset."""

    if not ObjectId.is_valid(dataset_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid dataset ID format."
        )

    result = await datasets_collection.delete_one(
        {"_id": ObjectId(dataset_id)}
    )

    if result.deleted_count == 0:
        raise HTTPException(
            status_code=404,
            detail="Dataset not found."
        )

    logger.info("Dataset %s deleted", dataset_id)

    return {
        "message": "Dataset deleted successfully",
        "id": dataset_id,
    }


# ─── Analysis History Endpoints ───────────────────────────


@app.get("/analyses")
async def list_analyses(skip: int = 0, limit: int = 20):
    """List all saved analyses (without full analysis data) with pagination."""

    analyses = []

    cursor = analyses_collection.find(
        {}, {"analysis": 0}  # Exclude full analysis for performance
    ).skip(skip).limit(limit)

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        analyses.append(doc)

    total = await analyses_collection.count_documents({})

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "analyses": analyses,
    }


@app.get("/analyses/{analysis_id}")
async def get_analysis(analysis_id: str):
    """Retrieve a specific analysis result."""

    if not ObjectId.is_valid(analysis_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid analysis ID format."
        )

    doc = await analyses_collection.find_one(
        {"_id": ObjectId(analysis_id)}
    )

    if not doc:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found."
        )

    doc["_id"] = str(doc["_id"])
    return doc