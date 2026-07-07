from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from bson import ObjectId
import pandas as pd
import numpy as np
import os
from datetime import datetime

from database import datasets_collection
from models import DatasetMetadata, DatasetResponse

app = FastAPI(title="Dataset Analyzer API")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def analyze_dataset(df):

    result = {}

    # Shape
    result["rows"] = len(df)
    result["columns"] = len(df.columns)

    # Column Information
    column_info = []

    for col in df.columns:
        column_info.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "missing_values": int(df[col].isnull().sum()),
            "missing_percentage": round(
                (df[col].isnull().sum()/len(df))*100,2
            ),
            "unique_values": int(df[col].nunique())
        })

    result["column_info"] = column_info

    # Missing Values
    result["missing_values"] = df.isnull().sum().to_dict()

    # Missing Percentage
    result["missing_percentage"] = (
        (df.isnull().sum()/len(df))*100
    ).round(2).to_dict()

    # Duplicate Rows
    result["duplicate_rows"] = int(df.duplicated().sum())

    # Memory Usage
    result["memory_usage_MB"] = round(
        df.memory_usage(deep=True).sum()/1024/1024,
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

    # Outlier Detection (Noise)
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


# ─── Existing Endpoints ───────────────────────────────────


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    filename = file.filename

    if filename.endswith(".csv"):

        df = pd.read_csv(file.file)

    elif filename.endswith(".xlsx"):

        df = pd.read_excel(file.file)

    else:

        raise HTTPException(
            status_code=400,
            detail="Only CSV or XLSX file supported."
        )

    return analyze_dataset(df)


@app.get("/")
def home():
    return {
        "message": "Dataset Analyzer API Running"
    }


# ─── MongoDB Dataset CRUD Endpoints ───────────────────────


@app.post("/save-dataset", response_model=DatasetResponse)
async def save_dataset(
    file: UploadFile = File(...),
    name: str = Form(None),
    description: str = Form(None),
):
    """Upload a CSV/XLSX file and save it to MongoDB."""

    filename = file.filename

    # Determine file type and parse
    if filename.endswith(".csv"):
        df = pd.read_csv(file.file)
        file_type = "csv"
    elif filename.endswith(".xlsx"):
        df = pd.read_excel(file.file)
        file_type = "xlsx"
    else:
        raise HTTPException(
            status_code=400,
            detail="Only CSV or XLSX file supported."
        )

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
        uploaded_at=datetime.utcnow(),
    )

    # Build the document to insert
    document = {
        "metadata": metadata.model_dump(),
        "data": records,
    }

    # Insert into MongoDB
    result = await datasets_collection.insert_one(document)

    return DatasetResponse(
        id=str(result.inserted_id),
        message="Dataset saved successfully",
        metadata=metadata,
    )


@app.get("/datasets")
async def list_datasets():
    """List all saved datasets (metadata only)."""

    datasets = []

    cursor = datasets_collection.find(
        {}, {"data": 0}  # Exclude row data for performance
    )

    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        datasets.append(doc)

    return {"total": len(datasets), "datasets": datasets}


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

    return {
        "message": "Dataset deleted successfully",
        "id": dataset_id,
    }