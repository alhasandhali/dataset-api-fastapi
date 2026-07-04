from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import numpy as np
import os

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