import pandas as pd
import numpy as np

def analyze_dataset(df: pd.DataFrame) -> dict:
    """Perform comprehensive statistical analysis on a DataFrame."""
    result = {}
    row_count = len(df)
    result["rows"] = row_count
    result["columns"] = len(df.columns)
    
    column_info = []
    for col in df.columns:
        missing = int(df[col].isnull().sum())
        column_info.append({
            "column_name": col,
            "dtype": str(df[col].dtype),
            "missing_values": missing,
            "missing_percentage": round((missing / row_count) * 100, 2) if row_count > 0 else 0,
            "unique_values": int(df[col].nunique())
        })
    result["column_info"] = column_info
    result["duplicate_rows"] = int(df.duplicated().sum())
    result["memory_usage_MB"] = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)
    
    numeric_cols = df.select_dtypes(include=np.number)
    if not numeric_cols.empty:
        result["numeric_summary"] = numeric_cols.describe().to_dict()
        
    categorical_cols = df.select_dtypes(include="object")
    cat_summary = {}
    for col in categorical_cols.columns:
        cat_summary[col] = {
            "top_value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            "top_frequency": int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0
        }
    result["categorical_summary"] = cat_summary
    
    outliers = {}
    for col in numeric_cols.columns:
        q1 = numeric_cols[col].quantile(0.25)
        q3 = numeric_cols[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = numeric_cols[(numeric_cols[col] < lower) | (numeric_cols[col] > upper)][col].count()
        outliers[col] = int(count)
    result["outliers"] = outliers
    
    return result

def make_json_safe(data):
    """Convert NaN, Infinity values to None for JSON serialization."""
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(item) for item in data]
    elif isinstance(data, float) and (np.isnan(data) or np.isinf(data)):
        return None
    return data
