"""Pure Pandas transformation functions.

These functions take a DataFrame in and return a DataFrame out.
No I/O, no database, no HTTP — purely testable data transformations.
"""

import pandas as pd
import numpy as np


def clean_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Remove duplicate rows. Returns (cleaned_df, rows_removed)."""
    initial_rows = len(df)
    df = df.drop_duplicates()
    return df, initial_rows - len(df)


def solve_missing_values(df: pd.DataFrame, drop_threshold: float = 0.5) -> pd.DataFrame:
    """Dynamically impute missing values based on column characteristics.

    Strategy per column:
    - >50% missing → drop the column entirely
    - Numeric + high skew (|skew| > 1) → fill with median
    - Numeric + normal distribution → fill with mean
    - Categorical → fill with mode (or "Unknown" if no mode)
    """
    for col in list(df.columns):
        missing_pct = df[col].isnull().mean()
        if missing_pct == 0:
            continue

        if missing_pct > drop_threshold:
            df = df.drop(columns=[col])
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            skewness = df[col].skew()
            if pd.notna(skewness) and abs(skewness) > 1:
                fill_val = df[col].median()
            else:
                fill_val = df[col].mean()
            df[col] = df[col].fillna(fill_val)
        else:
            mode_series = df[col].mode()
            if not mode_series.empty:
                df[col] = df[col].fillna(mode_series.iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")

    return df


def clean_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Clean noisy data using intelligent heuristics.

    Steps:
    1. Text Standardization — lowercase + strip whitespace
    2. Data Type Enforcement — coerce object columns to numeric where possible
    3. Logical Error Fix — abs() on columns matching age/price/salary/height/weight
    4. Outlier Capping — IQR Winsorization on all numeric columns
    """
    # 1. Text Standardization (Categorical columns)
    for col in df.select_dtypes(include=["object"]).columns:
        try:
            df[col] = df[col].astype(str).str.lower().str.strip()
        except Exception:
            pass

    # 2. Data Type Enforcement
    for col in df.select_dtypes(include=["object"]).columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > len(df) * 0.5:
            df[col] = converted

    # 3. Logical Errors (Heuristics)
    keywords = ["age", "price", "salary", "height", "weight"]
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].abs()

    # 4. Outlier Handling (IQR Winsorization)
    for col in df.select_dtypes(include=["number"]).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    return df
