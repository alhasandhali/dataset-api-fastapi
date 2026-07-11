"""Pure Pandas transformation functions.

These functions take a DataFrame in and return a DataFrame out.
No I/O, no database, no HTTP — purely testable data transformations.
"""

import pandas as pd
import numpy as np

from app.schemas.datasets import MLPrepRequest


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


def automated_ml_prep(df: pd.DataFrame, config: MLPrepRequest) -> pd.DataFrame:
    """Apply automated machine learning data preparation."""
    
    # 1. Drop Irrelevant
    if config.drop_irrelevant:
        cols_to_drop = []
        for col in df.columns:
            n_unique = df[col].nunique()
            # Drop if all values are unique (like IDs) or only 1 unique value (constant)
            if n_unique == len(df) or n_unique <= 1:
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)

    # 2. Parse Dates
    if config.parse_dates:
        datetime_cols = list(df.select_dtypes(include=['datetime64', 'datetimetz']).columns)
        
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            parsed = pd.to_datetime(df[col], errors='coerce')
            if parsed.notna().mean() > 0.5:
                df[col] = parsed
                datetime_cols.append(col)
                
        new_date_cols = {}
        cols_to_drop = []
        for col in set(datetime_cols):
            if col in df.columns:
                # memory-efficient nullable ints
                new_date_cols[f'{col}_year'] = df[col].dt.year.astype('Int16')
                new_date_cols[f'{col}_month'] = df[col].dt.month.astype('Int8')
                new_date_cols[f'{col}_day'] = df[col].dt.day.astype('Int8')
                cols_to_drop.append(col)
                
        if new_date_cols:
            df = df.drop(columns=cols_to_drop)
            # Create dataframe from dict to avoid fragmentation
            date_df = pd.DataFrame(new_date_cols, index=df.index)
            df = pd.concat([df, date_df], axis=1)

    # 3. Encode Categorical
    if config.encode_categorical:
        cat_cols = df.select_dtypes(include=['object', 'category']).columns
        dummies_list = []
        cols_to_drop = []
        for col in cat_cols:
            n_unique = df[col].nunique()
            if n_unique == 2:
                # Binary Label Encoding
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
                    df[col] = df[col].map(mapping).astype('Int8')
            elif 2 < n_unique < 15:
                # One-Hot Encoding
                dummies = pd.get_dummies(df[col], prefix=col, dtype='int8')
                dummies_list.append(dummies)
                cols_to_drop.append(col)
                
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            if dummies_list:
                df = pd.concat([df] + dummies_list, axis=1)

    # 4. Scale Features
    if config.scale_features:
        num_cols = df.select_dtypes(include=['number']).columns
        for col in num_cols:
            if df[col].nunique() <= 2:
                continue # Skip binary/OHE columns
            
            c_min = df[col].min()
            c_max = df[col].max()
            if c_max > c_min:
                df[col] = (df[col] - c_min) / (c_max - c_min)
                df[col] = df[col].astype('float32')

    return df
