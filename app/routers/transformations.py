"""Dataset transformation router — clean-duplicates, solve-missing, clean-noise.

Each endpoint follows the same pattern:
  1. get_validated_doc()  → validate ID + fetch from DB
  2. load_dataframe()     → download from S3 + parse
  3. transformation.xxx() → pure Pandas logic
  4. save_transformed()   → serialize + S3 + MongoDB + background pipeline

Adding a new transformation (e.g., encoding, scaling) requires only:
  - A new pure function in services/transformation.py
  - A ~10-line endpoint here
"""

import logging
from fastapi import APIRouter, BackgroundTasks, Depends

from app.core.auth import get_current_user
from app.services import dataset_service, transformation
from app.schemas.datasets import TransformationResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/datasets", tags=["Transformations"])


@router.post("/{dataset_id}/clean-duplicates")
async def clean_duplicates(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> TransformationResponse | dict:
    """Remove duplicate rows and save as a new dataset."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)
    df, metadata = await dataset_service.load_dataframe(doc)

    df, rows_removed = transformation.clean_duplicates(df)

    if rows_removed == 0:
        return {"message": "No duplicates found to remove", "dataset_id": dataset_id}

    return await dataset_service.save_transformed(
        df, metadata, "_cleaned", user_id, background_tasks,
    )


@router.post("/{dataset_id}/solve-missing-values")
async def solve_missing_values(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> TransformationResponse | dict:
    """Dynamically impute missing values and save as a new dataset."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)
    df, metadata = await dataset_service.load_dataframe(doc)

    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        return {"message": "No missing values found", "dataset_id": dataset_id}

    df = transformation.solve_missing_values(df)

    return await dataset_service.save_transformed(
        df, metadata, "_nomissing", user_id, background_tasks,
    )


@router.post("/{dataset_id}/clean-noise")
async def clean_noise(
    dataset_id: str,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user),
) -> TransformationResponse:
    """Clean noisy data and save as a new dataset."""
    doc = await dataset_service.get_validated_doc(dataset_id, user_id)
    df, metadata = await dataset_service.load_dataframe(doc)

    df = transformation.clean_noise(df)

    return await dataset_service.save_transformed(
        df, metadata, "_cleaned_noise", user_id, background_tasks,
    )
