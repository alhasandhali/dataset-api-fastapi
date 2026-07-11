"""Background ML pipeline worker."""

import logging
from datetime import datetime, timezone

import pandas as pd
from io import BytesIO
from starlette.concurrency import run_in_threadpool

from app.core.storage import download_from_s3
from app.core.analysis import analyze_dataset, make_json_safe
from app.repositories import analysis_repo

logger = logging.getLogger(__name__)


async def run_ml_pipeline(task_id: str, s3_key: str, user_id: str, filename: str):
    """Run the ML analysis pipeline in the background.

    All sync operations (S3 download, Pandas parsing, analysis) are
    offloaded to a threadpool to avoid blocking the event loop.
    """
    logger.info("Starting ML pipeline for file: %s", filename)
    try:
        # Download data (non-blocking)
        file_bytes = await run_in_threadpool(download_from_s3, s3_key)
        if not file_bytes:
            raise ValueError("Empty or missing dataset from S3")

        # Parse (non-blocking)
        if filename.endswith(".csv"):
            df = await run_in_threadpool(pd.read_csv, BytesIO(file_bytes))
        else:
            df = await run_in_threadpool(pd.read_excel, BytesIO(file_bytes))

        # Analyze (non-blocking)
        analysis_result = await run_in_threadpool(analyze_dataset, df)
        safe_result = make_json_safe(analysis_result)

        # Update Mongo document
        await analysis_repo.update_status(task_id, {
            "analysis": safe_result,
            "status": "completed",
            "analyzed_at": datetime.now(timezone.utc),
        })
        logger.info("ML pipeline completed for task_id: %s", task_id)

    except Exception as e:
        logger.error("Error in ML pipeline: %s", e)
        await analysis_repo.update_status(task_id, {
            "status": "failed",
            "error": str(e),
            "analyzed_at": datetime.now(timezone.utc),
        })
