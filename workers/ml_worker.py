import logging
from datetime import datetime, timezone
import pandas as pd
from io import BytesIO
from bson import ObjectId

from core.storage import download_dataset_from_s3
from core.analysis import analyze_dataset, make_json_safe
from database import analyses_collection

logger = logging.getLogger(__name__)

async def run_ml_pipeline(task_id: str, s3_key: str, user_id: str, filename: str):
    logger.info(f"Starting ML pipeline for file: {filename}")
    try:
        # Download data
        file_bytes = download_dataset_from_s3(s3_key)
        if not file_bytes:
            raise ValueError("Empty or missing dataset from S3")
            
        # Parse it
        if filename.endswith(".csv"):
            df = pd.read_csv(BytesIO(file_bytes))
        else:
            df = pd.read_excel(BytesIO(file_bytes))
            
        # Analyze
        analysis_result = analyze_dataset(df)
        safe_result = make_json_safe(analysis_result)
        
        # Update Mongo document
        await analyses_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {
                "analysis": safe_result,
                "status": "completed",
                "analyzed_at": datetime.now(timezone.utc),
            }}
        )
        logger.info(f"ML pipeline completed for task_id: {task_id}")
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {e}")
        await analyses_collection.update_one(
            {"_id": ObjectId(task_id)},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "analyzed_at": datetime.now(timezone.utc),
            }}
        )
