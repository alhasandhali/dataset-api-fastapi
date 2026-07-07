import logging
from datetime import datetime, timezone
import pandas as pd
from io import BytesIO
from core.celery_app import celery_app
from core.storage import download_dataset_from_s3
from core.analysis import analyze_dataset, make_json_safe
from database import analyses_collection

logger = logging.getLogger(__name__)

@celery_app.task(bind=True)
def run_ml_pipeline(self, s3_key: str, user_id: str, filename: str):
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
        
        # Save to Mongo directly since it's a worker (needs motor event loop but celery is sync, so we need to use pymongo or asyncio.run)
        import asyncio
        async def save_to_mongo():
            analysis_doc = {
                "user_id": user_id,
                "filename": filename,
                "analysis": safe_result,
                "analyzed_at": datetime.now(timezone.utc),
            }
            db_result = await analyses_collection.insert_one(analysis_doc)
            return str(db_result.inserted_id)
            
        inserted_id = asyncio.run(save_to_mongo())
        safe_result["id"] = inserted_id
        
        return {"status": "completed", "result": safe_result}
        
    except Exception as e:
        logger.error(f"Error in ML pipeline: {e}")
        return {"status": "failed", "error": str(e)}
