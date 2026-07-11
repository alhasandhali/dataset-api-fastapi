"""Task status and file upload/analysis router."""

import logging
from datetime import datetime, timezone

from bson import ObjectId
from fastapi import APIRouter, UploadFile, File, BackgroundTasks, Depends
from starlette.concurrency import run_in_threadpool

from app.core.auth import get_current_user
from app.core.storage import upload_to_s3
from app.config import settings
from app.core.exceptions import FileTooLargeError
from app.repositories import analysis_repo
from app.workers.ml_worker import run_ml_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tasks"])


@router.post("/analyze")
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user),
) -> dict:
    """Upload a file and trigger asynchronous ML analysis."""
    file_bytes = await file.read()

    if len(file_bytes) > settings.MAX_FILE_SIZE:
        raise FileTooLargeError(settings.MAX_FILE_SIZE // (1024 * 1024))

    s3_key = await run_in_threadpool(upload_to_s3, file_bytes, file.filename, user_id)

    task_id = await analysis_repo.insert({
        "user_id": user_id,
        "filename": file.filename,
        "status": "processing",
        "created_at": datetime.now(timezone.utc),
    })

    background_tasks.add_task(run_ml_pipeline, task_id, s3_key, user_id, file.filename)

    return {"task_id": task_id, "status": "processing", "s3_key": s3_key}


@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: str = Depends(get_current_user),
) -> dict:
    """Poll the status of a background analysis task."""
    if not ObjectId.is_valid(task_id):
        return {"status": "failed", "error": "Invalid task ID"}

    doc = await analysis_repo.find_by_id(task_id, current_user)
    if not doc:
        return {"status": "failed", "error": "Task not found"}

    if doc.get("status") == "processing":
        return {"status": "processing"}
    elif doc.get("status") == "completed":
        doc["analysis"]["id"] = str(doc["_id"])
        return {"status": "completed", "result": doc.get("analysis")}
    else:
        return {"status": "failed", "error": doc.get("error", "Unknown error")}
