"""Analysis history router."""

import logging
from fastapi import APIRouter, Depends
from bson import ObjectId

from app.core.auth import get_current_user
from app.core.exceptions import InvalidIdError, AnalysisNotFoundError
from app.repositories import analysis_repo

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/analyses", tags=["Analyses"])


@router.get("")
async def list_analyses(
    skip: int = 0,
    limit: int = 20,
    user_id: str = Depends(get_current_user),
) -> dict:
    """List saved analyses with pagination."""
    analyses, total = await analysis_repo.find_many(user_id, skip, limit)
    return {"total": total, "skip": skip, "limit": limit, "analyses": analyses}


@router.get("/{analysis_id}")
async def get_analysis(
    analysis_id: str,
    user_id: str = Depends(get_current_user),
) -> dict:
    """Retrieve a specific analysis result."""
    if not ObjectId.is_valid(analysis_id):
        raise InvalidIdError("analysis", analysis_id)

    doc = await analysis_repo.find_by_id(analysis_id, user_id)
    if not doc:
        raise AnalysisNotFoundError(analysis_id)

    doc["_id"] = str(doc["_id"])
    return doc
