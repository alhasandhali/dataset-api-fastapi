"""Repository layer for analyses collection."""

import logging
from bson import ObjectId
from app.database import analyses_collection

logger = logging.getLogger(__name__)


async def find_by_id(analysis_id: str, user_id: str) -> dict | None:
    """Find a single analysis by ID and user."""
    return await analyses_collection.find_one(
        {"_id": ObjectId(analysis_id), "user_id": user_id}
    )


async def insert(document: dict) -> str:
    """Insert an analysis document and return its string ID."""
    result = await analyses_collection.insert_one(document)
    return str(result.inserted_id)


async def update_status(task_id: str, update_fields: dict) -> None:
    """Update an analysis document's fields."""
    await analyses_collection.update_one(
        {"_id": ObjectId(task_id)},
        {"$set": update_fields},
    )


async def find_many(
    user_id: str,
    skip: int = 0,
    limit: int = 20,
) -> tuple[list[dict], int]:
    """List analyses with pagination. Returns (docs, total_count)."""
    query_filter = {"user_id": user_id}

    cursor = analyses_collection.find(
        query_filter, {"analysis": 0}
    ).skip(skip).limit(limit)

    analyses = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        analyses.append(doc)

    total = await analyses_collection.count_documents(query_filter)
    return analyses, total
