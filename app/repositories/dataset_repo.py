"""Repository layer for datasets collection."""

import logging
from bson import ObjectId
from app.database import datasets_collection

logger = logging.getLogger(__name__)


async def find_by_id(dataset_id: str, user_id: str) -> dict | None:
    """Find a single dataset by ID and user."""
    return await datasets_collection.find_one(
        {"_id": ObjectId(dataset_id), "user_id": user_id}
    )


async def insert(document: dict) -> str:
    """Insert a dataset document and return its string ID."""
    result = await datasets_collection.insert_one(document)
    return str(result.inserted_id)


async def delete_by_id(dataset_id: str, user_id: str) -> int:
    """Delete a dataset and return the deleted count."""
    result = await datasets_collection.delete_one(
        {"_id": ObjectId(dataset_id), "user_id": user_id}
    )
    return result.deleted_count


async def find_many(
    user_id: str,
    skip: int = 0,
    limit: int = 20,
    sort_by: str = "uploaded_at",
) -> tuple[list[dict], int]:
    """List datasets with pagination. Returns (docs, total_count)."""
    query_filter = {"user_id": user_id}

    sort_order = [("metadata.uploaded_at", -1)]
    if sort_by == "name":
        sort_order = [("metadata.name", 1)]

    cursor = datasets_collection.find(
        query_filter, {"data": 0}
    ).sort(sort_order).skip(skip).limit(limit)

    datasets = []
    async for doc in cursor:
        doc["_id"] = str(doc["_id"])
        datasets.append(doc)

    total = await datasets_collection.count_documents(query_filter)
    return datasets, total
