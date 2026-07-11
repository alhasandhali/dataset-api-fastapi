"""MongoDB connection and collection references."""

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from app.config import settings

logger = logging.getLogger(__name__)

client: AsyncIOMotorClient = AsyncIOMotorClient(settings.MONGO_URI, tz_aware=True)
db = client[settings.DB_NAME]
datasets_collection = db["datasets"]
analyses_collection = db["analyses"]


async def create_indexes() -> None:
    """Create database indexes for query performance."""
    await datasets_collection.create_index("user_id")
    await analyses_collection.create_index("user_id")
    logger.info("Database indexes ensured")


async def close_connection() -> None:
    """Close the MongoDB client connection."""
    client.close()
    logger.info("MongoDB connection closed")
