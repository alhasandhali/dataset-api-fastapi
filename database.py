"""MongoDB connection and collection references for Dataset API."""

import os
import logging

from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

DB_USER: str | None = os.getenv("DB_USER")
DB_PASS: str | None = os.getenv("DB_PASS")

MONGO_URI: str = (
    f"mongodb+srv://{DB_USER}:{DB_PASS}"
    f"@cluster-1.dymuola.mongodb.net/?appName=Cluster-1"
)

client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
db = client["dataset_api_db"]
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
