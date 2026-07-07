import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

MONGO_URI = (
    f"mongodb+srv://{DB_USER}:{DB_PASS}"
    f"@cluster-1.dymuola.mongodb.net/?appName=Cluster-1"
)

client = AsyncIOMotorClient(MONGO_URI)
db = client["dataset_api_db"]
datasets_collection = db["datasets"]
