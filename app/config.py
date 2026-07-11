"""Application configuration via pydantic-settings."""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central configuration. All values read from environment variables."""

    # MongoDB
    DB_USER: str = os.getenv("DB_USER", "")
    DB_PASS: str = os.getenv("DB_PASS", "")
    MONGO_URI: str = (
        f"mongodb+srv://{os.getenv('DB_USER', '')}:{os.getenv('DB_PASS', '')}"
        f"@cluster-1.dymuola.mongodb.net/?appName=Cluster-1"
    )
    DB_NAME: str = os.getenv("DB_NAME", "dataset_api_db")

    # AWS S3
    AWS_ACCESS_KEY_ID: str | None = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY: str | None = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET_NAME: str | None = os.getenv("S3_BUCKET_NAME")

    # Auth
    SECRET_KEY: str = os.getenv("SECRET_KEY", "")
    ALGORITHM: str = "HS256"

    # Upload limits
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50 MB

    # CORS
    CORS_ORIGINS: list[str] = ["*"]


settings = Settings()
