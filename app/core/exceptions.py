"""Custom exception classes and centralized error handling."""

from fastapi import Request
from fastapi.responses import JSONResponse


class AppException(Exception):
    """Base exception for application-level errors."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class InvalidIdError(AppException):
    def __init__(self, resource: str, resource_id: str):
        super().__init__(400, f"Invalid {resource} ID format: '{resource_id}'")


class DatasetNotFoundError(AppException):
    def __init__(self, dataset_id: str):
        super().__init__(404, f"Dataset '{dataset_id}' not found.")


class AnalysisNotFoundError(AppException):
    def __init__(self, analysis_id: str):
        super().__init__(404, f"Analysis '{analysis_id}' not found.")


class MissingS3KeyError(AppException):
    def __init__(self, dataset_id: str):
        super().__init__(400, f"Missing S3 key for dataset '{dataset_id}'.")


class StorageFileNotFoundError(AppException):
    def __init__(self):
        super().__init__(
            404,
            "Dataset file not found in storage. "
            "It may have been deleted due to ephemeral storage restart. "
            "Please re-upload the dataset.",
        )


class FileTooLargeError(AppException):
    def __init__(self, max_mb: int):
        super().__init__(413, f"File too large. Maximum size is {max_mb}MB.")


class UnsupportedFileTypeError(AppException):
    def __init__(self):
        super().__init__(400, "Only CSV or XLSX files are supported.")


class EmptyDatasetError(AppException):
    def __init__(self):
        super().__init__(400, "Uploaded file contains no data rows.")


# ─── Global Exception Handler ────────────────────────────


async def app_exception_handler(request: Request, exc: AppException) -> JSONResponse:
    """Return a structured JSON error response for all AppExceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
