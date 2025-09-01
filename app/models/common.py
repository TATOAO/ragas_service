from sqlmodel import SQLModel, Field
from typing import Optional, Dict, Any
from datetime import datetime


class PaginationParams(SQLModel):
    page: int = Field(1, ge=1, description="Page number")
    size: int = Field(10, ge=1, le=100, description="Page size")


class ErrorResponse(SQLModel):
    error: Dict[str, Any]
    timestamp: str
    request_id: str


class HealthResponse(SQLModel):
    status: str
    services: Dict[str, str]
    timestamp: str


class RootResponse(SQLModel):
    message: str
    version: str
    docs: str


# Additional common schemas for metrics
class LLMProvidersResponse(SQLModel):
    providers: list[Dict[str, Any]]


class EmbeddingProvidersResponse(SQLModel):
    providers: list[Dict[str, Any]]
