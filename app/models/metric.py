from sqlmodel import SQLModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from sqlalchemy import JSON, func
from app.core.timezone import get_current_time_utc_plus_8
from .base import TimezoneAwareModel
import uuid


class MetricBase(SQLModel):
    name: str = Field(..., min_length=1, max_length=255, description="Metric name")
    description: Optional[str] = Field(None, description="Metric description")
    metric_type: str = Field(..., description="Metric type: llm_based, embedding_based, rule_based")
    supported_sample_types: str = Field(..., description="List of supported sample types")
    parameters: Optional[str] = Field(None, description="Default parameters")
    default_config: Optional[str] = Field(None, description="Default LLM/embedding config")
    example_usage: Optional[str] = Field(None, description="Example usage")


class MetricCreate(MetricBase):
    pass


class MetricUpdate(SQLModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Metric name")
    description: Optional[str] = Field(None, description="Metric description")
    metric_type: Optional[str] = Field(None, description="Metric type")
    supported_sample_types: Optional[List[str]] = Field(None, description="List of supported sample types")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Default parameters")
    default_config: Optional[Dict[str, Any]] = Field(None, description="Default LLM/embedding config")
    example_usage: Optional[Dict[str, Any]] = Field(None, description="Example usage")
    is_active: Optional[bool] = Field(None, description="Whether metric is active")


class Metric(MetricBase, table=True):
    __tablename__ = "metrics"
    
    metric_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    is_active: bool = Field(default=True, description="Whether metric is active")
    created_at: datetime = Field(default_factory=get_current_time_utc_plus_8, sa_column_kwargs={"server_default": func.now()})
    updated_at: datetime = Field(default_factory=get_current_time_utc_plus_8, sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()})


class MetricResponse(MetricBase, TimezoneAwareModel):
    metric_id: str
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class MetricListResponse(SQLModel):
    metrics: List[MetricResponse]
    total: int
    page: int
    size: int


class MetricDeleteResponse(SQLModel):
    message: str
    metric_id: str
