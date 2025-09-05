import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from sqlmodel import JSON, Column, Field, Relationship, SQLModel
from sqlalchemy import func, TypeDecorator
from app.core.timezone import get_current_time_utc_plus_8
from .base import TimezoneAwareModel
import json

if TYPE_CHECKING:
    from .dataset import Dataset
    from .evaluation import EvaluationResult


class JSONEncodedDict(TypeDecorator):
    """Represents an immutable structure as a json-encoded string."""
    
    impl = JSON
    
    def process_bind_param(self, value, dialect):
        if value is not None:
            value = json.dumps(value)
        return value
    
    def process_result_value(self, value, dialect):
        if value is not None:
            value = json.loads(value)
        return value


class Message(SQLModel):
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content")


class SampleBase(SQLModel):
    """
    SampleBase model for the database.
    fields are the same as Ragas SingleTurnSample.
    """
    user_input: str = Field(..., description="User input - can be string or list of messages")
    retrieved_contexts: Optional[List[str]] = Field(None, description="List of retrieved contexts", sa_column=Column(JSONEncodedDict))
    reference_contexts: Optional[List[str]] = Field(None, description="List of reference contexts", sa_column=Column(JSONEncodedDict))
    response: Optional[str] = Field(None, description="Generated response")
    multi_responses: Optional[List[str]] = Field(None, description="List of multiple responses", sa_column=Column(JSONEncodedDict))
    reference: Optional[str] = Field(None, description="Reference answer")
    rubrics: Optional[Dict[str, Any]] = Field(None, description="Dictionary of rubrics", sa_column=Column(JSONEncodedDict))


class SampleCreate(SampleBase):
    dataset_id: Optional[str] = Field(None, description="Dataset ID")


class SampleUpdate(SQLModel):
    user_input: Optional[Union[str, List[Dict[str, Any]]]] = Field(None, description="User input")
    retrieved_contexts: Optional[List[str]] = Field(None, description="List of retrieved contexts")
    reference_contexts: Optional[List[str]] = Field(None, description="List of reference contexts")
    response: Optional[str] = Field(None, description="Generated response")
    multi_responses: Optional[List[str]] = Field(None, description="List of multiple responses")
    reference: Optional[str] = Field(None, description="Reference answer")
    rubrics: Optional[Dict[str, Any]] = Field(None, description="Dictionary of rubrics")


class Sample(SampleBase, table=True):  # pyright: ignore[reportCallIssue, reportGeneralTypeIssues]
    __tablename__ = "samples"
    
    sample_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    dataset_id: str = Field(foreign_key="datasets.dataset_id", index=True)
    created_at: datetime = Field(default_factory=get_current_time_utc_plus_8, sa_column_kwargs={"server_default": func.now()})
    updated_at: datetime = Field(default_factory=get_current_time_utc_plus_8, sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()})
    
    # Relationships
    dataset: "Dataset" = Relationship(back_populates="samples")
    evaluation_results: List["EvaluationResult"] = Relationship(back_populates="sample", cascade_delete=True)


class SampleResponse(SampleBase, TimezoneAwareModel):
    sample_id: str
    dataset_id: str
    created_at: datetime
    updated_at: datetime


class SampleListResponse(SQLModel):
    samples: List[SampleResponse]
    total: int
    page: int
    size: int


class SampleDeleteResponse(SQLModel):
    message: str
    sample_id: str


class SampleBulkDeleteResponse(SQLModel):
    message: str
    dataset_name: str


class SampleBulkCreate(SQLModel):
    samples: List[SampleCreate]


class SampleBulkResponse(SQLModel):
    inserted_count: int
    failed_count: int
    errors: List[Dict[str, Any]]
