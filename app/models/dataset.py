from sqlmodel import SQLModel, Field, Relationship
from pydantic import model_validator
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from datetime import datetime
from sqlalchemy import JSON, Column, func
from app.core.timezone import get_current_time_utc_plus_8
from .base import TimezoneAwareModel
import uuid
import json

if TYPE_CHECKING:
    from .sample import Sample
    from .evaluation import Evaluation


class DatasetBase(SQLModel):
    name: str = Field(..., min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    sample_type: str = Field(..., regex="^(single_turn|multi_turn)$", description="Sample type")
    metadata_json: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))


class DatasetCreate(DatasetBase):
    pass


class DatasetUpdate(SQLModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255, description="Dataset name")
    description: Optional[str] = Field(None, description="Dataset description")
    metadata_json: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class Dataset(DatasetBase, table=True):
    __tablename__ = "datasets"
    
    dataset_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    created_at: datetime = Field(default_factory=get_current_time_utc_plus_8, sa_column_kwargs={"server_default": func.now()})
    updated_at: datetime = Field(default_factory=get_current_time_utc_plus_8, sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()})
    
    # Relationships
    samples: List["Sample"] = Relationship(back_populates="dataset", cascade_delete=True)
    evaluations: List["Evaluation"] = Relationship(back_populates="dataset", cascade_delete=True)
    
    @property
    def sample_count(self) -> int:
        return len(self.samples) if self.samples else 0
    
    @property
    def metadata_json_dict(self) -> Dict[str, Any]:
        """Ensure metadata_json is always returned as a dictionary"""
        metadata = getattr(self, 'metadata_json', None)
        if metadata is None:
            return {}
        
        # If it's already a dict, return it
        if isinstance(metadata, dict):
            return metadata
        
        # If it's a string, try to parse it
        if isinstance(metadata, str):
            try:
                parsed = json.loads(metadata)
                if isinstance(parsed, dict):
                    return parsed
            except (json.JSONDecodeError, TypeError):
                pass
        
        # Fallback to empty dict
        return {}


class DatasetResponse(DatasetBase, TimezoneAwareModel):
    dataset_id: str
    created_at: datetime
    updated_at: datetime
    sample_count: int
    metadata_json: Dict[str, Any] = Field(default_factory=dict)
    
    @model_validator(mode='before')
    @classmethod
    def convert_metadata_json(cls, values: DatasetBase):
        """Convert metadata_json from string to dict if needed"""
        
        metadata = values.metadata_json
        if isinstance(metadata, str):
            try:
                values.metadata_json = json.loads(metadata)
            except (json.JSONDecodeError, TypeError):
                values.metadata_json = {}
        elif metadata is None:
            values.metadata_json = {}
        return values


class DatasetListResponse(SQLModel):
    datasets: List[DatasetResponse]
    total: int
    page: int
    size: int


class DatasetDeleteResponse(SQLModel):
    message: str
    dataset_id: str
