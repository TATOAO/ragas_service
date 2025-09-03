from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from pydantic import SecretStr
from datetime import datetime
from sqlalchemy import JSON, func
import uuid

if TYPE_CHECKING:
    from .dataset import Dataset
    from .sample import Sample


# Additional evaluation schemas
class MetricConfig(SQLModel):
    name: str = Field(..., description="Metric name")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Metric parameters")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

class LLMConfig(SQLModel):
    provider: str = Field(..., description="LLM provider")
    model: str = Field(..., description="Model name")
    api_key: str = Field(..., description="API key")

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return getattr(self, key, default)

class EvaluationBase(SQLModel):
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    metrics: List[MetricConfig] = Field(..., description="Metrics to evaluate", sa_type=JSON)
    llm_config: Optional[LLMConfig] = Field(None, description="LLM configuration", sa_type=JSON)
    embeddings_config: Optional[str] = Field(None, description="Embeddings configuration")
    batch_size: int = Field(default=10, description="Batch size")


class EvaluationCreate(EvaluationBase):
    dataset_id: Optional[str] = Field(default=None, description="Dataset ID")
    dataset_name: Optional[str] = Field(default=None, description="Dataset name")

class EvaluationUpdate(SQLModel):
    experiment_name: Optional[str] = Field(None, description="Experiment name")
    metrics: Optional[Any] = Field(None, description="List of metric configurations")
    llm_config: Optional[Any] = Field(None, description="LLM configuration")
    embeddings_config: Optional[Dict[str, Any]] = Field(None, description="Embeddings configuration")


class Evaluation(EvaluationBase, table=True):
    __tablename__ = "evaluations"
    
    evaluation_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    dataset_id: str = Field(foreign_key="datasets.dataset_id", index=True)
    
    # Status
    status: str = Field(default="pending", description="Evaluation status: pending, running, completed, failed")
    progress: float = Field(default=0.0, description="Evaluation progress (0.0 to 1.0)")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    
    # Results
    overall_scores: Optional[Dict[str, Any]] = Field(None, description="Overall metric scores", sa_type=JSON)
    cost_analysis: Optional[Dict[str, Any]] = Field(None, description="Cost information", sa_type=JSON)
    traces: Optional[Dict[str, Any]] = Field(None, description="Trace URLs", sa_type=JSON)
    
    # Timestamps
    started_at: Optional[datetime] = Field(None, description="When evaluation started")
    completed_at: Optional[datetime] = Field(None, description="When evaluation completed")
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.now()})
    updated_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.now(), "onupdate": func.now()})
    
    # Relationships
    dataset: "Dataset" = Relationship(back_populates="evaluations")
    results: List["EvaluationResult"] = Relationship(back_populates="evaluation", cascade_delete=True)


class EvaluationResponse(EvaluationBase):
    evaluation_id: str
    dataset_id: str
    status: str
    progress: float
    error_message: Optional[str]
    overall_scores: Optional[str] = Field(None, description="Overall metric scores")
    cost_analysis: Optional[str] = Field(None, description="Cost information")
    traces: Optional[str] = Field(None, description="Trace URLs")
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class EvaluationListResponse(SQLModel):
    evaluations: List[EvaluationResponse]
    total: int
    page: int
    size: int


class EvaluationDeleteResponse(SQLModel):
    message: str
    evaluation_id: str


# Evaluation Result Models
class EvaluationResultBase(SQLModel):
    scores: Dict[str, Any] = Field(..., description="Metric scores for this sample", sa_type=JSON)
    reasoning: Optional[str] = Field(None, description="Reasoning for scores")
    cost: Optional[Dict[str, Any]] = Field(None, description="Cost for this sample", sa_type=JSON)


class EvaluationResultCreate(EvaluationResultBase):
    evaluation_id: str = Field(..., description="Evaluation ID")
    sample_id: str = Field(..., description="Sample ID")


class EvaluationResult(EvaluationResultBase, table=True):
    __tablename__ = "evaluation_results"
    
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()), primary_key=True)
    evaluation_id: str = Field(foreign_key="evaluations.evaluation_id", index=True)
    sample_id: str = Field(foreign_key="samples.sample_id", index=True)
    created_at: datetime = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.now()})
    
    # Relationships
    evaluation: Evaluation = Relationship(back_populates="results")
    sample: "Sample" = Relationship(back_populates="evaluation_results")


class EvaluationResultResponse(EvaluationResultBase):
    result_id: str
    evaluation_id: str
    sample_id: str
    created_at: datetime
    
    class Config:
        from_attributes = True



class EmbeddingsConfig(SQLModel):
    provider: str = Field(..., description="Embeddings provider")
    model: str = Field(..., description="Model name")
    api_key: str = Field(..., description="API key")


class EvaluationStatusResponse(SQLModel):
    evaluation_id: str
    status: str
    progress: float
    started_at: Optional[str]
    completed_at: Optional[str]
    error_message: Optional[str]


class SampleScore(SQLModel):
    sample_id: str
    scores: Dict[str, float]


class EvaluationResultsResponse(SQLModel):
    evaluation_id: str
    dataset_id: str
    experiment_name: Optional[str]
    metrics: Any
    sample_scores: List[SampleScore]
    cost_analysis: Optional[Dict[str, Any]]
    traces: List[Dict[str, str]]
    created_at: str


class SingleEvaluationRequest(SQLModel):
    sample: Dict[str, Any] = Field(..., description="Sample to evaluate")
    metrics: Any = Field(..., description="Metrics to evaluate")
    llm_config: Optional[Any] = Field(None, description="LLM configuration")


class SingleEvaluationResponse(SQLModel):
    sample_id: str
    scores: Dict[str, float]
    reasoning: Optional[Dict[str, str]]
    cost: Optional[Dict[str, Any]]


class EvaluationComparisonRequest(SQLModel):
    evaluation_ids: List[str] = Field(..., min_items=2, description="Evaluation IDs to compare")
    metrics: Any = Field(..., description="Metrics to compare")


class EvaluationComparisonResponse(SQLModel):
    comparison: Dict[str, Any]
