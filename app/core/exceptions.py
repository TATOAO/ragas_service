from typing import Dict, Any, Optional


class RAGASException(Exception):
    """Base exception for RAGAS service"""
    
    def __init__(
        self,
        message: str,
        error_code: str,
        status_code: int = 500,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.status_code = status_code
        self.details = details or {}
        super().__init__(self.message)


class ValidationError(RAGASException):
    """Validation error exception"""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class DatasetNotFoundError(RAGASException):
    """Dataset not found exception"""
    
    def __init__(self, dataset_id: str):
        super().__init__(
            message=f"Dataset with id {dataset_id} not found",
            error_code="DATASET_NOT_FOUND",
            status_code=404,
            details={"dataset_id": dataset_id}
        )


class EvaluationNotFoundError(RAGASException):
    """Evaluation not found exception"""
    
    def __init__(self, evaluation_id: str):
        super().__init__(
            message=f"Evaluation with id {evaluation_id} not found",
            error_code="EVALUATION_NOT_FOUND",
            status_code=404,
            details={"evaluation_id": evaluation_id}
        )


class MetricNotSupportedError(RAGASException):
    """Metric not supported exception"""
    
    def __init__(self, metric_name: str):
        super().__init__(
            message=f"Metric {metric_name} is not supported",
            error_code="METRIC_NOT_SUPPORTED",
            status_code=400,
            details={"metric_name": metric_name}
        )


class LLMError(RAGASException):
    """LLM API error exception"""
    
    def __init__(self, message: str, provider: str):
        super().__init__(
            message=message,
            error_code="LLM_ERROR",
            status_code=500,
            details={"provider": provider}
        )


class RateLimitExceededError(RAGASException):
    """Rate limit exceeded exception"""
    
    def __init__(self, limit: int, window: str):
        super().__init__(
            message=f"Rate limit exceeded: {limit} requests per {window}",
            error_code="RATE_LIMIT_EXCEEDED",
            status_code=429,
            details={"limit": limit, "window": window}
        )


class AuthenticationError(RAGASException):
    """Authentication error exception"""
    
    def __init__(self, message: str = "Invalid authentication credentials"):
        super().__init__(
            message=message,
            error_code="AUTHENTICATION_ERROR",
            status_code=401
        )


class AuthorizationError(RAGASException):
    """Authorization error exception"""
    
    def __init__(self, message: str = "Insufficient permissions"):
        super().__init__(
            message=message,
            error_code="AUTHORIZATION_ERROR",
            status_code=403
        )
