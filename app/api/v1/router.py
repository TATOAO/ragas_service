from fastapi import APIRouter

from app.api.v1.datasets import router as datasets_router
from app.api.v1.evaluation import router as evaluation_router
from app.api.v1.metrics import router as metrics_router
from app.api.v1.config import router as config_router

api_router = APIRouter()

# Include all route modules
api_router.include_router(datasets_router, prefix="/datasets", tags=["datasets"])
api_router.include_router(evaluation_router, prefix="/evaluate", tags=["evaluation"])
api_router.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
api_router.include_router(config_router, prefix="/config", tags=["configuration"])
