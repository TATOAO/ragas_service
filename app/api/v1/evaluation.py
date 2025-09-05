from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from sqlalchemy.orm import Session
from typing import List, Optional
import logging
import asyncio
from datetime import datetime, timedelta
import uuid

from app.core.database import get_db
from app.core.auth import get_current_user_api_key
from app.core.exceptions import DatasetNotFoundError, EvaluationNotFoundError, MetricNotSupportedError
from app.models.dataset import Dataset
from app.models.evaluation import Evaluation, EvaluationResult
from app.models.sample import Sample
from app.models import (
    EvaluationCreate, EvaluationResponse, EvaluationListResponse,
    EvaluationStatusResponse, EvaluationResultsResponse, SingleEvaluationRequest,
    SingleEvaluationResponse, EvaluationComparisonRequest, EvaluationComparisonResponse,
    MetricConfig, LLMConfig, EmbeddingsConfig, PaginationParams
)
from app.services.ragas_service import RAGASService

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/dataset", response_model=EvaluationResponse)
async def evaluate_dataset(
    evaluation_data: EvaluationCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Start a dataset evaluation"""
    # Verify dataset exists

    # if dataset id is none, use the dataset name
    if evaluation_data.dataset_id is None:
        dataset = db.query(Dataset).filter(Dataset.name == evaluation_data.dataset_name).first()
        if not dataset:
            raise DatasetNotFoundError(evaluation_data.dataset_name)
        evaluation_data.dataset_id = dataset.dataset_id


    dataset = db.query(Dataset).filter(Dataset.dataset_id == evaluation_data.dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(evaluation_data.dataset_id)
    
    # Create evaluation record
    evaluation = Evaluation(
        dataset_id=evaluation_data.dataset_id,
        experiment_name=evaluation_data.experiment_name,
        metrics=[m.model_dump() for m in evaluation_data.metrics],
        llm_config=evaluation_data.llm_config.model_dump() if evaluation_data.llm_config else None,
        embeddings_config=evaluation_data.embeddings_config.dict() if evaluation_data.embeddings_config else None,
        status="pending"
    )
    
    db.add(evaluation)
    db.commit()
    db.refresh(evaluation)

    start_time = datetime.utcnow()
    
    # Start background evaluation
    # background_tasks.add_task(
    #     run_evaluation,
    # Add evaluation to background tasks
    background_tasks.add_task(
        run_evaluation,
        evaluation.evaluation_id,
        evaluation_data.dataset_id,
        [m.model_dump() for m in evaluation_data.metrics],
        evaluation_data.llm_config.model_dump() if evaluation_data.llm_config else None,
        evaluation_data.embeddings_config.model_dump() if evaluation_data.embeddings_config else None,
        evaluation_data.batch_size
    )
    
    logger.info(f"Started evaluation: {evaluation.evaluation_id}")
    
    # Calculate estimated completion time
    samples_count = db.query(Sample).filter(Sample.dataset_id == evaluation_data.dataset_id).count()
    estimated_time = datetime.utcnow() + timedelta(minutes=samples_count * 2)  # Rough estimate
    

    if evaluation.llm_config and evaluation.llm_config.get('api_key'):
        evaluation.llm_config['api_key'] = "********"
    if evaluation.embeddings_config and evaluation.embeddings_config.get('api_key'):
        evaluation.embeddings_config['api_key'] = "********"

    return {
        "experiment_name": evaluation.experiment_name,
        "metrics": evaluation.metrics,
        "llm_config": evaluation.llm_config,
        "embeddings_config": evaluation.embeddings_config,
        "batch_size": evaluation.batch_size,
        "evaluation_id": evaluation.evaluation_id,
        "dataset_id": evaluation.dataset_id,
        "status": "running",
        "progress": 0.0,
        "estimated_completion": estimated_time.isoformat(),
        "started_at": start_time.isoformat(),
        "completed_at": None,
        "created_at": start_time.isoformat(),
        "updated_at": start_time.isoformat(),
        "results_url": f"/api/v1/evaluations/{evaluation.evaluation_id}/results",
        "overall_scores": None,
        "cost_analysis": None,
        "traces": None,
        "error_message": None
    }



@router.get("/evaluations/{evaluation_id}", response_model=EvaluationStatusResponse)
async def get_evaluation_status(
    evaluation_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Get evaluation status"""
    evaluation = db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first()
    if not evaluation:
        raise EvaluationNotFoundError(evaluation_id)
    
    return {
        "evaluation_id": evaluation.evaluation_id,
        "status": evaluation.status,
        "progress": evaluation.progress,
        "started_at": evaluation.started_at.isoformat() if evaluation.started_at else None,
        "completed_at": evaluation.completed_at.isoformat() if evaluation.completed_at else None,
        "error_message": evaluation.error_message
    }


@router.get("/evaluations/{evaluation_id}/results", response_model=EvaluationResultsResponse)
async def get_evaluation_results(
    evaluation_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Get evaluation results"""
    evaluation = db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first()
    if not evaluation:
        raise EvaluationNotFoundError(evaluation_id)
    
    if evaluation.status != "completed":
        raise HTTPException(status_code=400, detail="Evaluation not completed yet")
    
    # Get sample scores
    results = db.query(EvaluationResult).filter(EvaluationResult.evaluation_id == evaluation_id).all()
    sample_scores = [
        {
            "sample_id": result.sample_id,
            "scores": result.scores
        }
        for result in results
    ]
    
    return {
        "evaluation_id": evaluation.evaluation_id,
        "dataset_id": evaluation.dataset_id,
        "experiment_name": evaluation.experiment_name,
        "metrics": evaluation.overall_scores or {},
        "sample_scores": sample_scores,
        "cost_analysis": evaluation.cost_analysis,
        "traces": evaluation.traces or [],
        "created_at": evaluation.created_at.isoformat() if evaluation.created_at else None
    }


@router.post("/single", response_model=SingleEvaluationResponse)
async def evaluate_single_sample(
    request: SingleEvaluationRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Evaluate a single sample"""
    try:
        # Initialize RAGAS service
        ragas_service = RAGASService()
        
        # Run evaluation
        result = await ragas_service.evaluate_single_sample(
            sample=request.sample,
            metrics=request.metrics,
            llm_config=request.llm_config
        )
        
        return {
            "sample_id": str(uuid.uuid4()),
            "scores": result["scores"],
            "reasoning": result.get("reasoning"),
            "cost": result.get("cost")
        }
        
    except Exception as e:
        logger.error(f"Error evaluating single sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/evaluations", response_model=EvaluationListResponse)
async def list_evaluations(
    pagination: PaginationParams = Depends(),
    dataset_id: Optional[str] = Query(None),
    dataset_name: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """List evaluations with pagination and filtering"""

    query = db.query(Evaluation)

    if dataset_id is None and dataset_name is not None:
        dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
        if not dataset:
            raise DatasetNotFoundError(dataset_name)
        dataset_id = dataset.dataset_id
    
    # Apply filters
    if dataset_id is not None:
        query = query.filter(Evaluation.dataset_id == dataset_id)
    if status:
        query = query.filter(Evaluation.status == status)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    evaluations = query.offset((pagination.page - 1) * pagination.size).limit(pagination.size).all()
    
    evaluation_list = []
    for eval in evaluations:
        # Get dataset name
        dataset = db.query(Dataset).filter(Dataset.dataset_id == eval.dataset_id).first()
        dataset_name = dataset.name if dataset else "Unknown"

        llm_config = eval.llm_config if hasattr(eval, 'llm_config') else None
        embeddings_config = eval.embeddings_config if hasattr(eval, 'embeddings_config') else None
        if llm_config and llm_config.get('api_key'):
            llm_config['api_key'] = "********"
        if embeddings_config and embeddings_config.get('api_key'):
            embeddings_config['api_key'] = "********"
        
        evaluation_list.append({
            "evaluation_id": eval.evaluation_id,
            "dataset_id": eval.dataset_id,
            "dataset_name": dataset_name,
            "experiment_name": eval.experiment_name,
            "status": eval.status,
            "metrics": eval.metrics if eval.metrics else [],
            "progress": eval.progress if hasattr(eval, 'progress') else 0.0,
            "error_message": eval.error_message if hasattr(eval, 'error_message') else None,
            "overall_scores": eval.overall_scores if hasattr(eval, 'overall_scores') else None,
            "cost_analysis": eval.cost_analysis if hasattr(eval, 'cost_analysis') else None,
            "traces": eval.traces if hasattr(eval, 'traces') else None,
            "started_at": eval.started_at if hasattr(eval, 'started_at') else None,
            "completed_at": eval.completed_at if hasattr(eval, 'completed_at') else None,
            "created_at": eval.created_at if eval.created_at else datetime.utcnow(),
            "updated_at": eval.updated_at if hasattr(eval, 'updated_at') else datetime.utcnow(),
            "llm_config": eval.llm_config if hasattr(eval, 'llm_config') else None,
            "embeddings_config": eval.embeddings_config if hasattr(eval, 'embeddings_config') else None,
            "batch_size": eval.batch_size if hasattr(eval, 'batch_size') else 10
        })
    
    return {
        "evaluations": evaluation_list,
        "total": total,
        "page": pagination.page,
        "size": pagination.size
    }


@router.post("/evaluations/compare", response_model=dict)
async def compare_evaluations(
    request: EvaluationComparisonRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Compare multiple evaluations"""
    evaluations = []
    
    for eval_id in request.evaluation_ids:
        evaluation = db.query(Evaluation).filter(Evaluation.evaluation_id == eval_id).first()
        if not evaluation:
            raise EvaluationNotFoundError(eval_id)
        evaluations.append(evaluation)
    
    # Build comparison data
    comparison = {
        "evaluation_ids": request.evaluation_ids,
        "metrics": {},
        "improvements": {}
    }
    
    # Extract metric scores
    for metric in request.metrics:
        comparison["metrics"][metric] = {}
        for eval in evaluations:
            if eval.overall_scores and metric in eval.overall_scores:
                comparison["metrics"][metric][eval.evaluation_id] = eval.overall_scores[metric]
    
    # Calculate improvements
    if len(evaluations) >= 2:
        for i in range(len(evaluations) - 1):
            current_eval = evaluations[i]
            next_eval = evaluations[i + 1]
            improvement_key = f"{current_eval.evaluation_id}_to_{next_eval.evaluation_id}"
            comparison["improvements"][improvement_key] = {}
            
            for metric in request.metrics:
                if (current_eval.overall_scores and next_eval.overall_scores and 
                    metric in current_eval.overall_scores and metric in next_eval.overall_scores):
                    improvement = next_eval.overall_scores[metric] - current_eval.overall_scores[metric]
                    comparison["improvements"][improvement_key][metric] = f"{improvement:+.3f}"
    
    return {"comparison": comparison}


async def run_evaluation(
    evaluation_id: str,
    dataset_id: str,
    metrics: List[dict],
    llm_config: Optional[dict],
    embeddings_config: Optional[dict],
    batch_size: int,
):
    """Background task to run evaluation"""
    from app.core.database import SessionLocal
    from sqlalchemy.exc import PendingRollbackError, DisconnectionError
    
    def get_fresh_db_session():
        """Get a fresh database session with proper error handling"""
        return SessionLocal()
    
    def safe_db_operation(db, operation, *args, **kwargs):
        """Safely execute database operations with rollback handling"""
        try:
            return operation(db, *args, **kwargs)
        except (PendingRollbackError, DisconnectionError) as e:
            logger.warning(f"Database connection issue, rolling back and retrying: {e}")
            db.rollback()
            return operation(db, *args, **kwargs)
        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            db.rollback()
            raise
    
    def update_evaluation_status(db, evaluation_id, status, **updates):
        """Update evaluation status with proper error handling"""
        try:
            evaluation = db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first()
            if evaluation:
                for key, value in updates.items():
                    setattr(evaluation, key, value)
                db.commit()
                return evaluation
        except (PendingRollbackError, DisconnectionError) as e:
            logger.warning(f"Database connection issue during status update, rolling back: {e}")
            db.rollback()
            # Try to get a fresh session and retry
            fresh_db = get_fresh_db_session()
            try:
                evaluation = fresh_db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first()
                if evaluation:
                    for key, value in updates.items():
                        setattr(evaluation, key, value)
                    fresh_db.commit()
                    return evaluation
            finally:
                fresh_db.close()
        except Exception as e:
            logger.error(f"Failed to update evaluation status: {e}")
            db.rollback()
            raise
    
    db = get_fresh_db_session()
    try:
        # Update status to running
        evaluation = safe_db_operation(db, lambda db: db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first())
        if not evaluation:
            logger.error(f"Evaluation {evaluation_id} not found")
            return
        
        # Update status to running
        update_evaluation_status(db, evaluation_id, "running", started_at=datetime.utcnow())
        
        # Get samples
        samples = safe_db_operation(db, lambda db: db.query(Sample).filter(Sample.dataset_id == dataset_id).all())
        total_samples = len(samples)
        
        if total_samples == 0:
            update_evaluation_status(db, evaluation_id, "completed", completed_at=datetime.utcnow())
            return
        
        # Initialize RAGAS service
        ragas_service = RAGASService()
        
        # Process samples in batches
        overall_scores = {}
        total_cost = {"tokens": 0, "cost": 0.0, "currency": "USD"}
        traces = []
        
        for i in range(0, total_samples, batch_size):
            batch = samples[i:i + batch_size]
            
            try:
                # Run evaluation on batch
                ragas_service._setup_llm(llm_config)
                ragas_service._setup_embeddings(embeddings_config)
                batch_results = await ragas_service.evaluate_batch(
                    samples=batch,
                    metrics=metrics,
                )
                
                # Store results with fresh database session for each batch
                batch_db = get_fresh_db_session()
                try:
                    for j, result in enumerate(batch_results):
                        sample = batch[j]
                        
                        # Create evaluation result
                        eval_result = EvaluationResult(
                            evaluation_id=evaluation_id,
                            sample_id=sample.sample_id,
                            scores=result["scores"],
                            reasoning=result.get("reasoning"),
                            cost=result.get("cost")
                        )

                        batch_db.add(eval_result)
                        
                        # Update cost tracking
                        if result.get("cost"):
                            tokens = result["cost"].get("tokens")
                            cost_value = result["cost"].get("cost")
                            if tokens is not None:
                                total_cost["tokens"] += tokens
                            if cost_value is not None:
                                total_cost["cost"] += cost_value
                    
                    # Update progress
                    progress = min((i + len(batch)) / total_samples, 1.0)
                    update_evaluation_status(batch_db, evaluation_id, "running", progress=progress)
                    
                    logger.info(f"Evaluation {evaluation_id}: {progress:.1%} complete")
                    
                except Exception as e:
                    logger.error(f"Error storing batch results: {e}")
                    batch_db.rollback()
                    raise
                finally:
                    batch_db.close()
                
            except Exception as e:
                logger.error(f"Error processing batch in evaluation {evaluation_id}: {e}")
                continue
        
        # Calculate overall scores with fresh session
        final_db = get_fresh_db_session()
        try:
            evaluation = final_db.query(Evaluation).filter(Evaluation.evaluation_id == evaluation_id).first()
            if evaluation and evaluation.results:
                for metric in metrics:
                    metric_name = metric["name"]
                    scores = [r.scores.get(metric_name) for r in evaluation.results if r.scores]
                    # Filter out None values and calculate average
                    valid_scores = [s for s in scores if s is not None]
                    if valid_scores:
                        overall_scores[metric_name] = sum(valid_scores) / len(valid_scores)
            
            # Update final status
            update_evaluation_status(
                final_db, 
                evaluation_id, 
                "completed", 
                completed_at=datetime.utcnow(),
                overall_scores=overall_scores,
                cost_analysis=total_cost,
                traces=traces
            )
            
            logger.info(f"Evaluation {evaluation_id} completed successfully")
            
        finally:
            final_db.close()
        
    except Exception as e:
        logger.error(f"Error in evaluation {evaluation_id}: {e}")
        try:
            update_evaluation_status(
                db, 
                evaluation_id, 
                "failed", 
                error_message=str(e),
                completed_at=datetime.utcnow()
            )
        except Exception as update_error:
            logger.error(f"Failed to update evaluation status to failed: {update_error}")
    finally:
        try:
            db.close()
        except Exception as e:
            logger.warning(f"Error closing database session: {e}")


# python -m app.api.v1.evaluation
def main():
    """Unit test function for evaluation routes"""
    from fastapi.testclient import TestClient
    from main import app
    from app.core.config import settings
    
    client = TestClient(app)
    
    # Test single sample evaluation
    """
    test_single_eval = {
        "sample": {
            "user_input": "What is the capital of France?",
            "retrieved_contexts": ["Paris is the capital of France."],
            "response": "The capital of France is Paris."
        },
        "metrics": [
            {"name": "answer_relevancy", "parameters": {}}
        ],
        "llm_config": {
            "provider": "openai",
            "model": "qwen-plus",
            "api_key": settings.OPENAI_API_KEY,
            "base_url": settings.OPENAI_BASE_URL
        },
    }
    
    # Test single evaluation
    response = client.post("/api/v1/evaluate/single", json=test_single_eval, headers={"Authorization": "Bearer test-api-key"})
    assert response.status_code == 200
    
    print("Evaluation routes test passed!")
    """


    # Test evaluation on dataset
    """
    test_evaluation = {
        # "dataset_id": "3e9554eb-402d-4cff-bf75-14f1b7a19bca",
        "dataset_name": "Jinyao_recall_dataset",
        "metrics": [
            {"name": "custom_context_recall", "parameters": {}}
        ],
        "llm_config": {
            "provider": "openai",
            "model": "qwen-plus",
            "api_key": settings.OPENAI_API_KEY,
            "base_url": settings.OPENAI_BASE_URL
        },
        "experiment_name": "Test Evaluation"
    }
    
    response = client.post("/api/v1/evaluate/dataset", json=test_evaluation, headers={"Authorization": "Bearer test-api-key"})
    print('result', response.json())
    assert response.status_code == 200
    
    print("Evaluation routes test passed!")
    """


    # Test get evaluation results with dataset name
    response = client.get("/api/v1/evaluate/evaluations?dataset_name=Jinyao_recall_dataset", headers={"Authorization": "Bearer test-api-key"})
    print('result', response.json())
    assert response.status_code == 200



# python -m app.api.v1.evaluation
if __name__ == "__main__":
    main()
