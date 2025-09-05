import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.auth import get_current_user_api_key
from app.core.database import get_db
from app.core.exceptions import DatasetNotFoundError, ValidationError
from app.models import (
    DatasetCreate,
    DatasetDeleteResponse,
    DatasetListResponse,
    DatasetResponse,
    DatasetUpdate,
    PaginationParams,
    SampleBulkCreate,
    SampleBulkDeleteResponse,
    SampleBulkResponse,
    SampleCreate,
    SampleDeleteResponse,
    SampleResponse,
    SampleUpdate,
)
from app.models.dataset import Dataset
from app.models.sample import Sample

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health")
async def health(
    db: Session = Depends(get_db),
):
    # check db connection is health
    try:
        # Execute a simple query to verify database connection
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail="Database connection failed"
        )

@router.post("/", response_model=DatasetResponse)
async def create_dataset(
    dataset_data: DatasetCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Create a new dataset"""
    try:
        # Check if dataset with same name exists
        existing_dataset = db.query(Dataset).filter(Dataset.name == dataset_data.name).first()
        if existing_dataset:
            raise ValidationError("Dataset with this name already exists")
        
        # Create new dataset
        dataset = Dataset.model_validate(dataset_data)
        
        db.add(dataset)
        db.commit()
        db.refresh(dataset)
        
        logger.info(f"Created dataset: {dataset.dataset_id}")
        return dataset
        
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating dataset: {e}")
        raise


@router.get("/{dataset_id}", response_model=DatasetResponse)
async def get_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Get a specific dataset"""
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    return dataset


@router.get("/", response_model=DatasetListResponse)
async def list_datasets(
    pagination: PaginationParams = Depends(),
    sample_type: Optional[str] = Query(None, pattern="^(single_turn|multi_turn)$"),
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """List datasets with pagination and filtering"""
    query = db.query(Dataset)
    
    # Apply filters
    if sample_type:
        query = query.filter(Dataset.sample_type == sample_type)
    
    # Get total count
    total = query.count()
    
    # Apply pagination
    datasets = query.offset((pagination.page - 1) * pagination.size).limit(pagination.size).all()
    
    return {
        "datasets": [DatasetResponse.from_orm(dataset) for dataset in datasets],
        "total": total,
        "page": pagination.page,
        "size": pagination.size
    }


@router.put("/{dataset_id}", response_model=DatasetResponse)
async def update_dataset(
    dataset_id: str,
    dataset_data: DatasetUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Update a dataset"""
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    # Update fields
    if dataset_data.name is not None:
        dataset.name = dataset_data.name
    if dataset_data.description is not None:
        dataset.description = dataset_data.description
    if dataset_data.metadata_json is not None:
        dataset.metadata_json = dataset_data.metadata_json
    
    db.commit()
    db.refresh(dataset)
    
    logger.info(f"Updated dataset: {dataset_id}")
    return dataset


@router.delete("/{dataset_id}", response_model=DatasetDeleteResponse)
async def delete_dataset(
    dataset_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Delete a dataset"""
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    db.delete(dataset)
    db.commit()

    # delete all samples in the dataset
    samples = db.query(Sample).filter(Sample.dataset_id == dataset_id).all()
    for sample in samples:
        db.delete(sample)
    db.commit()
    
    logger.info(f"Deleted dataset: {dataset_id}")
    return {"message": "Dataset deleted successfully", "dataset_id": dataset_id}

@router.delete("/datasetname/{dataset_name}/samples", response_model=SampleBulkDeleteResponse)
async def delete_samples(
    dataset_name: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Delete all samples in a dataset"""
    dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_name)
    
    samples = db.query(Sample).filter(Sample.dataset_id == dataset.dataset_id).all()
    for sample in samples:
        db.delete(sample)
    db.commit()
    
    logger.info(f"Deleted samples in dataset: {dataset_name}")
    return {"message": "Samples deleted successfully", "dataset_name": dataset_name}


@router.post("/{dataset_id}/samples", response_model=SampleResponse)
async def insert_sample(
    dataset_id: str,
    sample_data: SampleCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Insert a single sample into a dataset"""
    # Verify dataset exists

    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    # Create sample
    sample = Sample.model_validate(sample_data)
    sample.dataset_id = dataset_id
    
    db.add(sample)
    db.commit()
    db.refresh(sample)
    
    logger.info(f"Inserted sample: {sample.sample_id} into dataset: {dataset_id}")
    return sample


@router.post("/datasetname/{dataset_name}/samples", response_model=SampleResponse)
async def insert_sample_by_name(
    dataset_name: str,
    sample_data: SampleCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Insert a single sample into a dataset by dataset name"""
    # Verify dataset exists by name
    dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with name '{dataset_name}' not found")
    
    # Create sample
    sample = Sample.model_validate(sample_data.model_dump() | {"dataset_id": dataset.dataset_id})
    
    db.add(sample)
    db.commit()
    db.refresh(sample)
    
    logger.info(f"Inserted sample: {sample.sample_id} into dataset: {dataset_name} (ID: {dataset.dataset_id})")
    return sample

@router.get("/datasetname/{dataset_name}/samples", response_model=List[SampleResponse])
async def get_samples_by_dataset_name(
    dataset_name: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Get samples by dataset name"""
    dataset = db.query(Dataset).filter(Dataset.name == dataset_name).first()
    if not dataset:
        raise HTTPException(status_code=404, detail=f"Dataset with name '{dataset_name}' not found")
    
    samples = db.query(Sample).filter(Sample.dataset_id == dataset.dataset_id).all()
    return samples


@router.post("/{dataset_id}/samples/bulk", response_model=SampleBulkResponse)
async def bulk_insert_samples(
    dataset_id: str,
    bulk_data: SampleBulkCreate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Bulk insert samples into a dataset"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    inserted_count = 0
    failed_count = 0
    errors = []
    
    for i, sample_data in enumerate(bulk_data.samples):
        try:
            sample = Sample.model_validate(sample_data)
            sample.dataset_id = dataset_id
            if sample_data.rubrics is None:
                sample.rubrics = {}
            db.add(sample)
            inserted_count += 1
        except Exception as e:
            failed_count += 1
            errors.append({"index": i, "error": str(e)})
    
    db.commit()
    
    logger.info(f"Bulk inserted {inserted_count} samples, failed {failed_count} into dataset: {dataset_id}")
    return {
        "inserted_count": inserted_count,
        "failed_count": failed_count,
        "errors": errors
    }


@router.put("/{dataset_id}/samples/{sample_id}", response_model=SampleResponse)
async def update_sample(
    dataset_id: str,
    sample_id: str,
    sample_data: SampleUpdate,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Update a sample"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    # Get sample
    sample = db.query(Sample).filter(
        Sample.sample_id == sample_id,
        Sample.dataset_id == dataset_id
    ).first()
    
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    # Update fields
    if sample_data.user_input is not None:
        sample.user_input = sample_data.user_input
    if sample_data.retrieved_contexts is not None:
        sample.retrieved_contexts = sample_data.retrieved_contexts
    if sample_data.reference_contexts is not None:
        sample.reference_contexts = sample_data.reference_contexts
    if sample_data.response is not None:
        sample.response = sample_data.response
    if sample_data.multi_responses is not None:
        sample.multi_responses = sample_data.multi_responses
    if sample_data.reference is not None:
        sample.reference = sample_data.reference
    if sample_data.rubrics is not None:
        sample.rubrics = sample_data.rubrics
    
    db.commit()
    db.refresh(sample)
    
    logger.info(f"Updated sample: {sample_id}")
    return sample


@router.delete("/{dataset_id}/samples/{sample_id}", response_model=SampleDeleteResponse)
async def delete_sample(
    dataset_id: str,
    sample_id: str,
    db: Session = Depends(get_db),
    current_user: dict = Depends(get_current_user_api_key)
):
    """Delete a sample"""
    # Verify dataset exists
    dataset = db.query(Dataset).filter(Dataset.dataset_id == dataset_id).first()
    if not dataset:
        raise DatasetNotFoundError(dataset_id)
    
    # Get sample
    sample = db.query(Sample).filter(
        Sample.sample_id == sample_id,
        Sample.dataset_id == dataset_id
    ).first()
    
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    
    db.delete(sample)
    db.commit()
    
    logger.info(f"Deleted sample: {sample_id}")
    return {"message": "Sample deleted successfully", "sample_id": sample_id}


# python -m app.api.v1.datasets
def main():
    """Unit test function for dataset routes"""

    from fastapi.testclient import TestClient
    from main import app
    
    client = TestClient(app)
    
    # Test data
    test_dataset = {
        "name": "Jinyao_recall_dataset",
        "description": "A recall dataset for Jinyao",
        "sample_type": "single_turn",
        "metadata_json": {"source": "test"}
    }
    
    
    # Test create dataset
    response = client.post("/api/v1/datasets/", json=test_dataset, headers={"Authorization": "Bearer test-api-key"})
    print(response.json())
    assert response.status_code == 200
    dataset_id = response.json()["dataset_id"]

    # Test get dataset
    """
    response = client.get(f"/api/v1/datasets/{dataset_id}", headers={"Authorization": "Bearer test-api-key"})
    print("xxxxxxxxxxxxxxxxxxxxxxxxx get dataset xxxxxxxxxxxxxxxxxxxxxx")
    print(response.json())
    assert response.status_code == 200



    test_sample = {
        "dataset_id": dataset_id,
        "user_input": "What is the capital of France?",
        "retrieved_contexts": ["Paris is the capital of France."],
        "response": "The capital of France is Paris.",
        "reference": "Paris"
    }
    """
    
    # Test insert sample
    """
    response = client.post(f"/api/v1/datasets/{dataset_id}/samples", json=test_sample, headers={"Authorization": "Bearer test-api-key"})
    print("xxxxxxxxxxxxxxxxxxxxxxxxx insert sample xxxxxxxxxxxxxxxxxxxxxx")
    print(response.json())
    assert response.status_code == 200
    """



    # Test delete dataset
    """
    response = client.delete(f"/api/v1/datasets/{dataset_id}", headers={"Authorization": "Bearer test-api-key"})
    print("xxxxxxxxxxxxxxxxxxxxxxxxx delete dataset xxxxxxxxxxxxxxxxxxxxxx")
    print(response.json())
    assert response.status_code == 200
    """


    # test insert sample by name
    """
    test_sample = {
        "user_input": "What is the capital of Japan?",
        "retrieved_contexts": ["Tokyo is the capital of Japan."],
        "response": "The capital of Japan is Tokyo.",
        "reference": "Tokyo"
    }

    # Test insert sample by name
    response = client.post(f"/api/v1/datasets/datasetname/{test_dataset['name']}/samples", json=test_sample, headers={"Authorization": "Bearer test-api-key"})
    # print("xxxxxxxxxxxxxxxxxxxxxxxxx insert sample by name xxxxxxxxxxxxxxxxxxxxxx")
    print(response.json())
    assert response.status_code == 200
    """


    # test get samples by dataset name
    """
    response = client.get(f"/api/v1/datasets/datasetname/{test_dataset['name']}/samples", headers={"Authorization": "Bearer test-api-key"})
    print("xxxxxxxxxxxxxxxxxxxxxxxxx get samples by dataset name xxxxxxxxxxxxxxxxxxxxxx")
    import json
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    assert response.status_code == 200
    """


    # test delete samples by dataset name
    response = client.delete(f"/api/v1/datasets/datasetname/用友测试集100条-4B打分/samples", headers={"Authorization": "Bearer test-api-key"})
    print("xxxxxxxxxxxxxxxxxxxxxxxxx delete samples by dataset name xxxxxxxxxxxxxxxxxxxxxx")
    print(response.json())
    assert response.status_code == 200

# python -m app.api.v1.datasets
if __name__ == "__main__":
    main()
