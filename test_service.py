#!/usr/bin/env python3
"""
Test script for RAGAS FastAPI Service
This script tests the basic functionality of the service
"""

from typing import Any, Dict

import requests

# Service configuration
BASE_URL = "http://localhost:8000"
API_KEY = "test-api-key"  # Default test API key

def make_request(method: str, endpoint: str, data: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Make HTTP request to the service"""
    url = f"{BASE_URL}{endpoint}"
    default_headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    
    if headers:
        default_headers.update(headers)
    
    try:
        if method.upper() == "GET":
            response = requests.get(url, headers=default_headers)
        elif method.upper() == "POST":
            response = requests.post(url, headers=default_headers, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, headers=default_headers, json=data)
        elif method.upper() == "DELETE":
            response = requests.delete(url, headers=default_headers)
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")
        
        response.raise_for_status()
        return response.json()
    
    except requests.exceptions.RequestException as e:
        print(f"Request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text}")
        return None

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    result = make_request("GET", "/health")
    if result:
        print("✅ Health check passed")
        print(f"Status: {result.get('status')}")
        return True
    else:
        print("❌ Health check failed")
        return False

def test_create_dataset():
    """Test dataset creation"""
    print("\nTesting dataset creation...")
    dataset_data = {
        "name": "Test Dataset",
        "description": "A test dataset for evaluation",
        "sample_type": "single_turn",
        "metadata": {"source": "test", "version": "1.0"}
    }
    
    result = make_request("POST", "/api/v1/datasets/", dataset_data)
    if result:
        print("✅ Dataset created successfully")
        print(f"Dataset ID: {result.get('dataset_id')}")
        return result.get('dataset_id')
    else:
        print("❌ Dataset creation failed")
        return None

def test_add_sample(dataset_id: str):
    """Test adding a sample to the dataset"""
    print(f"\nTesting sample addition to dataset {dataset_id}...")
    sample_data = {
        "user_input": "What is the capital of France?",
        "retrieved_contexts": ["Paris is the capital of France."],
        "reference_contexts": ["Paris is the capital and largest city of France."],
        "response": "The capital of France is Paris.",
        "reference": "Paris",
        "rubrics": {"accuracy": "high", "completeness": "medium"}
    }
    
    result = make_request("POST", f"/api/v1/datasets/{dataset_id}/samples", sample_data)
    if result:
        print("✅ Sample added successfully")
        print(f"Sample ID: {result.get('sample_id')}")
        return result.get('sample_id')
    else:
        print("❌ Sample addition failed")
        return None

def test_list_metrics():
    """Test listing available metrics"""
    print("\nTesting metrics listing...")
    result = make_request("GET", "/api/v1/metrics/")
    if result:
        print("✅ Metrics listed successfully")
        metrics = result.get('metrics', [])
        print(f"Found {len(metrics)} metrics:")
        for metric in metrics[:3]:  # Show first 3 metrics
            print(f"  - {metric.get('name')}: {metric.get('description')}")
        return True
    else:
        print("❌ Metrics listing failed")
        return False

def test_single_evaluation():
    """Test single sample evaluation"""
    print("\nTesting single sample evaluation...")
    evaluation_data = {
        "sample": {
            "user_input": "What is the capital of France?",
            "retrieved_contexts": ["Paris is the capital of France."],
            "response": "The capital of France is Paris."
        },
        "metrics": [
            {"name": "answer_relevancy", "parameters": {}}
        ]
    }
    
    result = make_request("POST", "/api/v1/evaluate/single", evaluation_data)
    if result:
        print("✅ Single evaluation completed")
        scores = result.get('scores', {})
        print(f"Scores: {scores}")
        return True
    else:
        print("❌ Single evaluation failed")
        return False

def test_get_dataset(dataset_id: str):
    """Test getting dataset details"""
    print(f"\nTesting get dataset {dataset_id}...")
    result = make_request("GET", f"/api/v1/datasets/{dataset_id}")
    if result:
        print("✅ Dataset retrieved successfully")
        print(f"Name: {result.get('name')}")
        print(f"Sample count: {result.get('sample_count')}")
        return True
    else:
        print("❌ Dataset retrieval failed")
        return False

def test_list_datasets():
    """Test listing datasets"""
    print("\nTesting dataset listing...")
    result = make_request("GET", "/api/v1/datasets/")
    if result:
        print("✅ Datasets listed successfully")
        datasets = result.get('datasets', [])
        print(f"Found {len(datasets)} datasets")
        return True
    else:
        print("❌ Dataset listing failed")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting RAGAS FastAPI Service Tests")
    print("=" * 50)
    
    # Test health check
    if not test_health_check():
        print("❌ Service is not running or not accessible")
        return
    
    # Test metrics
    test_list_metrics()
    
    # Test dataset operations
    dataset_id = test_create_dataset()
    if dataset_id:
        test_add_sample(dataset_id)
        test_get_dataset(dataset_id)
    
    # Test listing
    test_list_datasets()
    
    # Test evaluation (this might fail if no LLM API key is configured)
    print("\nNote: Single evaluation test requires LLM API keys to be configured")
    test_single_evaluation()
    
    print("\n" + "=" * 50)
    print("✅ All tests completed!")
    print(f"Service is running at: {BASE_URL}")
    print(f"API Documentation: {BASE_URL}/docs")

if __name__ == "__main__":
    main()
