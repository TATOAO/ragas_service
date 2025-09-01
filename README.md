# RAGAS FastAPI Service

A FastAPI service that provides REST API endpoints for RAGAS evaluation capabilities, allowing other applications to request evaluations programmatically.

## Overview

This service exposes RAGAS evaluation functionality through HTTP endpoints, enabling:
- Dataset creation and management
- Data insertion and updates
- Batch evaluation of datasets
- Single sample evaluation
- Metric management and configuration
- Result storage and retrieval

## Core Services

### 1. Dataset Management Service
- Create, read, update, delete datasets
- Bulk data insertion
- Dataset validation and schema enforcement
- Dataset versioning and history

### 2. Evaluation Service
- Run evaluations with configurable metrics
- Support for both single-turn and multi-turn samples
- Async evaluation for large datasets
- Progress tracking and status updates

### 3. Metric Management Service
- List available metrics
- Configure metric parameters
- Custom metric registration
- Metric performance tracking

### 4. Result Management Service
- Store and retrieve evaluation results
- Result aggregation and analysis
- Export results in various formats
- Historical result comparison

### 5. LLM/Embedding Configuration Service
- Manage LLM configurations
- Embedding model management
- API key management
- Model performance tracking

## API Endpoints

### Dataset Management

#### Create Dataset
```http
POST /api/v1/datasets
```

**Request Body:**
```json
{
  "name": "string",
  "description": "string",
  "sample_type": "single_turn" | "multi_turn",
  "metadata": {
    "source": "string",
    "version": "string",
    "tags": ["string"]
  }
}
```

**Response:**
```json
{
  "dataset_id": "uuid",
  "name": "string",
  "description": "string",
  "sample_type": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "sample_count": 0,
  "metadata": {}
}
```

#### Get Dataset
```http
GET /api/v1/datasets/{dataset_id}
```

**Response:**
```json
{
  "dataset_id": "uuid",
  "name": "string",
  "description": "string",
  "sample_type": "string",
  "created_at": "datetime",
  "updated_at": "datetime",
  "sample_count": 0,
  "metadata": {},
  "samples": [
    {
      "user_input": "string",
      "retrieved_contexts": ["string"],
      "reference_contexts": ["string"],
      "response": "string",
      "multi_responses": ["string"],
      "reference": "string",
      "rubrics": {}
    }
  ]
}
```

#### List Datasets
```http
GET /api/v1/datasets?page=1&size=10&sample_type=single_turn
```

**Response:**
```json
{
  "datasets": [
    {
      "dataset_id": "uuid",
      "name": "string",
      "description": "string",
      "sample_type": "string",
      "created_at": "datetime",
      "updated_at": "datetime",
      "sample_count": 0
    }
  ],
  "total": 100,
  "page": 1,
  "size": 10
}
```

#### Delete Dataset
```http
DELETE /api/v1/datasets/{dataset_id}
```

**Response:**
```json
{
  "message": "Dataset deleted successfully",
  "dataset_id": "uuid"
}
```

### Data Management

#### Insert Single Sample
```http
POST /api/v1/datasets/{dataset_id}/samples
```

**Request Body (Single Turn):**
```json
{
  "user_input": "What is the capital of France?",
  "retrieved_contexts": ["Paris is the capital of France."],
  "reference_contexts": ["Paris is the capital and largest city of France."],
  "response": "The capital of France is Paris.",
  "reference": "Paris",
  "rubrics": {
    "accuracy": "high",
    "completeness": "medium"
  }
}
```

**Request Body (Multi Turn):**
```json
{
  "user_input": [
    {
      "role": "user",
      "content": "What is the weather like?"
    },
    {
      "role": "assistant", 
      "content": "I don't have access to real-time weather data."
    },
    {
      "role": "user",
      "content": "Can you check the weather for New York?"
    }
  ],
  "response": "I cannot provide real-time weather information.",
  "reference": "The assistant should explain it cannot access weather data."
}
```

**Response:**
```json
{
  "sample_id": "uuid",
  "dataset_id": "uuid",
  "created_at": "datetime"
}
```

#### Bulk Insert Samples
```http
POST /api/v1/datasets/{dataset_id}/samples/bulk
```

**Request Body:**
```json
{
  "samples": [
    {
      "user_input": "string",
      "retrieved_contexts": ["string"],
      "response": "string",
      "reference": "string"
    }
  ]
}
```

**Response:**
```json
{
  "inserted_count": 10,
  "failed_count": 0,
  "errors": []
}
```

#### Update Sample
```http
PUT /api/v1/datasets/{dataset_id}/samples/{sample_id}
```

**Request Body:** Same as insert single sample

**Response:**
```json
{
  "sample_id": "uuid",
  "updated_at": "datetime"
}
```

#### Delete Sample
```http
DELETE /api/v1/datasets/{dataset_id}/samples/{sample_id}
```

**Response:**
```json
{
  "message": "Sample deleted successfully",
  "sample_id": "uuid"
}
```

### Evaluation Service

#### Evaluate Dataset
```http
POST /api/v1/evaluate/dataset
```

**Request Body:**
```json
{
  "dataset_id": "uuid",
  "metrics": [
    {
      "name": "answer_relevancy",
      "parameters": {}
    },
    {
      "name": "context_precision", 
      "parameters": {}
    },
    {
      "name": "faithfulness",
      "parameters": {}
    },
    {
      "name": "context_recall",
      "parameters": {}
    }
  ],
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "string"
  },
  "embeddings_config": {
    "provider": "openai",
    "model": "text-embedding-3-small",
    "api_key": "string"
  },
  "experiment_name": "string",
  "batch_size": 10,
  "raise_exceptions": false
}
```

**Response:**
```json
{
  "evaluation_id": "uuid",
  "status": "running",
  "progress": 0.0,
  "estimated_completion": "datetime",
  "results_url": "/api/v1/evaluations/{evaluation_id}/results"
}
```

#### Get Evaluation Status
```http
GET /api/v1/evaluations/{evaluation_id}
```

**Response:**
```json
{
  "evaluation_id": "uuid",
  "status": "completed" | "running" | "failed",
  "progress": 0.85,
  "started_at": "datetime",
  "completed_at": "datetime",
  "error_message": "string"
}
```

#### Get Evaluation Results
```http
GET /api/v1/evaluations/{evaluation_id}/results
```

**Response:**
```json
{
  "evaluation_id": "uuid",
  "dataset_id": "uuid",
  "experiment_name": "string",
  "metrics": {
    "answer_relevancy": 0.874,
    "context_precision": 0.817,
    "faithfulness": 0.892,
    "context_recall": 0.756
  },
  "sample_scores": [
    {
      "sample_id": "uuid",
      "answer_relevancy": 0.9,
      "context_precision": 0.8,
      "faithfulness": 0.85,
      "context_recall": 0.7
    }
  ],
  "cost_analysis": {
    "total_tokens": 15000,
    "total_cost": 0.045,
    "currency": "USD"
  },
  "traces": [
    {
      "sample_id": "uuid",
      "trace_url": "string"
    }
  ],
  "created_at": "datetime"
}
```

#### Evaluate Single Sample
```http
POST /api/v1/evaluate/single
```

**Request Body:**
```json
{
  "sample": {
    "user_input": "What is the capital of France?",
    "retrieved_contexts": ["Paris is the capital of France."],
    "response": "The capital of France is Paris.",
    "reference": "Paris"
  },
  "metrics": [
    {
      "name": "answer_relevancy",
      "parameters": {}
    },
    {
      "name": "faithfulness",
      "parameters": {}
    }
  ],
  "llm_config": {
    "provider": "openai",
    "model": "gpt-4o",
    "api_key": "string"
  }
}
```

**Response:**
```json
{
  "sample_id": "uuid",
  "scores": {
    "answer_relevancy": 0.9,
    "faithfulness": 0.85
  },
  "reasoning": {
    "answer_relevancy": "The response directly answers the question about France's capital.",
    "faithfulness": "The response is consistent with the provided context."
  },
  "cost": {
    "tokens": 150,
    "cost": 0.00045,
    "currency": "USD"
  }
}
```

### Metric Management

#### List Available Metrics
```http
GET /api/v1/metrics
```

**Response:**
```json
{
  "metrics": [
    {
      "name": "answer_relevancy",
      "description": "Measures how relevant the answer is to the question",
      "type": "llm_based",
      "supported_sample_types": ["single_turn"],
      "parameters": {
        "llm_required": true,
        "embeddings_required": false
      }
    },
    {
      "name": "context_precision",
      "description": "Measures the precision of retrieved contexts",
      "type": "embedding_based",
      "supported_sample_types": ["single_turn"],
      "parameters": {
        "llm_required": false,
        "embeddings_required": true
      }
    }
  ]
}
```

#### Get Metric Details
```http
GET /api/v1/metrics/{metric_name}
```

**Response:**
```json
{
  "name": "answer_relevancy",
  "description": "Measures how relevant the answer is to the question",
  "type": "llm_based",
  "supported_sample_types": ["single_turn"],
  "parameters": {
    "llm_required": true,
    "embeddings_required": false
  },
  "default_config": {
    "llm": "gpt-4o",
    "embeddings": null
  },
  "example_usage": {
    "sample": {
      "user_input": "What is the capital of France?",
      "response": "The capital of France is Paris."
    },
    "expected_score": 0.9
  }
}
```

### Configuration Management

#### List LLM Providers
```http
GET /api/v1/config/llm-providers
```

**Response:**
```json
{
  "providers": [
    {
      "name": "openai",
      "models": [
        {
          "name": "gpt-4o",
          "max_tokens": 4096,
          "cost_per_1k_tokens": 0.03
        },
        {
          "name": "gpt-3.5-turbo",
          "max_tokens": 4096,
          "cost_per_1k_tokens": 0.002
        }
      ]
    },
    {
      "name": "anthropic",
      "models": [
        {
          "name": "claude-3-opus",
          "max_tokens": 4096,
          "cost_per_1k_tokens": 0.015
        }
      ]
    }
  ]
}
```

#### List Embedding Providers
```http
GET /api/v1/config/embedding-providers
```

**Response:**
```json
{
  "providers": [
    {
      "name": "openai",
      "models": [
        {
          "name": "text-embedding-3-small",
          "dimensions": 1536,
          "cost_per_1k_tokens": 0.00002
        }
      ]
    }
  ]
}
```

### Result Management

#### List Evaluations
```http
GET /api/v1/evaluations?dataset_id=uuid&page=1&size=10
```

**Response:**
```json
{
  "evaluations": [
    {
      "evaluation_id": "uuid",
      "dataset_id": "uuid",
      "dataset_name": "string",
      "experiment_name": "string",
      "status": "completed",
      "metrics_count": 4,
      "created_at": "datetime",
      "completed_at": "datetime"
    }
  ],
  "total": 50,
  "page": 1,
  "size": 10
}
```

#### Export Results
```http
GET /api/v1/evaluations/{evaluation_id}/export?format=csv
```

**Response:** File download (CSV, JSON, or Excel)

#### Compare Evaluations
```http
POST /api/v1/evaluations/compare
```

**Request Body:**
```json
{
  "evaluation_ids": ["uuid1", "uuid2", "uuid3"],
  "metrics": ["answer_relevancy", "faithfulness"]
}
```

**Response:**
```json
{
  "comparison": {
    "evaluation_ids": ["uuid1", "uuid2", "uuid3"],
    "metrics": {
      "answer_relevancy": {
        "uuid1": 0.874,
        "uuid2": 0.892,
        "uuid3": 0.856
      },
      "faithfulness": {
        "uuid1": 0.892,
        "uuid2": 0.901,
        "uuid3": 0.878
      }
    },
    "improvements": {
      "uuid1_to_uuid2": {
        "answer_relevancy": "+0.018",
        "faithfulness": "+0.009"
      }
    }
  }
}
```

## Additional Services Needed

### 1. Authentication & Authorization Service
- API key management
- User/tenant management
- Rate limiting
- Access control for datasets and evaluations

### 2. Storage Service
- Dataset persistence (database + file storage)
- Evaluation result storage
- Backup and archival
- Data versioning

### 3. Queue Management Service
- Async evaluation job processing
- Progress tracking
- Job scheduling and retry logic
- Resource management

### 4. Monitoring & Observability Service
- Request/response logging
- Performance metrics
- Error tracking and alerting
- Usage analytics

### 5. Caching Service
- LLM response caching
- Embedding caching
- Evaluation result caching
- Configuration caching

### 6. Notification Service
- Evaluation completion notifications
- Error alerts
- Progress updates
- Webhook support

## Error Handling

All endpoints return consistent error responses:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "user_input",
      "issue": "Field is required"
    }
  },
  "timestamp": "datetime",
  "request_id": "uuid"
}
```

Common error codes:
- `VALIDATION_ERROR`: Invalid request parameters
- `DATASET_NOT_FOUND`: Dataset doesn't exist
- `EVALUATION_NOT_FOUND`: Evaluation doesn't exist
- `METRIC_NOT_SUPPORTED`: Metric not available
- `LLM_ERROR`: LLM API error
- `RATE_LIMIT_EXCEEDED`: Too many requests
- `INTERNAL_ERROR`: Server error

## Rate Limiting

- Dataset operations: 100 requests/minute
- Evaluation requests: 10 requests/minute
- Single sample evaluation: 50 requests/minute
- Configuration operations: 200 requests/minute

## Authentication

All endpoints require API key authentication:

```http
Authorization: Bearer your-api-key-here
```

## Webhooks

Configure webhooks for evaluation completion:

```http
POST /api/v1/webhooks
```

**Request Body:**
```json
{
  "url": "https://your-app.com/webhook",
  "events": ["evaluation.completed", "evaluation.failed"],
  "secret": "webhook-secret"
}
```

## Deployment

### Docker
```bash
docker build -t ragas-service .
docker run -p 8000:8000 ragas-service
```

### Environment Variables
```bash
DATABASE_URL=postgresql://user:pass@localhost/ragas
REDIS_URL=redis://localhost:6379
OPENAI_API_KEY=your-key
LANGCHAIN_API_KEY=your-key
ANTHROPIC_API_KEY=your-key
JWT_SECRET=your-secret
```

### Health Check
```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "services": {
    "database": "connected",
    "redis": "connected",
    "llm_providers": "available"
  },
  "timestamp": "datetime"
}
```
