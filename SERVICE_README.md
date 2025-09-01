# RAGAS FastAPI Service

A production-ready FastAPI service that provides REST API endpoints for RAGAS evaluation capabilities, allowing other applications to request evaluations programmatically.

## Features

- **Dataset Management**: Create, read, update, delete datasets with support for single-turn and multi-turn conversations
- **Sample Management**: Insert, update, delete individual samples or bulk import
- **Evaluation Engine**: Run RAGAS evaluations with configurable metrics and LLM providers
- **Async Processing**: Background evaluation processing for large datasets
- **Metric Management**: List and configure available RAGAS metrics
- **Provider Support**: Support for OpenAI, Anthropic, and other LLM/embedding providers
- **Authentication**: API key-based authentication
- **Rate Limiting**: Built-in rate limiting for API endpoints
- **Health Monitoring**: Health check endpoints and monitoring
- **Docker Support**: Complete Docker and docker-compose setup

## Quick Start

### Using Docker (Recommended)

1. **Clone and navigate to the service directory**:
   ```bash
   cd service
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your API keys and configuration
   ```

3. **Start the service with Docker Compose**:
   ```bash
   cd docker
   docker-compose up -d
   ```

4. **Access the service**:
   - API: http://localhost:8000
   - Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/health

### Using Local Development

1. **Set up Python environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Set up environment variables**:
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

3. **Start the service**:
   ```bash
   ./start.sh
   # Or manually: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

## API Usage

### Authentication

All API endpoints require authentication using API keys:

```bash
curl -H "Authorization: Bearer your-api-key" http://localhost:8000/api/v1/datasets/
```

### Create a Dataset

```bash
curl -X POST "http://localhost:8000/api/v1/datasets/" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Test Dataset",
    "description": "A test dataset for evaluation",
    "sample_type": "single_turn",
    "metadata": {"source": "test", "version": "1.0"}
  }'
```

### Add Samples

```bash
curl -X POST "http://localhost:8000/api/v1/datasets/{dataset_id}/samples" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "What is the capital of France?",
    "retrieved_contexts": ["Paris is the capital of France."],
    "response": "The capital of France is Paris.",
    "reference": "Paris"
  }'
```

### Run Evaluation

```bash
curl -X POST "http://localhost:8000/api/v1/evaluate/dataset" \
  -H "Authorization: Bearer your-api-key" \
  -H "Content-Type: application/json" \
  -d '{
    "dataset_id": "your-dataset-id",
    "metrics": [
      {"name": "answer_relevancy", "parameters": {}},
      {"name": "faithfulness", "parameters": {}}
    ],
    "llm_config": {
      "provider": "openai",
      "model": "gpt-4o",
      "api_key": "your-openai-key"
    },
    "experiment_name": "My Evaluation"
  }'
```

### Check Evaluation Status

```bash
curl -X GET "http://localhost:8000/api/v1/evaluate/evaluations/{evaluation_id}" \
  -H "Authorization: Bearer your-api-key"
```

### Get Evaluation Results

```bash
curl -X GET "http://localhost:8000/api/v1/evaluate/evaluations/{evaluation_id}/results" \
  -H "Authorization: Bearer your-api-key"
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | `postgresql://user:password@localhost/ragas` |
| `REDIS_URL` | Redis connection string | `redis://localhost:6379` |
| `OPENAI_API_KEY` | OpenAI API key | None |
| `ANTHROPIC_API_KEY` | Anthropic API key | None |
| `SECRET_KEY` | JWT secret key | `your-secret-key-here` |
| `DEBUG` | Enable debug mode | `false` |
| `HOST` | Service host | `0.0.0.0` |
| `PORT` | Service port | `8000` |

### Supported Metrics

- **answer_relevancy**: Measures how relevant the answer is to the question
- **context_precision**: Measures the precision of retrieved contexts
- **faithfulness**: Measures how faithful the answer is to the provided context
- **context_recall**: Measures the recall of retrieved contexts
- **answer_correctness**: Measures the correctness of the answer
- **answer_similarity**: Measures similarity between generated and reference answers
- **context_relevancy**: Measures the relevancy of retrieved contexts
- **critique_tone**: Evaluates the tone and style of the answer

### Supported LLM Providers

- **OpenAI**: GPT-4, GPT-3.5-turbo, GPT-4o
- **Anthropic**: Claude-3-Opus, Claude-3-Sonnet, Claude-3-Haiku
- **Google**: Gemini Pro

### Supported Embedding Providers

- **OpenAI**: text-embedding-3-small, text-embedding-3-large
- **Cohere**: embed-english-v3.0
- **Sentence Transformers**: all-MiniLM-L6-v2

## Architecture

### Service Structure

```
service/
├── app/
│   ├── api/v1/           # API routes organized by use case
│   │   ├── datasets.py   # Dataset management
│   │   ├── evaluation.py # Evaluation endpoints
│   │   ├── metrics.py    # Metric management
│   │   └── config.py     # Configuration endpoints
│   ├── core/             # Core functionality
│   │   ├── config.py     # Configuration management
│   │   ├── database.py   # Database connection
│   │   ├── auth.py       # Authentication
│   │   └── exceptions.py # Custom exceptions
│   ├── models/           # Database models
│   ├── schemas/          # Pydantic schemas
│   └── services/         # Business logic
│       └── ragas_service.py # RAGAS integration
├── docker/               # Docker configuration
├── main.py               # Application entry point
├── requirements.txt      # Python dependencies
└── start.sh             # Startup script
```

### Database Schema

- **datasets**: Dataset metadata and configuration
- **samples**: Individual evaluation samples
- **evaluations**: Evaluation runs and metadata
- **evaluation_results**: Individual sample evaluation results
- **metrics**: Available metrics and their configurations

## Development

### Running Tests

Each route file includes a `main()` function for unit testing:

```bash
# Test dataset routes
python app/api/v1/datasets.py

# Test evaluation routes
python app/api/v1/evaluation.py

# Test metrics routes
python app/api/v1/metrics.py
```

### Code Organization

The service follows a clean architecture pattern:

1. **Routes** (`app/api/v1/`): Handle HTTP requests and responses
2. **Schemas** (`app/schemas/`): Define request/response models
3. **Services** (`app/services/`): Business logic and external integrations
4. **Models** (`app/models/`): Database models and ORM
5. **Core** (`app/core/`): Configuration, authentication, and utilities

### Adding New Features

1. **New Routes**: Add to appropriate file in `app/api/v1/`
2. **New Models**: Create in `app/models/` and update database
3. **New Schemas**: Define in `app/schemas/`
4. **New Services**: Add to `app/services/`

## Production Deployment

### Docker Deployment

1. **Build and run with docker-compose**:
   ```bash
   cd docker
   docker-compose up -d
   ```

2. **Scale services**:
   ```bash
   docker-compose up -d --scale ragas-service=3
   ```

### Environment Setup

1. **Set production environment variables**:
   ```bash
   export DATABASE_URL="postgresql://user:pass@prod-db/ragas"
   export REDIS_URL="redis://prod-redis:6379"
   export SECRET_KEY="your-production-secret-key"
   ```

2. **Configure reverse proxy** (nginx):
   - Use provided `nginx.conf`
   - Set up SSL certificates
   - Configure rate limiting

3. **Set up monitoring**:
   - Health checks: `/health`
   - Metrics: Prometheus endpoints
   - Logging: Structured logging with structlog

### Security Considerations

- Use strong, unique API keys
- Enable HTTPS in production
- Set up proper CORS policies
- Implement rate limiting
- Use secure database connections
- Regular security updates

## Troubleshooting

### Common Issues

1. **Database Connection Errors**:
   - Check `DATABASE_URL` configuration
   - Ensure PostgreSQL is running
   - Verify database permissions

2. **LLM API Errors**:
   - Verify API keys are correct
   - Check API rate limits
   - Ensure model names are valid

3. **Evaluation Failures**:
   - Check sample data format
   - Verify metric configurations
   - Review error logs

### Logs

- **Application logs**: Check container logs with `docker-compose logs ragas-service`
- **Database logs**: `docker-compose logs postgres`
- **Redis logs**: `docker-compose logs redis`

### Health Checks

- **Service health**: `GET /health`
- **Database health**: Check PostgreSQL connection
- **Redis health**: Check Redis connection

## Contributing

1. Follow the existing code structure
2. Add tests for new features
3. Update documentation
4. Use consistent formatting (black, isort)
5. Follow FastAPI best practices

## License

This project is part of the RAGAS evaluation toolkit.
