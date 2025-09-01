# Use Python 3.11 slim image
FROM --platform=linux/amd64 swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/tiangolo/uvicorn-gunicorn:python3.11-slim
# FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv
# Install python packages
RUN uv pip install --system \
    fastapi \
    uvicorn[standard] \
    pydantic \
    pydantic-settings \
    sqlalchemy \
    sqlmodel \
    psycopg2-binary \
    alembic \
    python-jose[cryptography] \
    passlib[bcrypt] \
    python-multipart \
    httpx \
    requests \
    pandas \
    numpy \
    ragas \
    langchain \
    langchain-openai \
    langchain-anthropic \
    langchain-cohere \
    cohere \
    redis \
    celery \
    structlog \
    prometheus-client \
    pytest \
    pytest-asyncio \
    black \
    isort \
    flake8 \
    mypy \
    python-dotenv \
    loguru



# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --upgrade pip \
    && uv pip install --system -r requirements.txt 

# Copy project
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser \
    && chown -R appuser:appuser /app
USER appuser

# Create uploads directory
RUN mkdir -p /app/uploads

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
# uvicorn main:app --host 0.0.0.0 --port 8000 --loop asyncio
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--loop", "asyncio"]
