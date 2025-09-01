#!/bin/bash

# RAGAS FastAPI Service Startup Script

set -e

echo "Starting RAGAS FastAPI Service..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Warning: .env file not found. Copying from env.example..."
    cp env.example .env
    echo "Please update .env file with your configuration before starting the service."
fi

# Install dependencies if not already installed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

# Create uploads directory if it doesn't exist
mkdir -p uploads

# Run database migrations (if using Alembic)
# alembic upgrade head

echo "Starting the service..."
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
