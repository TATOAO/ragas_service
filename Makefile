.PHONY: help install test run docker-build docker-run docker-stop clean

help: ## Show this help message
	@echo "RAGAS FastAPI Service - Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	pip install -r requirements.txt

test: ## Run tests
	python test_service.py

run: ## Run the service in development mode
	uvicorn main:app --host 0.0.0.0 --port 8000 --reload

docker-build: ## Build Docker image
	docker build -f docker/Dockerfile -t ragas-service .

docker-run: ## Run with Docker Compose
	cd docker && docker-compose up -d

docker-stop: ## Stop Docker Compose services
	cd docker && docker-compose down

docker-logs: ## View Docker logs
	cd docker && docker-compose logs -f

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	rm -rf .pytest_cache
	rm -rf uploads/*

setup: ## Initial setup
	cp env.example .env
	python3 -m venv venv
	@echo "Please activate the virtual environment and run 'make install'"

format: ## Format code with black and isort
	black .
	isort .

lint: ## Run linting checks
	flake8 .
	mypy .

check: format lint test ## Run all checks
