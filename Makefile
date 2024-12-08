.PHONY: help install clean test lint format docker-build docker-run mlflow docs

# Variables
PYTHON := python
PIP := pip
PROJECT := graph-attention-networks
DOCKER_COMPOSE := docker compose

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

clean:  ## Clean up generated files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "*.egg" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/

test:  ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:  ## Run code linting
	flake8 src/ tests/
	mypy src/ tests/
	black --check src/ tests/
	isort --check-only src/ tests/

format:  ## Format code
	black src/ tests/
	isort src/ tests/

# Docker commands
docker-build:  ## Build Docker images
	$(DOCKER_COMPOSE) build

docker-run:  ## Run Docker containers
	$(DOCKER_COMPOSE) up

docker-stop:  ## Stop Docker containers
	$(DOCKER_COMPOSE) down

docker-clean:  ## Clean Docker resources
	$(DOCKER_COMPOSE) down -v --remove-orphans

# MLflow commands
mlflow-ui:  ## Start MLflow UI
	mlflow ui --port 5000

mlflow-clean:  ## Clean MLflow artifacts
	rm -rf mlruns/

# Development commands
dev-setup:  ## Setup development environment
	$(PIP) install -e ".[dev]"
	pre-commit install
	mkdir -p data/{raw,processed,interim}
	mkdir -p models/artifacts
	touch .env

dev-start:  ## Start development server
	uvicorn src.api.main:app --reload --port 8080

# Data commands
data-download:  ## Download required datasets
	$(PYTHON) -m src.data.download

data-clean:  ## Clean data directories
	rm -rf data/processed/*
	rm -rf data/interim/*
	touch data/processed/.gitkeep
	touch data/interim/.gitkeep

# Documentation commands
docs-build:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Training commands
train:  ## Train the model
	$(PYTHON) -m src.training.train

train-gpu:  ## Train the model on GPU
	CUDA_VISIBLE_DEVICES=0 $(PYTHON) -m src.training.train

# Deployment commands
deploy-prod:  ## Deploy to production
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml up -d

deploy-stop:  ## Stop production deployment
	$(DOCKER_COMPOSE) -f docker-compose.prod.yml down

# Environment setup
setup-env:  ## Setup virtual environment
	$(PYTHON) -m venv .venv
	. .venv/bin/activate && $(PIP) install --upgrade pip
	. .venv/bin/activate && $(PIP) install -e ".[dev]"

requirements:  ## Update requirements files
	pip-compile requirements/base.in -o requirements/base.txt
	pip-compile requirements/dev.in -o requirements/dev.txt
	pip-compile requirements/prod.in -o requirements/prod.txt