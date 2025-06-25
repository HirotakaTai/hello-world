.PHONY: help build run test clean deploy

# Default target
help:
	@echo "Available targets:"
	@echo "  build    - Build the Docker image"
	@echo "  run      - Run the application locally"
	@echo "  test     - Run tests"
	@echo "  clean    - Clean up Docker images and containers"
	@echo "  deploy   - Deploy to production"
	@echo "  dev      - Start development environment"

# Build Docker image
build:
	docker build -t vscode-devcontainer-python:latest .

# Run application locally
run:
	docker run --rm -p 8000:8000 vscode-devcontainer-python:latest

# Run tests
test:
	uv run python -m pytest

# Clean up Docker resources
clean:
	docker system prune -f
	docker image prune -f

# Deploy to production
deploy:
	docker-compose -f docker-compose.prod.yml up -d

# Start development environment
dev:
	uv sync --dev
	uv run python main.py

# Security scan
security-scan:
	docker run --rm -v $(PWD):/app clair-scanner:latest
