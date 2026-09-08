.PHONY: help install dev test lint seed-corpus clean

ENGINE_DIR := apps/engine
PYTHON     := uv run python

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies via uv
	cp -n .env.example .env 2>/dev/null || true
	uv sync --all-extras
	@echo "✅ Dependencies installed. Edit .env to add your API keys."

dev: ## Start the engine API server (hot-reload)
	cd $(ENGINE_DIR) && uv run uvicorn engine.main:app --host 0.0.0.0 --port 8000 --reload --reload-dir src

test: ## Run all tests
	cd $(ENGINE_DIR) && uv run pytest tests/ -v

lint: ## Run ruff linter + formatter check
	uv run ruff check apps/engine/src apps/engine/tests
	uv run ruff format --check apps/engine/src apps/engine/tests

format: ## Auto-format with ruff
	uv run ruff format apps/engine/src apps/engine/tests
	uv run ruff check --fix apps/engine/src apps/engine/tests

seed-corpus: ## Seed the corpus with sample documents
	$(PYTHON) scripts/seed_corpus.py

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.db" -delete 2>/dev/null || true

demo-crash: ## Run the adversarial query to reproduce the GraphRecursionError
	@echo "🔥 Running adversarial query — expect a crash (baseline behaviour)..."
	curl -s -X POST http://localhost:8000/runs \
	  -H "Content-Type: application/json" \
	  -d '{"query": "What is the exact CO2 reduction achieved per country per year from 2015 to 2025?"}' \
	  | python3 -m json.tool

demo-normal: ## Run a normal query — should complete successfully
	@echo "✅ Running normal query — expect a clean answer..."
	curl -s -X POST http://localhost:8000/runs \
	  -H "Content-Type: application/json" \
	  -d '{"query": "What is solar energy and how does it work?"}' \
	  | python3 -m json.tool
