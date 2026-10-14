.PHONY: help install dev test lint format seed-corpus clean demo demo-crash demo-protected demo-normal

ENGINE_DIR := apps/engine

# Detect if uv is available in PATH, otherwise use the local virtualenv directly
UV := $(shell command -v uv 2>/dev/null)

ifeq ($(UV),)
VENV_BIN := $(CURDIR)/.venv/bin
PYTHON   := $(VENV_BIN)/python
PYTEST   := $(VENV_BIN)/pytest
RUFF     := $(VENV_BIN)/ruff
UVICORN  := $(VENV_BIN)/uvicorn
else
PYTHON   := uv run python
PYTEST   := uv run pytest
RUFF     := uv run ruff
UVICORN  := uv run uvicorn
endif

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install all dependencies
	cp -n .env.example .env 2>/dev/null || true
ifeq ($(UV),)
	@echo "Using existing .venv at $(VENV_BIN)"
else
	uv sync --all-extras
endif
	@echo "✅ Dependencies ready. Edit .env to add your API keys."

dev: ## Start the engine API server (hot-reload)
	$(UVICORN) engine.main:app --app-dir $(ENGINE_DIR)/src --host 0.0.0.0 --port 8000 --reload

test: ## Run all tests across library and engine
	$(PYTEST) packages/agentic-circuit-breaker/tests/ apps/engine/tests/ -v

lint: ## Run ruff linter + formatter check
	$(RUFF) check packages/agentic-circuit-breaker apps/engine/src apps/engine/tests scripts
	$(RUFF) format --check packages/agentic-circuit-breaker apps/engine/src apps/engine/tests scripts

format: ## Auto-format with ruff
	$(RUFF) format packages/agentic-circuit-breaker apps/engine/src apps/engine/tests scripts
	$(RUFF) check --fix packages/agentic-circuit-breaker apps/engine/src apps/engine/tests scripts

seed-corpus: ## Seed the corpus with sample documents
	$(PYTHON) scripts/seed_corpus.py

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.db" -delete 2>/dev/null || true

demo: ## Run the full interactive live demo (Baseline vs Circuit Breaker)
	$(PYTHON) scripts/demo.py

demo-crash: ## Run the adversarial query to reproduce the GraphRecursionError (Baseline)
	$(PYTHON) scripts/run_query.py --baseline

demo-protected: ## Run the adversarial query with circuit breaker enabled (Protected)
	$(PYTHON) scripts/run_query.py --protected

demo-normal: ## Run a normal query — should complete successfully
	$(PYTHON) scripts/run_query.py --query "What is solar energy and how does it work?"

