FROM python:3.11-slim

WORKDIR /app

# Install uv for fast dependency installation
RUN pip install uv --no-cache-dir

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml ./

# Install dependencies
RUN uv pip install --system --no-cache .

# Copy source code
COPY src/ ./src/

# Create data directory (mounted as volume in docker-compose)
RUN mkdir -p /app/data/corpus /app/data/adversarial_queries

EXPOSE 8000

CMD ["uvicorn", "engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
