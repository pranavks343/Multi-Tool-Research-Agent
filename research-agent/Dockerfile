FROM python:3.12-slim

WORKDIR /app

# Install uv (fast, matches your local workflow)
RUN pip install --no-cache-dir uv

# Copy dependency manifests first — layer caching: deps only reinstall when these change
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy the application
COPY app/ ./app/
COPY scripts/ ./scripts/

EXPOSE 8000

# Bind to 0.0.0.0 so the container is reachable from outside
CMD ["uv", "run", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]