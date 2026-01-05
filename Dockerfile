FROM python:3.11-slim

# Install basics
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into the system environment (no venv needed for Docker)
RUN uv sync --frozen --system

# Copy application code
COPY src ./src
COPY config ./config
COPY main.py .

# Create results directory
RUN mkdir -p results

# Default command: Run training by default, but can be overridden
CMD ["python", "main.py", "--train"]
