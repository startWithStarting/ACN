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

ENV MPLBACKEND=Agg \
    SDL_VIDEODRIVER=dummy \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

# Install only the API runtime dependencies. The simulation/training stack stays
# in the host uv environment; this image is intentionally small for trace API use.
RUN uv venv .venv && \
    uv pip install --python .venv/bin/python \
    "fastapi>=0.115.0" \
    "uvicorn[standard]>=0.30.0" \
    "psycopg[binary]>=3.2.0" \
    "matplotlib>=3.0.0" \
    "numpy>=1.21"

# Copy application code
COPY src ./src
COPY config ./config
COPY main.py .
COPY main_parallel.py .
COPY run.py .

# Create results directory
RUN mkdir -p results

# Default command: Run the trace API, but can be overridden for simulations.
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
