# Installation

ACN uses `uv` for dependency management.

## Prerequisites

* Python 3.11+
* uv package manager

### Install uv

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

# Or via pip
pip install uv

```
## Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/acn.git
cd acn

# Install dependencies (creates virtual environment)
uv sync

# Copy environment file
cp .env.example .env

# Verify installation
uv run python -c "import src; print('ACN installed successfully')"

```
## Running Simulations

```bash
# Run with default config in parallel mode
uv run python run.py --mode parallel

# Run with a specific config
uv run python run.py --mode parallel --config config/aggressive_config.yaml

# Run AEC mode
uv run python run.py --mode aec --config config/experiment_config.yaml

# Persist simulation trace directly to Postgres
uv run python run.py --mode parallel --config config/experiment_config.yaml --persist

# Run PPO training mode
uv run python run.py --mode train --config config/avoidant_config.yaml

```
`main.py` and `main_parallel.py` remain available as compatibility wrappers,
but they emit deprecation warnings and delegate to the older mode-specific paths.

## Docker

ACN includes a Docker Compose stack for the trace API and Postgres database:

```bash
# Start Postgres and the FastAPI trace service
docker compose up -d --build

# Check API health
curl http://localhost:8000/health

```

The API is served on `http://localhost:8000`, and Postgres is exposed on
`localhost:5432`. The API container is intentionally lightweight and focused on
trace querying, ingestion, and on-demand plotting; simulation and training runs
are normally launched from the host `uv` environment.
## Google Cloud Setup

See `docs/gcp_setup.md` for running on Google Cloud Platform.

## Updating Dependencies

```bash
# Add a new dependency
uv add package-name

# Update all dependencies
uv sync --upgrade

# Remove a dependency
uv remove package-name
```
