Installation
============

ACN uses ``uv`` for dependency management.

Prerequisites
------------

* Python 3.11+
* uv package manager

Install uv
~~~~~~~~~~

.. code-block:: bash

   # macOS / Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Windows (PowerShell)
   irm https://astral.sh/uv/install.ps1 | iex

   # Or via pip
   pip install uv

Setup
-----

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-repo/acn.git
   cd acn

   # Install dependencies (creates virtual environment)
   uv sync

   # Copy environment file
   cp .env.example .env

   # Verify installation
   uv run python -c "import src; print('ACN installed successfully')"

Running Simulations
-------------------

.. code-block:: bash

   # Run with default config
   uv run python main.py

   # Run with specific config
   uv run python main.py --config config/aggressive_config.yaml

   # Run parallel mode
   uv run python main_parallel.py --config config/experiment_config.yaml

Docker
------

ACN includes a Dockerfile for containerized execution:

.. code-block:: bash

   # Build image
   docker build -t acn:latest .

   # Run container
   docker run -it acn:latest

Google Cloud Setup
-------------------

See :doc:`gcp_setup` for running on Google Cloud Platform.

Updating Dependencies
---------------------

.. code-block:: bash

   # Add a new dependency
   uv add package-name

   # Update all dependencies
   uv sync --upgrade

   # Remove a dependency
   uv remove package-name