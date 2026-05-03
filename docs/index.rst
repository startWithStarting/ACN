ACN Documentation
==================

**Agent Communication Networks (ACN)** is a multi-agent simulation framework
designed for researching communication protocols and distributed control
mechanisms in autonomous agent systems.

.. image:: ../UMLdiagram.svg
   :alt: ACN Architecture
   :width: 100%

Overview
--------

ACN provides a PettingZoo-based environment where:

* **Blue Agents** are defensive agents with VAR-based prediction models to track red agents
* **Red Agents** are mobile agents with configurable movement strategies
* Agents can develop ego-centric world models and optionally share them via communication protocols

The framework supports both parallel and AEC (Agent-Environment-Communication) API modes,
making it suitable for various multi-agent reinforcement learning scenarios.

Quick Start
-----------

.. code-block:: bash

   # Install dependencies
   uv sync

   # Run simulation
   uv run python main.py

   # Or specify a config
   uv run python main.py --config config/aggressive_config.yaml

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   architecture
   agents
   environment
   physics
   configuration
   strategies
   api
   development

Indices and Tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`