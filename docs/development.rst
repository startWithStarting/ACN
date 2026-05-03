Development Guide
=================

This guide covers development practices for contributing to ACN.

Development Setup
-----------------

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-repo/acn.git
   cd acn

   # Install uv (if not already installed)
   curl -LsSf https://astral.sh/uv/install.sh | sh

   # Install dependencies
   uv sync

   # Copy environment file
   cp .env.example .env

Running Tests
-------------

.. code-block:: bash

   # Run all tests
   uv run python run_tests.py

   # Run specific test file
   uv run python -m pytest tests/test_physics.py -v

   # Run with coverage
   uv run python -m pytest --cov=src tests/

Code Style
----------

ACN follows these conventions:

* **Type hints**: All functions should have type annotations
* **Docstrings**: Use Google-style docstrings
* **Naming**: snake_case for functions/variables, PascalCase for classes
* **Line length**: Maximum 100 characters

Example:

.. code-block:: python

   def calculate_distance(
       pos1: Tuple[float, float],
       pos2: Tuple[float, float]
   ) -> float:
       """Calculate Euclidean distance between two points.

       Args:
           pos1: First position (x, y)
           pos2: Second position (x, y)

       Returns:
           Distance between the two positions
       """
       return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

Adding New Strategies
---------------------

1. Create a new file in ``src/agents/strategies/``

2. Implement the strategy function with decorator:

   .. code-block:: python

      from src.agents.registry import register_strategy

      @register_strategy("my_strategy", side="red")
      def my_strategy(
          current_pos: Optional[Tuple[float, float]],
          grid_center: Optional[Tuple[float, float]]
      ) -> Dict[str, Any]:
          # Implementation
          return {'direction': direction, 'speed': speed}

3. Add to configuration:

   .. code-block:: yaml

      red_agents:
        strategy: "my_strategy"

Adding New Agents
------------------

1. Extend ``BaseAgent`` in ``src/agents/``

2. Implement required methods:

   * ``choose_action()``: Select action based on observation
   * ``get_observation()``: Generate observation of environment

3. Register in ``AgentFactory``

Contributing
-----------

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

Project Structure Summary
-------------------------

* ``src/agents/``: Agent implementations
* ``src/env/``: PettingZoo environments
* ``src/physics/``: Physics simulation
* ``src/training/``: RL training utilities
* ``src/benchmark/``: Performance metrics
* ``src/utils/``: Helper functions
* ``config/``: YAML configuration files
* ``tests/``: Unit tests
* ``docs/``: Documentation