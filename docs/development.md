# Development Guide

This guide covers development practices for contributing to ACN.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/your-repo/acn.git
cd acn

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Copy environment file
cp .env.example .env

```
## Running Tests

```bash
# Run all tests
uv run python run_tests.py

# Run specific test file
uv run python -m pytest tests/test_physics.py -v

# Run with coverage
uv run python -m pytest --cov=src tests/

```
## Merge Gate

Run the full gate before committing implementation work:

```bash
./scripts/gate.sh
```

It runs, in order: `ruff check .`, the full `run_tests.py` suite, and the
golden-trace regression check (`scripts/golden_traces.py --check`). The gate
must pass before a commit lands on `main`.

## Code Style

ACN follows these conventions:

* **Type hints**: All functions should have type annotations
* **Docstrings**: Use Google-style docstrings
* **Naming**: snake_case for functions/variables, PascalCase for classes
* **Line length**: Maximum 100 characters

Example:

```python
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

```
## Adding New Strategies

1. Create a new file in `src/agents/strategies/`

2. Implement the strategy function with decorator:

   ```python
   from src.agents.registry import register_strategy

   @register_strategy("my_strategy", side="red")
   def my_strategy(
       current_pos: Optional[Tuple[float, float]],
       grid_center: Optional[Tuple[float, float]]
   ) -> Dict[str, Any]:
       # Implementation
       return {'direction': direction, 'speed': speed}
   ```

3. Wire the strategy into `RedAgent.choose_action()` or refactor strategy
   dispatch to call `src.agents.registry.get_strategy()`.

4. Add to configuration:

   ```yaml
   agents:
     red_agents:
       - count: 4
         strategy_type: "my_strategy"
   ```

## Adding New Agents

1. Extend `BaseAgent` in `src/agents/`

2. Implement required methods:

   * `choose_action()`: Select action based on observation
   * `get_observation()`: Generate observation of environment

3. If the new type must be selectable by name, decorate it with
   `@register_agent` and update `create_agents_from_config` to construct it
   from the project YAML schema.

## Adding New Runtime Capabilities

* For rewards, the config-gated benchmark modes live in `src.env.rewards`
  behind `environment.reward` (parallel environment only); new reward terms
  should extend that path. The older `create_reward_function()` factory is
  still not wired into the environments.
* For richer movement, extend the existing `PhysicsEngine` integration with
  scenario-specific body settings, fields, obstacles, and inertial controls.
* For communication experiments, add a new scheme builder registered with
  `src.communication.registry.register_communication_scheme` and import it
  from `src.communication.schemes` (mirroring the agent-strategy registry
  pattern). Differentiable schemes must mark their plans as such so the
  environment never executes them. `src.communication.models` is legacy; do
  not build on it.
* For training research, extend the TorchRL MARL trainer (`src.training.marl`)
  — e.g. new `OnlineMethod` implementations behind the `training:` config
  block — rather than the legacy SB3 path.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Project Structure Summary

* `src/agents/`: Agent implementations
* `src/env/`: PettingZoo environments
* `src/physics/`: Physics simulation
* `src/communication/`: Communication runtime and schemes
* `src/training/`: RL training (`marl/` TorchRL trainer + legacy SB3 path)
* `src/benchmark/`: Performance metrics
* `src/utils/`: Helper functions
* `config/`: YAML configuration files
* `infra/`: Remote training on Modal
* `scripts/`: Merge gate (`gate.sh`) and golden-trace tooling
* `tests/`: Unit tests
* `docs/`: Documentation
