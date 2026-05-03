# ACN (Agent Communication Networks)

A multi-agent simulation framework for researching communication protocols and distributed control in autonomous agent systems.

![ACN Architecture](UMLdiagram.svg)

## Overview

ACN is built on [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and provides:

* **Blue Agents**: Defensive units with VAR-based prediction models to forecast red agent movements
* **Red Agents**: Mobile units with configurable movement strategies
* **Physics Engine**: Custom 2D physics with collisions, drag, and force fields
* **Communication Models**: Framework for agent-to-agent message passing
* **Benchmarking**: Performance metrics and scenario comparison

## Features

### Agent Types

| Agent | Description |
|-------|-------------|
| **Blue** | Defensive agent with prediction capabilities. Uses Vector Auto-Regressive (VAR) models to predict red agent trajectories. |
| **Red** | Mobile agent with configurable behavior. Can pursue, avoid, flock, or target the grid center. |

### Red Agent Strategies

Located in `src/agents/strategies/`:

| Strategy | Description |
|----------|-------------|
| `center` | Move toward grid center, maintaining 10-unit minimum distance |
| `avoidant` | Detect and steer away from blue agents |
| `aggressive` | Pursue the nearest blue agent |
| `team` | Move toward average position of visible red teammates |
| `flocking` | Full boids-style behavior (cohesion, alignment, separation) |

### Blue Agent Strategies

Located in `src/agents/blue_strategies/`:

| Strategy | Description |
|----------|-------------|
| `static` | Remain stationary, continue tracking |
| `pursuit` | Move toward average predicted red position |

### Communication Models

Located in `src/communication/`:

* `NoCommunicationModel`: Local observations only (baseline)
* `GNNCommunicationModel`: (Experimental) Graph neural network-based message passing

### Training

Located in `src/training/`:

* `Trainer`: MARL training loop integration with Stable-Baselines3
* `BaseTrainer`: Abstract base for custom trainers

## Quick Start

```bash
# Install dependencies
uv sync

# Run simulation
uv run python main.py

# Specify config
uv run python main.py --config config/aggressive_config.yaml

# Run parallel mode
uv run python main_parallel.py --config config/experiment_config.yaml
```

## Project Structure

```
acn/
├── src/
│   ├── agents/           # Agent implementations
│   │   ├── base_agent.py
│   │   ├── blue_agent.py
│   │   ├── red_agent.py
│   │   ├── factory.py    # Agent creation from config
│   │   ├── registry.py   # Strategy registration
│   │   ├── strategies/   # Red agent behaviors
│   │   └── blue_strategies/  # Blue agent behaviors
│   ├── env/              # PettingZoo environments
│   │   ├── aec_env.py    # Alternating agent order
│   │   ├── parallel_env.py  # Simultaneous steps
│   │   ├── common_env_logic.py  # Shared logic
│   │   ├── observation.py  # Builder pattern for obs
│   │   └── rewards.py    # Reward functions
│   ├── physics/          # Physics simulation
│   │   ├── engine.py     # Euler integration, collisions
│   │   ├── obstacles.py  # Rect, Circle obstacles
│   │   └── fields.py     # Force fields
│   ├── communication/    # Message passing
│   ├── training/         # RL training
│   ├── benchmark/        # Performance metrics
│   └── utils/            # Logging, config, geometry
├── config/               # YAML configs
├── tests/                # Unit tests
├── docs/                 # Full documentation
└── main.py               # Entry point (AEC mode)
```

## Configuration

Create `.env` from `.env.example`:

```bash
cp .env.example .env
```

Key settings:
* `ACN_CONFIG_PATH`: Default config file
* `ACN_RESULTS_DIR`: Output directory

Config format (`config/*.yaml`):

```yaml
grid_width: 100
grid_height: 100
max_cycles: 1000

blue_agents:
  count: 5
  detection_radius: 30
  strategy: "pursuit"

red_agents:
  count: 10
  strategy: "flocking"
  params:
    cohesion_weight: 1.0
    separation_weight: 1.5

physics:
  boundary_mode: "clamp"
  enable_collisions: true

render_mode: "human_matplotlib"
```

## Testing

```bash
# Run all tests
uv run python run_tests.py

# Run specific module
uv run pytest tests/test_physics.py -v
```

## Documentation

Full documentation is available in the `docs/` directory:

* [Installation](docs/installation.rst)
* [Architecture](docs/architecture.rst)
* [Agents](docs/agents.rst)
* [Environment](docs/environment.rst)
* [Physics](docs/physics.rst)
* [Configuration](docs/configuration.rst)
* [Strategies](docs/strategies.rst)
* [API Reference](docs/api.rst)
* [Development](docs/development.rst)

Or build with Sphinx:

```bash
cd docs && sphinx-build -b html . _build
```

## Future Work

- [ ] ML-based communication protocols between agents
- [ ] Enhanced world models with sophisticated prediction
- [ ] Cooperative and competitive scenarios
- [ ] Performance evaluation across protocols
- [ ] Distributed training across multiple machines