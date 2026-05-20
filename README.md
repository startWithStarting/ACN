# ACN (Agent Communication Networks)

A multi-agent simulation framework for researching communication protocols and distributed control in autonomous agent systems.

![ACN Architecture](UMLdiagram.svg)

## Overview

ACN is built on [PettingZoo](https://github.com/Farama-Foundation/PettingZoo) and provides:

* **Blue Agents**: Defensive units with VAR-based prediction models to forecast red agent movements
* **Red Agents**: Mobile units with configurable movement strategies
* **Physics Package**: Integrated 2D physics helpers with collisions, drag, obstacles, and force fields
* **Communication Models**: No-op and placeholder model interfaces for future message passing
* **Run Traces**: Training-oriented traces for observations, actions, rewards, states, and blue-agent prediction events, written either as local JSONL files or directly to Postgres
* **Benchmarking**: Scenario-comparison scaffolding and basic metrics

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
* `GNNCommunicationModel`: Placeholder for future graph neural network-based message passing

### Training

Located in `src/training/`:

* `Trainer`: MARL training loop integration with Stable-Baselines3
* `BaseTrainer`: Abstract base for custom trainers

The current training path is a parameter-sharing PPO path for red agents. It does
not yet implement CTDE, centralized critics, learned communication, or opponent
modeling.

## Quick Start

```bash
# Install dependencies
uv sync

# Run the default parallel simulation
uv run python run.py --mode parallel

# Persist a run directly to Postgres instead of writing JSONL trace files
uv run python run.py --mode parallel --config config/experiment_config.yaml --persist

# Specify config
uv run python run.py --mode parallel --config config/aggressive_config.yaml

# Run AEC mode
uv run python run.py --mode aec --config config/experiment_config.yaml

# Inspect one blue agent from a completed run
uv run python -m src.analysis.blue_history \
  --run-dir results/avoidant/avoidant_strategy_scaled_YYYYMMDD_HHMMSS_parallel \
  --blue-agent blue_0 \
  --plot trajectory
```

## Trace API

Run Postgres and the ACN trace API:

```bash
docker compose up --build
```

The API is served on `http://localhost:8000`. With `--persist`, simulations
write directly to Postgres using a UUID `run_id`; no `trace/*.jsonl` files are
created for that run.

Without `--persist`, simulations write timestamped local run folders under
`results/.../<run_id>/trace/`. Those local runs can still be ingested later:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"run_dir": "/app/results/experiment/basic_comm_test_YYYYMMDD_HHMMSS_parallel"}'
```

Useful endpoints:

* `GET /runs`
* `GET /runs/{run_id}/agents`
* `GET /runs/{run_id}/transitions?agent_id=blue_0`
* `GET /runs/{run_id}/events?event_type=prediction&source_agent_id=blue_0`
* `GET /runs/{run_id}/trajectory?agent_id=blue_0`
* `POST /runs/{run_id}/plots`

You can also ingest from the host Python environment:

```bash
uv run python -m src.storage.ingest \
  --run-dir results/experiment/basic_comm_test_YYYYMMDD_HHMMSS_parallel
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
│   ├── analysis/         # Run trace inspection and plotting utilities
│   ├── api/              # FastAPI trace service
│   ├── storage/          # Postgres schema, direct persistence, and ingestion utilities
│   └── utils/            # Logging, config, geometry
├── config/               # YAML configs
├── tests/                # Unit tests
├── docs/                 # Full documentation
├── run.py                # Unified entry point
├── main.py               # Deprecated AEC wrapper
└── main_parallel.py      # Deprecated parallel wrapper
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
experiment_name: "flocking_strategy"
results_base_dir: "results"

agents:
  blue_agents:
    - count: 5
      communication_bandwidth: 15
      processing_capability: 2
      detection_radius: 100.0
      strategy_type: "pursuit"
      prediction_interval: 10

  red_agents:
    - count: 30
      communication_bandwidth: 5
      processing_capability: 3
      detection_radius: 200.0
      strategy_type: "flocking"
      cohesion_weight: 15.0
      alignment_weight: 15.0
      separation_weight: 15.0
      separation_radius: 15.0
      max_speed: 5.0
      min_speed: 2.0
      max_force: 0.2
      wall_avoidance_weight: 5.0
      wall_detection_radius: 200.0

environment:
  width: 300
  height: 300
  max_cycles: 1550
  render_mode: "human_pygame"
  save_episode_gifs: true

analysis:
  trace:
    enabled: true
  plots:
    generate_after_run: false
```

Current implementation note: the default AEC/parallel environments now route movement
through `PhysicsEngine`. The legacy direct-kinematic path is still available with
`environment.physics.enabled: false`. The reward factory and GNN communication model
are present as extension modules, but the default environments still use hard-coded
attractor scoring and no learned message passing.

By default, completed runs write local trace files under `trace/` in the run
directory. Bulk prediction plot generation is opt-in through
`analysis.plots.generate_after_run: true`; use `src.analysis.blue_history` to
recreate individual blue-agent histories and generate targeted plots on demand.

## Testing

```bash
# Run all tests
uv run python run_tests.py

# Run specific module
uv run python -m pytest tests/test_physics.py -v
```

## Documentation

Full Markdown documentation is available in the `docs/` directory:

* [Installation](docs/installation.md)
* [Architecture](docs/architecture.md)
* [Agents](docs/agents.md)
* [Environment](docs/environment.md)
* [Physics](docs/physics.md)
* [Configuration](docs/configuration.md)
* [Strategies](docs/strategies.md)
* [API Reference](docs/api.md)
* [Development](docs/development.md)

The Markdown files can be read directly in GitHub or VS Code. Optionally, build
an HTML site with Sphinx:

```bash
uv run --with sphinx --with sphinx-rtd-theme --with myst-parser sphinx-build -b html docs docs/_build
```

## Future Work

- [ ] Integrate explicit communication channels and learned message policies
- [ ] Add CTDE baselines such as MAPPO/MADDPG-style centralized critics
- [ ] Wire modular rewards into the PettingZoo environments
- [ ] Add scenario-level physics sweeps for inertia, obstacles, collisions, and fields
- [ ] Build benchmark scenarios and statistical reporting for communication utility
- [ ] Add opponent/world-model baselines for prediction under partial observability
