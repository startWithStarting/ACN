# Architecture Overview

ACN is organized into several key modules that work together to provide
a complete multi-agent simulation environment.

## Directory Structure

```text
src/
├── agents/           # Agent implementations and strategies
│   ├── base_agent.py
│   ├── blue_agent.py
│   ├── red_agent.py
│   ├── factory.py
│   ├── registry.py
│   ├── strategies/       # Red agent strategies
│   └── blue_strategies/ # Blue agent strategies
├── env/              # PettingZoo environment
│   ├── aec_env.py
│   ├── parallel_env.py
│   ├── common_env_logic.py
│   ├── observation.py
│   └── rewards.py
├── physics/          # Physics simulation
│   ├── engine.py
│   ├── obstacles.py
│   └── fields.py
├── communication/    # Agent communication models
├── training/         # RL training framework
├── benchmark/        # Performance benchmarking
├── analysis/         # File-backed trace reconstruction and plotting
├── api/              # FastAPI trace query and plotting service
├── storage/          # Postgres schema, ingestion, and direct persistence
└── utils/            # Utilities (logging, config, etc.)
```

## Core Components

### Agents

The agent system uses a combination of inheritance and the strategy pattern:

* **BaseAgent**: Abstract base class defining common interface
* **BlueAgent**: Extends BaseAgent with VAR prediction models
* **RedAgent**: Extends BaseAgent with configurable strategies

Agent behaviors are implemented through strategies that can be swapped at runtime.
Red agents use the registry pattern to support different movement behaviors.

### Environment

The environment follows the PettingZoo API with two implementations:

* **ParallelGameEnv**: All agents step simultaneously
* **AECGameEnv**: PettingZoo Agent Environment Cycle execution

Common logic is extracted into `ACNEnvironmentLogic` mixin to reduce duplication.
The current runtime builds observations, applies movement, computes rewards, and
renders through this mixin.

### Physics Engine

A custom physics engine provides:

* Euler integration for movement
* Drag and friction models
* Rigid body collisions with restitution between registered bodies
* Configurable boundary handling (clamp, bounce, stop)
* Force fields and obstacle geometry helpers

The default PettingZoo environments instantiate `PhysicsEngine` on reset and
register each agent as a physics body. Movement actions are interpreted as
velocity controls by default, then the engine applies boundaries, collisions,
obstacles, and force fields before positions are copied back to agents. The
legacy direct-kinematic path remains available by setting
`environment.physics.enabled: false`.

## Design Patterns

1. **Strategy Pattern**: Agent behaviors are implemented as strategies
   registered via `@register_strategy` decorator

2. **Factory Function**: `create_agents_from_config` builds the default
   blue and red agent populations from YAML configuration

3. **Builder Pattern**: `ObservationBuilder` defines reusable observation
   builders, although current environments still call `ACNEnvironmentLogic._get_observation`
   directly

4. **Registry Pattern**: Global registries for agent types and strategies

5. **Composition**: Reward functions compose multiple reward types

## Runtime Boundaries

Simulation, persistence, and analysis are intentionally separated:

* Default simulations write timestamped local trace folders under `results/`.
* `run.py --persist` writes the same logical trace records directly to Postgres
  with a UUID run ID and does not create local `trace/*.jsonl` files.
* The FastAPI service reads from Postgres and creates plot artifacts on demand.
* File-backed traces can still be ingested later with `src.storage.ingest`.

The repository contains several research-facing abstractions that are not yet
fully connected to the live simulation:

* `src.communication.models` defines no-op and GNN communication model classes,
  but no environment-level message channel exists yet.
* `src.env.rewards` defines configurable reward functions, but the environments
  currently use hard-coded attractor-ring scoring and a blue passive reward.
* `src.training` provides PPO integration for red-agent parameter sharing, but
  does not yet implement centralized critics, learned communication, or opponent
  modeling.

## Extensibility

To add a new red agent strategy:

1. Create a new file in `src/agents/strategies/`
2. Implement the strategy function with `@register_strategy` decorator
3. Wire it into `RedAgent.choose_action()` or refactor strategy dispatch to use
   `get_strategy()`
4. Use it in config by setting `strategy_type: "your_strategy_name"`

To add a new reward function:

1. Implement `RewardFunction` protocol in `src/env/rewards.py`
2. Add factory case in `create_reward_function()`
3. Wire the factory into environment reward calculation
4. Document and validate the resulting YAML keys
