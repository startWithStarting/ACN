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
├── communication/    # Communication runtime (schemes, topology, transport, processors)
├── training/         # RL training (marl/ TorchRL trainer + legacy SB3 path)
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

The parallel environment additionally executes the configured communication
scheme inside `step(actions)` (see Communication below); the AEC environment
raises `NotImplementedError` when communication or a benchmark reward mode is
enabled.

### Communication

`src/communication` implements the benchmark communication runtime from
`docs/communication_implementation_plan.md`. A named scheme configured under
`environment.communication` is compiled into a `CommunicationPlan` (topology,
transport, processor, `R` rounds per step, `C`-round message cache) by the
scheme registry. Execution ownership depends on differentiability:

* **Engineered schemes** (`one_hop_direct`, `one_hop_mean`, `multihop_relay`)
  run inside `ParallelGameEnv.step()` before movement, over a same-team radius
  graph frozen at the pre-move positions. Deliveries land in per-agent message
  caches and are attached to the NEXT observation under a `communication` key.
* **The differentiable scheme** (`multihop_gnn`) is compiled and validated by
  the environment but never executed there: the MARL trainer runs the
  GraphSAGE message-passing rounds inside the actor's forward pass so
  gradients reach the message functions. Observations carry no
  `communication` key for this scheme.

### Training

`src/training/marl` is the TorchRL-backed MARL trainer (`run.py --mode train`
with `training.backend: "marl"`). It binds one PPO algorithm to exactly one
trainable team (blue or red) against the scripted opponent team, supporting
shared or separate actors and local critics (IPPO) or one central critic over
training-only privileged simulator state (MAPPO-style CTDE). The privileged
tensor never reaches an actor. Runs are seeded, headless, and checkpointed
with exact resume. The legacy SB3 red-team parameter-sharing path
(`src/training/trainer.py`) remains for backward compatibility.

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

Implementation status of the research-facing subsystems:

* `src.communication` provides the live runtime described above; the parallel
  environment executes engineered schemes and the MARL actor executes the
  differentiable one. The older `src.communication.models` classes are a
  legacy placeholder layer kept for backward compatibility.
* `src.env.rewards` provides the config-gated benchmark reward modes
  (`environment.reward`, parallel environment only) alongside the default
  legacy attractor-ring scoring and blue passive reward. The older
  `create_reward_function()` factory is still not wired into the environments.
* `src.training.marl` implements CTDE (MAPPO-style privileged central critic),
  IPPO-style local critics, and both learning directions (trainable blue or
  trainable red) plus learned communication for `multihop_gnn`. Opponent
  modeling is not implemented. The legacy SB3 red-team path remains as a
  compatibility adapter.

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
