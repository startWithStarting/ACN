# Environment

ACN provides PettingZoo-compatible environments for multi-agent simulation. The
runtime code is in `src/env/`.

## Overview

The environment implements the standard multi-agent RL loop:

* Agents observe the environment state
* Agents select actions based on observations
* The environment applies movement, computes rewards, and advances time
* The environment returns rewards, terminations, truncations, and next observations

## PettingZoo APIs

ACN supports both PettingZoo execution styles:

| API | Runtime class | Source | Behavior |
| --- | --- | --- | --- |
| Parallel | `ParallelGameEnv` | [`parallel_env.py`](../src/env/parallel_env.py) | All active agents act in the same step. |
| AEC | `AECGameEnv` | [`aec_env.py`](../src/env/aec_env.py) | Agents act through the PettingZoo Agent Environment Cycle. |

Both implementations share most behavior through `ACNEnvironmentLogic`
([source](../src/env/common_env_logic.py)).

The benchmark features are parallel-only: `AECGameEnv` raises
`NotImplementedError` when `environment.communication` enables a scheme or
when a benchmark reward mode is configured.

## Common Runtime Logic

`ACNEnvironmentLogic` is responsible for:

* Reading environment dimensions and episode length
* Declaring base action and observation spaces
* Initializing agent positions
* Building observations through `_get_observation()`
* Applying actions through `_apply_action()`
* Registering and stepping `PhysicsEngine` when physics is enabled
* Calculating rewards through `_calculate_reward()`
* Rendering through Matplotlib or Pygame paths
* Saving GIFs and optional debug outputs

## Observation Building

The live environments currently construct observations through
`ACNEnvironmentLogic._get_observation()`.

`src.env.observation` ([source](../src/env/observation.py)) provides a
builder-pattern module for future refactors:

* `ObservationBuilder`
* `BlueObservationBuilder`
* `RedObservationBuilder`
* `FlockingObservationBuilder`
* `create_observation_builder()`

Those builders are useful extension points, but they are not the active
observation path in `AECGameEnv` or `ParallelGameEnv`.

## Reward Functions

`src.env.rewards` ([source](../src/env/rewards.py)) provides a composable reward
protocol and factory:

* `RewardFunction`
* `AttractorReward`
* `DistanceReward`
* `DetectionReward`
* `CompositeReward`
* `create_reward_function()`

The current `AECGameEnv` and `ParallelGameEnv` do not call
`create_reward_function()`. By default they use a fixed attractor-ring rule:
red agents score when their distance from the grid center is close to `50.0`
and they are not inside any active blue agent's detection radius. If no red
agent scores in a step/cycle, active blue agents receive a passive reward of
`0.1`.

The optional `environment.reward` section switches either team to the
config-gated benchmark reward mode (dense team/individual blue terms and
score/track/progress red terms), computed by `blue_benchmark_reward()` and
`red_benchmark_reward()` in `src.env.rewards` over a shared per-step
`DetectionState`. Benchmark modes are supported by the parallel environment
only; `AECGameEnv` raises `NotImplementedError` when one is configured. See
[Reward Configuration](configuration.md#reward-configuration).

## Action Space

Built-in agents output continuous action dictionaries:

* `direction`: 2D normalized vector `[x, y]`
* `speed`: scalar value or one-element array

The optional `environment.action_space` section instead declares one flat
`Discrete(N)` movement space per agent (evenly spaced headings x speed
levels); integer actions are decoded through that spec, and dict actions
remain accepted in both modes. The discrete space is required by the MARL
trainer. See
[Action Space Configuration](configuration.md#action-space-configuration).

In the default continuous mode the declared action space allows speed values
in `[0, max_speed]` (default `10`). Movement is applied
through `PhysicsEngine` by default. The default `physics.control_mode` is
`velocity`, which treats `direction * speed` as a desired velocity for the
current step, then applies physics boundaries, collisions, obstacles, and force
fields. Set `environment.physics.control_mode: "force"` to treat the same vector
as a force and preserve inertia, or set `environment.physics.enabled: false` to
use the legacy direct-kinematic path.

## Observation Space

Returned observations are dictionaries containing:

* `position`: agent's current position `[x, y]`
* `grid_center`: center of the simulation grid
* `timestamp`: current step count
* `red_agents`: visible red-agent positions for blue agents
* `blue_agents`: visible blue-agent positions for red agents
* `red_teammates`: visible red-team positions for red agents
* Flocking strategy parameters such as `cohesion_weight` and `max_force` when
  the observing red agent uses `strategy_type: "flocking"`

Visibility is limited by the observing agent's `detection_radius`.

Two config-gated additions change this layout:

* `environment.observation.blue_sensor: "bearing_only"` replaces the blue
  `red_agents` dict with anonymous `contact_reports` (see
  [Observation Configuration](configuration.md#observation-configuration)).
* When the parallel environment executes a communication scheme, every
  observation additionally carries a `communication` key with the agent's
  delivered-message view (see
  [Communication Configuration](configuration.md#communication-configuration)).

The declared Gymnasium observation space currently covers only the base fields
(`position`, `grid_center`, and `timestamp`). Returned observations include
additional dictionaries for agent-relative information. This should be tightened
before strict RL-library training relies on observation-space validation.

## Rendering

The environment supports multiple render modes:

* `human`
* `human_matplotlib`
* `human_matplotlib_pred`
* `human_pygame`

Rendering and GIF export are useful for inspection, but benchmark throughput
should be measured in headless mode once that path is added.
