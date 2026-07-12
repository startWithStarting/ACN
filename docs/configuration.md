# Configuration

ACN uses YAML configuration files to define experiments without code changes.
The runtime configuration is grouped under top-level `agents` and
`environment` sections.

## Configuration Files

Example configs are provided in the `config/` directory:

* `experiment_config.yaml`: Mixed blue/red strategy groups
* `center_config.yaml`: Red agents use the center strategy
* `avoidant_config.yaml`: Red agents avoid blue agents at larger scale
* `aggressive_config.yaml`: Red agents pursue visible blue agents
* `team_config.yaml`: Red agents move toward visible teammates
* `flocking_config.yaml`: Red agents use boids-style flocking

## Configuration Structure

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
      prediction_timeout: 50
      observation_window_size: 5

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
  debug_mode: false
  physics:
    enabled: true
    control_mode: "velocity"
    boundary_mode: "clamp"
    default_drag: 0.0
    default_mass: 1.0
    default_max_speed: 10.0
    default_max_force: 10.0
    default_radius: 0.5

analysis:
  trace:
    enabled: true
  plots:
    generate_after_run: false

training:
  algorithm: "PPO"
  num_episodes: 1
  learning_rate: 0.0003

```
## Agent Configuration

Each entry under `agents.blue_agents` or `agents.red_agents` defines a group.
The factory expands each group into `count` concrete agents.

Shared agent keys:

* `count`: Number of agents to create for the group
* `communication_bandwidth`: Metadata for communication-capacity experiments
* `processing_capability`: Metadata for red agents; for blue agents, the cap on
  autoregressive prediction lags
* `detection_radius`: Radius used by local observation and strategy logic
* `strategy_type`: Runtime strategy selector
* `max_speed`: Per-team movement speed cap, default `10.0` (the historic cap).
  The factory resolves it per group and writes the resolved value onto every
  agent. It sets the upper bound of the continuous `speed` Box and the top
  discrete speed level (see `environment.action_space`). For red flocking
  groups the same key is also forwarded into the observation as before.

Blue `strategy_type` values:

* `pursuit`: Move toward the smoothed mean of predicted red positions
* `static`: Remain stationary while still tracking detected red agents

Additional blue keys:

* `prediction_interval`: Number of blue decision steps between prediction updates
* `prediction_timeout`: Steps after which unseen red agents are pruned from the
  predictor state
* `observation_window_size`: Sliding history length for VAR fitting

Red `strategy_type` values:

* `center`: Move toward, then maintain distance from, the grid center
* `avoidant`: Blend center seeking with avoidance of visible blue agents
* `aggressive`: Blend center seeking with pursuit of the nearest visible blue agent
* `team`: Blend center seeking, visible-teammate cohesion, and blue avoidance
* `flocking`: Use cohesion, alignment, separation, and wall avoidance
* `trainable`: Return a no-op from `choose_action`; intended for external policies

Flocking strategy keys are read from the matching red group and injected into that
agent's observation. The current implementation supports `cohesion_weight`,
`alignment_weight`, `separation_weight`, `separation_radius`, `max_speed`,
`min_speed`, `max_force`, `wall_avoidance_weight`, and
`wall_detection_radius`.

## Environment Configuration

The `environment` section is passed to `AECGameEnv` or `ParallelGameEnv`.

Supported keys used by the current environments:

* `width` and `height`: Cartesian grid dimensions
* `max_cycles`: Episode length in AEC cycles or parallel steps
* `render_mode`: `human`, `human_matplotlib`, `human_matplotlib_pred`,
  or `human_pygame`
* `save_episode_gifs`: Save rendered episode GIFs when a results directory exists
* `gif_figsize`: Matplotlib figure size for GIF rendering
* `debug_mode`: Enables extra position-record export in `main_parallel.py`
* `action_space`: Movement action-space selection (see Action Space
  Configuration below)

## Analysis Configuration

The `analysis` section controls trace recording and expensive derived plots.
The default storage backend is file-backed JSONL. Pass `--persist` to `run.py`
to write the same trace records directly to Postgres instead.

Supported keys:

* `analysis.trace.enabled`: Defaults to `true`. Without `--persist`, writes
  local trace files under each timestamped run directory:
  * `trace/manifest.json`: Run, config, and agent metadata
  * `trace/agent_transitions.jsonl`: One training-style row per agent per step,
    including observation, action, reward, next observation, done flags, and
    privileged before/after state
  * `trace/events.jsonl`: Relational events, currently blue observations,
    prediction targets, and future-position predictions
* `analysis.plots.generate_after_run`: Defaults to `false`. Set to `true` to
  generate the older full set of per-blue/per-red PNGs after each run.

With `--persist`, run history is inserted directly into Postgres using a UUID
`run_id`; local `trace/*.jsonl` files are not created. On-demand API plots are
still written as artifact PNGs under `results/api_artifacts/<run_id>/`.

The recommended workflow is to keep `generate_after_run: false` and generate
only the plots needed for inspection:

```bash
uv run python -m src.analysis.blue_history \
  --run-dir results/avoidant/avoidant_strategy_scaled_YYYYMMDD_HHMMSS_parallel \
  --blue-agent blue_0 \
  --target red_30 \
  --plot all \
  --export-history
```

For DB-backed persisted runs, use the trace API instead:

```bash
uv run python run.py --mode parallel --config config/experiment_config.yaml --persist
curl http://localhost:8000/runs
```

### Action Space Configuration

The `environment.action_space` section selects the movement action space built
for every agent (`src/agents/action_spaces.py`). Omitting the block keeps the
legacy continuous spaces, so existing scenarios are unchanged.

```yaml
environment:
  action_space:
    type: "discrete"   # "continuous" (default) | "discrete"
    headings: 8        # H evenly spaced unit headings (discrete mode)
    speed_levels: 4    # S speed levels evenly spaced 0..max_speed (discrete mode)
```

Supported keys:

* `type`: Defaults to `continuous`, the legacy
  `Dict{direction: Box[-1, 1]^2, speed: Box[0, max_speed]}` per-agent space.
  `discrete` builds one flat `Discrete(N)` with `N = 1 + headings *
  (speed_levels - 1)`.
* `headings`: Number of evenly spaced unit headings
  `(cos(2*pi*k/H), sin(2*pi*k/H))`, starting at `(1, 0)`. Default `8`.
* `speed_levels`: Number of speed levels evenly spaced from `0` to the agent's
  resolved `max_speed`, including the zero level. Default `4`, i.e.
  `{0, 1/3, 2/3, 1} * max_speed`. Must be at least 2.

Discrete index layout: index `0` is the shared `stay` action; index
`1 + k*(S-1) + (j-1)` is heading `k` at nonzero speed level `j`, decoding to
the target velocity `j * max_speed/(S-1) * (cos(theta_k), sin(theta_k))`.
Headings are unit vectors, so the speed cap is isotropic.

Action handling in both environments:

* Dict actions (`{'direction', 'speed'}`, the scripted-controller format) are
  always accepted, in both modes.
* Integer actions are decoded through the discrete spec when `type: discrete`
  is configured. In continuous mode integer actions are rejected with an error.
* `max_speed` resolves per agent group (see Agent Configuration), so teams may
  have different speed caps over the same `Discrete(N)` index layout.

The discrete speed level equals the achieved speed only in `velocity` physics
control mode (the default). In `force` control mode the decoded command acts
as a force and the effective top speed emerges from the force/drag balance;
the discrete cap is not enforced there.

### Physics Configuration

The `environment.physics` section controls the runtime physics path.

Supported keys:

* `enabled`: Defaults to `true`. Set to `false` to use the legacy direct
  `position += direction * speed` movement path.
* `control_mode`: `"velocity"` treats `direction * speed` as this step's
  velocity command. `"force"` treats it as a force and preserves inertia.
* `dt`: Physics time step, default `1.0`.
* `boundary_mode`: `clamp`, `bounce`, or `stop`.
* `default_drag`, `default_mass`, `default_max_speed`, `default_max_force`,
  `default_radius`, `default_restitution`: Defaults for registered bodies.
* `enable_collisions` and `enable_obstacles`: Toggle body-body and body-obstacle
  collision handling.
* `obstacles`: List of rectangular or circular obstacle configs accepted by
  `src.physics.obstacles.create_obstacle`.
* `force_fields` or `fields`: List of field configs accepted by
  `src.physics.fields.create_force_field`.

You can override body settings globally with `agent_*` keys, by type with nested
`red` or `blue` dictionaries, or per agent group with a nested `physics`
dictionary under that group.

## Current Runtime Caveats

Several modules are present as extension points but are not yet wired into the
default simulation loop:

* `src.env.rewards` contains a modular reward factory, but the environments use
  a hard-coded attractor-ring reward and blue passive reward.
* `src.communication.models` contains no-op and GNN placeholders, but there is no
  runtime communication channel that delivers messages between agents.
* `src.env.observation` contains observation builder classes, but the runtime
  currently builds observations through `ACNEnvironmentLogic._get_observation`.
* PPO hyperparameters in the bundled configs are stored under `training`. The
  current `Trainer` reads most PPO hyperparameters from the top level, so nested
  training values are descriptive until the schema is unified.

## Environment Variables

ACN uses the following environment variables:

* `ACN_CONFIG_PATH`: Default config file used by command-line entry points
* `ACN_RESULTS_DIR`: Reserved for results-directory configuration
* `ACN_LOG_LEVEL`: Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`)
* `ACN_DATABASE_URL`: Postgres URL used by `src.storage.ingest` and the trace
  API. The Docker Compose default is
  `postgresql://acn:acn@postgres:5432/acn`; the host default is
  `postgresql://acn:acn@localhost:5432/acn`.
* `ACN_ARTIFACT_DIR`: Optional output directory for API-generated plot
  artifacts from DB-only runs. Defaults to `results/api_artifacts`.

## Loading Configuration

```python
from src.utils.config_loader import load_config

config = load_config("config/experiment_config.yaml")

```
Command-line overrides are handled by the entry point:

```bash
uv run python run.py --mode parallel --config config/center_config.yaml
uv run python run.py --mode parallel --config config/center_config.yaml --persist
```
