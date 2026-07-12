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
* `reward`: Optional per-team reward mode selection (see
  [Reward Configuration](#reward-configuration))

### Observation Configuration

The `environment.observation` section gates the blue sensor model. An absent
block reproduces current behavior exactly.

Supported keys:

* `blue_sensor`: `"legacy"` (default) or `"bearing_only"`.
  * `legacy`: blue observations carry the historical `red_agents` dict of
    visible red positions and distances.
  * `bearing_only`: blue observations replace `red_agents` with
    `contact_reports`, a variable-size list with one anonymous report per red
    inside the observer's `detection_radius`. Each report is
    `{"payload": [observer_x, observer_y, direction_x, direction_y],
    "metadata": {"observer": <blue name>, "step": <t>}}`, where the direction
    is the global-frame unit vector toward the red (`cos`/`sin` of the
    bearing). No red identity, position, range, or velocity appears in any
    policy-visible field; missing contacts mean not-visible. Reports are
    sorted by bearing angle so traces are reproducible without revealing
    which red produced which report. The simulator retains per-report red
    identity outside the observation: the step/reset `infos` dict carries
    `infos[<blue name>]["ground_truth_contacts"]` (report index -> red name,
    aligned with the report order) for tracking metrics and evaluation. Red
    team observations are unchanged.
* `scripted_blue_privileged`: Defaults to `true`; only meaningful with
  `blue_sensor: bearing_only`. When true, blue observations additionally carry
  `privileged_red_agents` (the legacy dict) so scripted blue controllers such
  as the VAR pursuit blue keep working. This field is for scripted controllers
  and traces only — learned policies must never consume it (enforced later at
  the trainer boundary). Set to `false` for the strict limited-sensing
  contract with no privileged fields in the observation.

Example:

```yaml
environment:
  observation:
    blue_sensor: "bearing_only"
    scripted_blue_privileged: true
```

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

### Communication Configuration

The `environment.communication` section selects and parameterizes a named
communication scheme (see `docs/communication_implementation_plan.md`). It is
parsed and validated by `src.communication.config.parse_communication_config`
and compiled into a runnable plan by
`src.communication.registry.create_communication_plan`. An absent block,
`enabled: false`, or `scheme: "none"` all mean "no communication" and leave
existing scenarios unchanged. When enabled, the parallel environment compiles
the plan at construction and executes it through
`src.communication.runtime.CommunicationRuntime` inside every `step(actions)`.

```yaml
environment:
  communication:
    enabled: true
    scheme: "one_hop_direct"
    rounds_per_step: 1
    cache_window: 0

    topology:
      type: "radius"
      radius_rule: "sender"        # mutual | sender | receiver | minimum
      include_self_edges: false
      freeze_within_step: true

    payload:
      type: "engineered_vector"
      dimension: 4
      coordinate_frame: "global"

    processor:
      backend: "pyg"
      aggregation: "none"
      update: "identity"

    transport:
      type: "slotted_radius"
      delivery: "broadcast"
      free: true
```

Supported keys:

* `enabled`: Defaults to `false`. Communication only runs when explicitly
  enabled with a non-`none` scheme.
* `scheme`: Named scheme. Implemented: `none`, `one_hop_direct`. Reserved for
  later delivery phases: `one_hop_mean`, `multihop_relay`, `multihop_gnn`.
* `rounds_per_step` (R): Synchronous communication rounds per movement step,
  run over a graph frozen at the start of the step. One round moves a message
  at most one graph hop. `one_hop_direct` requires exactly `1`.
* `cache_window` (C): Sliding per-agent message-cache window in rounds. The
  window spans movement-step boundaries; `0` retains nothing. Relay schemes
  must satisfy `cache_window >= processor.ttl`.
* `topology.type`: Graph builder; the initial builder is `radius`
  (`src.communication.topology.RadiusTopology`).
* `topology.radius_rule`: Which agent's communication radius validates a
  directed edge: `sender` (default), `receiver`, `mutual`, or `minimum`.
* `topology.include_self_edges`: Defaults to `false`.
* `topology.freeze_within_step`: Defaults to `true`; per-round graph
  rebuilding is not supported yet.
* `payload.type` / `payload.dimension`: `engineered_vector` with dimension 4
  is the anonymous bearing report
  `[observer_x, observer_y, direction_x, direction_y]` produced by
  `src.communication.sources.EngineeredBearingSource`.
* `processor.aggregation`: Must be `none` for `one_hop_direct` (its contract
  preserves distinct messages); aggregating schemes arrive in Phase 2.
* `processor.ttl`, `processor.forwarding`, `processor.packet_relay`: Relay
  settings; rejected for `one_hop_direct` (use `multihop_relay` later).
* `transport.type` / `transport.delivery`: The initial transport is
  `slotted_radius` with `broadcast` delivery and no loss, queues, or cost.

#### Step Flow (Parallel Environment)

`ParallelGameEnv.step(actions)` runs the communication phase BEFORE movement,
on the frozen pre-move state, per the runtime model in
`docs/communication_implementation_plan.md`:

1. The current local observations (the ones the caller just acted on) are
   rebuilt at the pre-move positions.
2. The payload source derives each agent's outbox from its own observation
   only (`src.communication.sources.EngineeredBearingSource` for
   `engineered_vector`: one anonymous 4-float bearing report per locally
   visible opponent).
3. `CommunicationRuntime.run_step` builds the frozen same-team radius graph
   and runs the `R` synchronous rounds; deliveries enter the per-agent
   message caches.
4. Movement actions are applied and physics advances once, as before.
5. The step's post-move observations are returned with each agent's
   communication view attached, so delivered messages become available to
   the NEXT decision.

The AEC environment does not support communication: constructing `AECGameEnv`
with an enabled scheme raises `NotImplementedError` (AEC agent iteration order
must not emulate synchronous communication rounds).

#### Observation And Infos Contract

When communication is enabled, every observation dict (from `reset` and
`step`) carries a `communication` key; the disabled path never adds the key,
so legacy observation dicts stay structurally identical. The view is:

```python
observation["communication"] == {
    "scheme": "one_hop_direct",   # configured scheme name
    "inbox": EdgeMessageBatch,    # this step's per-agent entry of the
                                  # CommunicationResult.agent_output (the
                                  # preserved inbox for one_hop_direct);
                                  # an empty tuple () at reset or when the
                                  # agent was off the graph
    "agent_ids": ("blue_0", ...), # node-index -> agent-id decode table for
                                  # the inbox's sender/origin/receiver indices
    "cache": MessageCache,        # the agent's persistent message-cache handle
    "cache_window": 0,            # its configured C window in rounds
}
```

Reset observations carry the explicit empty view (same shape, zero messages),
implementing the "no-communication baseline receives an explicit empty view"
rule. Step infos surface per-step summary counts:

```python
infos[agent]["communication"] == {"messages_delivered": 4}
```

The step's communication trace records (graph snapshot plus one record per
delivered message copy, built by `src.communication.tracing`) are kept on the
environment as `last_communication_trace_records`. Episode reset clears every
message cache through `CommunicationRuntime.reset()`.

#### Per-Agent Communication Radius

`create_agents_from_config` resolves each agent's communication radius with
this precedence and writes the resolved float onto every agent as
`communication_radius` (which `RadiusTopology` reads directly):

1. the group spec's `communication_radius`
   (`agents.blue_agents[i].communication_radius`);
2. the team default (`agents.team_defaults.<team>.communication_radius`);
3. the environment fallback (`environment.communication_radius`, default
   `15.0`).

```yaml
agents:
  team_defaults:
    blue:
      communication_radius: 12.0
  blue_agents:
    - count: 2
      communication_radius: 20.0   # spec value wins over the team default
    - count: 1                     # uses the blue team default (12.0)
  red_agents:
    - count: 2                     # uses environment.communication_radius
environment:
  communication_radius: 15.0
```

### Reward Configuration

The `environment.reward` section selects the reward mode per team. It is
optional; when absent, both teams use the `legacy` rewards and runtime behavior
is unchanged. The `benchmark` modes implement the formulas from
`docs/communication_decision_log.md` ("Blue Reward Design" and "Red Reward
Design") and are parsed/computed in `src.env.rewards`.

```yaml
environment:
  reward:
    blue: "benchmark"       # "legacy" (default) | "benchmark"
    red: "benchmark"        # "legacy" (default) | "benchmark"
    weights:
      pin: 0.5              # blue: team pin-coverage weight
      track: 1.0            # blue: team track-coverage weight
      shape: 0.1            # blue: individual undercoverage shaping weight
      score_penalty: 2.0    # blue: REQUIRED when blue is "benchmark"
      red_score: 1.0        # red: undetected on-ring occupancy weight
      red_track: 0.5        # red: REQUIRED when red is "benchmark"
      red_progress: 0.25    # red: REQUIRED when red is "benchmark"
```

Semantics:

* `blue: "legacy"`: blue agents receive the passive `+0.1` bonus on every step
  in which no red scores (current behavior).
* `blue: "benchmark"`: every blue agent `i` receives
  `pin * pin_coverage + track * track_coverage + shape * shape_i -
  score_penalty * red_score_fraction`. A red is *pinned* when at least 3 blues
  detect it, and *tracked* after 3 consecutive pinned steps; coverages and the
  shaping term are normalized by the number of active reds, and
  `red_score_fraction` is the number of reds newly scoring this step divided by
  the initial red count. The benchmark blue reward **replaces** the passive
  bonus (never both).
* `red: "legacy"`: a red receives `+1` on every step it occupies the attractor
  ring undetected (current behavior).
* `red: "benchmark"`: every red agent `i` receives
  `red_score * on_ring_undetected_i - red_track * tracked_i + red_progress *
  (phi(s') - phi(s))` with the potential `phi = -|distance_to_center -
  ring_radius|` (potential-based progress-to-ring shaping).

The detection state (who detects whom, pin counts, streaks) is computed once
per step and shared by both teams' benchmark rewards. Benchmark modes are only
supported by the parallel environment; the AEC environment raises
`NotImplementedError` when a benchmark mode is configured. Missing required
weights or unknown keys fail fast at environment construction.

## Current Runtime Caveats

Several modules are present as extension points but are not yet wired into the
default simulation loop:

* `src.env.rewards` also contains an older modular reward factory
  (`create_reward_function`) that is not wired into the environments; the
  runtime uses the legacy attractor-ring reward and blue passive reward by
  default, plus the config-gated benchmark modes described above.
* `src.communication`'s Phase 1 runtime (radius topology, slotted transport,
  fixed-round scheduler, and the `one_hop_direct` scheme) is executed by the
  parallel environment inside `step(actions)` when
  `environment.communication` enables a scheme; communication defaults to
  disabled, and the AEC environment reports enabled communication as
  unsupported. The legacy `src.communication.models` placeholders remain for
  backward compatibility.
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
