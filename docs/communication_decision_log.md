# Communication And MARL Decision Log

This file records design decisions already made for ACN's limited-sensing
communication and first MARL benchmark direction. It is a design record, not a
description of implemented runtime behavior.

## Current Runtime Context

Updated 2026-07: phases 0 through 4 of the
[Communication Implementation Plan](communication_implementation_plan.md) are
implemented. `src.communication` provides the runtime (radius topology,
slotted transport, fixed-round scheduler, message caches) and the registered
schemes `none`, `one_hop_direct`, `one_hop_mean`, `multihop_relay`, and
`multihop_gnn`; the parallel environment executes the engineered schemes
inside `step(actions)` and the MARL trainer runs the differentiable
`multihop_gnn` inside the actor. `src.communication.models` is a legacy
placeholder layer kept for backward compatibility. The decision sections below
are the design record those implementations follow; they are kept as decided,
not rewritten to match the code.

## Observation Contract

Decision: learning-facing observations remain local dictionaries/lists of
detections, not environment-level graph observations.

Rationale:

- A dict/list observation directly represents limited sensing: the agent sees
  only currently visible contacts.
- Missing contacts mean "not visible," not "does not exist."
- The simulator can still convert observations to graph/tensor forms in policy
  or trainer adapters.

Implications:

- The environment should preserve direct local sensing as the source of truth.
- Graph representations are optional model-layer adapters, especially for GNN
  policies and communication processors.
- Observation code must avoid leaking hidden team size through padding masks or
  global IDs unless explicitly configured.

Decision: policy-facing encoding of variable-size observations and received
messages is part of the selected communication scheme, not the canonical sensor
contract.

- A scheme may preserve every report, compute fixed statistics such as a mean
  or standard deviation, apply attention or a permutation-invariant set
  encoder, or construct graph features.
- Each scheme must declare the shape and semantics of the communication output
  presented to the movement actor.
- This permits representation experiments without changing what the simulated
  sensor observed or silently discarding information for every scheme.

## Identity And Evaluability

Decision: simulator identities remain internal for logging, tracing, and
evaluation, while policy-visible identity is configurable.

Defaults:

- Teammates may have policy-visible identity, especially for communication,
  routing, and role coordination.
- Opponent identities should be hidden or remapped by default.
- Ground-truth opponent IDs remain available to experiment traces so prediction
  and tracking quality can still be evaluated.

Implications:

- The learning agent gets limited sensing.
- The experiment still gets full evaluability.
- Policy-visible contact IDs must be treated separately from simulator IDs.

## Sensor Model

Decision: the initial blue sensor is bearing-only and returns a variable-size
collection of anonymous red detections.

Settled semantics:

- Each detection contains a bearing to one visible red.
- Bearings are represented as global-frame unit directions
  `(cos(theta), sin(theta))`, avoiding scalar-angle wraparound.
- Each detected red produces one anonymous contact report with numerical
  payload `[observer_x, observer_y, direction_x, direction_y]`.
- Contact-report metadata contains the originating teammate identity and
  observation timestep, but no opponent identity.
- Multiple detections produce multiple reports, so receiving policies consume a
  variable-size collection rather than a padded opponent roster.
- The sensor does not expose red position, range, velocity, or confidence.
- The simulator retains the corresponding red identity as privileged metadata
  for rewards, tracking metrics, and experiment evaluation.
- The policy-facing observation does not expose persistent red identity.
- Bearing measurements are deterministic initially; probabilistic sensor noise
  is deferred.
- Global-frame bearings assume a shared world orientation; the physical analogue
  is an agent-local compass or IMU heading.

Rationale:

- Bearing-only sensing is physically compatible with passive directional or
  vision-based detection of non-cooperative targets.
- It supports multiple contacts without assuming cooperative ranging beacons.
- Combining reports from different blue positions makes communication directly
  relevant to localization.

## Action Space

Decision: the benchmark movement action is discrete, replacing the current
continuous `Dict{direction, speed}`. Discrete is chosen for broad
learning-algorithm compatibility: value-based (single Q head) and policy-gradient
(single categorical) families both consume it natively, which supports the
algorithm-swap goal.

Current implementation for reference: `RedAgent`/`BlueAgent` each hardcode
`Dict{'direction': Box[-1,1]^2, 'speed': Box[0,10]}`, and `_parse_movement_action`
computes `velocity = direction * speed` without normalizing direction. That makes
diagonals faster than axis-aligned moves and makes speed ambiguous. The discrete
design removes both problems.

Structure:

- Heading: `H` evenly-spaced directions. The environment decodes heading index
  `k` to the unit vector `(cos theta_k, sin theta_k)`. Unit-norm headings make the
  maximum speed isotropic, removing the old anisotropy. Default `H = 8`,
  configurable.
- Speed: `S` levels evenly spaced from `0` to the agent's `max_speed`; level `0`
  is stationary. Default `S = 4`, i.e. `{0, 1/3, 2/3, 1} * max_speed`.
- Decoding: `(heading k, speed j) -> target velocity v = level_j *
  (cos theta_k, sin theta_k)`, consumed by physics. The mapping is one-to-one, so
  the redundancy of the continuous `direction * speed` parameterization is gone.

Representation: a single flat `Discrete(N)` with `N = 1 + H * (S - 1)` — one shared
`stay` action (all speed-0 choices collapse to it) plus `H * (nonzero speed
levels)` moving actions. With `H = 8`, `S = 4` this is `N = 25`. Flat `Discrete`
is preferred over `MultiDiscrete([H, S])` because it is the most universally
compatible target across algorithm families and avoids `H` redundant stay actions.

Max speed:

- `max_speed` is a per-team parameter: identical within a team, and may differ
  between teams. It sets the top speed level and thus the isotropic speed cap.
- Asymmetric team max speeds are a deliberate difficulty lever. Raising the
  opponent's relative speed makes the task harder for a team: faster reds stress
  blue tracking; faster blues stress red evasion.
- Numeric `max_speed` values (and `H`, `S` if changed from the defaults) are tuned
  and frozen with the benchmark configuration, deferred like reward weights.

Control-mode caveat: `level = speed` holds exactly in `velocity` control mode
(the benchmark default). In `force` mode the levels are forces and the true top
speed emerges from force/drag balance, so a hard speed cap there needs
post-integration magnitude clipping. Documented as a boundary, not a blocker.

Architecture:

- Action-space construction moves out of the agent classes into a config- and
  scheme-aware builder, fixed for the lifetime of an environment instance.
- The discrete-to-velocity/force decoding replaces the continuous parse path.
- For explicit learned-message schemes the action becomes
  `Dict{movement: Discrete(N), message: Box}`; the movement adapter extracts the
  discrete movement field. This stays reserved until the Phase 5 message-action
  work.

## Communication Semantics

Decision: baseline communication is same-team, radius-limited, synchronous, and
round-based.

Settled semantics:

- Direct communication edges exist only between same-team agents within
  communication radius.
- Communication radius is a per-agent runtime property. Configuration resolves
  it with agent/spec-specific values taking precedence over team defaults, and
  the resolved value is written onto every agent.
- Communication and movement remain separate channels. The public environment
  still uses the standard PettingZoo `step(actions)` transition.
- Initial messages are numerical vectors.
- Structured text and LLM-produced messages are future payload types.
- Communication is initially free: no reward penalty, energy cost, or message
  budget penalty.
- Agents know and may communicate their own global position. Detected reds are
  communicated through bearing reports rather than ground-truth positions.
- The direct engineered scheme transports each contact report unchanged and
  does not average or otherwise aggregate reports in the environment.
- A receiver obtains a variable-size inbox of reports. Its policy or model
  adapter is responsible for converting that collection into any required
  fixed-size representation through pooling, attention, a set encoder, or a
  padded tensor and mask.
- Learned or explicitly aggregated communication schemes may transform reports,
  but that transformation is part of the selected scheme rather than baseline
  transport semantics.
- At decision epoch `t`, every learned or rule-based method receives the agent's
  current policy-visible local observation/state, persistent communication
  memory delivered by previous transitions, and optional recurrent state.
- Movement and outgoing communication are produced by separate functions. The
  direct engineered source message is generated from the current local
  observation, `u_i(t) = g_comm(o_i(t))`, while movement is selected as
  `a_i(t) = f_move(o_i(t), C_i(t), h_i(t))`.
- A learned source-message function may additionally consume communication or
  recurrent state when its scheme explicitly defines that behavior. Relay
  protocols may emit previously received packets. Neither case makes outgoing
  communication an output of the movement function.
- For deterministic schemes such as `one_hop_direct`, the public agent action
  contains movement only and the configured communication source derives the
  outbox from the current local observation. For explicit learned-message
  schemes, the declared action space contains separate `movement` and `message`
  fields.
- The action space is fixed for the lifetime of an environment instance and
  matches its configured communication scheme. Rule-based, value-based, and
  policy-gradient methods interact through the same `reset()` and
  `step(actions)` API.
- Within movement step `t`, the `R` communication rounds run first over the
  frozen same-team graph, seeded by each agent's local observation `o_i(t)` and
  the message cache carried from `t-1`. Movement `a_i(t)` is then selected from
  the post-communication state, so this step's communication does affect this
  step's movement. This supersedes an earlier draft in which communication only
  became available at `t+1` and did not affect `a_i(t)`. The cache carried from
  `t-1` preserves the intent that movement depends on prior communication, while
  same-step rounds are what make learned communication differentiable end to end
  with the current action. See the Round And Cache Model below.
- A no-communication baseline receives an explicit empty communication result
  rather than using a different movement interface.
- Delivered messages persist in capacity-limited per-agent memory across
  movement steps instead of being cleared at each step boundary.
- Each stored message retains its observation/creation step. Its policy-visible
  age is `current_step - observation_step`, so age advances once per movement
  step without rewriting immutable message contents.
- Message memory is cleared on episode reset. Its capacity is distinct from
  communication bandwidth. When memory is full, the oldest observation is
  evicted first; equal-age messages are ordered by delivery sequence.
- Each movement step has a fixed number `R` of communication rounds.
- One communication round permits at most one physical graph hop.
- AEC ordering should not define baseline communication semantics.
- Asynchronous/event-driven communication is deferred.

Implications:

- The baseline communication graph is frozen within a movement step.
- Multi-hop relay must happen through explicit communication rounds, not through
  agent iteration order.
- Communication traces should record graph edges, rounds, sender, receiver,
  origin, and payload/protocol metadata.

### Round And Cache Model

Decision: communication is organized on two independent axes, and every named
scheme is a configuration of them.

- `R` is the number of synchronous communication rounds per movement step. It is
  justified physically by communication being much faster than movement, so many
  message exchanges occur between two movement updates. The same speed ratio
  justifies freezing the same-team graph within the step, which means the
  bearings fused within a step are effectively simultaneous. `R` is also an
  experimental control on propagation depth; for a learned GNN, `R` equals the
  number of message-passing layers.
- `C` is a message cache expressed as a sliding window of the last `C` rounds.
  The window spans movement-step boundaries, so a round may read messages emitted
  in previous steps. The cache is the protocol memory used for relaying,
  duplicate suppression, temporal filtering, and persistence. Its capacity is a
  round window (optionally bounded further by a per-round message cap); when full,
  the oldest round is evicted first.

The cache is distinct from a processor's intrinsic round-to-round state. The `R`
rounds always propagate hidden state from one round to the next through the
processor itself, and that happens even when `C = 0`. `C` governs only how much
past message history is retained and re-readable, not whether within-step
propagation occurs. The first learned scheme is the `C = 0` corner: three
GraphSAGE rounds with sum aggregation, no cache, no relay, differentiable in a
single transition.

Decision: relay correctness requires `C >= TTL`.

- One round permits at most one graph hop, so a time-to-live of `TTL` hops is a
  lifetime of `TTL` rounds; a copy of a message can still arrive up to `TTL`
  rounds after creation.
- The forwarding buffer only needs a message for about one round, but duplicate
  suppression must remember it for its whole lifetime. If `C < TTL`, a still-live
  message is evicted and then re-accepted as new, breaking duplicate suppression
  and allowing cycles. `C >= TTL` is therefore both necessary and sufficient for
  correct relay. Only the message identity must survive `TTL` rounds; retaining
  the full payload that long is an optional simplification.
- When `TTL > R` a relay necessarily continues across movement steps, which is
  exactly what the cross-step cache window enables. This resolves the earlier
  open question of whether relay TTL may continue across steps: it may, provided
  `C >= TTL`.
- The scheme compiler must reject a relay configuration with `C < TTL`.

`C` may exceed `TTL` when a scheme needs a longer window for temporal features;
`C >= TTL` is only the relay-correctness floor.

## PyG And Protocol Boundary

Decision: ACN owns communication semantics; PyTorch Geometric supplies reusable
graph-processing primitives where appropriate.

Decision: direct delivery, unchanged relay, and learned GNN communication use
one common synchronous slotted radius transport. They are not implemented as
separate network protocols.

The common transport:

- accepts zero or more frames from each agent in each round;
- delivers each frame over at most one valid graph edge per round;
- preserves transport metadata separately from the application payload;
- supports broadcast or addressed delivery;
- provides the extension point for later queues, bandwidth, loss, delay,
  fragmentation, and medium-access behavior.

Schemes differ through their source and receiver processing rules:

- direct communication stores a received frame and does not re-emit it;
- relay stores and re-emits first-seen frames unchanged subject to duplicate and
  hop-limit rules;
- GNN communication aggregates received payloads, updates communication state,
  and emits a newly encoded payload in the next round.

Only the common transport and direct processor are required initially. Relay
and learned processors are later additions over the same transport.

ACN owns:

- topology;
- round timing;
- inbox preservation;
- packet/protocol state;
- relay semantics;
- tracing;
- policy-visible versus privileged fields.

PyG may provide:

- aggregation modules;
- `MessagePassing` implementations;
- learned message/update layers;
- graph batching utilities;
- attention and GNN components.

Important boundary:

- Direct delivery must preserve distinct messages. Aggregation is optional and
  configured by the communication scheme.
- PyG should not become the protocol state machine.
- Packet-like relay still needs explicit protocol state such as origin,
  previous hop, TTL, duplicate suppression, and queues.

Named scheme direction:

- `none`: no communication.
- `one_hop_direct`: one-hop same-team message delivery with distinct inboxes.
- `one_hop_mean`: one-hop delivery followed by PyG-backed aggregation.
- `multihop_relay`: unchanged packet relay across fixed rounds.
- `multihop_gnn`: learned message/aggregate/update over fixed rounds. The first
  learned configuration is a three-layer GraphSAGE model with sum aggregation
  over the frozen same-team radius graph.

The first `multihop_gnn` configuration performs learned multihop propagation,
but no unchanged packet relay. It therefore has no relay queue, forwarding
cache, duplicate suppression, or packet TTL. Three GraphSAGE layers permit
information to influence nodes up to three graph hops away through successively
updated hidden states.

## First MARL Task

Decision: the first serious MARL benchmark supports single-team learning in
either direction, with exactly one trainable team per experiment.

Valid baseline modes:

- learned blue agents against scripted red agents;
- learned red agents against scripted blue agents.

These modes are mutually exclusive within one baseline training run. The
training runner binds the selected algorithm to a configured team rather than
assuming that the learner is always blue.

Goal:

- The selected team learns cooperatively under its policy-visible local sensing
  and configured communication scheme.
- The opposing team remains stationary at the policy level by using a scripted
  controller, making learning curves and communication ablations interpretable.
- Blue-learning experiments retain the limited-sensing tracking and
  communication benchmark already defined below.
- Red-learning experiments test scoring and evasion against existing scripted
  blue pursuit strategies, including the VAR-prediction pursuit blue, through the
  same algorithm interface.

Deferred:

- Simultaneously learning blue and red policies in the same run.
- Alternating, population-based, or league self-play.
- Federated/distributed training.
- Advanced learned communication before simple baselines exist.

## Scripted Opponent Curricula

Decision: each learning direction uses a scripted-opponent seed followed by a
curriculum, then evaluates against a fixed suite for that direction.

Blue-learning stages:

1. Train blue against one stable red setup first, likely fast avoidant reds.
1. Introduce randomized avoidant variants after the seed environment works.
1. Add scripted red mixtures once training, reward, trace, and communication
   mechanics are stable.

Red-learning stages:

1. Train red against one stable scripted blue pursuit setup first.
1. Introduce randomized parameters for that blue controller.
1. Add a fixed mixture of scripted blue strategies after the seed setup is
   stable.

Decision: the existing VAR-prediction pursuit blue is preserved as a first-class
scripted-blue controller and is the concrete seed opponent for red-learning. The
trainable benchmark must retain a non-gradient blue controller path, not only a
learned blue team. This keeps the repo's current default behavior available as a
baseline rather than discarding it during the pivot to trainable agents.

Implementation note: the current VAR blue consumes observed red positions over a
window of past timesteps (`src/utils/regressor.py`; `processing_capability` sets
the window) and pursues the averaged predicted position. The bearing-only sensor
decision removes direct red positions from the policy-visible observation, so
keeping VAR blue as a scripted opponent requires one of:

- granting the scripted controller privileged position access, which is permitted
  because a scripted opponent is not the object of study in the red-learning
  direction and preserves current behavior with minimal change; or
- fusing shared same-team bearings into triangulated position estimates before the
  VAR step, which keeps the baseline inside limited sensing and is the
  limited-sensing-faithful upgrade for later.

Either path implements the same non-gradient online-method interface already
reserved for VAR and other rule-based baselines in the training algorithm boundary.

Evaluation:

- Evaluate saved policies against the fixed scripted-opponent suite for their
  selected learning direction.
- The blue-learning suite includes at least the seed avoidant-red setup and
  held-out scripted red variants.
- The red-learning suite includes at least the seed scripted-blue pursuit setup
  and held-out scripted blue variants.
- Training curriculum and evaluation suite must be tracked separately.
- Exact maps, team sizes, scripts, seeds, and train/test manifests are selected
  and frozen when the benchmark is implemented rather than guessed in the
  architecture document.

Rationale:

- A single seed opponent makes reward and trainer failures easier to debug.
- Curriculum adds robustness without live self-play instability.
- Fixed evaluation prevents changing the test as training improves.

## Blue Reward Design

Decision: blue reward is hybrid. The primary objective is a shared team reward
for reliable triangulated tracking. Individual shaping is small and normalized.

Definitions for each step `t`:

```text
N_R = number of active red agents
k_r(t) = number of blue agents detecting red r
pinned_r(t) = 1[k_r(t) >= 3]
streak_r(t) = consecutive steps where pinned_r(t) is true
tracked_r(t) = 1[streak_r(t) >= 3]
```

Team components:

```text
pin_coverage =
  mean over active reds of pinned_r(t)

track_coverage =
  mean over active reds of tracked_r(t)

red_score_fraction =
  number of red agents that newly score at t / initial red count
```

Individual shaping for blue agent `i`:

```text
shape_i =
  (1 / N_R) * sum over active reds r:
    1[blue_i detects r]
    * 1[k_r(t) >= 3]
    * max(0, 4 - (k_r(t) - 1)) / 2
```

Initial reward formula:

```text
blue_reward_i =
  0.5 * pin_coverage
  + 1.0 * track_coverage
  + 0.1 * shape_i
  - score_penalty_weight * red_score_fraction
```

Normalization decisions:

- Normalize `pin_coverage` by active red count.
- Normalize `track_coverage` by active red count.
- Normalize `shape_i` by active red count.
- Normalize red scoring by the initial red count and apply the penalty only on
  the step where a red newly scores, not on every later step.
- Do not normalize the final weighted sum by total weight initially.
- The benchmark reward replaces the current passive blue bonus for every step
  in which red does not score; the passive bonus and event penalty must not both
  be active.

Rationale:

- The real desired behavior is reliable pinpointing: enough blue agents detect
  each red at the same time and maintain that coverage over time.
- `k_r(t) >= 3` represents triangulation.
- `streak_r(t) >= 3` rewards sustained tracking rather than momentary detection.
- Individual shaping helps credit assignment without replacing the team
  objective.
- The shaping term does not reward one- or two-blue detections because it is
  gated by `k_r(t) >= 3`.
- The undercoverage term rewards useful participation around 3 or 4 observers
  and stops rewarding over-covered reds.
- A red scoring event is directly adverse to the blue task and therefore gives
  every blue agent the same negative team component. This makes the task
  outcome adversarial without requiring every dense shaping term to be exactly
  zero-sum with a future learned red reward.
- The initial benchmark reward uses detection count and detection streak only;
  it does not add a localization-error reward.
- The numerical `score_penalty_weight` is tuned and frozen with the benchmark
  configuration after checking component scales.

### Dense Reward And Reward/Metric Decoupling

Decision: blue reward stays dense (per-step detection count and streak) for the
first benchmark. Reward faithfulness is supplied by evaluation metrics, not by
adding cost or noise to the reward signal.

Rationale and supporting decisions:

- Dense is the natural form for this task, not a shaping crutch. Reliable
  tracking is sustained custody, which is an integral of a per-step coverage
  quantity. A per-step reward therefore measures the objective directly rather
  than acting as a proxy for a sparse terminal goal.
- The count/streak reward is computationally cheap: it reduces over the
  blue-red detection state the environment already computes every step to build
  observations. Only an error- or geometry-based reward would add real per-step
  cost, because it would require evaluating the predictor on every step.
- Over-coverage is already de-incentivized: the shaping term is `1.0` at three
  detectors, `0.5` at four, and `0` at five or more, so it does not reward
  piling many blue agents onto one red. The streak requirement means momentary
  detection does not pay, so sustained custody is the intended behavior rather
  than a reward hack.
- The one un-closed Goodhart hole is geometry: three near-collinear detectors
  satisfy the count gate but triangulate poorly. The targeted fix, deferred
  until traces show collinear-but-pinned behavior, is to gate `pinned_r` on a
  minimum angular diversity of the detectors around the red instead of a raw
  count.

Decision: separate the reward signal from the evaluation metric.

- The reward uses the cheap, robust count/streak proxy.
- Faithfulness lives in evaluation: localization error, triangulation geometry,
  and prediction ADE/FDE with versus without communication are tracked as
  first-class metrics from the first benchmark.
- An error-based localization reward is the explicit v2 upgrade. It is adopted
  only if a trained policy games the count while the localization metric stays
  poor; that measured proxy/metric divergence is the trigger, not an up-front
  assumption. Folding error into the reward up front is avoided because it
  couples the reward to the estimator and injects noise that would muddy the
  no-communication versus engineered versus learned-communication ablation.

## Red Reward Design

This section applies when red is the trainable team (the scripted-blue plus
learned-red direction). Scripted red controllers such as avoidant red need no
reward gradient and are unaffected.

Current scoring semantics in code (`_calculate_reward`): a red receives `+1` on
every step where `|distance_to_center - REWARD_ATTRACTOR_DISTANCE| <=
REWARD_ATTRACTOR_TOLERANCE` (currently a ring at radius `50.0` with a `±1.0`
band) and it is not detected by any blue agent. Two consequences follow:

- The reward is already evasion-coupled: detection zeroes it, so red is rewarded
  only for occupying the ring while hidden. There is no separate evasion target
  to add.
- The raw reward is far too sparse for a learner: a two-unit ring band gated on
  being undetected gives a randomly-initialized red almost no signal. It is
  adequate for scripted red but not for learned red.

Decision: keep per-step dwell scoring; do not switch to a one-time breach event.

- Scoring here is occupying a ring, not reaching a goal, so dwell is the faithful
  form.
- It is symmetric with blue: blue integrates coverage per step, red integrates
  undetected ring-occupation per step.
- Degenerate camping is already prevented by the geometry: the target is a ring
  locus rather than a single safe point, and blue pursues, so a stationary red
  gets pinned. Red must keep repositioning to stay hidden, which is the sustained
  cat-and-mouse the benchmark is meant to elicit and to test blue tracking
  against. One-time scoring would collapse this into a single breach event.

Decision: learned red receives dense shaping, mirroring the blue reward
philosophy. This is required in practice because of the thin scoring band.

- Potential-based progress-to-ring: a small reward for reducing
  `|distance_to_center - REWARD_ATTRACTOR_DISTANCE|`. Potential-based so it does
  not move the optimum, only guides exploration onto the ring.
- Dense evasion penalty: a small per-step negative for being pinned/tracked, i.e.
  blue's `pinned_r`/`tracked_r` with a minus sign. It is cheap (same detection
  state the environment already computes) and directly opposes blue.
- Reward/metric decoupling as for blue: the reward uses cheap detection-based
  shaping, while evaluation measures true evasion quality (undetected-dwell time,
  score rate, distance at first detection).
- Both shaping terms stay small relative to the ring-score term so they guide
  without dominating the true objective.

Initial reward formula for a learned red agent `i`:

```text
red_reward_i =
    w_score * on_ring_undetected_i
  - w_track * tracked_i
  + w_prog  * potential_progress_to_ring_i
```

Decision: red reward is not literal zero-sum with blue.

- The benchmark pits learned red against a fixed scripted blue, not against a
  learned blue, so it is not self-play and blue's reward is not used for learning
  in red-mode. Red's reward is therefore a standalone design.
- Only outcome terms are naturally opposed: red scoring is adverse to blue, and
  red being tracked is adverse to red. Blue's internal coverage shaping
  (`1.0` at three detectors, `0.5` at four, `0` at five or more) is a blue
  credit-assignment device with no natural red meaning and must not be negated
  into red's reward.
- Outcome-level zero-sum coupling is reserved for the deferred simultaneous
  self-play phase, and even there only scoring/tracking outcomes are coupled, not
  internal shaping.

Deferred to the implementation PR: `w_score`, `w_track`, `w_prog` values, and
whether the `±1.0` scoring band is widened for the learned-red scenario, are
tuned and frozen with the benchmark configuration after checking component
scales.

## Training Algorithm Boundary

Decision: environment interaction and algorithm-specific learning are separated
through a common transition contract.

Per environment step:

1. Snapshot policy inputs and any training-only privileged state at time `t`.
1. Ask the selected method for environment actions.
1. Apply `env.step(actions)` once.
1. Snapshot next observations, communication memory, privileged state, rewards,
   and termination flags at `t + 1`.
1. Build one complete transition and pass it to the selected learning method.

The learning method decides how to use transitions:

- PPO/IPPO/MAPPO append them to an on-policy rollout buffer.
- Value-based and off-policy methods insert them into replay storage and sample
  according to their own update schedule.
- World models store and sample temporal sequences.
- VAR and other rule-based baselines may maintain method-specific history and
  perform no gradient update.

The common online method interface provides episode reset, action selection,
transition ingestion, update readiness, optimization, and checkpointing. It does
not standardize network architecture, storage type, loss function, or update
schedule across algorithm families.

Decision: the training stack is built so that training algorithms can be
benchmarked against each other. The method interface, the per-step transition
contract, and the discrete action space together form a fixed comparison
surface: every method family — on-policy policy gradient (PPO/IPPO/MAPPO),
value-based, model-based/world-model, and non-gradient baselines such as the
VAR pursuit controller — implements the same interface and consumes identical
observations, transitions, rewards, and actions. Adding a new algorithm must
not require changing the environment API, the benchmark scenarios, or the
reward definitions, so that measured differences between algorithms are
attributable to the algorithms rather than the harness. PPO is the only
implemented method at this stage; this constraint governs how every future
method is added.

Decision: actor parameter assignment and critic information scope are
independent configuration axes.

Required actor modes:

- `shared`: one actor and optimizer are used by all agents in a configured
  homogeneous group;
- `separate`: each configured agent has its own actor and optimizer;
- `grouped` may later map roles or capability classes to shared actors.

Required critic scopes:

- `local`: critic inputs contain only policy-visible local information, giving
  an IPPO-style configuration;
- `global`: a training-only critic may consume privileged simulator state,
  giving a MAPPO/CTDE-style configuration.

One PPO implementation should support shared/local, separate/local,
shared/global, and separate/global combinations. Shared weights never imply
shared observations, message memory, or recurrent state.

Decision: the first trainable benchmark configuration is MAPPO with a shared
actor for whichever team is selected as trainable and a privileged global
critic. Shared-actor IPPO is the principal local-critic control. Separate actors
remain a supported configuration rather than the initial benchmark default.

The baseline runner enforces an exclusive choice:

```text
trainable_team = blue  XOR  trainable_team = red
opponent_controller = scripted
```

Future checkpointed or learned opponent controllers can implement the same
controller boundary without changing the environment API.

Decision: TorchRL is the primary implementation backend for new trainable RL
algorithms.

- ACN retains the common online method interface and transition contract as the
  stable boundary exposed to the environment and experiment runner.
- PettingZoo remains the public multi-agent environment API. An adapter converts
  ACN observations and transitions into the TensorDict representation needed by
  TorchRL without changing the simulator's dict/list observation contract.
- TorchRL may provide model modules, losses, rollout and replay storage, return
  estimation, batching, and optimization utilities. ACN retains control of the
  environment step sequence, communication phase, actor assignment, critic
  scope, and algorithm configuration.
- Native scripted, VAR, and other non-gradient methods remain valid
  implementations of the same ACN interface and are not forced through
  TorchRL.
- The existing Stable-Baselines3 path may remain as a compatibility adapter,
  but it is not the foundation for new MARL implementations. RLlib is not an
  initial dependency.
- PyTorch Geometric may provide differentiable graph and aggregation operations
  inside a communicating actor. TorchRL computes the RL losses and optimizes the
  complete actor, including its PyG communication parameters, so communication
  and movement can be learned end to end. ACN separately defines transport
  semantics such as topology, delivery timing, TTL, storage, and eviction.

## Remote Training Infrastructure

Decision: training runs remotely on Modal; the laptop (no GPU) is only the
control surface. GCP via SkyPilot is the documented scale-out path, deferred
until run economics demand it.

Workload facts that shaped the choice:

- CTDE does not require distributed infrastructure. The centralized part of
  MAPPO is the critic's privileged input, not the placement of computation; one
  training run is one process, so the requirement is a single adequate machine
  per experiment, not a cluster.
- The bottleneck is CPU env-stepping (Python physics, per-agent observation
  building, communication rounds), not GPU math. Networks are small (MLP actors,
  three-layer GraphSAGE over tens of nodes, one central critic). The right
  machine is many CPU cores plus a modest GPU (T4/L4 class).

Why Modal over GCP-first:

- Manual setup is one browser authorization; there are no quota requests, IAM,
  images, or instance lifecycle to manage.
- Invocations ship the local working tree, including uncommitted changes, so
  experiment iteration does not require commit/push.
- Per-second billing with scale-to-zero fits hours-long, experiment-shaped runs.
- Raw GCP spot capacity is cheaper per hour; that matters only for multi-day
  runs. When that pressure exists, SkyPilot moves the same entrypoint onto
  GCP/AWS spot without code changes. GCP infrastructure must not be built
  before then.

Implementation (in-repo, `infra/`):

- `infra/modal_train.py` defines the Modal app: image from
  `pyproject.toml`/`uv.lock`, headless env vars, the persistent `acn-results`
  volume mounted over `results/` so `run.py` writes its normal run folders
  unchanged, a CPU `smoke` function running the preview scenario end to end,
  and a `train` function (T4 by default, `ACN_MODAL_GPU` overrides).
- Artifacts are pulled back with `modal volume get` and ingested locally
  through the existing recorder/ingest boundary; database infrastructure stays
  local, consistent with the storage-boundary rule.

Decision: the training entrypoint must be cloud-runnable from day one. The
TorchRL benchmark trainer is designed headless, fully config-driven, seeded,
and checkpoint/resume-capable, with artifacts written through the recorder
factory to a configurable directory. Remote execution is a property of the
entrypoint, not a retrofit.

## Development Sequence

Decision: implement and validate the benchmark before selecting the longer-term
research direction.

The initial benchmark establishes reproducible results for both single-team
learning directions. Blue communication experiments compare no communication,
engineered communication, and learned GraphSAGE under fixed scripted opponents.
Later work on self-play, world models, value-based methods, heterogeneous
agents, or federated/distributed optimization should be motivated by measured
benchmark limitations rather than preselected as part of the first
implementation.

## Implementation-PR Decisions

The following lower-level choices are documented but intentionally resolved in
the communication implementation PR, where their interaction with concrete
types, configuration, and tests can be evaluated:

- Whether teammate identity is a learned input feature or routing metadata only.
- Shared versus team-specific communication encoders.
- Stateless versus recurrent communication state.
- Future meaning and enforcement of `communication_bandwidth`.
- Asynchronous/event-driven communication fidelity.
- Exact benchmark scenario manifest, seeds, and final reward component weights.
- Red-learning reward shaping beyond the existing scoring reward.

### Resolved By The multihop_gnn Implementation

The learned-scheme implementation fixed the following of the deferred
choices for the first `multihop_gnn` configuration:

- Round weights: separate GraphSAGE weights per round are the default; a
  single weight set shared (recurrent) across all `R` rounds is selectable
  with `processor.shared_weights: true`. The round count — and therefore the
  hop limit — is unchanged by weight sharing.
- Communication encoders: the trainable team uses ONE communication encoder
  inside its shared actor (parameter sharing extends to the communicator).
  Team-specific encoders arise trivially because exactly one team trains per
  run; a shared-across-teams encoder question only exists for the deferred
  self-play mode.
- Stateless versus recurrent communication state: the first learned scheme is
  stateless across movement steps by definition (`C = 0`; hidden state exists
  only within the step's `R` rounds). Recurrent cross-step communication
  state remains a future scheme.
- Execution ownership: differentiable communication runs in the
  policy/trainer forward path using the compiled plan's own topology and
  round definitions; the environment compiles and validates the plan but
  never executes it (its processor slot is an execution-refusing marker).
  Rollouts store raw node features and the frozen edge_index so PPO updates
  recompute the communication forward pass; detached embeddings are never
  the stored training input.
