# Communication Module Implementation Plan

## Status

This document defines the planned communication architecture for ACN. It is an
implementation plan, not a description of the current runtime. At present,
`src.communication.models` contains placeholders and neither PettingZoo
environment executes communication rounds.

The plan is based on the following agreed design decisions:

- The full decision record is maintained in
  [Communication And MARL Decision Log](communication_decision_log.md).
- Learning-facing observations remain local dictionaries/lists of detections.
- Simulator identities remain available for tracing and evaluation.
- Teammates may have policy-visible identities; opponent identity exposure is
  configurable and is not required for ground-truth evaluation.
- Direct communication is limited to same-team agents within a communication
  radius.
- Communication radius is resolved per agent, with an agent or agent-spec value
  overriding its team default.
- Messages are numerical tensors initially. Structured text and LLM-generated
  messages are future payload types.
- Communication and movement actions are separate.
- Communication has no reward cost in the initial implementation.
- Agents know and may communicate their own global positions.
- Initial blue sensing is bearing-only: each local observation contains a
  variable-size collection of anonymous red bearings, without red position,
  range, velocity, or confidence.
- Delivered messages persist across movement steps in capacity-limited
  per-agent memory and expose age derived from their observation step.
- Each movement step contains a fixed number `R` of synchronous communication
  rounds.
- One communication round permits at most one physical graph hop.
- ACN owns communication semantics; PyTorch Geometric supplies reusable graph
  processing and aggregation implementations where appropriate.

## Objectives

The communication module must support a progression from simple, inspectable
baselines to learned and protocol-aware communication without changing the
environment's fundamental communication contract.

The initial architecture must support:

1. No communication.
1. One-hop unchanged vector delivery.
1. Multi-hop unchanged relay.
1. Rule-based aggregation and update functions.
1. Learnable message, aggregation, and update functions.
1. Layered GNN communication using PyG.
1. Explicit packet and protocol simulation.
1. Differentiable and non-differentiable communication training modes.
1. Deterministic tracing of graph edges, packets, messages, and derived states.
1. Future asynchronous, federated, distributed, and text-message extensions.

## Non-Goals For The First Release

The first communication release will not implement:

- continuous-time or event-driven asynchronous networking;
- radio propagation, interference, or medium-access control;
- finite bandwidth, communication reward penalties, or energy costs;
- text or LLM payloads;
- federated parameter optimization;
- learned routing;
- cross-team communication;
- a complete catalogue of PyG convolution layers.

These features must remain possible without redesigning the core interfaces.

## Architectural Principles

### Separate Delivery From Processing

Communication delivery answers:

```text
Which payloads arrive at which agent, from whom, during which round?
```

Communication processing answers:

```text
How does an agent transform its local state and delivered payloads into a new
communication state?
```

Delivery must preserve individual messages. Aggregation is a configurable
processing decision, not an implicit property of the channel.

### Keep Simulation Semantics Outside PyG

ACN defines:

- active agents and teams;
- communication radius and valid directed edges;
- graph snapshot timing;
- number and ordering of communication rounds;
- packet lifetime and protocol state;
- message delivery, loss, queues, relays, and traces;
- which information is policy-visible or privileged.

PyG may implement:

- edge-message construction;
- standard and custom aggregation;
- learned node updates;
- attention and graph convolutions;
- batching utilities;
- differentiable multi-round processing.

No experiment's physical communication meaning may depend on an undocumented
property of a particular PyG layer.

### Preserve Decentralized Information Boundaries

An agent's communication processor may consume only:

- its local observation;
- its own recurrent or protocol state;
- messages delivered to it by the communication runtime;
- public configuration such as the round number.

Privileged simulator identity and global state may be logged or used by a
centralized critic, but must not silently enter decentralized actor inputs.

### Make Communication Rounds Explicit

The baseline runtime is synchronous and double-buffered. During round `r`, all
agents read state from round `r` and produce outputs for round `r + 1`.

This prevents PettingZoo agent iteration order from changing information flow.
AEC execution must not be used to emulate multi-hop communication in the
baseline implementation.

## Runtime Model

For transition `t -> t + 1`, the parallel environment will execute:

```text
1. Present each agent with local observation o_i(t), the message cache C_i
   carried from the previous step, and optional recurrent state h_i(t).
2. Freeze agent positions and build the same-team radius graph G_t.
3. Run R synchronous communication rounds over G_t. In each round agents emit
   source messages (from o_i(t), the current round state, and the C-round cache
   window), messages propagate at most one hop, and each agent updates its round
   state and cache. For a learned processor this is the R-layer message-passing
   forward pass; for relay it is the packet state transition below.
4. Select movement a_i(t) for every agent from its post-communication state, so
   this step's communication affects this step's movement.
5. Apply all movement actions a_i(t) simultaneously and advance physics once,
   producing environment state S(t+1).
6. Age the cache and drop rounds outside the C-round window (oldest round first);
   the retained window carries to t+1.
7. Compute rewards, terminations, traces, and local observations o_i(t+1).
```

The communication graph is frozen across all `R` rounds for the baseline. This
means an agent cannot move into communication range partway through the same
movement step. Dynamic per-round graph rebuilding is a later configurable
extension.

### Round Semantics

For learned or aggregating communication, a round may use:

```text
m_ji^r = message(h_j^r, h_i^r, edge_ji, round_context)
q_i^r  = aggregate({m_ji^r | j -> i})
h_i^(r+1) = update(h_i^r, q_i^r, local_context_i)
```

For unchanged relay, a round instead performs packet state transitions:

```text
outgoing queue -> valid next-hop transmissions -> receiver inbox
                -> duplicate/TTL handling -> next-round relay queue
```

Both forms use the same frozen topology and fixed round scheduler.

## Planned Package Structure

```text
src/communication/
├── __init__.py
├── config.py          # Typed config parsing and validation
├── types.py           # Messages, packets, graph snapshots, results
├── topology.py        # Radius graph and future topology builders
├── runtime.py         # Fixed-round scheduler and orchestration
├── registry.py        # Named scheme registry and compiler
├── plans.py           # CommunicationPlan and component composition
├── sources.py         # Observation-derived and explicit message sources
├── inbox.py           # Distinct-message grouping and inbox views
├── transport.py       # Common synchronous slotted radius transport
├── packets.py         # Frame metadata and fragmentation primitives
├── processors/
│   ├── __init__.py
│   ├── base.py        # CommunicationProcessor interface
│   ├── identity.py    # No-op and unchanged-message processors
│   ├── relay.py       # First-seen unchanged forwarding behavior
│   ├── torch.py       # Native Torch message/update modules
│   └── pyg.py         # PyG MessagePassing/Aggregation adapters
└── tracing.py         # Communication-specific trace records
```

The existing `models.py` classes will be deprecated after compatibility
adapters exist. They should not remain the central abstraction because they mix
partner selection and message processing without defining timing or transport.

## Core Data Types

All runtime types should use dataclasses or typed containers. Tensor fields
must support CPU and accelerator devices without hidden NumPy conversion.

### CommunicationGraph

```python
@dataclass(frozen=True)
class CommunicationGraph:
    agent_ids: tuple[str, ...]
    edge_index: torch.LongTensor  # shape [2, E], source -> receiver
    edge_distance: torch.Tensor  # shape [E]
    edge_features: torch.Tensor | None  # shape [E, F_e]
    team_index: torch.LongTensor  # shape [N]
    step: int
```

Requirements:

- Only same-team radius-valid edges appear.
- Self-edges are configurable and disabled by default for transport.
- Edge ordering is deterministic for a fixed state and seed.
- Global agent IDs are runtime metadata, not automatically model features.
- Empty and disconnected graphs are valid.

### EdgeMessageBatch

```python
@dataclass
class EdgeMessageBatch:
    payload: torch.Tensor  # shape [M, D] or structured payload
    sender_index: torch.LongTensor  # immediate sender
    receiver_index: torch.LongTensor
    origin_index: torch.LongTensor  # original source
    message_id: torch.LongTensor
    round_index: torch.LongTensor
    metadata: dict[str, torch.Tensor]
```

This is the core delivery representation. Messages remain distinct until a
processor explicitly aggregates them.

### InboxBatch

`InboxBatch` provides receiver-oriented views without changing the underlying
message identity:

```python
@dataclass
class InboxBatch:
    messages: EdgeMessageBatch
    receiver_ptr: torch.LongTensor | None
```

It must support:

- sparse iteration for protocol and debugging code;
- dense payload plus mask conversion for policies requiring fixed tensors;
- PyG aggregation using `receiver_index`;
- zero-message receivers;
- sender and origin metadata access.

The variable-size inbox is the canonical delivery result. Fixed-size conversion
is performed by the receiving policy or its model adapter, not by the transport
layer. Different policies may therefore use pooling, attention, permutation-
invariant set encoders, or dense payload-and-mask conversion without changing
communication semantics.

### Persistent Message Memory

Each agent owns a capacity-limited message memory that survives movement-step
boundaries. Newly delivered messages are added without changing their payloads.
Every message stores its observation or creation step, and consumers derive
`age = current_step - observation_step` when reading it. Episode reset clears
all message memory.

Memory capacity is separate from transmission bandwidth and is configured as a
scenario or agent parameter. When memory is full, evict the message with the
oldest observation step first; break equal-age ties by delivery sequence.
Round-local deliveries must remain distinguishable from older stored messages
so recurrent or relay processors do not accidentally treat the full history as
newly received traffic.

### PacketBatch

Protocol simulation extends edge messages with persistent transport state:

```python
@dataclass
class PacketBatch:
    payload: torch.Tensor
    packet_id: torch.LongTensor
    message_id: torch.LongTensor
    origin_index: torch.LongTensor
    previous_hop_index: torch.LongTensor
    destination_index: torch.LongTensor  # sentinel for broadcast
    sequence_number: torch.LongTensor
    ttl: torch.LongTensor
    fragment_index: torch.LongTensor
    fragment_count: torch.LongTensor
    created_step: torch.LongTensor
    created_round: torch.LongTensor
```

Payload tensors and protocol metadata must remain separate. IDs and TTL values
must not enter learned payload features unless a scheme explicitly selects them.

### CommunicationResult

```python
@dataclass
class CommunicationResult:
    agent_output: dict[str, object]
    final_state: torch.Tensor | None
    delivered_messages: EdgeMessageBatch
    protocol_state: object | None
    diagnostics: dict[str, object]
```

`agent_output` may contain a tensor embedding, a preserved inbox, structured
reports, or another policy-facing object depending on the scheme.

## Topology Layer

### RadiusTopology

The first topology builder creates directed edges between same-team agents when:

```text
distance(sender, receiver) <= receiver/sender communication-radius rule
```

The radius ownership rule must be configurable:

- `mutual`: both agents' ranges must cover the link;
- `sender`: sender transmission radius controls the edge;
- `receiver`: receiver range controls the edge;
- `minimum`: use the minimum of both configured radii.

The initial default should be `sender`, because it directly represents a
transmission range. Symmetric configurations still produce bidirectional edges.

The runtime stores an effective radius on every agent. Configuration resolution
uses this precedence:

```text
agent or agent-spec communication_radius
team default communication_radius
environment fallback communication_radius
```

Team defaults are expanded onto agents during construction, keeping topology
code independent of configuration inheritance.

### Topology API

```python
class CommunicationTopology(Protocol):
    def build(self, agents, step: int) -> CommunicationGraph: ...
```

Future implementations may include line-of-sight, obstacle attenuation,
directed antenna range, fixed infrastructure nodes, or externally supplied
graphs.

## Processing Layer

### CommunicationProcessor

```python
class CommunicationProcessor(nn.Module):
    def initialize(self, local_observations, graph, context): ...

    def process_round(self, state, inbox, graph, round_index, context): ...

    def finalize(self, state, inbox, graph, context): ...
```

Rule-based processors may subclass `nn.Module` without trainable parameters.
This keeps one calling convention and permits device-aware tensor execution.

### Inbox-Preserving Processing

`one_hop_direct` must preserve one payload per incoming edge. Its processor must
not sum or average messages. It may return an `InboxBatch` or dense payload and
mask pair to the movement policy.

For the engineered bearing scheme, every anonymous contact report is delivered
unchanged. A receiver performs any size-consistent encoding after delivery.
Aggregation performed by a learned or explicitly aggregated scheme remains part
of that scheme's processor rather than the transport layer.

The implementation may reuse PyG utilities such as sparse indexing and
`to_dense_batch`, but must not call an aggregation that loses message identity.

### PyG Aggregation Processing

Where aggregation is required, use PyG's public aggregation modules directly:

```python
aggregated = aggregation(
    messages.payload,
    index=messages.receiver_index,
    dim_size=num_agents,
)
```

Supported initial names:

- `sum`
- `mean`
- `max`
- `min`
- `softmax`
- `attention`
- `multi`
- `deepsets`
- `set_transformer`
- `custom`

Only a tested subset needs to be enabled in the first release. Registry entries
must validate required dimensions and parameters.

### Full PyG Message Passing

Learned multihop schemes may subclass PyG `MessagePassing` or compose existing
convolution layers. One configured processing layer corresponds to one ACN
communication round unless the scheme explicitly documents another meaning.

The adapter must expose:

- encoder module;
- message module or convolution;
- aggregation module;
- update module;
- recurrent/shared weights across rounds option;
- separate weights per round option;
- residual and normalization options.

PyG internals must not be monkey-patched. Use public composition, subclassing,
or a small vendored adaptation with provenance when public APIs are inadequate.

## Common Transport Protocol

All initial communication schemes use one synchronous slotted radius transport.
It moves opaque frames over valid one-hop edges and does not prescribe how an
agent interprets or transforms the payload.

### Transport Interface

```python
class RoundTransport(Protocol):
    def reset(self, graph, context): ...

    def transmit_round(
        self,
        outboxes,
        graph,
        transport_state,
        round_index,
        context,
    ) -> DeliveredFrameBatch: ...
```

Each frame keeps transport metadata separate from its payload:

```text
message ID
origin agent
immediate sender
creation step and round
hop count and optional hop limit
destination or broadcast marker
opaque payload
```

### Direct Processing

Initial semantics:

- Each sender creates zero or more application contact reports per step, one for
  each locally detected opponent.
- The message is copied to every outgoing communication edge.
- Delivery occurs within the current communication round.
- No loss, queue limit, fragmentation, retransmission, or cost is applied.
- The receiving processor stores the frame and does not re-emit it.

### Unchanged Relay Processing

Relay later reuses the same transport with this processor behavior:

- Each packet has an origin, unique packet ID, previous hop, and TTL.
- A receiver records the packet ID in a duplicate cache.
- First-seen packets may be delivered to the local application and queued for
  forwarding in the next communication round.
- A relay sends to valid neighbours other than the previous hop.
- TTL is decremented once per hop; packets with exhausted TTL are dropped.
- Duplicate packets are dropped before application delivery and forwarding.

### Learned GNN Processing

Learned multihop communication also reuses the same transport. After each
round, the processor aggregates delivered payloads, updates local communication
state, and emits a newly encoded payload for the next round. The fixed round
count limits information depth; packet TTL is not required for transformed GNN
state.

### Fragmentation And Reassembly

Fragmentation is deferred until the common transport and unchanged-relay
processor are stable. The design must support:

- configurable maximum payload units per packet;
- deterministic fragment numbering;
- receiver reassembly keyed by origin, message, and sequence number;
- incomplete-message expiry;
- trace events for fragment creation and reassembly.

### Future Protocol Features

Later phases may add:

- unicast routing tables;
- acknowledgements and retransmission;
- finite queues and scheduling;
- delivery delay and jitter;
- packet loss and corruption;
- data-rate limits;
- TDMA/CSMA-style medium access;
- learned routing and scheduling decisions;
- network coding or message compression.

Communication remaining "free" means no reward penalty. It does not prevent
later protocol experiments from applying physical capacity constraints.

## Scheme Registry And Compiler

Users should select named communication schemes instead of manually assembling
every component for standard experiments.

### CommunicationPlan

```python
@dataclass
class CommunicationPlan:
    topology: CommunicationTopology
    transport: RoundTransport
    processor: CommunicationProcessor
    rounds: int
    graph_update: str
    differentiable: bool
    output_contract: str
```

### Registry API

```python
@register_communication_scheme("one_hop_direct")
def build_one_hop_direct(config) -> CommunicationPlan: ...


def create_communication_plan(config) -> CommunicationPlan: ...
```

The compiler must reject logically inconsistent combinations, including:

- `rounds < 1` for an enabled scheme;
- payload dimension mismatch;
- inbox-preserving output passed to a policy expecting one embedding;
- learned differentiable processing with forced NumPy conversion;
- a relay scheme whose `TTL` exceeds its cache window `C` (correct duplicate
  suppression requires `C >= TTL`; `TTL > R` is allowed because the cache window
  spans movement steps);
- protocol metadata exposed as model features without explicit selection;
- dynamic topology requested by an unsupported runtime.

### Initial Named Schemes

#### `none`

```text
topology: none
transport: none
processor: none
rounds: 0
output: empty communication result
```

#### `one_hop_direct`

```text
topology: same-team radius
transport: slotted radius broadcast
processor: preserve inbox
rounds: 1
output: distinct incoming payloads plus sender identity
```

#### `one_hop_mean`

```text
topology: same-team radius
transport: slotted radius broadcast
processor: PyG mean aggregation
rounds: 1
output: one aggregated vector per agent
```

#### `multihop_relay`

```text
topology: same-team radius
transport: slotted radius broadcast
processor: first-seen unchanged forwarding with duplicate suppression and TTL
rounds: R
output: distinct origin-preserving payloads
```

#### `multihop_gnn`

```text
topology: same-team radius
transport: slotted radius broadcast
processor: three-layer PyG GraphSAGE with sum aggregation
rounds: 3
output: one learned communication embedding per agent
```

This first learned scheme propagates updated hidden states for three graph hops,
but does not forward unchanged packets. It does not use relay queues, duplicate
suppression, forwarding caches, or packet TTL. Other GNN processors remain
configuration-driven extensions after this reference model is working.

#### `hybrid_relay_gnn`

Deferred scheme combining packet-preserving relay with learned payload encoding,
selection, transformation, or final inbox aggregation.

## Scheme-Specific Input Encoding

The raw variable-size sensor reports and delivered inbox remain the simulator
contract. Their policy-facing encoding belongs to the selected communication
scheme.

Examples include:

- preserving every report for a set encoder or attention module;
- mean aggregation;
- standard-deviation or combined mean-and-standard-deviation summaries;
- masked dense conversion;
- learned graph or permutation-invariant encoders.

Every compiled `CommunicationPlan` must declare the shape and semantics of its
policy-facing output. This keeps the movement actor interface explicit while
allowing representation choices to vary between communication experiments.

## Payload Generation

Communication and movement actions remain separate. The communication module
must support multiple payload sources:

### Engineered Payload

A deterministic encoder produces fields such as:

```text
metadata:
    origin teammate identity
    observation timestep

payload:
    observer global x
    observer global y
    global bearing direction x = cos(theta)
    global bearing direction y = sin(theta)
```

Each visible opponent produces one anonymous contact report. Multiple detections
therefore produce a variable-size collection of reports without padding to a
global opponent count. A report contains no opponent identity, range, opponent
velocity, confidence, or sender velocity. The simulator may attach privileged
opponent identity outside the policy-visible message for evaluation. This is
the first inspectable communication baseline.

### Learned Differentiable Payload

A local `nn.Module` encodes the local observation into a message vector. The
encoder, graph processor, and action policy may be trained end to end.

### Communication Action Payload

A policy explicitly samples or emits a message vector as a communication action.
The environment transports the vector. Non-differentiable protocol choices can
then be trained with policy gradients.

### Future Structured Payload

Structured text or LLM-generated messages require a payload codec and explicit
serialization/size semantics. The transport layer should treat them as opaque
payloads after encoding.

## Training Integration

The architecture must not assume CTDE, but it must support it.

### Differentiable Shared Communication

For a batched team policy:

```text
local observations -> local encoders -> R PyG rounds -> actor outputs
```

The complete communication computation runs inside the actor's forward pass.
Autograd trains encoders, message functions, aggregators, updates, and action
heads together.

Rollout storage must retain enough information to reconstruct the communication
forward pass during PPO updates:

- local observations;
- communication graph or deterministic graph-building inputs;
- recurrent communication state, when used;
- sampled communication actions, when applicable;
- log probabilities for stochastic message actions;
- masks for inactive agents and empty inboxes.

Do not store detached final communication embeddings as the only training input,
because this prevents updated models from recomputing messages during learning.

### Communication As An RL Action

For non-differentiable channels or independently acting agents:

```text
policy -> sampled communication action -> ACN protocol runtime -> receiver input
```

This supports packet loss, discrete routing, queue choices, or local policies
without cross-agent autograd. Credit assignment is harder and requires explicit
rollout fields and policy-gradient handling.

### Distributed And Federated Training

The runtime must avoid assumptions that require one global optimizer. Agent
modules may later have separate parameter copies and optimizers. Federated or
distributed learners may exchange parameters or gradients independently of the
communication payload channel.

This is a future trainer concern. It does not change the topology, protocol, or
round interfaces.

### Centralized Critics

A centralized critic may consume privileged state and communication diagnostics
during training. The actor must still consume only policy-visible local inputs
and delivered messages. Critic-only fields must be marked explicitly in rollout
storage.

The initial trainable benchmark uses MAPPO with one shared actor for the
selected trainable team and a privileged global critic. Shared-actor IPPO is
the local-critic control, and separate actors remain supported through
configuration.

### Trainable Team Selection

The baseline runner supports exactly one trainable team per run:

```yaml
training:
  trainable_team: "blue"  # blue | red
  opponent:
    type: "scripted"
    strategy: "configured_strategy_name"
```

The two valid modes are learned blue against scripted red and learned red
against scripted blue. Algorithm construction, transition collection, storage,
optimization, and checkpointing use the same team-parameterized interfaces in
both modes. Team-specific reward functions, observation adapters,
communication plans, and scripted strategies are selected through
configuration.

The runner must reject a baseline configuration that marks both teams as
trainable. Simultaneous or alternating self-play is a later training mode, not
an implicit consequence of making both teams compatible with learned
controllers.

## Environment Integration

### Parallel Environment

`ParallelGameEnv` is the primary communication runtime. Communication memory is
initialized in `reset()`. The public API remains the standard PettingZoo
`reset()` and `step(actions)` contract. The configured communication plan
determines how source messages are obtained without adding public communication
substeps or a second `step` argument.

The intended step flow is:

```python
observations, infos = env.reset()

while env.agents:
    actions = decision_method(observations)
    observations, rewards, terminations, truncations, infos = env.step(actions)
```

Inside `step(actions)`, ACN executes:

```python
movement_actions = movement_action_adapter.extract(actions)
source_outboxes = communication_plan.source.create_outboxes(
    observations=current_local_observations,
    actions=actions,
)
next_message_memory = communication_runtime.run(
    source_outboxes=source_outboxes,
    current_memory=current_message_memory,
    graph=current_frozen_graph,
)
next_physical_state = apply_movement_and_physics(movement_actions)
```

For `one_hop_direct`, `source.create_outboxes` is the deterministic
bearing-report encoder and consumes only current local observations; the public
action remains the existing movement action. For a learned explicit-message
scheme, the per-agent action space is a `Dict` with separate `movement` and
`message` fields, and the source reads the message field. Relay protocols add
forwarding traffic from packet state after source-message creation.

The action space is selected when the environment is constructed and remains
stable for that environment instance. External rule-based and learned methods
only need to return values accepted by the declared action space. Wrappers may
flatten or batch those spaces for a particular RL library without changing the
environment transition semantics.

### AEC Environment

AEC must initially use the same synchronous communication runtime once per full
agent cycle, or explicitly report communication as unsupported. It must not
deliver messages immediately after each individual agent acts.

True asynchronous networking should later use a discrete-event runtime rather
than relying on AEC selection order.

## Observation Integration

Communication output is distinct from direct sensing:

```python
@dataclass
class AgentDecisionInput:
    observation: LocalObservation
    communication: CommunicationView
    internal_state: object | None = None
```

Every learned, value-based, planning, or rule-based movement method receives an
`AgentDecisionInput` at the start of the transition. The communication view may
contain raw message memory, a deterministic processed result, or a learned
embedding produced by earlier transitions. A no-communication method receives
an explicit empty view, keeping the movement interface identical across
ablations.

Direct observations retain lists/dictionaries of locally visible teammates and
opponents. Received messages must not be merged into direct detections without
provenance.

Each received item should preserve at least:

- immediate sender identity;
- origin identity when relay semantics expose it;
- creation and delivery time/round;
- payload;
- optional protocol metadata selected for policy visibility.

Ground-truth opponent IDs remain available in traces even when hidden from the
policy.

## Configuration Schema

Proposed baseline configuration:

```yaml
environment:
  communication:
    enabled: true
    scheme: "one_hop_direct"
    rounds_per_step: 1

    topology:
      type: "radius"
      radius_rule: "sender"
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

Learned GNN example:

```yaml
environment:
  communication:
    enabled: true
    scheme: "multihop_gnn"
    rounds_per_step: 3

    topology:
      type: "radius"
      radius_rule: "sender"
      freeze_within_step: true

    payload:
      type: "learned"
      dimension: 32

    processor:
      backend: "pyg"
      model: "graph_sage"
      layers: 3
      aggregation: "sum"
      hidden_dimension: 64
      packet_relay: false

    transport:
      type: "slotted_radius"
      delivery: "broadcast"
      free: true
```

Relay example:

```yaml
environment:
  communication:
    enabled: true
    scheme: "multihop_relay"
    rounds_per_step: 3

    topology:
      type: "radius"
      radius_rule: "sender"
      freeze_within_step: true

    payload:
      type: "engineered_vector"
      dimension: 4

    processor:
      backend: "identity"
      aggregation: "none"
      forwarding: "first_seen_unchanged"
      ttl: 3
      duplicate_suppression: true

    transport:
      type: "slotted_radius"
      delivery: "broadcast"
      free: true
```

Per-agent communication radius should be added as an explicit agent setting,
with team defaults resolved by the agent factory.
The existing `communication_bandwidth` remains metadata until a later phase
defines its unit and enforcement semantics.

## Tracing And Evaluation

Communication must be observable as an experimental variable.

### Trace Records

Add communication records containing:

- step and communication round;
- graph edges and distances;
- message ID and packet ID;
- origin, immediate sender, and receiver;
- payload summary or full payload according to trace policy;
- protocol decision and drop reason;
- TTL, sequence, and fragment metadata;
- aggregation name and processor output summary;
- policy-visible versus privileged fields;
- communication scheme and configuration hash.

Large tensor payloads should support configurable sampling or separate artifact
storage rather than bloating transition JSONL records.

### Metrics

Initial metrics:

- messages generated, sent, delivered, relayed, duplicated, and dropped;
- graph degree and connectivity statistics;
- successful source-to-destination reachability by round;
- communication latency measured in rounds;
- number of unique origins reaching each agent;
- payload volume, even while communication has no reward cost;
- task performance difference versus no communication;
- prediction ADE/FDE with and without communication;
- runtime throughput and communication processing time.

Learned communication metrics may later include entropy, attention weights,
message ablation sensitivity, intervention tests, and representation similarity.

## Testing Strategy

### Unit Tests

Add focused tests under `tests/test_communication_*.py`.

Topology tests:

- same-team edges only;
- exact boundary inclusion;
- asymmetric radius rules;
- deterministic edge ordering;
- no edges and disconnected components;
- inactive agents excluded.

Inbox tests:

- one distinct message per valid edge;
- correct sender, receiver, and origin metadata;
- empty inbox handling;
- dense conversion masks;
- permutation-independent grouping.

Aggregation tests:

- PyG sum/mean/max match hand-computed values;
- custom aggregation receives correct receiver indices;
- no aggregation preserves all messages;
- gradients reach sender encoder, aggregator, update, and actor;
- empty-neighbour gradients and output defaults are valid.

Protocol tests:

- one-hop direct does not relay;
- relay advances exactly one edge per round;
- TTL is decremented once per hop;
- duplicate suppression works on cycles;
- origin and previous-hop identities remain distinct;
- fragmentation/reassembly once implemented;
- no agent-order dependence.

Registry tests:

- all named schemes compile;
- invalid component combinations fail with actionable errors;
- config round-trip is deterministic;
- unknown schemes and aggregators fail early.

### Integration Tests

- A-B-C chain: `one_hop_direct` reaches only B after one round.
- A-B-C chain: `multihop_relay` reaches C only after round two.
- A-B-C chain: two-layer GNN permits A information to affect C.
- Disconnected graph prevents cross-component information flow.
- Parallel simulation produces identical communication traces under stable seed.
- AEC compatibility does not depend on agent iteration order.
- Learning-facing inputs contain no hidden opponent IDs in hidden-identity mode.
- Ground-truth traces still permit per-opponent prediction evaluation.

### Training Tests

- Tiny differentiable graph verifies end-to-end gradient flow.
- PPO rollout can reconstruct graph communication during update.
- Parameter-shared actors produce per-agent actions using local inputs only.
- Communication-action log probabilities are stored when that mode is enabled.
- No-communication and direct-communication baselines can overfit a toy task.

### Performance Tests

Benchmark:

- agent count;
- edge count;
- payload dimension;
- communication rounds;
- preserved inbox versus aggregation;
- CPU versus accelerator execution;
- PyG versus simple native Torch operations for baseline aggregations.

Performance optimization should follow measurement. Correct semantics and trace
reproducibility take priority in the first release.

## Delivery Phases

### Phase 0: Contract And Test Fixtures

Deliverables:

- typed config and core data types;
- A-B-C chain and disconnected-team fixtures;
- communication plan registry skeleton;
- `none` plan;
- documented step and round semantics.

Acceptance criteria:

- data types support empty tensors and device movement;
- fixed graph fixtures produce deterministic edge indices;
- no existing simulations change when communication is disabled.

### Phase 1: Radius Graph And One-Hop Direct Delivery

Deliverables:

- `RadiusTopology`;
- fixed-round runtime;
- common slotted radius transport;
- direct store-without-forwarding processor;
- distinct-message inboxes;
- engineered vector payload provider;
- `one_hop_direct` registry entry;
- parallel environment integration for scripted policies;
- graph and delivery traces.

Acceptance criteria:

- direct neighbours receive distinct payloads;
- non-neighbours receive nothing;
- no opponent receives a teammate message;
- no second-hop delivery occurs;
- movement and communication use the same pre-transition state;
- newly delivered messages appear in the next decision input.

### Phase 2: PyG Aggregations And Rule-Based Processing

Deliverables:

- PyG aggregation adapter;
- sum, mean, max, and configurable custom aggregation;
- identity and simple update modules;
- `one_hop_mean` and related plans;
- aggregation trace summaries.

Acceptance criteria:

- PyG outputs match hand calculations;
- inbox-preserving and aggregating schemes share the same delivery runtime;
- processor configuration is validated at startup.

### Phase 3: Multi-Hop Unchanged Relay

Deliverables:

- packet batch and forwarding state;
- first-seen unchanged-relay processor over the common transport;
- TTL and duplicate suppression;
- origin and immediate-sender distinction;
- `multihop_relay` plan;
- protocol traces and reachability metrics.

Acceptance criteria:

- packets travel no more than one graph edge per round;
- cycles do not cause unbounded duplicate delivery;
- fixed `R` caps within-step reachability;
- unchanged payload equality is verified end to end.

### Phase 4: Learned PyG Communication

Deliverables:

- local learnable encoder;
- three-layer PyG GraphSAGE processor with sum aggregation;
- no packet-relay state in the learned reference scheme;
- `multihop_gnn` plan;
- movement actor integration;
- differentiable rollout reconstruction;
- gradient-flow tests.

Acceptance criteria:

- encoder, GNN, and actor train jointly;
- actors receive no privileged information;
- three processing layers correspond to three configured rounds;
- information can propagate at most three graph hops;
- no unchanged packet forwarding occurs;
- no graph or tensor is silently detached during training.

### Phase 5: Communication As An Explicit RL Action

Deliverables:

- separate communication action specification;
- stochastic message heads and log-probability storage;
- protocol transport of sampled messages;
- support for non-differentiable channel decisions;
- communication-action evaluation and ablations.

Acceptance criteria:

- movement and communication actions have separate policy heads/contracts;
- protocol operations can remain non-differentiable;
- policy-gradient updates train message production.

### Phase 6: Protocol Realism

Deliverables selected incrementally:

- fragmentation and reassembly;
- queues and scheduling;
- loss, delay, acknowledgements, and retransmission;
- bandwidth semantics and capacity enforcement;
- optional communication reward cost;
- event-driven asynchronous runtime proposal.

Each feature requires a no-failure compatibility mode and targeted protocol
tests before combination with learned policies.

### Phase 7: Distributed And Text Communication

Potential deliverables:

- per-agent model and optimizer ownership;
- federated parameter aggregation;
- distributed RL trainer integration;
- structured message codecs;
- text and LLM payload providers;
- payload-size and serialization accounting.

## Migration And Compatibility

1. Communication defaults to disabled so existing configs retain current
   behavior.
1. Add explicit `communication_radius`; do not reinterpret
   `communication_bandwidth` as radius.
1. Preserve current agent names internally for environment indexing and traces.
1. Add policy-facing identity controls independently of internal IDs.
1. Deprecate `CommunicationModel.select_communication_partners` in favour of
   topology builders.
1. Deprecate `CommunicationModel.process_messages` in favour of processors and
   protocol runtimes.
1. Keep temporary adapters for `NoCommunicationModel` and any external imports.
1. Update README, architecture, configuration, API, and environment docs as each
   phase becomes runtime truth.

## Risks And Mitigations

### Environment And Policy Ownership Is Ambiguous

Risk: learned communication may be executed inside `env.step()` and detached
from the trainable actor.

Mitigation: scripted communication may run in the environment, but
differentiable communication must run in the policy/trainer forward path using
the same plan, topology, and round definitions.

### GNN Depth Is Confused With Physical Time

Risk: an arbitrary deep model permits undocumented information propagation.

Mitigation: every scheme documents whether one layer equals one round. The
default `multihop_gnn` enforces this equality.

### PyG Becomes The Protocol State Machine

Risk: packet queues, identity, and lifecycle become difficult to inspect and
test inside convolution abstractions.

Mitigation: keep protocol state in ACN data types and use PyG only for explicit
tensor transformations or learned decisions.

### Hidden Identity Leaks Through Messages

Risk: engineered payloads or protocol metadata reveal opponent simulator IDs.

Mitigation: payload schemas and policy-visible metadata are explicitly selected;
trace-only ground truth remains separate.

### Centralized Batching Violates Decentralized Execution

Risk: a batched implementation consumes information unavailable to an
individual deployed agent.

Mitigation: enforce actor input contracts and test equivalent per-agent and
batched outputs. Batching is an implementation optimization, not an information
permission.

### Scope Expands Into A Full Network Simulator

Risk: protocol realism blocks MARL progress.

Mitigation: deliver phases 1 through 4 before advanced channel and transport
features. Each protocol feature must answer a specific experiment question.

## Implementation-PR Decisions

The following communication-specific questions remain open and should be
resolved in the communication implementation PR, where concrete data types,
configuration validation, and tests make their consequences visible:

1. Whether teammate identities are embedded as learned categorical features or
   used only as routing metadata.
1. Shared versus separate communication encoders for red and blue teams.
1. Recurrent versus stateless communication state across movement steps.
1. Initial asynchronous simulation fidelity and event model.
1. Future unit and enforcement meaning of `communication_bandwidth`.

## Definition Of Done For The Initial Communication Module

The initial communication module is complete when phases 0 through 3 are
implemented and:

- existing no-communication scenarios remain reproducible;
- same-team radius graphs are deterministic and traced;
- one-hop messages remain distinct and inspectable;
- PyG aggregation can be selected without changing delivery semantics;
- unchanged packets can relay over exactly one edge per round;
- multi-hop traces preserve origin and immediate sender;
- no hidden opponent identity enters policy inputs;
- communication effects can be compared against a no-communication baseline;
- tests cover topology, inboxes, aggregation, relay, identity boundaries, and
  parallel step integration.
