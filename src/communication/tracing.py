"""Communication trace records (Phase 1).

Builds plain, JSON-serializable dicts describing the communication graph and
per-message deliveries of a movement step, ready for the existing recorder
pattern (``src.utils.history`` writes plain dict rows to JSONL/Postgres).
This module only *builds* records; it performs no I/O and never imports
storage or API code (see the architectural boundary in ``CLAUDE.md``).

Record types:

- ``communication_graph``: one record per movement step with the frozen
  edges and their distances.
- ``communication_delivery``: one record per delivered message copy with
  step, round, message id, origin, immediate sender, receiver, hop count,
  and the scheme name.
- ``communication_relay``: one record per relay protocol decision (Phase 3),
  built by :func:`relay_decision_record`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.communication.types import CommunicationGraph, EdgeMessageBatch

__all__ = [
    "GRAPH_RECORD_TYPE",
    "DELIVERY_RECORD_TYPE",
    "RELAY_RECORD_TYPE",
    "RELAY_DECISIONS",
    "communication_graph_record",
    "communication_delivery_records",
    "relay_decision_record",
]

#: ``record_type`` value of graph snapshot records.
GRAPH_RECORD_TYPE = "communication_graph"

#: ``record_type`` value of per-delivery records.
DELIVERY_RECORD_TYPE = "communication_delivery"

#: ``record_type`` value of relay protocol decision records.
RELAY_RECORD_TYPE = "communication_relay"

#: Valid ``decision`` values of a relay decision record.
RELAY_DECISIONS = (
    "delivered",
    "dropped_duplicate",
    "dropped_ttl",
    "forwarded",
    "dropped_no_target",
    "dropped_off_graph",
)


def communication_graph_record(
    graph: CommunicationGraph,
    scheme: Optional[str],
    episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the trace record for one frozen communication graph.

    Args:
        graph: The frozen graph snapshot of a movement step.
        scheme: The communication scheme name (e.g. ``"one_hop_direct"``).
        episode: Optional episode index to include in the record.

    Returns:
        A JSON-serializable dict with the step, scheme, agent ids, and one
        entry per directed edge carrying source id, receiver id, and distance.
    """
    sources = graph.edge_index[0].tolist()
    receivers = graph.edge_index[1].tolist()
    distances = graph.edge_distance.tolist()
    edges: List[Dict[str, Any]] = [
        {
            "source": graph.agent_ids[source],
            "receiver": graph.agent_ids[receiver],
            "distance": float(distance),
        }
        for source, receiver, distance in zip(sources, receivers, distances)
    ]
    record: Dict[str, Any] = {
        "record_type": GRAPH_RECORD_TYPE,
        "step": int(graph.step),
        "scheme": scheme,
        "agent_ids": list(graph.agent_ids),
        "num_edges": len(edges),
        "edges": edges,
    }
    if episode is not None:
        record["episode"] = int(episode)
    return record


def communication_delivery_records(
    delivered: EdgeMessageBatch,
    graph: CommunicationGraph,
    step: int,
    scheme: Optional[str],
    episode: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Build one trace record per delivered message copy.

    Args:
        delivered: Delivered messages (typically one round's batch, or the
            whole step's concatenation; the per-row ``round_index`` field
            records the round either way).
        graph: The frozen graph the messages were delivered over (maps node
            indices back to agent ids).
        step: The movement step of delivery.
        scheme: The communication scheme name.
        episode: Optional episode index to include in each record.

    Returns:
        JSON-serializable dicts in delivery order with step, round, message
        id, origin, immediate sender, receiver, scheme, and — when the
        transport provided them — hop count and creation step/round.
    """
    records: List[Dict[str, Any]] = []
    senders = delivered.sender_index.tolist()
    receivers = delivered.receiver_index.tolist()
    origins = delivered.origin_index.tolist()
    message_ids = delivered.message_id.tolist()
    rounds = delivered.round_index.tolist()
    metadata = {key: value.tolist() for key, value in delivered.metadata.items()}
    for row in range(delivered.num_messages):
        record: Dict[str, Any] = {
            "record_type": DELIVERY_RECORD_TYPE,
            "step": int(step),
            "round": int(rounds[row]),
            "message_id": int(message_ids[row]),
            "origin": graph.agent_ids[origins[row]],
            "sender": graph.agent_ids[senders[row]],
            "receiver": graph.agent_ids[receivers[row]],
            "scheme": scheme,
        }
        for key in ("hop_count", "created_step", "created_round", "observation_step"):
            if key in metadata:
                record[key] = int(metadata[key][row])
        if episode is not None:
            record["episode"] = int(episode)
        records.append(record)
    return records


def relay_decision_record(
    *,
    decision: str,
    step: int,
    round_index: int,
    message_id: int,
    origin: str,
    sender: str,
    receiver: Optional[str],
    previous_hop: str,
    ttl: int,
    scheme: Optional[str] = None,
    episode: Optional[int] = None,
) -> Dict[str, Any]:
    """Build one relay protocol decision record (Phase 3, ``multihop_relay``).

    Decisions and their field semantics:

    - ``delivered``: a first-seen copy was accepted into the receiver's
      application inbox view. ``sender`` is the immediate sender of the copy,
      which is also its ``previous_hop``.
    - ``dropped_duplicate``: a copy whose ``message_id`` was already seen by
      the receiver was suppressed *before* application delivery and
      forwarding. Fields as for ``delivered``.
    - ``dropped_ttl``: a first-seen copy was delivered but not enqueued for
      forwarding because its hop budget is exhausted (``ttl`` is ``0``).
      Fields as for ``delivered``.
    - ``forwarded``: the receiver of an earlier copy hands it, unchanged, to
      the transport. ``sender`` is the forwarding agent, ``receiver`` the
      intended forward target, and ``previous_hop`` the agent the forwarder
      received the message from (excluded from the target set). ``round`` is
      the round in which the forwarded copy is transmitted.
    - ``dropped_no_target``: a queued forward was consumed without ever being
      transmitted because previous-hop exclusion left no valid target (e.g.
      an isolated bidirectional pair). ``sender`` is the would-be forwarder
      and ``receiver`` is ``None`` (nothing was addressed).
    - ``dropped_off_graph``: a queued forward was consumed without ever being
      transmitted because the forwarder or the message origin is no longer on
      the current graph (agents may leave between movement steps while a
      cross-step carryover is pending). ``sender`` is the would-be forwarder
      and ``receiver`` is ``None``.

    ``ttl`` is the remaining hop budget carried by the copy *after* the
    recorded transmission/arrival, so the ``forwarded`` record of a copy and
    the delivery-side record of its arrival report the same value. For the
    consumed-without-transmission drops (``dropped_no_target`` /
    ``dropped_off_graph``) no transmission occurred, so ``ttl`` is the budget
    the frame still held when it was consumed.

    Args:
        decision: One of :data:`RELAY_DECISIONS`.
        step: Movement step of the decision.
        round_index: Communication round of the decision (see above).
        message_id: Stable transport identity of the message.
        origin: Agent id of the message's original source.
        sender: Immediate sender (delivery-side) or forwarder (``forwarded``).
        receiver: Receiving agent (delivery-side) or forward target; ``None``
            for the consumed-without-transmission drop decisions.
        previous_hop: The copy's previous hop at the deciding agent.
        ttl: Remaining hop budget after this transmission/arrival.
        scheme: The communication scheme name.
        episode: Optional episode index to include in the record.

    Returns:
        A JSON-serializable dict with ``record_type`` ``communication_relay``.

    Raises:
        ValueError: If ``decision`` is not a known relay decision.
    """
    if decision not in RELAY_DECISIONS:
        raise ValueError(
            "Unknown relay decision {!r}; expected one of {}.".format(decision, RELAY_DECISIONS)
        )
    record: Dict[str, Any] = {
        "record_type": RELAY_RECORD_TYPE,
        "decision": decision,
        "step": int(step),
        "round": int(round_index),
        "message_id": int(message_id),
        "origin": origin,
        "sender": sender,
        "receiver": receiver,
        "previous_hop": previous_hop,
        "ttl": int(ttl),
        "scheme": scheme,
    }
    if episode is not None:
        record["episode"] = int(episode)
    return record
