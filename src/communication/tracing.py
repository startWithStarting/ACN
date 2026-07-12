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
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.communication.types import CommunicationGraph, EdgeMessageBatch

__all__ = [
    "GRAPH_RECORD_TYPE",
    "DELIVERY_RECORD_TYPE",
    "communication_graph_record",
    "communication_delivery_records",
]

#: ``record_type`` value of graph snapshot records.
GRAPH_RECORD_TYPE = "communication_graph"

#: ``record_type`` value of per-delivery records.
DELIVERY_RECORD_TYPE = "communication_delivery"


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
