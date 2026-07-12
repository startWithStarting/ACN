"""Learned GraphSAGE communicator for the ``multihop_gnn`` scheme (Phase 4).

Implements the learned message/aggregate/update model of
``docs/communication_decision_log.md`` ("PyG And Protocol Boundary" and
"Round And Cache Model"): a local learnable encoder followed by ``R``
GraphSAGE rounds over the frozen same-team radius graph, finishing with a
linear projection to the per-agent communication embedding. ``R`` equals the
number of message-passing layers, so one configured layer corresponds to one
ACN communication round and information propagates AT MOST ``R`` graph hops —
guaranteed by construction, not by convention.

Execution boundary (``docs/communication_implementation_plan.md``, "Risks:
Environment And Policy Ownership Is Ambiguous"): this module is a plain
``nn.Module`` executed inside the policy/trainer forward pass. The
environment never runs it — the compiled ``multihop_gnn`` plan carries a
marker processor that raises if the env runtime tries to execute the rounds.

Empty and disconnected graphs are valid: PyG's ``SAGEConv`` keeps its root
transform (``lin_r``) for every node, so an isolated node's own encoding
still flows through the update path of every round and reaches the output
projection.

Weight layout: separate ``SAGEConv`` weights per round by default;
``shared_weights=True`` reuses ONE convolution across all rounds (the
recurrent-weights option of the implementation plan).
"""

from __future__ import annotations

from typing import Tuple

import torch
from torch import nn

from src.utils.logger import get_logger

logger = get_logger("acn.communication")

__all__ = ["SUPPORTED_GNN_AGGREGATIONS", "GraphSAGECommunicator"]

#: Aggregations accepted by :class:`GraphSAGECommunicator` (per the decision
#: log the reference configuration is ``sum``; ``mean`` and ``max`` are the
#: config-selectable alternatives).
SUPPORTED_GNN_AGGREGATIONS: Tuple[str, ...] = ("sum", "mean", "max")


class GraphSAGECommunicator(nn.Module):
    """Encoder -> R GraphSAGE rounds -> embedding, over a frozen edge_index.

    The forward pass is fully differentiable and torch end-to-end; gradients
    reach the encoder, every round's convolution, and any downstream action
    head jointly.

    Args:
        in_dim: Per-node input feature width (the agent's policy-visible
            local features).
        hidden_dim: GNN hidden width (``processor.hidden_dimension``).
        message_dim: Output embedding width (``payload.dimension``).
        rounds: ``R``, the number of message-passing rounds; equals the
            number of ``SAGEConv`` layers by construction.
        aggregation: Neighbour aggregation; one of
            :data:`SUPPORTED_GNN_AGGREGATIONS`.
        shared_weights: ``True`` reuses one convolution across all rounds;
            ``False`` (default) keeps separate weights per round.

    Raises:
        ValueError: If any dimension is non-positive, ``rounds < 1``, or the
            aggregation is unsupported.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        message_dim: int,
        rounds: int = 3,
        aggregation: str = "sum",
        shared_weights: bool = False,
    ) -> None:
        super().__init__()
        # torch_geometric is imported lazily (mirroring processors.pyg) so
        # code paths that never build a learned communicator do not pay the
        # import cost.
        from torch_geometric.nn import SAGEConv

        for name, value in (
            ("in_dim", in_dim),
            ("hidden_dim", hidden_dim),
            ("message_dim", message_dim),
            ("rounds", rounds),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    "GraphSAGECommunicator {} must be a positive integer, got {!r}.".format(
                        name, value
                    )
                )
        if aggregation not in SUPPORTED_GNN_AGGREGATIONS:
            raise ValueError(
                "GraphSAGECommunicator aggregation must be one of {} (got {!r}).".format(
                    ", ".join(SUPPORTED_GNN_AGGREGATIONS), aggregation
                )
            )

        self.rounds = int(rounds)
        self.aggregation = aggregation
        self.shared_weights = bool(shared_weights)
        self.encoder = nn.Linear(in_dim, hidden_dim)
        if self.shared_weights:
            self.convs = nn.ModuleList([SAGEConv(hidden_dim, hidden_dim, aggr=aggregation)])
        else:
            self.convs = nn.ModuleList(
                [SAGEConv(hidden_dim, hidden_dim, aggr=aggregation) for _ in range(self.rounds)]
            )
        self.output = nn.Linear(hidden_dim, message_dim)
        logger.debug(
            "GraphSAGECommunicator built: in={} hidden={} message={} rounds={} "
            "aggregation={} shared_weights={}",
            in_dim,
            hidden_dim,
            message_dim,
            self.rounds,
            aggregation,
            self.shared_weights,
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Run the R communication rounds over one (possibly batched) graph.

        Args:
            x: Node feature matrix ``[N, in_dim]``. For batched evaluation,
                pass the disjoint union of the per-sample graphs (block
                offsets in ``edge_index``); sum/mean/max aggregation never
                crosses disconnected components.
            edge_index: ``[2, E]`` long tensor of directed
                ``source -> receiver`` edges. ``E = 0`` (no communication
                edges) is valid: every node's embedding is then a function of
                its own features only.

        Returns:
            Per-node communication embedding ``[N, message_dim]``.
        """
        hidden = torch.tanh(self.encoder(x))
        for round_index in range(self.rounds):
            conv = self.convs[0] if self.shared_weights else self.convs[round_index]
            hidden = torch.tanh(conv(hidden, edge_index))
        return self.output(hidden)
