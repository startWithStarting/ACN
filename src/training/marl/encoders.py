"""Policy-input encoders: policy-visible observation -> fixed tensors + masks.

The environment's observation contract stays dict/list-shaped (limited
sensing); converting the variable-size collections to fixed-size tensors is a
policy/trainer adapter concern (``docs/communication_decision_log.md``,
"Observation Contract"). This module owns that conversion for the v1
benchmark encoders:

- **Base features** ``[4]``: own position normalized by the grid
  (``x/W, y/H``) plus the normalized grid center.
- **Contact slots** ``[K, 4]`` plus boolean mask ``[K]``: K padded contact
  slots (K configurable, default 8), deterministic order = report order,
  reports beyond K dropped in order. For a blue agent the slots hold its
  bearing-only ``contact_reports`` payloads with the observer position
  normalized. For a red agent the slots are derived bearing-style contacts
  toward its visible opponents (``blue_agents`` values only, sorted by
  bearing angle) so no opponent identity enters the tensor.
- **Communication features** ``[D]``: the scheme's declared policy-facing
  encoding. ``one_hop_mean`` already declares an aggregated
  ``{"vector", "count"}`` view — the vector is used directly. For
  ``one_hop_direct`` (and ``multihop_relay``) the delivered inbox preserves
  distinct messages by scheme contract; THIS policy adapter mean-pools the
  inbox payloads at the policy boundary, which is legitimate precisely
  because pooling here is the adapter's declared policy-facing encoding, not
  an environment-side transformation (the scheme's delivery keeps messages
  distinct for traces and other consumers). Scheme ``none``/absent yields
  zeros of the declared payload dimension, keeping the feature layout
  identical across communication ablations.

Privileged-leakage boundary: NO opponent identity and NO privileged field may
enter these encoders. :meth:`PolicyEncoder.encode` raises if the observation
carries ``privileged_red_agents`` or ``ground_truth_contacts`` — a trainable
team must be configured so those keys never exist in its observations
(``environment.observation.scripted_blue_privileged: false``).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import torch

from src.communication.processors.pyg import AggregatedVectorView
from src.communication.types import EdgeMessageBatch, InboxBatch
from src.utils.logger import get_logger

logger = get_logger("acn.training.encoders")

__all__ = ["PRIVILEGED_OBSERVATION_KEYS", "PolicyEncoder"]

#: Observation keys that must never reach a learned policy's encoder.
PRIVILEGED_OBSERVATION_KEYS: Tuple[str, ...] = (
    "privileged_red_agents",
    "ground_truth_contacts",
)

#: Per-slot payload width: [pos_x_norm, pos_y_norm, direction_x, direction_y].
CONTACT_PAYLOAD_DIM: int = 4


class PolicyEncoder:
    """Fixed-size policy-input encoder for one team's agents.

    One encoder instance is built per run from the environment configuration
    (grid size, communication scheme and payload dimension) plus the
    ``training.encoder`` settings, and applied uniformly to every agent of the
    trainable team.

    Attributes:
        grid_width: Environment grid width used for position normalization.
        grid_height: Environment grid height used for position normalization.
        contact_slots: K, the number of padded contact slots.
        comm_dim: D, the declared communication feature dimension (the
            configured payload dimension; used for the zeros encoding when
            communication is disabled or nothing was delivered).
        comm_scheme: The configured communication scheme name (``"none"``
            when communication is disabled).
    """

    def __init__(
        self,
        grid_width: float,
        grid_height: float,
        contact_slots: int = 8,
        comm_dim: int = 4,
        comm_scheme: str = "none",
    ) -> None:
        """Create the encoder.

        Args:
            grid_width: Positive grid width.
            grid_height: Positive grid height.
            contact_slots: K >= 1 padded contact slots.
            comm_dim: Declared communication feature dimension (>= 0).
            comm_scheme: Configured scheme name, for error messages.

        Raises:
            ValueError: If any dimension parameter is out of range.
        """
        if grid_width <= 0 or grid_height <= 0:
            raise ValueError(
                "PolicyEncoder grid dimensions must be positive, got {}x{}.".format(
                    grid_width, grid_height
                )
            )
        if contact_slots < 1:
            raise ValueError(
                "PolicyEncoder contact_slots must be >= 1, got {}.".format(contact_slots)
            )
        if comm_dim < 0:
            raise ValueError("PolicyEncoder comm_dim must be >= 0, got {}.".format(comm_dim))
        self.grid_width = float(grid_width)
        self.grid_height = float(grid_height)
        self.contact_slots = int(contact_slots)
        self.comm_dim = int(comm_dim)
        self.comm_scheme = comm_scheme

    @property
    def base_dim(self) -> int:
        """Width of the base feature vector."""
        return 4

    @property
    def feature_dim(self) -> int:
        """Width of the flat concatenated feature vector.

        Layout: base ``[4]`` + contacts flattened ``[K * 4]`` + contact mask
        ``[K]`` + communication ``[D]``.
        """
        return self.base_dim + self.contact_slots * (CONTACT_PAYLOAD_DIM + 1) + self.comm_dim

    def encode(self, observation: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        """Encode one agent's policy-visible observation.

        Args:
            observation: The agent's native dict observation as returned by
                the environment.

        Returns:
            Dict with keys ``base`` ``[4]``, ``contacts`` ``[K, 4]``,
            ``contacts_mask`` ``[K]`` (bool), ``comm`` ``[D]``, and
            ``features`` (the flat concatenation, ``[feature_dim]``), all
            float32 except the mask.

        Raises:
            ValueError: If the observation carries a privileged key (leakage
                guard) or a communication view of unexpected shape.
        """
        self._assert_no_privileged_fields(observation)
        base = self._encode_base(observation)
        contacts, mask = self._encode_contacts(observation)
        comm = self._encode_communication(observation)
        features = torch.cat([base, contacts.reshape(-1), mask.to(torch.float32), comm])
        return {
            "base": base,
            "contacts": contacts,
            "contacts_mask": mask,
            "comm": comm,
            "features": features,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _assert_no_privileged_fields(self, observation: Mapping[str, Any]) -> None:
        """Refuse to encode an observation that carries privileged fields."""
        present = [key for key in PRIVILEGED_OBSERVATION_KEYS if key in observation]
        if present:
            raise ValueError(
                "Observation contains privileged field(s) {} which must never reach a "
                "learned policy encoder. For a trainable blue team set "
                "environment.observation.scripted_blue_privileged: false so the key is "
                "never built.".format(", ".join(present))
            )

    def _encode_base(self, observation: Mapping[str, Any]) -> torch.Tensor:
        """Encode own position and grid center, both grid-normalized."""
        position = observation.get("position")
        center = observation.get("grid_center")
        if position is None or center is None:
            raise ValueError(
                "Observation is missing 'position'/'grid_center'; got keys {}.".format(
                    sorted(observation.keys())
                )
            )
        return torch.tensor(
            [
                float(position[0]) / self.grid_width,
                float(position[1]) / self.grid_height,
                float(center[0]) / self.grid_width,
                float(center[1]) / self.grid_height,
            ],
            dtype=torch.float32,
        )

    def _encode_contacts(
        self, observation: Mapping[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode the agent's contacts into K padded slots plus a mask.

        Blue agents provide bearing-only ``contact_reports``; red agents get
        derived bearing-style contacts toward visible opponents. Order is the
        deterministic report order (bearing-sorted by construction); reports
        beyond K are dropped in order.
        """
        payloads: List[List[float]]
        if "contact_reports" in observation:
            payloads = []
            for report in observation["contact_reports"]:
                payload = report.get("payload") if isinstance(report, Mapping) else None
                if payload is None or len(payload) != CONTACT_PAYLOAD_DIM:
                    raise ValueError(
                        "contact_reports entries must carry a 4-float payload, got "
                        "{!r}.".format(report)
                    )
                payloads.append(
                    [
                        float(payload[0]) / self.grid_width,
                        float(payload[1]) / self.grid_height,
                        float(payload[2]),
                        float(payload[3]),
                    ]
                )
        elif "blue_agents" in observation:
            payloads = self._derived_opponent_contacts(observation)
        else:
            payloads = []

        contacts = torch.zeros((self.contact_slots, CONTACT_PAYLOAD_DIM), dtype=torch.float32)
        mask = torch.zeros(self.contact_slots, dtype=torch.bool)
        kept = payloads[: self.contact_slots]
        if len(payloads) > self.contact_slots:
            logger.debug(
                "Dropping {} contact(s) beyond the {}-slot encoder window",
                len(payloads) - self.contact_slots,
                self.contact_slots,
            )
        for index, payload in enumerate(kept):
            contacts[index] = torch.tensor(payload, dtype=torch.float32)
            mask[index] = True
        return contacts, mask

    def _derived_opponent_contacts(self, observation: Mapping[str, Any]) -> List[List[float]]:
        """Build bearing-style contacts toward a red agent's visible opponents.

        Only the position VALUES of the ``blue_agents`` dict are read — the
        opponent identities (dict keys) never enter the tensor. Contacts are
        sorted by bearing angle in ``(-pi, pi]`` (mirroring
        ``src.env.sensors.build_contact_reports``), so slot order carries no
        identity either.
        """
        position = observation.get("position")
        if position is None:
            return []
        own_x = float(position[0])
        own_y = float(position[1])
        entries: List[Tuple[float, List[float]]] = []
        for info in observation["blue_agents"].values():
            opponent_pos = info.get("position") if isinstance(info, Mapping) else None
            if opponent_pos is None:
                continue
            angle = math.atan2(float(opponent_pos[1]) - own_y, float(opponent_pos[0]) - own_x)
            entries.append(
                (
                    angle,
                    [
                        own_x / self.grid_width,
                        own_y / self.grid_height,
                        math.cos(angle),
                        math.sin(angle),
                    ],
                )
            )
        entries.sort(key=lambda entry: entry[0])
        return [payload for _, payload in entries]

    def _encode_communication(self, observation: Mapping[str, Any]) -> torch.Tensor:
        """Encode the communication view to the declared fixed dimension ``D``.

        Scheme handling (see the module docstring):

        - absent view / scheme ``none`` / empty inbox -> zeros ``[D]``;
        - ``one_hop_mean`` aggregated view -> its ``vector``;
        - preserved-inbox batches (``one_hop_direct``, ``multihop_relay``) ->
          mean-pooled payloads at the policy boundary.
        """
        zeros = torch.zeros(self.comm_dim, dtype=torch.float32)
        view = observation.get("communication")
        if view is None:
            return zeros
        inbox = view.get("inbox", ()) if isinstance(view, Mapping) else ()
        return self._encode_inbox(inbox, zeros)

    def _encode_inbox(self, inbox: Any, zeros: torch.Tensor) -> torch.Tensor:
        """Encode one delivery view; see :meth:`_encode_communication`."""
        if isinstance(inbox, AggregatedVectorView):
            vector = inbox.vector.to(torch.float32)
            if vector.numel() != self.comm_dim:
                raise ValueError(
                    "Aggregated communication vector has dimension {} but the encoder "
                    "declares comm_dim={}. Align training.encoder with the configured "
                    "communication payload dimension.".format(vector.numel(), self.comm_dim)
                )
            return vector
        if isinstance(inbox, InboxBatch):
            inbox = inbox.messages
        if isinstance(inbox, EdgeMessageBatch):
            if inbox.num_messages == 0:
                return zeros
            payload = inbox.payload.to(torch.float32)
            if payload.dim() != 2 or payload.size(1) != self.comm_dim:
                raise ValueError(
                    "Inbox payload has shape {} but the encoder declares comm_dim={}. "
                    "Align training.encoder with the configured communication payload "
                    "dimension.".format(tuple(payload.shape), self.comm_dim)
                )
            return payload.mean(dim=0)
        if isinstance(inbox, Sequence) and len(inbox) == 0:
            # The explicit empty view (reset / off-graph agent / scheme none).
            return zeros
        raise ValueError(
            "Unsupported communication inbox type {} for scheme {!r}; the v1 policy "
            "encoders understand aggregated vectors, preserved inboxes, and the "
            "explicit empty view.".format(type(inbox).__name__, self.comm_scheme)
        )
