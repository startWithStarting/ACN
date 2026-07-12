"""Config-gated blue sensor models for environment observations.

This module implements the bearing-only blue sensor decided in
``docs/communication_decision_log.md`` ("Sensor Model", "Observation
Contract", "Identity And Evaluability") with the report layout from
``docs/communication_implementation_plan.md`` ("Payload Generation").

Two blue sensor modes exist, selected by ``environment.observation``:

- ``legacy`` (default): the blue observation carries the historical
  ``red_agents`` dict of visible red positions/distances. This is the exact
  pre-existing behavior; configs without an ``observation`` block use it.
- ``bearing_only``: the blue observation replaces ``red_agents`` with
  ``contact_reports``, a variable-size list holding one anonymous report per
  red inside the observer's detection radius. Each report is::

      {
          "payload": [observer_x, observer_y, direction_x, direction_y],
          "metadata": {"observer": <blue name>, "step": <t>},
      }

  ``(direction_x, direction_y)`` is the global-frame unit vector toward the
  red, i.e. ``(cos(theta), sin(theta))`` of the bearing. No red identity,
  position, range, or velocity appears in any policy-visible field; missing
  contacts mean "not visible". Reports are sorted by bearing angle (the
  ``atan2`` value in ``(-pi, pi]``) so traces are reproducible without
  leaking which red produced which report.

Privileged ground truth: the simulator retains the per-report red identity
outside the observation. The environments expose it through the step/reset
``infos`` dict under ``ground_truth_contacts`` (report index -> red name,
aligned with the report order) so tracking metrics and reward evaluation
still work.

Scripted-controller boundary: scripted blues (VAR pursuit) keep privileged
access. When ``bearing_only`` is on and ``scripted_blue_privileged`` is on
(the default), blue observations additionally carry ``privileged_red_agents``
(the legacy dict). That field is for scripted controllers and traces only;
learned policies must never consume it (enforced later at the trainer
boundary). Strict mode (``scripted_blue_privileged: false``) omits it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.utils.logger import get_logger

logger = get_logger("acn.env.observation")

__all__ = [
    "VALID_BLUE_SENSORS",
    "ObservationConfig",
    "build_contact_reports",
]

#: Blue sensor modes accepted by ``environment.observation.blue_sensor``.
VALID_BLUE_SENSORS: Tuple[str, ...] = ("legacy", "bearing_only")

_KNOWN_KEYS: Tuple[str, ...] = ("blue_sensor", "scripted_blue_privileged")


@dataclass(frozen=True)
class ObservationConfig:
    """Validated ``environment.observation`` configuration.

    The default-constructed instance reproduces legacy behavior exactly; an
    absent ``observation`` block parses to it.

    Attributes:
        blue_sensor: Blue sensor mode, one of :data:`VALID_BLUE_SENSORS`.
            ``"legacy"`` keeps the historical ``red_agents`` dict;
            ``"bearing_only"`` replaces it with anonymous ``contact_reports``.
        scripted_blue_privileged: Only meaningful in ``bearing_only`` mode.
            When true (default), blue observations also carry
            ``privileged_red_agents`` (the legacy dict) so scripted blue
            controllers keep working; learned policies must never consume it.
    """

    blue_sensor: str = "legacy"
    scripted_blue_privileged: bool = True

    def __post_init__(self) -> None:
        if self.blue_sensor not in VALID_BLUE_SENSORS:
            raise ValueError(
                "environment.observation.blue_sensor must be one of {} (got {!r}).".format(
                    ", ".join(VALID_BLUE_SENSORS), self.blue_sensor
                )
            )
        if not isinstance(self.scripted_blue_privileged, bool):
            raise ValueError(
                "environment.observation.scripted_blue_privileged must be a boolean "
                "(true/false), got {} ({!r}).".format(
                    type(self.scripted_blue_privileged).__name__, self.scripted_blue_privileged
                )
            )

    @property
    def bearing_only(self) -> bool:
        """True when the bearing-only blue sensor is selected."""
        return self.blue_sensor == "bearing_only"

    @classmethod
    def from_dict(cls, data: Optional[Mapping[str, Any]]) -> "ObservationConfig":
        """Parse the ``environment.observation`` block of a loaded YAML config.

        Args:
            data: The plain-dict observation block, or None when the block is
                absent. An absent block yields the legacy default.

        Returns:
            A validated, immutable ObservationConfig.

        Raises:
            ValueError: If the block is not a mapping or a field is invalid.
        """
        if data is None:
            return cls()
        if not isinstance(data, MappingABC):
            raise ValueError(
                "environment.observation must be a mapping of keys to values, got {} "
                "({!r}). Check the YAML indentation of this block.".format(
                    type(data).__name__, data
                )
            )
        unknown = sorted(set(data.keys()) - set(_KNOWN_KEYS))
        if unknown:
            logger.warning(
                "Ignoring unknown keys in environment.observation: {}. Known keys: {}",
                ", ".join(unknown),
                ", ".join(_KNOWN_KEYS),
            )
        blue_sensor = data.get("blue_sensor", "legacy")
        if blue_sensor is None:
            blue_sensor = "legacy"
        scripted_blue_privileged = data.get("scripted_blue_privileged", True)
        if scripted_blue_privileged is None:
            scripted_blue_privileged = True
        config = cls(
            blue_sensor=blue_sensor,
            scripted_blue_privileged=scripted_blue_privileged,
        )
        if config.bearing_only:
            logger.debug(
                "Parsed observation config: blue_sensor={} scripted_blue_privileged={}",
                config.blue_sensor,
                config.scripted_blue_privileged,
            )
        return config


def build_contact_reports(
    observer_name: str,
    observer_position: Tuple[float, float],
    visible_reds: Sequence[Tuple[str, Tuple[float, float]]],
    step: int,
) -> Tuple[List[Dict[str, Any]], Dict[int, str]]:
    """Build anonymous bearing-only contact reports for one blue observer.

    Each visible red produces one report whose payload is exactly four floats
    ``[observer_x, observer_y, direction_x, direction_y]`` with the direction
    the global-frame unit vector toward the red (``cos``/``sin`` of the
    ``atan2`` bearing). Reports are sorted by bearing angle in ``(-pi, pi]``;
    equal-bearing ties keep the input order (stable sort), which cannot leak
    identity because equal bearings produce identical payloads. A red exactly
    at the observer position gets the ``atan2(0, 0) == 0`` convention, i.e.
    direction ``(1.0, 0.0)``.

    Args:
        observer_name: Name of the observing blue agent (teammate identity is
            policy-visible metadata by design).
        observer_position: Global ``(x, y)`` position of the observer.
        visible_reds: ``(red_name, (x, y))`` pairs for every red inside the
            observer's detection radius. Names are used only for the
            privileged ground-truth mapping, never placed in a report.
        step: Observation timestep recorded in each report's metadata.

    Returns:
        A tuple ``(reports, ground_truth_contacts)`` where ``reports`` is the
        bearing-sorted list of policy-visible report dicts and
        ``ground_truth_contacts`` maps report index to the red name it
        corresponds to (privileged; expose via infos, never the observation).
    """
    observer_x = float(observer_position[0])
    observer_y = float(observer_position[1])
    contacts: List[Tuple[float, Dict[str, Any], str]] = []
    for red_name, red_position in visible_reds:
        angle = math.atan2(
            float(red_position[1]) - observer_y,
            float(red_position[0]) - observer_x,
        )
        report: Dict[str, Any] = {
            "payload": [observer_x, observer_y, math.cos(angle), math.sin(angle)],
            "metadata": {"observer": observer_name, "step": int(step)},
        }
        contacts.append((angle, report, red_name))
    contacts.sort(key=lambda contact: contact[0])
    reports = [report for _, report, _ in contacts]
    ground_truth = {index: red_name for index, (_, _, red_name) in enumerate(contacts)}
    return reports, ground_truth
