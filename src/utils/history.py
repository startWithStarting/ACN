"""Run-history recording utilities for analysis and offline training.

The default recorder writes a lightweight local trace under each run directory.
The format is intentionally close to training data:

* ``agent_transitions.jsonl`` stores one row per agent per environment step.
* ``events.jsonl`` stores relational events such as blue detections and
  predictions.
* ``manifest.json`` stores run/config/agent metadata.

The same logical rows can also be persisted directly to Postgres via the
``--persist`` runner flag.
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Iterable, Iterator, Mapping, Optional

import numpy as np

from src.utils.geometry import calculate_distance
from src.utils.logger import get_logger

logger = get_logger("acn.utils.history")

TRACE_SCHEMA_VERSION = 1


class NoOpRunHistoryRecorder:
    """Recorder with the same public API that intentionally writes nothing."""

    def __init__(self, run_id: Optional[str] = None):
        self.enabled = False
        self.run_id = run_id
        self.results_dir = None

    def close(self) -> None:
        pass

    def register_agents(self, agents: Iterable[Any]) -> None:
        pass

    def snapshot_agents(self, agent_objects: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        return snapshot_agents(agent_objects)

    def record_agent_transitions(self, **kwargs: Any) -> None:
        pass

    def record_blue_events(self, **kwargs: Any) -> None:
        pass

    def finish(self, *, duration_seconds: float, num_steps: int, status: str = "completed") -> None:
        pass


def to_jsonable(value: Any) -> Any:
    """Convert common simulation values to JSON-serializable objects."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(to_jsonable(item) for item in value)
    return value


def _agent_side(agent_obj: Any) -> Optional[str]:
    agent_type = getattr(agent_obj, "agent_type", None)
    return getattr(agent_type, "value", agent_type)


def _vector_from_direction_speed(direction: Any, speed: Any) -> tuple[float, float]:
    if direction is None:
        return 0.0, 0.0
    try:
        direction_arr = np.asarray(direction, dtype=np.float32).reshape(-1)
        speed_value = float(np.asarray(speed, dtype=np.float32).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0
    if direction_arr.shape[0] < 2:
        return 0.0, 0.0
    return float(direction_arr[0] * speed_value), float(direction_arr[1] * speed_value)


def snapshot_agent(agent_obj: Any) -> Dict[str, Any]:
    """Return a compact, JSON-ready physical and metadata snapshot for one agent."""
    direction = getattr(agent_obj, "direction", None)
    speed = getattr(agent_obj, "speed", 0.0)
    vx, vy = _vector_from_direction_speed(direction, speed)
    return {
        "agent_id": getattr(agent_obj, "name", None),
        "side": _agent_side(agent_obj),
        "strategy_type": getattr(agent_obj, "strategy_type", None),
        "x": float(agent_obj.x) if getattr(agent_obj, "x", None) is not None else None,
        "y": float(agent_obj.y) if getattr(agent_obj, "y", None) is not None else None,
        "vx": vx,
        "vy": vy,
        "speed": float(speed) if speed is not None else None,
        "direction": to_jsonable(direction),
        "active": bool(getattr(agent_obj, "is_active", True)),
    }


def snapshot_agents(agent_objects: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Snapshot all known agents keyed by agent id."""
    return {agent_id: snapshot_agent(agent) for agent_id, agent in agent_objects.items()}


def agent_metadata_rows(agents: Iterable[Any]) -> list[Dict[str, Any]]:
    """Return static agent metadata rows."""
    return [
        {
            "agent_id": getattr(agent, "name", None),
            "side": _agent_side(agent),
            "strategy_type": getattr(agent, "strategy_type", None),
            "communication_bandwidth": getattr(agent, "communication_bandwidth", None),
            "processing_capability": getattr(agent, "processing_capability", None),
            "detection_radius": getattr(agent, "detection_radius", None),
        }
        for agent in agents
    ]


def iter_agent_transition_rows(
    *,
    episode: int,
    step: int,
    observations: Mapping[str, Any],
    actions: Mapping[str, Any],
    next_observations: Mapping[str, Any],
    rewards: Mapping[str, Any],
    terminations: Mapping[str, Any],
    truncations: Mapping[str, Any],
    infos: Optional[Mapping[str, Any]],
    state_before: Mapping[str, Any],
    state_after: Mapping[str, Any],
    agent_objects: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    """Yield one transition row for each agent touched by this step."""
    agent_ids = set()
    for mapping in (
        observations,
        actions,
        next_observations,
        rewards,
        terminations,
        truncations,
        state_before,
        state_after,
    ):
        agent_ids.update(mapping.keys())

    for agent_id in sorted(agent_ids):
        agent_obj = agent_objects.get(agent_id)
        yield {
            "schema_version": TRACE_SCHEMA_VERSION,
            "episode": episode,
            "step": step,
            "agent_id": agent_id,
            "side": _agent_side(agent_obj) if agent_obj is not None else None,
            "strategy_type": getattr(agent_obj, "strategy_type", None)
            if agent_obj is not None
            else None,
            "observation": observations.get(agent_id),
            "action": actions.get(agent_id),
            "reward": rewards.get(agent_id),
            "terminated": bool(terminations.get(agent_id, False)),
            "truncated": bool(truncations.get(agent_id, False)),
            "next_observation": next_observations.get(agent_id),
            "info": infos.get(agent_id, {}) if infos is not None else {},
            "state_before": state_before.get(agent_id),
            "state_after": state_after.get(agent_id),
        }


def iter_blue_event_rows(
    *,
    episode: int,
    step: int,
    observations: Mapping[str, Any],
    agent_objects: Mapping[str, Any],
) -> Iterator[Dict[str, Any]]:
    """Yield blue-agent observation, target, and prediction events."""
    for agent_id, observation in observations.items():
        blue_agent = agent_objects.get(agent_id)
        if blue_agent is None or _agent_side(blue_agent) != "blue":
            continue

        blue_x = getattr(blue_agent, "x", None)
        blue_y = getattr(blue_agent, "y", None)

        for red_id, red_data in observation.get("red_agents", {}).items():
            position = red_data.get("position")
            if position is None or blue_x is None or blue_y is None:
                continue

            position_arr = np.asarray(position, dtype=np.float32).reshape(-1)
            if position_arr.shape[0] < 2:
                continue

            distance = calculate_distance((blue_x, blue_y), (position_arr[0], position_arr[1]))
            visible = bool(blue_agent.is_within_detection_radius((position_arr[0], position_arr[1])))
            if not visible:
                continue

            yield {
                "schema_version": TRACE_SCHEMA_VERSION,
                "episode": episode,
                "step": step,
                "phase": "observe",
                "event_type": "observation",
                "source_agent_id": agent_id,
                "target_agent_id": red_id,
                "receiver_agent_id": agent_id,
                "distance": float(distance),
                "visible": visible,
                "delivered": True,
                "payload": {
                    "x": float(position_arr[0]),
                    "y": float(position_arr[1]),
                    "timestamp": observation.get("timestamp"),
                },
            }

        target_position = getattr(blue_agent, "current_target_position", None)
        if target_position is not None:
            target_arr = np.asarray(target_position, dtype=np.float32).reshape(-1)
            if target_arr.shape[0] >= 2 and np.all(np.isfinite(target_arr[:2])):
                yield {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "episode": episode,
                    "step": step,
                    "phase": "decide",
                    "event_type": "target",
                    "source_agent_id": agent_id,
                    "target_agent_id": None,
                    "receiver_agent_id": agent_id,
                    "payload": {
                        "x": float(target_arr[0]),
                        "y": float(target_arr[1]),
                    },
                }

        predicted_positions = getattr(blue_agent, "predicted_positions", {})
        for red_id, future_positions in predicted_positions.items():
            for horizon, predicted_position in enumerate(future_positions, start=1):
                pred_arr = np.asarray(predicted_position, dtype=np.float32).reshape(-1)
                if pred_arr.shape[0] < 2 or not np.all(np.isfinite(pred_arr[:2])):
                    continue
                yield {
                    "schema_version": TRACE_SCHEMA_VERSION,
                    "episode": episode,
                    "step": step,
                    "phase": "decide",
                    "event_type": "prediction",
                    "source_agent_id": agent_id,
                    "target_agent_id": red_id,
                    "receiver_agent_id": agent_id,
                    "payload": {
                        "x": float(pred_arr[0]),
                        "y": float(pred_arr[1]),
                        "horizon": horizon,
                        "prediction_for_step": step + horizon,
                    },
                }


def create_history_recorder(
    *,
    persist: bool,
    results_dir: Optional[str],
    config: Mapping[str, Any],
    mode: str,
    config_path: Optional[str] = None,
    enabled: bool = True,
    run_id: Optional[str] = None,
    database_url: Optional[str] = None,
) -> Any:
    """Create the selected run-history recorder backend."""
    if not enabled:
        return NoOpRunHistoryRecorder(run_id=run_id)

    if persist:
        persistent_run_id = run_id or str(uuid.uuid4())
        from src.storage.history import PostgresRunHistoryRecorder

        return PostgresRunHistoryRecorder(
            run_id=persistent_run_id,
            config=config,
            mode=mode,
            config_path=config_path,
            database_url=database_url,
        )

    if results_dir is None:
        raise ValueError("results_dir is required for file-backed run history")
    return RunHistoryRecorder(
        results_dir=results_dir,
        config=config,
        mode=mode,
        config_path=config_path,
        run_id=run_id,
    )


class RunHistoryRecorder:
    """Write local trace files for one simulation run."""

    def __init__(
        self,
        results_dir: str,
        config: Mapping[str, Any],
        mode: str,
        config_path: Optional[str] = None,
        enabled: bool = True,
        run_id: Optional[str] = None,
    ):
        self.enabled = enabled
        self.run_id = run_id or os.path.basename(os.path.abspath(results_dir))
        self.results_dir = results_dir
        self.trace_dir = os.path.join(results_dir, "trace")
        self.manifest_path = os.path.join(self.trace_dir, "manifest.json")
        self.transitions_path = os.path.join(self.trace_dir, "agent_transitions.jsonl")
        self.events_path = os.path.join(self.trace_dir, "events.jsonl")
        self._manifest: Dict[str, Any] = {}
        self._transition_file = None
        self._event_file = None

        if not self.enabled:
            return

        os.makedirs(self.trace_dir, exist_ok=True)
        self._transition_file = open(self.transitions_path, "w", encoding="utf-8")
        self._event_file = open(self.events_path, "w", encoding="utf-8")
        self._manifest = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "run_id": self.run_id,
            "mode": mode,
            "config_path": config_path,
            "config": to_jsonable(config),
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
            "duration_seconds": None,
            "num_steps": None,
            "status": "running",
            "files": {
                "agent_transitions": os.path.relpath(self.transitions_path, results_dir),
                "events": os.path.relpath(self.events_path, results_dir),
            },
            "agents": [],
        }
        self._write_manifest()
        logger.info("Trace recording enabled at {}", self.trace_dir)

    def close(self) -> None:
        """Close trace files without changing run status."""
        if self._transition_file is not None:
            self._transition_file.close()
            self._transition_file = None
        if self._event_file is not None:
            self._event_file.close()
            self._event_file = None

    def register_agents(self, agents: Iterable[Any]) -> None:
        """Store static agent metadata in the manifest."""
        if not self.enabled:
            return

        self._manifest["agents"] = to_jsonable(agent_metadata_rows(agents))
        self._write_manifest()

    def snapshot_agents(self, agent_objects: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Snapshot all known agents keyed by agent id."""
        return snapshot_agents(agent_objects)

    def record_agent_transitions(
        self,
        *,
        episode: int,
        step: int,
        observations: Mapping[str, Any],
        actions: Mapping[str, Any],
        next_observations: Mapping[str, Any],
        rewards: Mapping[str, Any],
        terminations: Mapping[str, Any],
        truncations: Mapping[str, Any],
        infos: Optional[Mapping[str, Any]],
        state_before: Mapping[str, Any],
        state_after: Mapping[str, Any],
        agent_objects: Mapping[str, Any],
    ) -> None:
        """Write one transition row for each agent touched by this step."""
        if not self.enabled or self._transition_file is None:
            return

        rows = iter_agent_transition_rows(
            episode=episode,
            step=step,
            observations=observations,
            actions=actions,
            next_observations=next_observations,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            infos=infos,
            state_before=state_before,
            state_after=state_after,
            agent_objects=agent_objects,
        )
        for row in rows:
            self._write_jsonl(self._transition_file, row)

    def record_blue_events(
        self,
        *,
        episode: int,
        step: int,
        observations: Mapping[str, Any],
        agent_objects: Mapping[str, Any],
    ) -> None:
        """Write blue-agent observation, target, and prediction events."""
        if not self.enabled or self._event_file is None:
            return

        rows = iter_blue_event_rows(
            episode=episode,
            step=step,
            observations=observations,
            agent_objects=agent_objects,
        )
        for row in rows:
            self._write_jsonl(self._event_file, row)

    def finish(self, *, duration_seconds: float, num_steps: int, status: str = "completed") -> None:
        """Mark the run trace as finished and close files."""
        if not self.enabled:
            return

        self._manifest["finished_at"] = datetime.now().isoformat()
        self._manifest["duration_seconds"] = float(duration_seconds)
        self._manifest["num_steps"] = int(num_steps)
        self._manifest["status"] = status
        self._write_manifest()
        self.close()

    def _write_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(self._manifest), f, indent=2, sort_keys=True)

    @staticmethod
    def _write_jsonl(file_obj: Any, row: Mapping[str, Any]) -> None:
        file_obj.write(json.dumps(to_jsonable(row), sort_keys=True) + "\n")
