"""Reconstruct and plot blue-agent histories from ACN trace files.

Usage:
    uv run python -m src.analysis.blue_history --run-dir results/... --blue-agent blue_0
    uv run python -m src.analysis.blue_history --run-dir results/... --blue-agent blue_0 --target red_30 --plot all
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.utils.history import to_jsonable


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file. Missing files return an empty list."""
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def trace_paths(run_dir: str) -> Dict[str, str]:
    """Return standard trace file paths for a run directory."""
    trace_dir = os.path.join(run_dir, "trace")
    return {
        "trace_dir": trace_dir,
        "manifest": os.path.join(trace_dir, "manifest.json"),
        "transitions": os.path.join(trace_dir, "agent_transitions.jsonl"),
        "events": os.path.join(trace_dir, "events.jsonl"),
    }


def load_manifest(run_dir: str) -> Dict[str, Any]:
    """Load trace manifest metadata."""
    path = trace_paths(run_dir)["manifest"]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_blue_agents(run_dir: str) -> List[str]:
    """List blue agents recorded in a run trace."""
    manifest = load_manifest(run_dir)
    return sorted(
        agent["agent_id"]
        for agent in manifest.get("agents", [])
        if agent.get("side") == "blue"
    )


def reconstruct_blue_history(run_dir: str, blue_agent_id: str) -> Dict[str, Any]:
    """Build one blue agent's history from transition and event trace files."""
    paths = trace_paths(run_dir)
    manifest = load_manifest(run_dir)
    transitions = load_jsonl(paths["transitions"])
    events = load_jsonl(paths["events"])

    blue_transitions = [
        row for row in transitions if row.get("agent_id") == blue_agent_id
    ]
    if not blue_transitions:
        raise ValueError(f"No transition rows found for blue agent '{blue_agent_id}'")

    red_state_by_step: Dict[str, Dict[int, Dict[str, Any]]] = defaultdict(dict)
    for row in transitions:
        if row.get("side") != "red":
            continue
        state_after = row.get("state_after") or {}
        red_state_by_step[row["agent_id"]][int(row["step"])] = state_after

    path = []
    actions = []
    rewards = []
    observations_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    predictions_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    prediction_errors_by_target: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    targets = []
    first_step = min(int(item["step"]) for item in blue_transitions)

    for row in blue_transitions:
        step = int(row["step"])
        state_before = row.get("state_before") or {}
        state_after = row.get("state_after") or {}
        if step == first_step and state_before:
            path.append(
                {
                    "step": step,
                    "phase": "before",
                    "x": state_before.get("x"),
                    "y": state_before.get("y"),
                    "vx": state_before.get("vx"),
                    "vy": state_before.get("vy"),
                    "speed": state_before.get("speed"),
                }
            )
        path.append(
            {
                "step": step,
                "phase": "after",
                "x": state_after.get("x"),
                "y": state_after.get("y"),
                "vx": state_after.get("vx"),
                "vy": state_after.get("vy"),
                "speed": state_after.get("speed"),
            }
        )
        actions.append({"step": step, "action": row.get("action")})
        rewards.append(
            {
                "step": step,
                "reward": row.get("reward"),
                "terminated": row.get("terminated"),
                "truncated": row.get("truncated"),
            }
        )

    for event in events:
        if event.get("source_agent_id") != blue_agent_id:
            continue

        event_type = event.get("event_type")
        target_id = event.get("target_agent_id")
        payload = event.get("payload") or {}
        step = int(event.get("step", 0))

        if event_type == "observation" and target_id:
            observations_by_target[target_id].append(
                {
                    "step": step,
                    "x": payload.get("x"),
                    "y": payload.get("y"),
                    "distance": event.get("distance"),
                    "timestamp": payload.get("timestamp"),
                }
            )
        elif event_type == "prediction" and target_id:
            horizon = int(payload.get("horizon", 1))
            predicted_for_step = int(payload.get("prediction_for_step", step + horizon))
            prediction = {
                "step": step,
                "prediction_for_step": predicted_for_step,
                "horizon": horizon,
                "x": payload.get("x"),
                "y": payload.get("y"),
            }
            predictions_by_target[target_id].append(prediction)

            actual_state = red_state_by_step.get(target_id, {}).get(predicted_for_step)
            if actual_state and actual_state.get("x") is not None and actual_state.get("y") is not None:
                error = float(
                    np.linalg.norm(
                        np.array([payload.get("x"), payload.get("y")], dtype=np.float32)
                        - np.array([actual_state.get("x"), actual_state.get("y")], dtype=np.float32)
                    )
                )
                prediction_errors_by_target[target_id].append(
                    {
                        "step": step,
                        "prediction_for_step": predicted_for_step,
                        "horizon": horizon,
                        "error": error,
                        "predicted_x": payload.get("x"),
                        "predicted_y": payload.get("y"),
                        "actual_x": actual_state.get("x"),
                        "actual_y": actual_state.get("y"),
                    }
                )
        elif event_type == "target":
            targets.append(
                {
                    "step": step,
                    "x": payload.get("x"),
                    "y": payload.get("y"),
                }
            )

    return {
        "run_dir": run_dir,
        "manifest": {
            "schema_version": manifest.get("schema_version"),
            "mode": manifest.get("mode"),
            "config_path": manifest.get("config_path"),
            "started_at": manifest.get("started_at"),
            "finished_at": manifest.get("finished_at"),
            "duration_seconds": manifest.get("duration_seconds"),
            "num_steps": manifest.get("num_steps"),
        },
        "blue_agent_id": blue_agent_id,
        "path": path,
        "actions": actions,
        "rewards": rewards,
        "observations_by_target": dict(observations_by_target),
        "predictions_by_target": dict(predictions_by_target),
        "prediction_errors_by_target": dict(prediction_errors_by_target),
        "targets": targets,
    }


def summarize_blue_history(history: Mapping[str, Any]) -> Dict[str, Any]:
    """Compute compact descriptive stats for one blue-agent history."""
    observations = history.get("observations_by_target", {})
    predictions = history.get("predictions_by_target", {})
    prediction_errors = history.get("prediction_errors_by_target", {})

    error_summary = {}
    for target_id, rows in prediction_errors.items():
        errors = [row["error"] for row in rows if row.get("error") is not None]
        if not errors:
            continue
        error_summary[target_id] = {
            "count": len(errors),
            "mean": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "max": float(np.max(errors)),
        }

    return {
        "blue_agent_id": history.get("blue_agent_id"),
        "path_points": len(history.get("path", [])),
        "actions": len(history.get("actions", [])),
        "targets_observed": {
            target_id: len(rows) for target_id, rows in sorted(observations.items())
        },
        "targets_predicted": {
            target_id: len(rows) for target_id, rows in sorted(predictions.items())
        },
        "prediction_error": error_summary,
    }


def _rows_for_target(history: Mapping[str, Any], target_id: str) -> tuple[list, list, list]:
    observations = history.get("observations_by_target", {}).get(target_id, [])
    predictions = history.get("predictions_by_target", {}).get(target_id, [])
    errors = history.get("prediction_errors_by_target", {}).get(target_id, [])
    if not observations and not predictions:
        raise ValueError(
            f"No observations or predictions for {history.get('blue_agent_id')} -> {target_id}"
        )
    return observations, predictions, errors


def plot_coordinates(history: Mapping[str, Any], target_id: str, output_dir: str) -> str:
    """Plot observed target coordinates and predicted coordinates over time."""
    observations, predictions, _ = _rows_for_target(history, target_id)
    os.makedirs(output_dir, exist_ok=True)
    blue_id = history["blue_agent_id"]
    path = os.path.join(output_dir, f"{blue_id}_observing_{target_id}_coordinates.png")

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    if observations:
        steps = [row["step"] for row in observations]
        axes[0].plot(steps, [row["x"] for row in observations], "ro-", label="Observed X")
        axes[1].plot(steps, [row["y"] for row in observations], "ro-", label="Observed Y")
    if predictions:
        pred_steps = [row["prediction_for_step"] for row in predictions]
        axes[0].plot(pred_steps, [row["x"] for row in predictions], "bo", label="Predicted X")
        axes[1].plot(pred_steps, [row["y"] for row in predictions], "bo", label="Predicted Y")

    axes[0].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("step")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle(f"{blue_id} observing {target_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_prediction_error(history: Mapping[str, Any], target_id: str, output_dir: str) -> Optional[str]:
    """Plot prediction error for one blue/target pair."""
    _, _, errors = _rows_for_target(history, target_id)
    if not errors:
        return None

    os.makedirs(output_dir, exist_ok=True)
    blue_id = history["blue_agent_id"]
    path = os.path.join(output_dir, f"{blue_id}_observing_{target_id}_prediction_error.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        [row["prediction_for_step"] for row in errors],
        [row["error"] for row in errors],
        "go-",
        label="Prediction error",
    )
    ax.set_xlabel("predicted step")
    ax.set_ylabel("Euclidean error")
    ax.set_title(f"{blue_id} prediction error for {target_id}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_trajectory(history: Mapping[str, Any], output_dir: str) -> str:
    """Plot one blue agent's path through the environment."""
    os.makedirs(output_dir, exist_ok=True)
    blue_id = history["blue_agent_id"]
    path = os.path.join(output_dir, f"{blue_id}_trajectory.png")
    rows = [row for row in history.get("path", []) if row.get("x") is not None and row.get("y") is not None]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([row["x"] for row in rows], [row["y"] for row in rows], "b.-", label=blue_id)
    if rows:
        ax.plot(rows[0]["x"], rows[0]["y"], "go", label="start")
        ax.plot(rows[-1]["x"], rows[-1]["y"], "ro", label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{blue_id} trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def generate_plots(
    history: Mapping[str, Any],
    *,
    target_id: Optional[str],
    output_dir: str,
    plot_type: str,
) -> List[str]:
    """Generate selected plots and return paths."""
    written = []
    if plot_type in ("trajectory", "all"):
        written.append(plot_trajectory(history, output_dir))

    target_ids: Iterable[str]
    if target_id:
        target_ids = [target_id]
    else:
        target_ids = sorted(
            set(history.get("observations_by_target", {}))
            | set(history.get("predictions_by_target", {}))
        )

    if plot_type in ("coordinates", "all"):
        for target in target_ids:
            written.append(plot_coordinates(history, target, output_dir))
    if plot_type in ("prediction-error", "all"):
        for target in target_ids:
            error_path = plot_prediction_error(history, target, output_dir)
            if error_path is not None:
                written.append(error_path)

    return written


def write_json(path: str, payload: Mapping[str, Any]) -> None:
    """Write JSON payload to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(payload), f, indent=2, sort_keys=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze blue-agent history from an ACN run trace.")
    parser.add_argument("--run-dir", required=True, help="Path to a completed ACN run directory.")
    parser.add_argument("--blue-agent", help="Blue agent id to inspect, for example blue_0.")
    parser.add_argument("--target", help="Optional target red agent id, for example red_30.")
    parser.add_argument(
        "--plot",
        choices=["none", "trajectory", "coordinates", "prediction-error", "all"],
        default="none",
        help="Plot type to generate.",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated plots/exports. Defaults to <run-dir>/analysis.",
    )
    parser.add_argument(
        "--export-history",
        action="store_true",
        help="Write reconstructed blue-agent history JSON.",
    )
    parser.add_argument(
        "--list-blue-agents",
        action="store_true",
        help="List available blue agents and exit.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_blue_agents:
        for agent_id in list_blue_agents(args.run_dir):
            print(agent_id)
        return 0

    if not args.blue_agent:
        parser.error("--blue-agent is required unless --list-blue-agents is used")

    output_dir = args.output_dir or os.path.join(args.run_dir, "analysis")
    history = reconstruct_blue_history(args.run_dir, args.blue_agent)
    summary = summarize_blue_history(history)
    print(json.dumps(to_jsonable(summary), indent=2, sort_keys=True))

    if args.export_history:
        output_path = os.path.join(output_dir, f"{args.blue_agent}_history.json")
        write_json(output_path, history)
        print(f"Wrote history: {output_path}")

    if args.plot != "none":
        paths = generate_plots(
            history,
            target_id=args.target,
            output_dir=output_dir,
            plot_type=args.plot,
        )
        for path in paths:
            print(f"Wrote plot: {path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
