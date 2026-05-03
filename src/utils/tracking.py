"""Experiment tracking for ACN.

This module provides experiment tracking with support for multiple backends:
- LocalTracker: JSONL + CSV to results directory (no dependencies)
- WandBTracker: Weights & Biases (lazy import)
- MLflowTracker: MLflow (lazy import)
- CompositeTracker: Delegate to multiple trackers

Usage:
    # Create tracker from config
    tracker = create_tracker(config.get("tracking", []), results_dir)

    # Use in training loop
    tracker.log_episode(episode_data)
    tracker.finish()
"""

import os
import json
import csv
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger("acn.utils.tracking")


class ExperimentTracker(ABC):
    """Abstract base class for experiment trackers."""

    @abstractmethod
    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics at a given step."""
        pass

    @abstractmethod
    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Log episode-level data."""
        pass

    @abstractmethod
    def finish(self) -> None:
        """Finalize tracking (flush buffers, close connections, etc.)."""
        pass


@dataclass
class EpisodeData:
    """Standardized episode data format."""
    episode: int
    total_steps: int
    red_team_avg_reward: float
    blue_team_avg_reward: float
    red_detections: int
    red_scores: int
    per_agent_rewards: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class LocalTracker(ExperimentTracker):
    """Local file-based tracker (JSONL + CSV). No external dependencies."""

    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        self.episodes_path = os.path.join(results_dir, "episodes.jsonl")
        self.metrics_path = os.path.join(results_dir, "metrics.csv")

        # Ensure directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Initialize CSV with headers if it doesn't exist
        if not os.path.exists(self.metrics_path):
            with open(self.metrics_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "metric", "value"])

        logger.info("LocalTracker initialized at: {}", results_dir)

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        """Log metrics to CSV."""
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    writer.writerow([step, key, value])

    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        """Log episode to JSONL."""
        # Normalize to EpisodeData
        if isinstance(episode_data, dict):
            # Ensure required fields exist
            episode_data.setdefault("timestamp", datetime.now().isoformat())
            ep = EpisodeData(**episode_data)
            data = asdict(ep)
        else:
            data = episode_data

        with open(self.episodes_path, "a") as f:
            f.write(json.dumps(data) + "\n")

        logger.debug("Logged episode {}", data.get("episode", "?"))

    def finish(self) -> None:
        """Finalize - nothing to do for LocalTracker."""
        logger.info("LocalTracker finished. Episodes: {}", self.episodes_path)


class WandBTracker(ExperimentTracker):
    """Weights & Biases tracker (lazy import)."""

    def __init__(self, results_dir: str, project: str = "acn-experiments", **kwargs):
        self.results_dir = results_dir
        self.project = project
        self.kwargs = kwargs
        self._run = None

    def _ensure_init(self):
        """Lazy import and init."""
        if self._run is not None:
            return

        try:
            import wandb
            wandb.init(project=self.project, dir=self.results_dir, **self.kwargs)
            self._run = wandb
            logger.info("WandBTracker initialized: {}", self.project)
        except ImportError:
            logger.warning("wandb not installed, falling back to no-op")
            self._run = None

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        self._ensure_init()
        if self._run:
            self._run.log({**metrics, "step": step})

    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        self._ensure_init()
        if self._run:
            self._run.log(episode_data)

    def finish(self) -> None:
        if self._run:
            self._run.finish()


class MLflowTracker(ExperimentTracker):
    """MLflow tracker (lazy import)."""

    def __init__(self, results_dir: str, experiment_name: str = "acn", **kwargs):
        self.results_dir = results_dir
        self.experiment_name = experiment_name
        self.kwargs = kwargs
        self._client = None

    def _ensure_init(self):
        if self._client is not None:
            return

        try:
            import mlflow
            mlflow.set_experiment(self.experiment_name)
            mlflow.start_run()
            self._client = mlflow
            logger.info("MLflowTracker initialized: {}", self.experiment_name)
        except ImportError:
            logger.warning("mlflow not installed, falling back to no-op")
            self._client = None

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        self._ensure_init()
        if self._client:
            self._client.log_metrics(metrics, step=step)

    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        self._ensure_init()
        if self._client:
            self._client.log_dict(episode_data, "episode.json")

    def finish(self) -> None:
        if self._client:
            self._client.end_run()


class CompositeTracker(ExperimentTracker):
    """Composite tracker that delegates to multiple trackers."""

    def __init__(self, trackers: List[ExperimentTracker]):
        self.trackers = trackers

    def log_metrics(self, metrics: Dict[str, Any], step: int) -> None:
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)

    def log_episode(self, episode_data: Dict[str, Any]) -> None:
        for tracker in self.trackers:
            tracker.log_episode(episode_data)

    def finish(self) -> None:
        for tracker in self.trackers:
            tracker.finish()


def create_tracker(config: List[Dict[str, Any]], results_dir: str) -> ExperimentTracker:
    """
    Factory function to create trackers from config.

    Args:
        config: List of tracker configs, e.g.:
            [
                {"type": "local"},
                {"type": "wandb", "project": "acn-experiments"}
            ]
        results_dir: Directory for local outputs

    Returns:
        An ExperimentTracker instance (CompositeTracker if multiple)
    """
    if not config:
        # Default to local only
        return LocalTracker(results_dir)

    trackers = []
    for tc in config:
        tracker_type = tc.get("type", "local")
        params = {k: v for k, v in tc.items() if k != "type"}

        if tracker_type == "local":
            trackers.append(LocalTracker(results_dir))
        elif tracker_type == "wandb":
            trackers.append(WandBTracker(results_dir, **params))
        elif tracker_type == "mlflow":
            trackers.append(MLflowTracker(results_dir, **params))
        else:
            logger.warning("Unknown tracker type: {}, skipping", tracker_type)

    if len(trackers) == 0:
        return LocalTracker(results_dir)
    elif len(trackers) == 1:
        return trackers[0]
    else:
        return CompositeTracker(trackers)


def snapshot_config(config: Dict[str, Any], results_dir: str) -> str:
    """
    Write config snapshot to results directory.

    Args:
        config: The resolved configuration dict
        results_dir: Directory to write snapshot

    Returns:
        Path to the snapshot file
    """
    import yaml

    # Compute hash of config for traceability
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:8]

    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "config_hash": config_hash,
        "config": config,
    }

    snapshot_path = os.path.join(results_dir, "config_snapshot.yaml")
    with open(snapshot_path, "w") as f:
        yaml.dump(snapshot, f, default_flow_style=False)

    logger.info("Config snapshot written to: {} (hash: {})", snapshot_path, config_hash)
    return snapshot_path
