"""Ingest local ACN trace directories into Postgres."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterator, Mapping, Optional

from src.storage.postgres import connect, init_db, jsonb


def iter_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Yield JSON rows from a JSONL file."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_manifest(run_dir: str) -> Dict[str, Any]:
    """Load a local trace manifest."""
    path = os.path.join(run_dir, "trace", "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Trace manifest not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def default_run_id(run_dir: str) -> str:
    """Derive a stable default run id from a run directory."""
    return os.path.basename(os.path.abspath(run_dir))


def _state_value(row: Mapping[str, Any], state_key: str, field: str) -> Optional[float]:
    state = row.get(state_key) or {}
    value = state.get(field)
    return float(value) if value is not None else None


def _experiment_name(manifest: Mapping[str, Any], run_dir: str) -> Optional[str]:
    config = manifest.get("config") or {}
    if config.get("experiment_name"):
        return config["experiment_name"]
    return os.path.basename(os.path.abspath(run_dir))


def ingest_run_dir(
    run_dir: str,
    *,
    database_url: Optional[str] = None,
    run_id: Optional[str] = None,
    replace: bool = True,
) -> Dict[str, Any]:
    """Load one local run trace into Postgres."""
    run_dir = os.path.abspath(run_dir)
    manifest = load_manifest(run_dir)
    run_id = run_id or manifest.get("run_id") or default_run_id(run_dir)
    transitions_path = os.path.join(run_dir, "trace", "agent_transitions.jsonl")
    events_path = os.path.join(run_dir, "trace", "events.jsonl")

    conn = connect(database_url)
    try:
        init_db(conn)
        with conn.transaction():
            with conn.cursor() as cur:
                if replace:
                    cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
                else:
                    cur.execute("SELECT 1 FROM runs WHERE run_id = %s", (run_id,))
                    if cur.fetchone():
                        raise ValueError(f"Run already exists: {run_id}")

                cur.execute(
                    """
                    INSERT INTO runs (
                        run_id, run_dir, experiment_name, mode, config_path,
                        config_json, schema_version, started_at, finished_at,
                        duration_seconds, num_steps, status
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        run_id,
                        run_dir,
                        _experiment_name(manifest, run_dir),
                        manifest.get("mode"),
                        manifest.get("config_path"),
                        jsonb(manifest.get("config") or {}),
                        manifest.get("schema_version"),
                        manifest.get("started_at"),
                        manifest.get("finished_at"),
                        manifest.get("duration_seconds"),
                        manifest.get("num_steps"),
                        manifest.get("status"),
                    ),
                )

                agent_count = 0
                for agent in manifest.get("agents", []):
                    cur.execute(
                        """
                        INSERT INTO agents (
                            run_id, agent_id, side, strategy_type,
                            communication_bandwidth, processing_capability,
                            detection_radius, params_json
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            agent.get("agent_id"),
                            agent.get("side"),
                            agent.get("strategy_type"),
                            agent.get("communication_bandwidth"),
                            agent.get("processing_capability"),
                            agent.get("detection_radius"),
                            jsonb(agent),
                        ),
                    )
                    agent_count += 1

                transition_count = 0
                for row in iter_jsonl(transitions_path):
                    cur.execute(
                        """
                        INSERT INTO agent_transitions (
                            run_id, episode, step, agent_id, side, strategy_type,
                            observation, action, reward, terminated, truncated,
                            next_observation, info, state_before, state_after,
                            x_before, y_before, vx_before, vy_before,
                            x_after, y_after, vx_after, vy_after
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s
                        )
                        """,
                        (
                            run_id,
                            row.get("episode"),
                            row.get("step"),
                            row.get("agent_id"),
                            row.get("side"),
                            row.get("strategy_type"),
                            jsonb(row.get("observation")),
                            jsonb(row.get("action")),
                            row.get("reward"),
                            row.get("terminated"),
                            row.get("truncated"),
                            jsonb(row.get("next_observation")),
                            jsonb(row.get("info")),
                            jsonb(row.get("state_before")),
                            jsonb(row.get("state_after")),
                            _state_value(row, "state_before", "x"),
                            _state_value(row, "state_before", "y"),
                            _state_value(row, "state_before", "vx"),
                            _state_value(row, "state_before", "vy"),
                            _state_value(row, "state_after", "x"),
                            _state_value(row, "state_after", "y"),
                            _state_value(row, "state_after", "vx"),
                            _state_value(row, "state_after", "vy"),
                        ),
                    )
                    transition_count += 1

                event_count = 0
                for row in iter_jsonl(events_path):
                    cur.execute(
                        """
                        INSERT INTO events (
                            run_id, episode, step, phase, event_type,
                            source_agent_id, target_agent_id, receiver_agent_id,
                            payload, scalar_value, distance, visible, delivered
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            run_id,
                            row.get("episode"),
                            row.get("step"),
                            row.get("phase"),
                            row.get("event_type"),
                            row.get("source_agent_id"),
                            row.get("target_agent_id"),
                            row.get("receiver_agent_id"),
                            jsonb(row.get("payload")),
                            row.get("scalar_value"),
                            row.get("distance"),
                            row.get("visible"),
                            row.get("delivered"),
                        ),
                    )
                    event_count += 1
        conn.commit()
        return {
            "run_id": run_id,
            "run_dir": run_dir,
            "agents": agent_count,
            "transitions": transition_count,
            "events": event_count,
        }
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest an ACN trace run into Postgres.")
    parser.add_argument("--run-dir", required=True, help="Path to an ACN run directory.")
    parser.add_argument("--run-id", help="Optional run id. Defaults to run directory basename.")
    parser.add_argument("--database-url", help="Postgres URL. Defaults to ACN_DATABASE_URL.")
    parser.add_argument(
        "--no-replace",
        action="store_true",
        help="Fail if the run id already exists instead of replacing it.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = ingest_run_dir(
        args.run_dir,
        database_url=args.database_url,
        run_id=args.run_id,
        replace=not args.no_replace,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
