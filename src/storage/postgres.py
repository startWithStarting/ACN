"""Postgres storage primitives for ACN run traces."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, List, Optional

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


DEFAULT_DATABASE_URL = "postgresql://acn:acn@localhost:5432/acn"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    run_dir TEXT,
    experiment_name TEXT,
    mode TEXT,
    config_path TEXT,
    config_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    schema_version INTEGER,
    started_at TIMESTAMPTZ,
    finished_at TIMESTAMPTZ,
    duration_seconds DOUBLE PRECISION,
    num_steps INTEGER,
    status TEXT,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS agents (
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    agent_id TEXT NOT NULL,
    side TEXT,
    strategy_type TEXT,
    communication_bandwidth INTEGER,
    processing_capability INTEGER,
    detection_radius DOUBLE PRECISION,
    params_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (run_id, agent_id)
);

CREATE TABLE IF NOT EXISTS agent_transitions (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    agent_id TEXT NOT NULL,
    side TEXT,
    strategy_type TEXT,
    observation JSONB,
    action JSONB,
    reward DOUBLE PRECISION,
    terminated BOOLEAN,
    truncated BOOLEAN,
    next_observation JSONB,
    info JSONB,
    state_before JSONB,
    state_after JSONB,
    x_before DOUBLE PRECISION,
    y_before DOUBLE PRECISION,
    vx_before DOUBLE PRECISION,
    vy_before DOUBLE PRECISION,
    x_after DOUBLE PRECISION,
    y_after DOUBLE PRECISION,
    vx_after DOUBLE PRECISION,
    vy_after DOUBLE PRECISION
);

CREATE INDEX IF NOT EXISTS idx_agent_transitions_run_agent_step
    ON agent_transitions (run_id, agent_id, episode, step);
CREATE INDEX IF NOT EXISTS idx_agent_transitions_run_step
    ON agent_transitions (run_id, episode, step);

CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    episode INTEGER NOT NULL,
    step INTEGER NOT NULL,
    phase TEXT,
    event_type TEXT NOT NULL,
    source_agent_id TEXT,
    target_agent_id TEXT,
    receiver_agent_id TEXT,
    payload JSONB,
    scalar_value DOUBLE PRECISION,
    distance DOUBLE PRECISION,
    visible BOOLEAN,
    delivered BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_events_run_type_step
    ON events (run_id, event_type, episode, step);
CREATE INDEX IF NOT EXISTS idx_events_run_source_target
    ON events (run_id, source_agent_id, target_agent_id);

CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_artifacts_run_kind
    ON artifacts (run_id, kind);
"""


def get_database_url() -> str:
    """Return the configured Postgres connection URL."""
    return os.getenv("ACN_DATABASE_URL", DEFAULT_DATABASE_URL)


def connect(database_url: Optional[str] = None) -> psycopg.Connection:
    """Open a Postgres connection using dictionary rows."""
    return psycopg.connect(database_url or get_database_url(), row_factory=dict_row)


def init_db(conn: psycopg.Connection) -> None:
    """Create tables and indexes if they do not exist."""
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
    conn.commit()


def jsonb(value: Any) -> Jsonb:
    """Wrap a value for JSONB insertion."""
    return Jsonb(value if value is not None else None)


def state_value(row: Mapping[str, Any], state_key: str, field: str) -> Optional[float]:
    """Extract a numeric field from a transition state snapshot."""
    state = row.get(state_key) or {}
    value = state.get(field)
    return float(value) if value is not None else None


def insert_run(
    conn: psycopg.Connection,
    *,
    run_id: str,
    run_dir: Optional[str],
    experiment_name: Optional[str],
    mode: Optional[str],
    config_path: Optional[str],
    config: Mapping[str, Any],
    schema_version: Optional[int],
    started_at: Optional[str],
    finished_at: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    num_steps: Optional[int] = None,
    status: str = "running",
    replace: bool = False,
) -> None:
    """Insert run metadata."""
    with conn.cursor() as cur:
        if replace:
            cur.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
        cur.execute(
            """
            INSERT INTO runs (
                run_id, run_dir, experiment_name, mode, config_path,
                config_json, schema_version, started_at, finished_at,
                duration_seconds, num_steps, status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id) DO UPDATE
            SET run_dir = EXCLUDED.run_dir,
                experiment_name = EXCLUDED.experiment_name,
                mode = EXCLUDED.mode,
                config_path = EXCLUDED.config_path,
                config_json = EXCLUDED.config_json,
                schema_version = EXCLUDED.schema_version,
                started_at = EXCLUDED.started_at,
                finished_at = EXCLUDED.finished_at,
                duration_seconds = EXCLUDED.duration_seconds,
                num_steps = EXCLUDED.num_steps,
                status = EXCLUDED.status
            """,
            (
                run_id,
                run_dir,
                experiment_name,
                mode,
                config_path,
                jsonb(config or {}),
                schema_version,
                started_at,
                finished_at,
                duration_seconds,
                num_steps,
                status,
            ),
        )


def update_run_completion(
    conn: psycopg.Connection,
    *,
    run_id: str,
    finished_at: str,
    duration_seconds: float,
    num_steps: int,
    status: str,
) -> None:
    """Mark a run finished."""
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE runs
            SET finished_at = %s,
                duration_seconds = %s,
                num_steps = %s,
                status = %s
            WHERE run_id = %s
            """,
            (finished_at, duration_seconds, num_steps, status, run_id),
        )


def insert_agent(conn: psycopg.Connection, *, run_id: str, agent: Mapping[str, Any]) -> None:
    """Insert one agent metadata row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO agents (
                run_id, agent_id, side, strategy_type,
                communication_bandwidth, processing_capability,
                detection_radius, params_json
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (run_id, agent_id) DO UPDATE
            SET side = EXCLUDED.side,
                strategy_type = EXCLUDED.strategy_type,
                communication_bandwidth = EXCLUDED.communication_bandwidth,
                processing_capability = EXCLUDED.processing_capability,
                detection_radius = EXCLUDED.detection_radius,
                params_json = EXCLUDED.params_json
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


def insert_transition(conn: psycopg.Connection, *, run_id: str, row: Mapping[str, Any]) -> None:
    """Insert one agent transition row."""
    with conn.cursor() as cur:
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
                state_value(row, "state_before", "x"),
                state_value(row, "state_before", "y"),
                state_value(row, "state_before", "vx"),
                state_value(row, "state_before", "vy"),
                state_value(row, "state_after", "x"),
                state_value(row, "state_after", "y"),
                state_value(row, "state_after", "vx"),
                state_value(row, "state_after", "vy"),
            ),
        )


def insert_event(conn: psycopg.Connection, *, run_id: str, row: Mapping[str, Any]) -> None:
    """Insert one relational event row."""
    with conn.cursor() as cur:
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


def _add_filter(
    clauses: List[str],
    params: List[Any],
    column: str,
    value: Any,
    operator: str = "=",
) -> None:
    if value is not None:
        clauses.append(f"{column} {operator} %s")
        params.append(value)


def fetch_runs(conn: psycopg.Connection, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """Return run summaries."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id, run_dir, experiment_name, mode, config_path, schema_version,
                   started_at, finished_at, duration_seconds, num_steps, status, ingested_at
            FROM runs
            ORDER BY ingested_at DESC
            LIMIT %s OFFSET %s
            """,
            (limit, offset),
        )
        return list(cur.fetchall())


def fetch_run(conn: psycopg.Connection, run_id: str) -> Optional[Dict[str, Any]]:
    """Return one run row."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM runs WHERE run_id = %s", (run_id,))
        return cur.fetchone()


def fetch_agents(conn: psycopg.Connection, run_id: str) -> List[Dict[str, Any]]:
    """Return agents for a run."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT agent_id, side, strategy_type, communication_bandwidth,
                   processing_capability, detection_radius, params_json
            FROM agents
            WHERE run_id = %s
            ORDER BY agent_id
            """,
            (run_id,),
        )
        return list(cur.fetchall())


def fetch_transitions(
    conn: psycopg.Connection,
    run_id: str,
    agent_id: Optional[str] = None,
    episode: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    limit: int = 1000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return transition rows with basic filters."""
    clauses = ["run_id = %s"]
    params: List[Any] = [run_id]
    _add_filter(clauses, params, "agent_id", agent_id)
    _add_filter(clauses, params, "episode", episode)
    _add_filter(clauses, params, "step", start_step, ">=")
    _add_filter(clauses, params, "step", end_step, "<=")
    params.extend([limit, offset])
    query = f"""
        SELECT *
        FROM agent_transitions
        WHERE {' AND '.join(clauses)}
        ORDER BY episode, step, agent_id
        LIMIT %s OFFSET %s
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def fetch_events(
    conn: psycopg.Connection,
    run_id: str,
    event_type: Optional[str] = None,
    source_agent_id: Optional[str] = None,
    target_agent_id: Optional[str] = None,
    receiver_agent_id: Optional[str] = None,
    episode: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    limit: int = 1000,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Return event rows with basic filters."""
    clauses = ["run_id = %s"]
    params: List[Any] = [run_id]
    _add_filter(clauses, params, "event_type", event_type)
    _add_filter(clauses, params, "source_agent_id", source_agent_id)
    _add_filter(clauses, params, "target_agent_id", target_agent_id)
    _add_filter(clauses, params, "receiver_agent_id", receiver_agent_id)
    _add_filter(clauses, params, "episode", episode)
    _add_filter(clauses, params, "step", start_step, ">=")
    _add_filter(clauses, params, "step", end_step, "<=")
    params.extend([limit, offset])
    query = f"""
        SELECT *
        FROM events
        WHERE {' AND '.join(clauses)}
        ORDER BY episode, step, id
        LIMIT %s OFFSET %s
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        return list(cur.fetchall())


def fetch_trajectory(
    conn: psycopg.Connection,
    run_id: str,
    agent_id: str,
    episode: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Return one agent trajectory from transition state snapshots."""
    clauses = ["run_id = %s", "agent_id = %s"]
    params: List[Any] = [run_id, agent_id]
    _add_filter(clauses, params, "episode", episode)
    query = f"""
        SELECT episode, step, x_before, y_before, vx_before, vy_before,
               x_after, y_after, vx_after, vy_after, reward, terminated, truncated
        FROM agent_transitions
        WHERE {' AND '.join(clauses)}
        ORDER BY episode, step
    """
    with conn.cursor() as cur:
        cur.execute(query, params)
        rows = list(cur.fetchall())

    trajectory: List[Dict[str, Any]] = []
    if not rows:
        return trajectory

    first = rows[0]
    trajectory.append(
        {
            "episode": first["episode"],
            "step": first["step"],
            "phase": "before",
            "x": first["x_before"],
            "y": first["y_before"],
            "vx": first["vx_before"],
            "vy": first["vy_before"],
        }
    )
    for row in rows:
        trajectory.append(
            {
                "episode": row["episode"],
                "step": row["step"],
                "phase": "after",
                "x": row["x_after"],
                "y": row["y_after"],
                "vx": row["vx_after"],
                "vy": row["vy_after"],
                "reward": row["reward"],
                "terminated": row["terminated"],
                "truncated": row["truncated"],
            }
        )
    return trajectory


def insert_artifact(
    conn: psycopg.Connection,
    *,
    artifact_id: str,
    run_id: str,
    kind: str,
    path: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Insert artifact metadata and return the row."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO artifacts (artifact_id, run_id, kind, path, metadata_json)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (artifact_id) DO UPDATE
            SET kind = EXCLUDED.kind,
                path = EXCLUDED.path,
                metadata_json = EXCLUDED.metadata_json
            RETURNING *
            """,
            (artifact_id, run_id, kind, path, jsonb(metadata)),
        )
        row = cur.fetchone()
    conn.commit()
    return row


def fetch_artifact(conn: psycopg.Connection, artifact_id: str) -> Optional[Dict[str, Any]]:
    """Return one artifact row."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM artifacts WHERE artifact_id = %s", (artifact_id,))
        return cur.fetchone()
