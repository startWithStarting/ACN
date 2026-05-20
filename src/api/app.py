"""FastAPI service for ACN trace retrieval and plotting."""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.storage.ingest import ingest_run_dir
from src.storage.postgres import (
    connect,
    fetch_agents,
    fetch_artifact,
    fetch_events,
    fetch_run,
    fetch_runs,
    fetch_trajectory,
    fetch_transitions,
    init_db,
    insert_artifact,
)


app = FastAPI(
    title="ACN Trace API",
    version="0.1.0",
    description="Query ACN run traces, retrieve training records, and generate selected plots.",
)


def _artifact_output_dir(run: Dict[str, Any]) -> str:
    run_dir = run.get("run_dir")
    if run_dir:
        return os.path.join(run_dir, "api_artifacts")
    artifact_root = os.getenv("ACN_ARTIFACT_DIR", os.path.join("results", "api_artifacts"))
    return os.path.join(artifact_root, run["run_id"])


class IngestRequest(BaseModel):
    run_dir: str
    run_id: Optional[str] = None
    replace: bool = True


class PlotRequest(BaseModel):
    plot_type: str = Field(pattern="^(trajectory|coordinates|prediction_error)$")
    agent_id: str
    target_agent_id: Optional[str] = None
    episode: Optional[int] = None
    format: str = Field(default="png", pattern="^png$")


def _conn():
    conn = connect()
    init_db(conn)
    return conn


@app.get("/health")
def health() -> Dict[str, Any]:
    with _conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 AS ok")
            row = cur.fetchone()
    return {"status": "ok", "database": row["ok"]}


@app.post("/db/init")
def initialize_database() -> Dict[str, str]:
    with connect() as conn:
        init_db(conn)
    return {"status": "initialized"}


@app.post("/ingest")
def ingest_trace(request: IngestRequest) -> Dict[str, Any]:
    if not os.path.isdir(request.run_dir):
        raise HTTPException(status_code=404, detail=f"Run directory not found: {request.run_dir}")
    try:
        return ingest_run_dir(request.run_dir, run_id=request.run_id, replace=request.replace)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.get("/runs")
def list_runs(
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    with _conn() as conn:
        return fetch_runs(conn, limit=limit, offset=offset)


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> Dict[str, Any]:
    with _conn() as conn:
        row = fetch_run(conn, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return row


@app.get("/runs/{run_id}/config")
def get_run_config(run_id: str) -> Dict[str, Any]:
    row = get_run(run_id)
    return row.get("config_json") or {}


@app.get("/runs/{run_id}/agents")
def get_run_agents(run_id: str) -> List[Dict[str, Any]]:
    with _conn() as conn:
        if fetch_run(conn, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return fetch_agents(conn, run_id)


@app.get("/runs/{run_id}/transitions")
def get_transitions(
    run_id: str,
    agent_id: Optional[str] = None,
    episode: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    with _conn() as conn:
        if fetch_run(conn, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return fetch_transitions(
            conn,
            run_id,
            agent_id=agent_id,
            episode=episode,
            start_step=start_step,
            end_step=end_step,
            limit=limit,
            offset=offset,
        )


@app.get("/runs/{run_id}/events")
def get_events(
    run_id: str,
    event_type: Optional[str] = None,
    source_agent_id: Optional[str] = None,
    target_agent_id: Optional[str] = None,
    receiver_agent_id: Optional[str] = None,
    episode: Optional[int] = None,
    start_step: Optional[int] = None,
    end_step: Optional[int] = None,
    limit: int = Query(default=1000, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
) -> List[Dict[str, Any]]:
    with _conn() as conn:
        if fetch_run(conn, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return fetch_events(
            conn,
            run_id,
            event_type=event_type,
            source_agent_id=source_agent_id,
            target_agent_id=target_agent_id,
            receiver_agent_id=receiver_agent_id,
            episode=episode,
            start_step=start_step,
            end_step=end_step,
            limit=limit,
            offset=offset,
        )


@app.get("/runs/{run_id}/trajectory")
def get_trajectory(
    run_id: str,
    agent_id: str,
    episode: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with _conn() as conn:
        if fetch_run(conn, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        return fetch_trajectory(conn, run_id, agent_id, episode=episode)


@app.get("/runs/{run_id}/blue/{blue_agent_id}/history")
def get_blue_history(
    run_id: str,
    blue_agent_id: str,
    target_agent_id: Optional[str] = None,
    episode: Optional[int] = None,
) -> Dict[str, Any]:
    with _conn() as conn:
        if fetch_run(conn, run_id) is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
        observations = fetch_events(
            conn,
            run_id,
            event_type="observation",
            source_agent_id=blue_agent_id,
            target_agent_id=target_agent_id,
            episode=episode,
            limit=10000,
        )
        predictions = fetch_events(
            conn,
            run_id,
            event_type="prediction",
            source_agent_id=blue_agent_id,
            target_agent_id=target_agent_id,
            episode=episode,
            limit=10000,
        )
        targets = fetch_events(
            conn,
            run_id,
            event_type="target",
            source_agent_id=blue_agent_id,
            episode=episode,
            limit=10000,
        )
        trajectory = fetch_trajectory(conn, run_id, blue_agent_id, episode=episode)

    return {
        "run_id": run_id,
        "blue_agent_id": blue_agent_id,
        "trajectory": trajectory,
        "observations": observations,
        "predictions": predictions,
        "targets": targets,
        "summary": {
            "trajectory_points": len(trajectory),
            "observations": len(observations),
            "predictions": len(predictions),
            "targets": len(targets),
        },
    }


@app.post("/runs/{run_id}/plots")
def create_plot(run_id: str, request: PlotRequest) -> Dict[str, Any]:
    with _conn() as conn:
        run = fetch_run(conn, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

        output_dir = _artifact_output_dir(run)
        os.makedirs(output_dir, exist_ok=True)

        if request.plot_type == "trajectory":
            path = _plot_trajectory(conn, run_id, request.agent_id, request.episode, output_dir)
        elif request.plot_type == "coordinates":
            if not request.target_agent_id:
                raise HTTPException(status_code=422, detail="target_agent_id is required")
            path = _plot_coordinates(
                conn,
                run_id,
                request.agent_id,
                request.target_agent_id,
                request.episode,
                output_dir,
            )
        elif request.plot_type == "prediction_error":
            if not request.target_agent_id:
                raise HTTPException(status_code=422, detail="target_agent_id is required")
            path = _plot_prediction_error(
                conn,
                run_id,
                request.agent_id,
                request.target_agent_id,
                request.episode,
                output_dir,
            )
        else:
            raise HTTPException(status_code=422, detail=f"Unsupported plot type: {request.plot_type}")

        artifact_id = str(uuid.uuid4())
        artifact = insert_artifact(
            conn,
            artifact_id=artifact_id,
            run_id=run_id,
            kind=f"plot:{request.plot_type}",
            path=path,
            metadata={
                "agent_id": request.agent_id,
                "target_agent_id": request.target_agent_id,
                "episode": request.episode,
                "format": request.format,
            },
        )
    return artifact


@app.get("/artifacts/{artifact_id}")
def get_artifact(artifact_id: str) -> FileResponse:
    with _conn() as conn:
        artifact = fetch_artifact(conn, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    path = artifact["path"]
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"Artifact file missing: {path}")
    return FileResponse(path)


def _plot_trajectory(
    conn,
    run_id: str,
    agent_id: str,
    episode: Optional[int],
    output_dir: str,
) -> str:
    rows = fetch_trajectory(conn, run_id, agent_id, episode=episode)
    rows = [row for row in rows if row.get("x") is not None and row.get("y") is not None]
    if not rows:
        raise HTTPException(status_code=404, detail="No trajectory rows found")

    path = os.path.join(output_dir, f"{agent_id}_trajectory.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([row["x"] for row in rows], [row["y"] for row in rows], "b.-", label=agent_id)
    ax.plot(rows[0]["x"], rows[0]["y"], "go", label="start")
    ax.plot(rows[-1]["x"], rows[-1]["y"], "ro", label="end")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{agent_id} trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_coordinates(
    conn,
    run_id: str,
    agent_id: str,
    target_agent_id: str,
    episode: Optional[int],
    output_dir: str,
) -> str:
    observations = fetch_events(
        conn,
        run_id,
        event_type="observation",
        source_agent_id=agent_id,
        target_agent_id=target_agent_id,
        episode=episode,
        limit=10000,
    )
    predictions = fetch_events(
        conn,
        run_id,
        event_type="prediction",
        source_agent_id=agent_id,
        target_agent_id=target_agent_id,
        episode=episode,
        limit=10000,
    )
    if not observations and not predictions:
        raise HTTPException(status_code=404, detail="No observation or prediction rows found")

    path = os.path.join(output_dir, f"{agent_id}_observing_{target_agent_id}_coordinates.png")
    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    if observations:
        steps = [row["step"] for row in observations]
        axes[0].plot(steps, [_payload_float(row, "x") for row in observations], "ro-", label="Observed X")
        axes[1].plot(steps, [_payload_float(row, "y") for row in observations], "ro-", label="Observed Y")
    if predictions:
        pred_steps = [
            int((row["payload"] or {}).get("prediction_for_step", row["step"]))
            for row in predictions
        ]
        axes[0].plot(pred_steps, [_payload_float(row, "x") for row in predictions], "bo", label="Predicted X")
        axes[1].plot(pred_steps, [_payload_float(row, "y") for row in predictions], "bo", label="Predicted Y")

    axes[0].set_ylabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_xlabel("step")
    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()
    fig.suptitle(f"{agent_id} observing {target_agent_id}")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _plot_prediction_error(
    conn,
    run_id: str,
    agent_id: str,
    target_agent_id: str,
    episode: Optional[int],
    output_dir: str,
) -> str:
    errors = _prediction_errors(conn, run_id, agent_id, target_agent_id, episode)
    if not errors:
        raise HTTPException(status_code=404, detail="No matched prediction errors found")

    path = os.path.join(output_dir, f"{agent_id}_observing_{target_agent_id}_prediction_error.png")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        [row["prediction_for_step"] for row in errors],
        [row["error"] for row in errors],
        "go-",
        label="Prediction error",
    )
    ax.set_xlabel("predicted step")
    ax.set_ylabel("Euclidean error")
    ax.set_title(f"{agent_id} prediction error for {target_agent_id}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def _prediction_errors(
    conn,
    run_id: str,
    agent_id: str,
    target_agent_id: str,
    episode: Optional[int],
) -> List[Dict[str, Any]]:
    predictions = fetch_events(
        conn,
        run_id,
        event_type="prediction",
        source_agent_id=agent_id,
        target_agent_id=target_agent_id,
        episode=episode,
        limit=10000,
    )
    if not predictions:
        return []

    rows = fetch_transitions(
        conn,
        run_id,
        agent_id=target_agent_id,
        episode=episode,
        limit=10000,
    )
    state_by_key = {
        (row["episode"], row["step"]): row
        for row in rows
        if row.get("x_after") is not None and row.get("y_after") is not None
    }

    errors = []
    for prediction in predictions:
        payload = prediction["payload"] or {}
        predicted_for_step = int(payload.get("prediction_for_step", prediction["step"]))
        actual = state_by_key.get((prediction["episode"], predicted_for_step))
        if actual is None:
            continue
        pred = np.array([payload.get("x"), payload.get("y")], dtype=np.float32)
        truth = np.array([actual["x_after"], actual["y_after"]], dtype=np.float32)
        errors.append(
            {
                "episode": prediction["episode"],
                "step": prediction["step"],
                "prediction_for_step": predicted_for_step,
                "error": float(np.linalg.norm(pred - truth)),
                "predicted_x": float(pred[0]),
                "predicted_y": float(pred[1]),
                "actual_x": actual["x_after"],
                "actual_y": actual["y_after"],
            }
        )
    return errors


def _payload_float(row: Dict[str, Any], key: str) -> Optional[float]:
    payload = row.get("payload") or {}
    value = payload.get(key)
    return float(value) if value is not None else None
