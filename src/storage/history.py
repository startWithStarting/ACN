"""Postgres-backed run-history recorder."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, Mapping, Optional

from src.storage.postgres import (
    connect,
    init_db,
    insert_agent,
    insert_event,
    insert_run,
    insert_transition,
    update_run_completion,
)
from src.utils.history import (
    TRACE_SCHEMA_VERSION,
    agent_metadata_rows,
    iter_agent_transition_rows,
    iter_blue_event_rows,
    snapshot_agents,
    to_jsonable,
)
from src.utils.logger import get_logger

logger = get_logger("acn.storage.history")


class PostgresRunHistoryRecorder:
    """Persist run history directly to Postgres without writing JSON trace files."""

    def __init__(
        self,
        *,
        run_id: str,
        config: Mapping[str, Any],
        mode: str,
        config_path: Optional[str] = None,
        database_url: Optional[str] = None,
    ):
        self.enabled = True
        self.run_id = run_id
        self.results_dir = None
        self._config = to_jsonable(config)
        self._conn = connect(database_url)
        init_db(self._conn)
        insert_run(
            self._conn,
            run_id=self.run_id,
            run_dir=None,
            experiment_name=self._config.get("experiment_name"),
            mode=mode,
            config_path=config_path,
            config=self._config,
            schema_version=TRACE_SCHEMA_VERSION,
            started_at=datetime.now().isoformat(),
            status="running",
            replace=False,
        )
        self._conn.commit()
        logger.info("DB trace persistence enabled for run_id={}", self.run_id)

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def register_agents(self, agents: Iterable[Any]) -> None:
        if not self.enabled or self._conn is None:
            return
        for agent in agent_metadata_rows(agents):
            insert_agent(self._conn, run_id=self.run_id, agent=to_jsonable(agent))
        self._conn.commit()

    def snapshot_agents(self, agent_objects: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
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
        if not self.enabled or self._conn is None:
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
            insert_transition(self._conn, run_id=self.run_id, row=to_jsonable(row))
        self._conn.commit()

    def record_blue_events(
        self,
        *,
        episode: int,
        step: int,
        observations: Mapping[str, Any],
        agent_objects: Mapping[str, Any],
    ) -> None:
        if not self.enabled or self._conn is None:
            return
        rows = iter_blue_event_rows(
            episode=episode,
            step=step,
            observations=observations,
            agent_objects=agent_objects,
        )
        for row in rows:
            insert_event(self._conn, run_id=self.run_id, row=to_jsonable(row))
        self._conn.commit()

    def finish(self, *, duration_seconds: float, num_steps: int, status: str = "completed") -> None:
        if not self.enabled or self._conn is None:
            return
        update_run_completion(
            self._conn,
            run_id=self.run_id,
            finished_at=datetime.now().isoformat(),
            duration_seconds=float(duration_seconds),
            num_steps=int(num_steps),
            status=status,
        )
        self._conn.commit()
        self.close()
