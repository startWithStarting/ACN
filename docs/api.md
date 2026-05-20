# API Reference

This page is a curated Markdown reference for the public modules in ACN. It is
kept readable in standard Markdown previews; for implementation details, follow
the source links.

## Agents

| Module | Public API | Source |
| --- | --- | --- |
| `src.agents.base_agent` | `AgentType`, `CommsType`, `BaseAgent` | [`base_agent.py`](../src/agents/base_agent.py) |
| `src.agents.blue_agent` | `BlueAgent` | [`blue_agent.py`](../src/agents/blue_agent.py) |
| `src.agents.red_agent` | `RedAgent` | [`red_agent.py`](../src/agents/red_agent.py) |
| `src.agents.factory` | `create_agents_from_config()` | [`factory.py`](../src/agents/factory.py) |
| `src.agents.registry` | `register_agent()`, `register_strategy()`, `create_agent()`, `get_strategy()`, `list_agent_types()`, `list_strategies()` | [`registry.py`](../src/agents/registry.py) |

Strategy modules:

| Module | Public API | Source |
| --- | --- | --- |
| `src.agents.strategies.red_agent_strategy` | `center_based_movement_strategy()` | [`red_agent_strategy.py`](../src/agents/strategies/red_agent_strategy.py) |
| `src.agents.strategies.avoidant_red_strategy` | `avoidant_red_strategy()` | [`avoidant_red_strategy.py`](../src/agents/strategies/avoidant_red_strategy.py) |
| `src.agents.strategies.aggressive_red_strategy` | `aggressive_red_strategy()` | [`aggressive_red_strategy.py`](../src/agents/strategies/aggressive_red_strategy.py) |
| `src.agents.strategies.team_based_red_strategy` | `team_based_red_strategy()` | [`team_based_red_strategy.py`](../src/agents/strategies/team_based_red_strategy.py) |
| `src.agents.strategies.flocking_red_strategy` | `flocking_red_strategy()`, `limit_magnitude()` | [`flocking_red_strategy.py`](../src/agents/strategies/flocking_red_strategy.py) |
| `src.agents.blue_strategies.static_strategy` | `static_blue_strategy()` | [`static_strategy.py`](../src/agents/blue_strategies/static_strategy.py) |
| `src.agents.blue_strategies.pursuit_strategy` | `pursuit_blue_strategy()` | [`pursuit_strategy.py`](../src/agents/blue_strategies/pursuit_strategy.py) |

## Environment

| Module | Public API | Source |
| --- | --- | --- |
| `src.env.parallel_env` | `env()`, `ParallelGameEnv` | [`parallel_env.py`](../src/env/parallel_env.py) |
| `src.env.aec_env` | `env()`, `AECGameEnv` | [`aec_env.py`](../src/env/aec_env.py) |
| `src.env.common_env_logic` | `ACNEnvironmentLogic` | [`common_env_logic.py`](../src/env/common_env_logic.py) |
| `src.env.observation` | `ObservationBuilder`, `BlueObservationBuilder`, `RedObservationBuilder`, `FlockingObservationBuilder`, `create_observation_builder()` | [`observation.py`](../src/env/observation.py) |
| `src.env.rewards` | `RewardFunction`, `AttractorRewardConfig`, `AttractorReward`, `DistanceReward`, `DetectionReward`, `CompositeReward`, `create_reward_function()` | [`rewards.py`](../src/env/rewards.py) |

## Physics

| Module | Public API | Source |
| --- | --- | --- |
| `src.physics.engine` | `BoundaryMode`, `PhysicsBody`, `PhysicsEngine` | [`engine.py`](../src/physics/engine.py) |
| `src.physics.obstacles` | `Obstacle`, `RectObstacle`, `CircleObstacle`, `create_obstacle()` | [`obstacles.py`](../src/physics/obstacles.py) |
| `src.physics.fields` | `ForceField`, `AttractorField`, `RepulsorField`, `FlowField`, `RadialFlowField`, `create_force_field()` | [`fields.py`](../src/physics/fields.py) |

## Communication

| Module | Public API | Source |
| --- | --- | --- |
| `src.communication.models` | `CommunicationModel`, `GNNCommunicationModel`, `NoCommunicationModel` | [`models.py`](../src/communication/models.py) |

`GNNCommunicationModel` is a placeholder and currently raises
`NotImplementedError`. Runtime environments do not yet provide a message channel.

## Training

| Module | Public API | Source |
| --- | --- | --- |
| `src.training.trainer` | `RedTeamWrapper`, `Trainer` | [`trainer.py`](../src/training/trainer.py) |
| `src.training.base_trainer` | `BaseTrainer`, `SB3Trainer`, `RLlibTrainer`, `create_trainer()` | [`base_trainer.py`](../src/training/base_trainer.py) |

The current training path provides PPO integration for red-agent parameter
sharing. It does not yet implement CTDE, centralized critics, learned
communication, or opponent modeling.

## Benchmark

| Module | Public API | Source |
| --- | --- | --- |
| `src.benchmark.runner` | `Scenario`, `AlgorithmResult`, `BenchmarkRunner`, `run_benchmark()` | [`runner.py`](../src/benchmark/runner.py) |
| `src.benchmark.metrics` | `avg_episode_reward()`, `detection_rate()`, `red_score_rate()`, `communication_utility()`, `convergence_speed()`, `calculate_metrics()` | [`metrics.py`](../src/benchmark/metrics.py) |

## Analysis

| Module | Public API | Source |
| --- | --- | --- |
| `src.analysis.blue_history` | `reconstruct_blue_history()`, `summarize_blue_history()`, `generate_plots()` | [`blue_history.py`](../src/analysis/blue_history.py) |

The blue-history module can also be run as a CLI:

```bash
uv run python -m src.analysis.blue_history --run-dir results/... --blue-agent blue_0 --plot trajectory
```

## Trace API Service

| Module | Public API | Source |
| --- | --- | --- |
| `src.api.app` | FastAPI `app` with run, transition, event, trajectory, artifact, ingest, and plot endpoints | [`app.py`](../src/api/app.py) |
| `src.storage.ingest` | `ingest_run_dir()` and CLI ingest utility | [`ingest.py`](../src/storage/ingest.py) |
| `src.storage.history` | `PostgresRunHistoryRecorder` for direct `--persist` writes | [`history.py`](../src/storage/history.py) |
| `src.storage.postgres` | `init_db()`, query helpers, and artifact insertion | [`postgres.py`](../src/storage/postgres.py) |

Run the API stack:

```bash
docker compose up --build
```

Persist a simulation directly to Postgres:

```bash
uv run python run.py --mode parallel --config config/experiment_config.yaml --persist
```

Persisted runs use UUID run IDs and do not create local `trace/*.jsonl` files.

For file-backed runs, ingest from the API container's mounted results path:

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"run_dir": "/app/results/experiment/basic_comm_test_YYYYMMDD_HHMMSS_parallel"}'
```

Generate a plot artifact:

```bash
curl -X POST http://localhost:8000/runs/<run_id>/plots \
  -H "Content-Type: application/json" \
  -d '{"plot_type": "prediction_error", "agent_id": "blue_0", "target_agent_id": "red_10"}'
```

## Utilities

| Module | Public API | Source |
| --- | --- | --- |
| `src.utils.config_loader` | `load_config()` | [`config_loader.py`](../src/utils/config_loader.py) |
| `src.utils.logger` | `configure_logging()`, `get_logger()` | [`logger.py`](../src/utils/logger.py) |
| `src.utils.geometry` | `calculate_distance()`, `is_within_detection_radius()` | [`geometry.py`](../src/utils/geometry.py) |
| `src.utils.tracking` | `ExperimentTracker`, `EpisodeData`, `LocalTracker`, `WandBTracker`, `MLflowTracker`, `CompositeTracker`, `create_tracker()`, `snapshot_config()` | [`tracking.py`](../src/utils/tracking.py) |
| `src.utils.regressor` | `VectorAutoRegressor` | [`regressor.py`](../src/utils/regressor.py) |
| `src.utils.experiment` | `setup_experiment_results_dir()`, `save_prediction_plots()`, `save_timing_stats()`, `should_record_trace()`, `should_generate_prediction_plots()` | [`experiment.py`](../src/utils/experiment.py) |
| `src.utils.history` | `RunHistoryRecorder`, `create_history_recorder()`, `snapshot_agent()`, `to_jsonable()` | [`history.py`](../src/utils/history.py) |
