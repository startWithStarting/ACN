# Repository Guidelines

## Project Structure & Module Organization

ACN is a Python multi-agent simulation framework. Core code lives in `src/`: `agents/` contains blue/red agents, factories, registries, and strategies; `env/` contains PettingZoo AEC and parallel environments; `physics/` contains movement, collisions, obstacles, and fields; `communication/`, `training/`, `benchmark/`, and `utils/` hold supporting systems. Scenario YAML files are in `config/`. Tests are in `tests/`, docs are in `docs/`, and top-level entry points include `main.py`, `main_parallel.py`, `run.py`, and `run_tests.py`. Generated outputs such as logs and experiment results should remain out of source changes unless explicitly needed.

## Build, Test, and Development Commands

- `uv sync`: install project dependencies from `pyproject.toml` and `uv.lock`.
- `cp .env.example .env`: create local environment settings.
- `uv run python main.py`: run the default AEC simulation.
- `uv run python main.py --config config/aggressive_config.yaml`: run a specific scenario.
- `uv run python main_parallel.py --config config/experiment_config.yaml`: run parallel-environment mode.
- `uv run python run_tests.py`: run the unittest discovery suite.
- `uv run python -m pytest tests/test_physics.py -v`: run one test file with pytest.
- `uv run --with sphinx --with sphinx-rtd-theme --with myst-parser sphinx-build -b html docs docs/_build`: build local Sphinx docs.

## Coding Style & Naming Conventions

Use Python 3.9+ syntax, 4-space indentation, type hints for public functions, and Google-style docstrings for non-trivial APIs. Follow `snake_case` for functions, variables, modules, and YAML strategy names; use `PascalCase` for classes and enums. Keep lines near 100 characters. Prefer existing registry/factory patterns when adding agents or strategies.

## Testing Guidelines

Tests currently use `unittest` classes and `test_*` methods, with pytest available for targeted runs and coverage. Add or update tests in `tests/` for changes to physics, strategies, environment behavior, communication models, or configuration loading. Name files `test_<module>.py` and keep fixtures small and deterministic.

## Commit & Pull Request Guidelines

Recent commits use concise, imperative summaries such as `Fix syntax error...`, `Sync uv.lock...`, or `Add...`. Keep commits focused and include lockfile changes when dependencies change. Pull requests should describe the behavior changed, list tests run, link related issues, and include screenshots or plots when rendering, docs, or simulation output changes.

## Configuration & Agent Notes

Keep reusable scenarios in `config/*.yaml` and document new options in `docs/configuration.md` when they affect users. For new strategies, register them through `src/agents/registry.py` and include a minimal config example.
