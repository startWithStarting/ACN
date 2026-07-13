# ACN Documentation

**Agent Communication Networks (ACN)** is a multi-agent simulation framework
designed for researching communication protocols and distributed control
mechanisms in autonomous agent systems.

![ACN simulation demo](media/acn_demo.gif)

## Overview

ACN provides a PettingZoo-based environment where:

* **Blue Agents** are defensive agents with VAR-based prediction models to track red agents
* **Red Agents** are mobile agents with configurable movement strategies
* Agents can be configured into partially observable pursuit/evasion and flocking scenarios

The framework supports both parallel and AEC (Agent Environment Cycle) API
modes. On top of the simulator, ACN implements a MARL-with-communication
benchmark stack: a communication runtime with engineered schemes
(`one_hop_direct`, `one_hop_mean`, `multihop_relay`) executed by the parallel
environment and a differentiable `multihop_gnn` scheme executed inside the
actor, a config-gated bearing-only blue sensor, benchmark reward modes, a
config-driven discrete action space, and a TorchRL MAPPO/IPPO trainer with
CTDE (trainable blue or red team) plus remote training on Modal. The AEC mode
does not support communication or the benchmark reward modes. The trainer is
implemented and smoke-tested; full benchmark training campaigns have not been
run yet.

## Quick Start

```bash
# Install dependencies
uv sync

# Run simulation
uv run python run.py --mode parallel

# Or specify a config
uv run python run.py --mode parallel --config config/aggressive_config.yaml

# Persist trace data directly to Postgres
uv run python run.py --mode parallel --config config/experiment_config.yaml --persist

# Train a team with the TorchRL MARL trainer (training.backend: "marl")
uv run python run.py --mode train --config config/benchmark_blue_mappo.yaml
```

## Contents

* [Installation](installation.md)
* [Google Cloud Setup](gcp_setup.md)
* [Architecture](architecture.md)
* [Agents](agents.md)
* [Environment](environment.md)
* [Physics](physics.md)
* [Configuration](configuration.md)
* [Communication And MARL Decision Log](communication_decision_log.md)
* [Communication Implementation Plan](communication_implementation_plan.md)
* [Technical Gap Analysis and Research Roadmap](technical_gap_analysis.md)
* [Strategies](strategies.md)
* [API Reference](api.md)
* [Development](development.md)
* [Remote Training on Modal](../infra/README.md)
