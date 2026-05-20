# Agents

ACN supports two primary runtime agent types: blue defensive agents and red
mobile agents. The implementation lives in `src/agents/`.

## Base Agent

All agents inherit from `BaseAgent`
([source](../src/agents/base_agent.py)), which provides:

* Position tracking (`x`, `y`)
* Movement state (`speed`, `direction`)
* Communication metadata (`communication_bandwidth`, `processing_capability`)
* Agent type identification through `AgentType`
* Basic `send_message()` and `receive_message()` storage helpers

The base class defines the `choose_action()` and `get_observation()` interface
that concrete agents override.

## Blue Agents

`BlueAgent` ([source](../src/agents/blue_agent.py)) represents defensive units.
Blue agents track red-agent trajectories, fit a VAR-style prediction model, and
can move according to a selected blue strategy.

Blue agents currently support:

* Local detection checks through `is_within_detection_radius()`
* Red-agent path recording through `record_red_agent_movement()`
* Prediction fitting through `fit_prediction_model()`
* Future-position prediction through `predict_future_position()`
* Strategy dispatch through `choose_action()`

Blue observations include only red agents within the blue agent's
`detection_radius`. The trace recorder stores those detections separately from
privileged true state so offline training can preserve partial observability.

## Blue Strategies

Blue strategy implementations are in `src/agents/blue_strategies/`.

| Strategy | Selector | Source | Runtime behavior |
| --- | --- | --- | --- |
| Static | `static` | [`static_strategy.py`](../src/agents/blue_strategies/static_strategy.py) | Remain stationary while continuing to track visible red agents. |
| Pursuit | `pursuit` | [`pursuit_strategy.py`](../src/agents/blue_strategies/pursuit_strategy.py) | Move toward the smoothed mean of predicted visible red positions. |

## Red Agents

`RedAgent` ([source](../src/agents/red_agent.py)) represents mobile red-team
units. Red agents dispatch movement through `strategy_type`, using local
observations of blue agents and red teammates.

Red agents currently support:

* Detection checks through `is_within_detection_radius()`
* Teammate movement history through `record_teammate_movement()`
* Strategy dispatch through `choose_action()`
* A `trainable` placeholder mode for externally supplied policies

The `trainable` strategy returns a no-op if `choose_action()` is called directly;
it is intended to be controlled by an external training wrapper.

## Red Strategies

Red strategy implementations are in `src/agents/strategies/`.

| Strategy | Selector | Source | Runtime behavior |
| --- | --- | --- | --- |
| Center | `center` | [`red_agent_strategy.py`](../src/agents/strategies/red_agent_strategy.py) | Move toward the grid center while maintaining a minimum distance. |
| Avoidant | `avoidant` | [`avoidant_red_strategy.py`](../src/agents/strategies/avoidant_red_strategy.py) | Blend center seeking with steering away from visible blue agents. |
| Aggressive | `aggressive` | [`aggressive_red_strategy.py`](../src/agents/strategies/aggressive_red_strategy.py) | Blend center seeking with pursuit of the nearest visible blue agent. |
| Team | `team` | [`team_based_red_strategy.py`](../src/agents/strategies/team_based_red_strategy.py) | Move toward visible red teammates while avoiding blue agents. |
| Flocking | `flocking` | [`flocking_red_strategy.py`](../src/agents/strategies/flocking_red_strategy.py) | Use cohesion, alignment, separation, inertia, and wall avoidance. |
| Trainable | `trainable` | [`red_agent.py`](../src/agents/red_agent.py) | Placeholder for external policies; direct calls return no movement. |

## Agent Factory

The factory function `create_agents_from_config()`
([source](../src/agents/factory.py)) expands YAML agent-group definitions into
concrete `BlueAgent` and `RedAgent` instances.

The factory reads:

* `agents.blue_agents[*].count`
* `agents.blue_agents[*].strategy_type`
* `agents.red_agents[*].count`
* `agents.red_agents[*].strategy_type`
* Shared detection, communication, prediction, and processing settings

It currently constructs the built-in blue/red classes directly. If a new agent
class must be selectable from config, update the factory schema and construction
logic.

## Agent Registry

The registry module ([source](../src/agents/registry.py)) provides
decorator-based registration helpers:

* `register_agent(agent_type)`: register an agent class by string name
* `register_strategy(name, side)`: register a red or blue strategy function
* `create_agent(agent_type, **kwargs)`: instantiate a registered agent type
* `get_strategy(name, side)`: retrieve a registered strategy function
* `list_agent_types()`: list registered agent type names
* `list_strategies(side=None)`: list registered strategies

Built-in strategies are registered at import time. The current red/blue
`choose_action()` implementations still use explicit dispatch logic, so adding a
strategy also requires wiring it into the relevant agent class unless strategy
dispatch is refactored to call `get_strategy()` directly.
