# Strategies Reference

Strategies define the built-in movement behavior for red and blue agents. The
strategy modules are ordinary Python functions, while agent classes perform the
runtime dispatch from `strategy_type`.

## Red Agent Strategies

### `center`

Source: [`red_agent_strategy.py`](../src/agents/strategies/red_agent_strategy.py)

Move toward the grid center while maintaining a minimum distance.

* Behavior: move toward `(50, 50)` until within 10 units, then reverse
* Speed: proportional to distance from center, capped at `5.0`
* Use case: simple exploration or food-seeking baseline

### `avoidant`

Source: [`avoidant_red_strategy.py`](../src/agents/strategies/avoidant_red_strategy.py)

Detect and avoid visible blue agents.

* Behavior: steer away from detected blue agents within range
* Key parameters:
  * `avoidance_radius`
  * `avoidance_strength`

### `aggressive`

Source: [`aggressive_red_strategy.py`](../src/agents/strategies/aggressive_red_strategy.py)

Pursue and intercept visible blue agents.

* Behavior: move toward the nearest detected blue agent
* Key parameters:
  * `pursuit_speed`
  * `detection_radius`

### `team`

Source: [`team_based_red_strategy.py`](../src/agents/strategies/team_based_red_strategy.py)

Coordinate with visible red teammates.

* Behavior: move toward the average position of visible red agents
* Key parameter:
  * `team_cohesion_weight`

The YAML selector is `strategy_type: "team"`. The registered function name is
currently `team_based`.

### `flocking`

Source: [`flocking_red_strategy.py`](../src/agents/strategies/flocking_red_strategy.py)

Use boids-style flocking behavior.

* Key parameters:
  * `cohesion_weight`
  * `alignment_weight`
  * `separation_weight`
  * `separation_radius`
  * `max_speed`
  * `max_force`
  * `inertia_weight`
  * `wall_avoidance_weight`
  * `wall_detection_radius`

## Blue Agent Strategies

### `static`

Source: [`static_strategy.py`](../src/agents/blue_strategies/static_strategy.py)

Remain stationary while continuing to observe, track, and predict red-agent
movement.

### `pursuit`

Source: [`pursuit_strategy.py`](../src/agents/blue_strategies/pursuit_strategy.py)

Move toward the average predicted position of detected red agents.

* Key parameter:
  * `pursuit_speed`

## Strategy Selection

Select strategies via configuration:

```yaml
agents:
  red_agents:
    - count: 8
      strategy_type: "flocking"
      cohesion_weight: 1.0
      alignment_weight: 1.0
      separation_weight: 1.5

  blue_agents:
    - count: 2
      strategy_type: "pursuit"
```

## Implementation Notes

The built-in `RedAgent` and `BlueAgent` classes dispatch strategies from their
`choose_action()` methods. The registry decorators populate `src.agents.registry`
for discovery and future dynamic dispatch, but adding a new strategy currently
also requires updating the relevant agent's dispatch logic or refactoring it to
call `get_strategy()`.
