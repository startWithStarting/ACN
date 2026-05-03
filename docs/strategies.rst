Strategies Reference
====================

This section details the available strategies for both agent types.

Red Agent Strategies
--------------------

center
~~~~~~

Move toward grid center while maintaining minimum distance.

* **Behavior**: Move toward (50, 50) until within 10 units, then reverse
* **Speed**: Proportional to distance from center (capped at 5.0)
* **Use case**: Exploration or food-seeking scenarios

.. function:: src.agents.strategies.red_agent_strategy.center_based_movement_strategy

avoidant
~~~~~~~~

Detect and avoid blue agents.

* **Behavior**: Steering away from detected blue agents within range
* **Parameters**:
  - ``avoidance_radius``: Detection range for blue agents
  - ``avoidance_strength``: How strongly to steer away

.. function:: src.agents.strategies.avoidant_red_strategy

aggressive
~~~~~~~~~~

Pursue and intercept blue agents.

* **Behavior**: Move toward nearest detected blue agent
* **Parameters**:
  - ``pursuit_speed``: Speed when pursuing
  - ``detection_radius``: Range to detect blue agents

.. function:: src.agents.strategies.aggressive_red_strategy

team
~~~~

Flock with red teammates.

* **Behavior**: Move toward average position of visible red agents
* **Parameters**:
  - ``team_cohesion_weight``: Strength of attraction to team

.. function:: src.agents.strategies.team_based_red_strategy

flocking
~~~~~~~~

Full boids-style flocking behavior.

* **Parameters**:
  - ``cohesion_weight``: Attraction to flock center (default: 1.0)
  - ``alignment_weight``: Matching velocity (default: 1.0)
  - ``separation_weight``: Avoiding crowding (default: 1.5)
  - ``separation_radius``: Separation distance (default: 10)
  - ``max_speed``: Maximum velocity (default: 5.0)
  - ``max_force``: Maximum steering force (default: 0.5)
  - ``inertia_weight``: Resistance to direction change
  - ``wall_avoidance_weight``: Steering away from boundaries
  - ``wall_detection_radius``: Distance to walls for avoidance

.. function:: src.agents.strategies.flocking_red_strategy

Blue Agent Strategies
---------------------

static
~~~~~~

Remain stationary.

* **Behavior**: No movement, continues tracking
* **Use case**: Fixed surveillance positions

.. function:: src.agents.blue_strategies.static_strategy

pursuit
~~~~~~~

Move toward predicted red agent positions.

* **Behavior**: Move toward average of predicted red positions
* **Parameters**:
  - ``pursuit_speed``: Movement speed

.. function:: src.agents.blue_strategies.pursuit_strategy

Strategy Selection
-----------------

Select strategies via configuration:

.. code-block:: yaml

   red_agents:
     strategy: "flocking"
     params:
       cohesion_weight: 1.0

   blue_agents:
     strategy: "pursuit"