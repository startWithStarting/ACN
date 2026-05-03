Environment
===========

ACN provides a PettingZoo-compatible environment for multi-agent simulation.

Overview
--------

The environment implements the standard multi-agent RL interface where:

* Agents observe the environment state
* Agents select actions based on observations
* Environment returns rewards and next state

PettingZoo API
--------------

ACN supports both PettingZoo APIs:

1. **Parallel API**: All agents step simultaneously
2. **AEC API**: Alternating Agent-Environment-Communication cycles

Parallel Environment
--------------------

.. autoclass:: src.env.parallel_env.ParallelGameEnv
   :members:

AEC Environment
---------------

.. autoclass:: src.env.aec_env.AECGameEnv
   :members:

Common Logic
-----------

Shared logic between environments:

.. autoclass:: src.env.common_env_logic.ACNEnvironmentLogic
   :members:

Observation Building
--------------------

Observations are constructed using a builder pattern:

.. automodule:: src.env.observation
   :members:

The system provides specialized builders:

* ``BlueObservationBuilder``: For blue agents with red agent tracking
* ``RedObservationBuilder``: For red agents with team awareness
* ``FlockingObservationBuilder``: Extended builder with flocking parameters

Reward Functions
----------------

Rewards follow a composable protocol:

.. automodule:: src.env.rewards
   :members:

Available reward types:

* ``AttractorReward``: Ring-based scoring for red agents
* ``DistanceReward``: Distance-proportional rewards
* ``DetectionReward``: Detection-based rewards for blue agents
* ``CompositeReward``: Combines multiple reward functions

Action Space
------------

Agents can output continuous actions:

* **Direction**: 2D normalized vector (x, y)
* **Speed**: Scalar value [0, max_speed]

Observation Space
-----------------

Observations are dictionaries containing:

* ``position``: Agent's current position [x, y]
* ``grid_center``: Center of the simulation grid
* ``timestamp``: Current step count
* ``red_agents``: Dict of detected red agents (blue agents only)
* ``blue_agents``: Dict of detected blue agents (red agents only)
* ``red_teammates``: Dict of red team positions (red agents only)

Rendering
---------

The environment supports multiple render modes:

* ``human``: Default visualization
* ``human_matplotlib``: Matplotlib-based rendering
* ``human_matplotlib_pred``: Matplotlib with predictions
* ``human_pygame``: Pygame-based rendering