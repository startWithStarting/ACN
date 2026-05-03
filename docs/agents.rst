Agents
======

ACN supports two primary agent types: Blue (defensive) and Red (mobile).

Base Agent
----------

All agents inherit from ``BaseAgent`` which provides:

* Position tracking (x, y)
* Movement (speed, direction)
* Communication capabilities (bandwidth, processing capability)
* Agent type identification

.. autoclass:: src.agents.base_agent.BaseAgent
   :members:
   :undoc-members:

Blue Agents
----------

Blue agents are defensive units equipped with Vector Auto-Regressive (VAR)
prediction models to forecast red agent movements.

.. autoclass:: src.agents.blue_agent.BlueAgent
   :members:
   :undoc-members:

Blue Agent Strategies
~~~~~~~~~~~~~~~~~~~~~

Blue agents can use different movement strategies:

**Static Strategy** (``static``)
   Blue agent remains stationary but continues tracking and predicting red agents.

**Pursuit Strategy** (``pursuit``)
   Blue agent moves toward the average predicted position of detected red agents.

.. automodule:: src.agents.blue_strategies.static_strategy
.. automodule:: src.agents.blue_strategies.pursuit_strategy

Red Agents
----------

Red agents are mobile units with configurable movement behaviors.

.. autoclass:: src.agents.red_agent.RedAgent
   :members:
   :undoc-members:

Red Agent Strategies
~~~~~~~~~~~~~~~~~~~~

Red agents support multiple strategies:

**Center Strategy** (``center``)
   Move toward grid center, maintaining minimum distance of 10 units.

**Avoidant Strategy** (``avoidant``)
   Detect blue agents and steer away from them.

**Aggressive Strategy** (``aggressive``)
   Detect and pursue blue agents, prioritizing the closest ones.

**Team Strategy** (``team``)
   Move toward the average position of visible red teammates.

**Flocking Strategy** (``flocking``)
   Complex behavior combining cohesion, alignment, and separation.

.. automodule:: src.agents.strategies.red_agent_strategy
.. automodule:: src.agents.strategies.avoidant_red_strategy
.. automodule:: src.agents.strategies.aggressive_red_strategy
.. automodule:: src.agents.strategies.team_based_red_strategy
.. automodule:: src.agents.strategies.flocking_red_strategy

Agent Factory
------------

The ``AgentFactory`` creates agents from configuration:

.. autoclass:: src.agents.factory.AgentFactory
   :members:

Agent Registry
-------------

The registry provides dynamic strategy lookup:

.. autoclass:: src.agents.registry.AgentRegistry
   :members: