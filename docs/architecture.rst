Architecture Overview
=====================

ACN is organized into several key modules that work together to provide
a complete multi-agent simulation environment.

Directory Structure
-------------------

::

    src/
    ├── agents/           # Agent implementations and strategies
    │   ├── base_agent.py
    │   ├── blue_agent.py
    │   ├── red_agent.py
    │   ├── factory.py
    │   ├── registry.py
    │   ├── strategies/       # Red agent strategies
    │   └── blue_strategies/ # Blue agent strategies
    ├── env/              # PettingZoo environment
    │   ├── aec_env.py
    │   ├── parallel_env.py
    │   ├── common_env_logic.py
    │   ├── observation.py
    │   └── rewards.py
    ├── physics/          # Physics simulation
    │   ├── engine.py
    │   ├── obstacles.py
    │   └── fields.py
    ├── communication/    # Agent communication models
    ├── training/         # RL training framework
    ├── benchmark/        # Performance benchmarking
    └── utils/            # Utilities (logging, config, etc.)

Core Components
--------------

Agents
~~~~~~

The agent system uses a combination of inheritance and the strategy pattern:

* **BaseAgent**: Abstract base class defining common interface
* **BlueAgent**: Extends BaseAgent with VAR prediction models
* **RedAgent**: Extends BaseAgent with configurable strategies

Agent behaviors are implemented through strategies that can be swapped at runtime.
Red agents use the registry pattern to support different movement behaviors.

Environment
~~~~~~~~~~~

The environment follows the PettingZoo API with two implementations:

* **ParallelGameEnv**: All agents step simultaneously
* **AECGameEnv**: Alternating Agent-Environment-Communication cycles

Common logic is extracted into ``ACNEnvironmentLogic`` mixin to reduce duplication.

Physics Engine
~~~~~~~~~~~~~~

A custom physics engine provides:

* Euler integration for movement
* Drag and friction models
* Rigid body collisions with restitution
* Configurable boundary handling (clamp, bounce, stop)
* Force fields and obstacles

Design Patterns
---------------

1. **Strategy Pattern**: Agent behaviors are implemented as strategies
   registered via ``@register_strategy`` decorator

2. **Factory Pattern**: ``AgentFactory`` creates agents from configuration

3. **Builder Pattern**: ``ObservationBuilder`` constructs agent observations

4. **Registry Pattern**: Global registries for agent types and strategies

5. **Composition**: Reward functions compose multiple reward types

Extensibility
-------------

To add a new red agent strategy:

1. Create a new file in ``src/agents/strategies/``
2. Implement the strategy function with ``@register_strategy`` decorator
3. Use it in config by setting ``strategy: "your_strategy_name"``

To add a new reward function:

1. Implement ``RewardFunction`` protocol in ``src/env/rewards.py``
2. Add factory case in ``create_reward_function()``
3. Configure via YAML