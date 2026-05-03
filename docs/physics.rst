Physics Engine
==============

ACN includes a custom physics engine for realistic agent movement and interactions.

Overview
--------

The physics engine provides:

* Euler integration for position updates
* Drag and friction models
* Rigid body collision detection and response
* Configurable boundary handling
* Force field support
* Obstacle collision

Core Components
---------------

Physics Engine
~~~~~~~~~~~~~~

.. autoclass:: src.physics.engine.PhysicsEngine
   :members:

Physics Body
~~~~~~~~~~~

.. autoclass:: src.physics.engine.PhysicsBody
   :members:

Boundary Modes
~~~~~~~~~~~~~~

The engine supports three boundary handling modes:

* ``clamp``: Stop at boundaries (default)
* ``bounce``: Reflect velocity with restitution
* ``stop``: Zero velocity at boundaries

.. autoclass:: src.physics.engine.BoundaryMode
   :members:

Obstacles
---------

.. automodule:: src.physics.obstacles
   :members:

Obstacles provide collision boundaries that agents cannot pass through.

**RectObstacle**: Axis-aligned rectangular obstacle

**CircleObstacle**: Circular obstacle

Force Fields
------------

.. automodule:: src.physics.fields
   :members:

Force fields apply continuous forces to agents within their influence:

**AttractorField**: Pulls agents toward a center point

**RepulsorField**: Pushes agents away from a center point

**FlowField**: Applies constant force in a direction

Usage Example
------------

.. code-block:: python

   from src.physics.engine import PhysicsEngine, BoundaryMode

   engine = PhysicsEngine(
       grid_width=100,
       grid_height=100,
       boundary_mode="bounce",
       default_drag=0.1
   )

   # Register an agent body
   engine.register_body(
       "agent1",
       position=np.array([50.0, 50.0]),
       velocity=np.array([1.0, 0.0]),
       mass=1.0,
       max_speed=5.0
   )

   # Apply force and step
   engine.apply_force("agent1", np.array([0.5, 0.0]))
   engine.step(dt=1.0)

   # Get updated position
   pos = engine.get_position("agent1")