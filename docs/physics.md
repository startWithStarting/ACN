# Physics Engine

ACN includes a custom physics package in `src/physics/` for more realistic
movement and interactions.

The default AEC and parallel environments now use this package for movement.
Agents are registered as physics bodies on reset, actions are converted into
velocity commands by default, and positions are synchronized back after each
physics step. The legacy direct-kinematic path remains available with
`environment.physics.enabled: false`.

## Overview

The physics package provides:

* Euler integration for position updates
* Drag and friction models
* Rigid body collision detection and response
* Configurable boundary handling
* Force field support
* Obstacle geometry helpers and collision response

## Core Components

| Component | Source | Purpose |
| --- | --- | --- |
| `PhysicsEngine` | [`engine.py`](../src/physics/engine.py) | Registers bodies, applies forces, steps motion, handles boundaries, and resolves body-body collisions. |
| `PhysicsBody` | [`engine.py`](../src/physics/engine.py) | Dataclass storing position, velocity, acceleration, mass, radius, restitution, and movement limits. |
| `BoundaryMode` | [`engine.py`](../src/physics/engine.py) | Enum for boundary handling: `clamp`, `bounce`, and `stop`. |
| `Obstacle` | [`obstacles.py`](../src/physics/obstacles.py) | Abstract geometry interface for obstacle containment, surface normals, and nearest points. |
| `ForceField` | [`fields.py`](../src/physics/fields.py) | Abstract interface for forces applied as a function of position. |

## Obstacles

Obstacle implementations provide geometry operations such as containment,
nearest-point lookup, surface normals, and distance-to-surface:

* `RectObstacle`: axis-aligned rectangular obstacle
* `CircleObstacle`: circular obstacle
* `create_obstacle(config)`: construct an obstacle from a config dictionary

`PhysicsEngine.add_obstacle()` stores obstacles and obstacle collision response
is applied during `PhysicsEngine.step()` when obstacle handling is enabled.

## Force Fields

Force field implementations apply continuous forces to agents within their
influence:

* `AttractorField`: pulls agents toward a center point
* `RepulsorField`: pushes agents away from a center point
* `FlowField`: applies constant force in one direction
* `RadialFlowField`: applies outward radial force from a center point
* `create_force_field(config)`: construct a force field from a config dictionary

## Usage Example

```python
import numpy as np

from src.physics.engine import PhysicsEngine

engine = PhysicsEngine(
    grid_width=100,
    grid_height=100,
    boundary_mode="bounce",
    default_drag=0.1,
)

engine.register_body(
    "agent1",
    position=np.array([50.0, 50.0]),
    velocity=np.array([1.0, 0.0]),
    mass=1.0,
    max_speed=5.0,
)

engine.apply_force("agent1", np.array([0.5, 0.0]))
engine.step(dt=1.0)

position = engine.get_position("agent1")
```

## Integration Status

The physics package is wired into `ACNEnvironmentLogic._apply_action()`.
Parallel environments queue all agent controls and advance physics once per
parallel step. AEC environments advance physics after the current agent's action.

Example environment configuration:

```yaml
environment:
  width: 100
  height: 80
  physics:
    enabled: true
    control_mode: "velocity"  # or "force" for inertial controls
    boundary_mode: "clamp"
    default_drag: 0.05
    default_radius: 0.5
    enable_collisions: true
    obstacles:
      - type: "rect"
        x: 40
        y: 30
        width: 10
        height: 20
    force_fields:
      - type: "flow"
        direction: [1.0, 0.0]
        strength: 0.1
```
