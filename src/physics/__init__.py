"""ACN Physics module.

Provides intermediate-level physics simulation:
- PhysicsBody dataclass and PhysicsEngine
- Obstacle system (RectObstacle, CircleObstacle)
- Force fields (AttractorField, RepulsorField, FlowField)
"""

from .engine import PhysicsEngine, PhysicsBody, BoundaryMode
from .obstacles import Obstacle, RectObstacle, CircleObstacle, create_obstacle
from .fields import ForceField, AttractorField, RepulsorField, FlowField, create_force_field

__all__ = [
    # Engine
    "PhysicsEngine",
    "PhysicsBody",
    "BoundaryMode",
    # Obstacles
    "Obstacle",
    "RectObstacle",
    "CircleObstacle",
    "create_obstacle",
    # Fields
    "ForceField",
    "AttractorField",
    "RepulsorField",
    "FlowField",
    "create_force_field",
]
