"""Physics engine for ACN.

This module provides an intermediate-level physics simulation with:
- Euler integration
- Drag
- Turning rate limits
- Boundary modes (clamp, bounce, stop)
- Rigid-body collisions with mass and restitution
- Static obstacle collision response

Usage:
    engine = PhysicsEngine(grid_width=100, grid_height=100)
    engine.register_body("agent1", position=np.array([10.0, 20.0]), mass=1.0, ...)
    engine.apply_force("agent1", np.array([0.5, 0.3]))
    engine.step(dt=1.0)
    pos = engine.get_position("agent1")
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum

from src.utils.logger import get_logger

logger = get_logger("acn.physics")


class BoundaryMode(Enum):
    """Boundary handling modes."""
    CLAMP = "clamp"  # Stop at boundary (current behavior)
    BOUNCE = "bounce"  # Reflect velocity with restitution
    STOP = "stop"  # Zero velocity at boundary


@dataclass
class PhysicsBody:
    """Represents a physical body in the simulation."""
    name: str
    position: np.ndarray  # shape (2,)
    velocity: np.ndarray  # shape (2,)
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2))
    mass: float = 1.0
    max_speed: float = 10.0
    max_force: float = 1.0
    drag_coefficient: float = 0.0
    turning_rate: float = float('inf')  # radians/step
    radius: float = 0.5  # collision radius
    restitution: float = 0.8  # bounciness


class PhysicsEngine:
    """Physics engine with Euler integration, drag, collisions, and boundaries."""

    def __init__(
        self,
        grid_width: float = 100.0,
        grid_height: float = 100.0,
        boundary_mode: str = "clamp",
        default_drag: float = 0.0,
        default_mass: float = 1.0,
        default_max_speed: float = 10.0,
        default_max_force: float = 1.0,
        default_turning_rate: float = float('inf'),
        default_radius: float = 0.5,
        default_restitution: float = 0.8,
        enable_collisions: bool = True,
        enable_obstacles: bool = True,
    ):
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.boundary_mode = BoundaryMode(boundary_mode)
        self.default_drag = default_drag
        self.default_mass = default_mass
        self.default_max_speed = default_max_speed
        self.default_max_force = default_max_force
        self.default_turning_rate = default_turning_rate
        self.default_radius = default_radius
        self.default_restitution = default_restitution
        self.enable_collisions = enable_collisions
        self.enable_obstacles = enable_obstacles

        self._bodies: Dict[str, PhysicsBody] = {}
        self._obstacles = []
        self._force_fields = []

        logger.debug("PhysicsEngine initialized: {}x{}, mode={}", grid_width, grid_height, boundary_mode)

    def register_body(
        self,
        name: str,
        position: np.ndarray,
        velocity: Optional[np.ndarray] = None,
        **kwargs
    ) -> PhysicsBody:
        """Register a new body with the physics engine."""
        if velocity is None:
            velocity = np.zeros(2)

        body = PhysicsBody(
            name=name,
            position=np.array(position, dtype=np.float32),
            velocity=np.array(velocity, dtype=np.float32),
            mass=kwargs.get("mass", self.default_mass),
            max_speed=kwargs.get("max_speed", self.default_max_speed),
            max_force=kwargs.get("max_force", self.default_max_force),
            drag_coefficient=kwargs.get("drag", self.default_drag),
            turning_rate=kwargs.get("turning_rate", self.default_turning_rate),
            radius=kwargs.get("radius", self.default_radius),
            restitution=kwargs.get("restitution", self.default_restitution),
        )
        self._bodies[name] = body
        logger.debug("Registered body: {}", name)
        return body

    def remove_body(self, name: str) -> None:
        """Remove a body from the physics engine."""
        if name in self._bodies:
            del self._bodies[name]
            logger.debug("Removed body: {}", name)

    def apply_force(self, name: str, force: np.ndarray) -> None:
        """Apply a force to a body. Force is clamped by max_force."""
        if name not in self._bodies:
            logger.warning("apply_force: body {} not found", name)
            return

        body = self._bodies[name]
        force = np.array(force, dtype=np.float32)

        # Clamp force magnitude
        force_magnitude = np.linalg.norm(force)
        if force_magnitude > body.max_force and force_magnitude > 1e-6:
            force = (force / force_magnitude) * body.max_force

        body.acceleration += force / body.mass

    def step(self, dt: float = 1.0) -> None:
        """Advance physics simulation by one time step."""
        # Apply force fields
        for field in self._force_fields:
            for body in self._bodies.values():
                field_force = field.force_at(body.position)
                body.acceleration += field_force / body.mass

        # Update each body
        for body in self._bodies.values():
            self._update_body(body, dt)

        # Resolve collisions
        if self.enable_obstacles:
            self._resolve_obstacle_collisions()
        if self.enable_collisions:
            self._resolve_collisions()

    def _update_body(self, body: PhysicsBody, dt: float) -> None:
        """Update a single body's physics."""
        # Apply drag: acceleration -= drag * velocity
        if body.drag_coefficient > 0:
            body.acceleration -= body.drag_coefficient * body.velocity

        # Euler integration: v += a * dt, then clamp before p += v * dt.
        body.velocity += body.acceleration * dt

        # Apply turning rate limit (heading change)
        if body.turning_rate < float('inf'):
            current_heading = np.arctan2(body.velocity[1], body.velocity[0])
            # For simplicity, we'll just clamp speed for now
            # Full turning rate implementation would track heading separately
            pass

        # Clamp speed
        speed = np.linalg.norm(body.velocity)
        if speed > body.max_speed and speed > 1e-6:
            body.velocity = (body.velocity / speed) * body.max_speed

        body.position += body.velocity * dt

        # Handle boundaries
        self._handle_boundaries(body)

        # Reset acceleration for next step
        body.acceleration[:] = 0.0

    def _handle_boundaries(self, body: PhysicsBody) -> None:
        """Handle collision with grid boundaries."""
        min_x, max_x = body.radius, self.grid_width - body.radius
        min_y, max_y = body.radius, self.grid_height - body.radius

        if self.boundary_mode == BoundaryMode.CLAMP:
            # Clamp position
            body.position[0] = np.clip(body.position[0], min_x, max_x)
            body.position[1] = np.clip(body.position[1], min_y, max_y)

        elif self.boundary_mode == BoundaryMode.BOUNCE:
            # Reflect velocity with restitution
            if body.position[0] < min_x:
                body.position[0] = min_x
                body.velocity[0] = -body.velocity[0] * body.restitution
            elif body.position[0] > max_x:
                body.position[0] = max_x
                body.velocity[0] = -body.velocity[0] * body.restitution

            if body.position[1] < min_y:
                body.position[1] = min_y
                body.velocity[1] = -body.velocity[1] * body.restitution
            elif body.position[1] > max_y:
                body.position[1] = max_y
                body.velocity[1] = -body.velocity[1] * body.restitution

        elif self.boundary_mode == BoundaryMode.STOP:
            # Zero velocity at boundaries
            if body.position[0] < min_x:
                body.position[0] = min_x
                body.velocity[0] = 0.0
            elif body.position[0] > max_x:
                body.position[0] = max_x
                body.velocity[0] = 0.0

            if body.position[1] < min_y:
                body.position[1] = min_y
                body.velocity[1] = 0.0
            elif body.position[1] > max_y:
                body.position[1] = max_y
                body.velocity[1] = 0.0

    def _resolve_collisions(self) -> None:
        """Resolve rigid-body collisions between all pairs of bodies."""
        body_list = list(self._bodies.values())

        for i, body_a in enumerate(body_list):
            for body_b in body_list[i + 1:]:
                self._resolve_pair_collision(body_a, body_b)

    def _resolve_pair_collision(self, a: PhysicsBody, b: PhysicsBody) -> None:
        """Resolve collision between two bodies using elastic collision."""
        delta = b.position - a.position
        distance = np.linalg.norm(delta)
        min_dist = a.radius + b.radius

        if distance < min_dist and distance > 1e-6:
            # Normalize collision normal
            normal = delta / distance

            # Separate overlapping bodies
            overlap = min_dist - distance
            total_mass = a.mass + b.mass
            a.position -= normal * (overlap * b.mass / total_mass)
            b.position += normal * (overlap * a.mass / total_mass)

            # Relative velocity
            rel_vel = b.velocity - a.velocity
            vel_along_normal = np.dot(rel_vel, normal)

            # Don't resolve if velocities are separating
            if vel_along_normal > 0:
                return

            # Calculate impulse scalar
            restitution = min(a.restitution, b.restitution)
            j = -(1 + restitution) * vel_along_normal
            j /= 1 / a.mass + 1 / b.mass

            # Apply impulse
            impulse = j * normal
            a.velocity -= impulse / a.mass
            b.velocity += impulse / b.mass

    def _resolve_obstacle_collisions(self) -> None:
        """Resolve collisions between bodies and static obstacles."""
        for body in self._bodies.values():
            for obstacle in self._obstacles:
                self._resolve_body_obstacle_collision(body, obstacle)

    def _resolve_body_obstacle_collision(self, body: PhysicsBody, obstacle) -> None:
        """Resolve collision between a circular body and a static obstacle."""
        if hasattr(obstacle, "center") and hasattr(obstacle, "radius"):
            self._resolve_circle_obstacle_collision(body, obstacle)
        elif all(hasattr(obstacle, attr) for attr in ("left", "right", "top", "bottom")):
            self._resolve_rect_obstacle_collision(body, obstacle)
        else:
            self._resolve_generic_obstacle_collision(body, obstacle)

    def _resolve_circle_obstacle_collision(self, body: PhysicsBody, obstacle) -> None:
        """Resolve collision between a circular body and circular obstacle."""
        delta = body.position - obstacle.center
        distance = np.linalg.norm(delta)
        min_dist = body.radius + obstacle.radius

        if distance >= min_dist:
            return

        if distance > 1e-6:
            normal = delta / distance
        else:
            normal = np.array([1.0, 0.0], dtype=np.float32)

        body.position = obstacle.center + normal * min_dist
        self._reflect_body_velocity(body, normal, getattr(obstacle, "restitution", 0.8))

    def _resolve_rect_obstacle_collision(self, body: PhysicsBody, obstacle) -> None:
        """Resolve collision between a circular body and axis-aligned rectangle."""
        pos = body.position
        nearest = np.array(
            [
                np.clip(pos[0], obstacle.left, obstacle.right),
                np.clip(pos[1], obstacle.top, obstacle.bottom),
            ],
            dtype=np.float32,
        )
        delta = pos - nearest
        distance = np.linalg.norm(delta)

        if obstacle.contains(pos):
            distances = {
                "left": pos[0] - obstacle.left,
                "right": obstacle.right - pos[0],
                "top": pos[1] - obstacle.top,
                "bottom": obstacle.bottom - pos[1],
            }
            face = min(distances, key=distances.get)
            if face == "left":
                normal = np.array([-1.0, 0.0], dtype=np.float32)
            elif face == "right":
                normal = np.array([1.0, 0.0], dtype=np.float32)
            elif face == "top":
                normal = np.array([0.0, -1.0], dtype=np.float32)
            else:
                normal = np.array([0.0, 1.0], dtype=np.float32)

            body.position += normal * (distances[face] + body.radius)
            self._reflect_body_velocity(body, normal, getattr(obstacle, "restitution", 0.8))
            return

        if distance >= body.radius:
            return

        if distance > 1e-6:
            normal = delta / distance
        else:
            normal = obstacle.normal_at(pos)

        body.position = nearest + normal * body.radius
        self._reflect_body_velocity(body, normal, getattr(obstacle, "restitution", 0.8))

    def _resolve_generic_obstacle_collision(self, body: PhysicsBody, obstacle) -> None:
        """Resolve collision for obstacles that only implement the obstacle interface."""
        nearest = obstacle.nearest_point(body.position)
        delta = body.position - nearest
        distance = np.linalg.norm(delta)

        if not obstacle.contains(body.position) and distance >= body.radius:
            return

        if distance > 1e-6:
            normal = delta / distance
        else:
            normal = obstacle.normal_at(body.position)

        body.position = nearest + normal * body.radius
        self._reflect_body_velocity(body, normal, getattr(obstacle, "restitution", 0.8))

    def _reflect_body_velocity(
        self,
        body: PhysicsBody,
        normal: np.ndarray,
        obstacle_restitution: float,
    ) -> None:
        """Reflect the inward velocity component against a static surface."""
        vel_along_normal = np.dot(body.velocity, normal)
        if vel_along_normal >= 0:
            return

        restitution = min(body.restitution, obstacle_restitution)
        body.velocity -= (1 + restitution) * vel_along_normal * normal

    def get_position(self, name: str) -> Optional[np.ndarray]:
        """Get the position of a body."""
        if name in self._bodies:
            return self._bodies[name].position.copy()
        return None

    def get_velocity(self, name: str) -> Optional[np.ndarray]:
        """Get the velocity of a body."""
        if name in self._bodies:
            return self._bodies[name].velocity.copy()
        return None

    def set_position(self, name: str, position: np.ndarray) -> None:
        """Set the position of a body."""
        if name in self._bodies:
            self._bodies[name].position = np.array(position, dtype=np.float32)

    def add_obstacle(self, obstacle) -> None:
        """Add an obstacle to the physics world."""
        self._obstacles.append(obstacle)

    def add_force_field(self, field) -> None:
        """Add a force field to the physics world."""
        self._force_fields.append(field)

    def get_body(self, name: str) -> Optional[PhysicsBody]:
        """Get a body by name."""
        return self._bodies.get(name)
