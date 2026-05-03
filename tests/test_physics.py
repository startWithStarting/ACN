"""Tests for physics engine."""

import unittest
import numpy as np

from src.physics.engine import PhysicsEngine, BoundaryMode
from src.physics.obstacles import RectObstacle, CircleObstacle
from src.physics.fields import AttractorField, FlowField


class TestPhysicsEngine(unittest.TestCase):
    """Test cases for PhysicsEngine."""

    def setUp(self):
        """Set up test fixtures."""
        self.engine = PhysicsEngine(
            grid_width=100,
            grid_height=100,
            boundary_mode="clamp",
        )

    def test_register_body(self):
        """Test registering a body."""
        body = self.engine.register_body(
            "agent1",
            position=np.array([50.0, 50.0]),
            velocity=np.array([1.0, 0.0]),
            mass=1.0,
        )
        self.assertEqual(body.name, "agent1")
        np.testing.assert_array_equal(body.position, [50.0, 50.0])

    def test_euler_integration(self):
        """Test that Euler integration works."""
        self.engine.register_body(
            "agent1",
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.0, 0.0]),
        )
        self.engine.step(dt=1.0)

        pos = self.engine.get_position("agent1")
        np.testing.assert_array_almost_equal(pos, [1.0, 0.0])

    def test_drag(self):
        """Test that drag works."""
        engine = PhysicsEngine(
            grid_width=100,
            grid_height=100,
            default_drag=0.5,
        )
        engine.register_body(
            "agent1",
            position=np.array([0.0, 0.0]),
            velocity=np.array([10.0, 0.0]),
        )
        engine.step(dt=1.0)

        # Velocity should decrease due to drag
        vel = self.engine.get_velocity("agent1")
        self.assertLess(vel[0], 10.0)

    def test_boundary_clamp(self):
        """Test clamp boundary mode."""
        self.engine.register_body(
            "agent1",
            position=np.array([99.0, 50.0]),
            velocity=np.array([5.0, 0.0]),
        )
        self.engine.step(dt=1.0)

        pos = self.engine.get_position("agent1")
        self.assertLessEqual(pos[0], 100.0)

    def test_boundary_bounce(self):
        """Test bounce boundary mode."""
        engine = PhysicsEngine(
            grid_width=100,
            grid_height=100,
            boundary_mode="bounce",
        )
        engine.register_body(
            "agent1",
            position=np.array([99.0, 50.0]),
            velocity=np.array([5.0, 0.0]),
        )
        engine.step(dt=1.0)

        # Velocity should be reflected
        vel = engine.get_velocity("agent1")
        self.assertLess(vel[0], 0.0)

    def test_collision(self):
        """Test rigid-body collision."""
        self.engine.register_body(
            "agent1",
            position=np.array([0.0, 0.0]),
            velocity=np.array([10.0, 0.0]),
            radius=1.0,
            mass=1.0,
        )
        self.engine.register_body(
            "agent2",
            position=np.array([2.0, 0.0]),
            velocity=np.array([0.0, 0.0]),
            radius=1.0,
            mass=1.0,
        )
        self.engine.step(dt=1.0)

        # Agents should have collided and changed velocities
        vel1 = self.engine.get_velocity("agent1")
        vel2 = self.engine.get_velocity("agent2")

        # Both should have non-zero velocity after collision
        self.assertFalse(np.allclose(vel1, [10.0, 0.0]))


class TestObstacles(unittest.TestCase):
    """Test cases for obstacles."""

    def test_rect_contains(self):
        """Test RectObstacle contains check."""
        rect = RectObstacle(x=40, y=30, width=10, height=20)

        # Inside
        self.assertTrue(rect.contains(np.array([45.0, 40.0])))
        # Outside
        self.assertFalse(rect.contains(np.array([10.0, 10.0])))

    def test_rect_normal(self):
        """Test RectObstacle normal calculation."""
        rect = RectObstacle(x=40, y=30, width=10, height=20)

        # Left face
        normal = rect.normal_at(np.array([40.0, 40.0]))
        np.testing.assert_array_almost_equal(normal, [-1.0, 0.0])

    def test_circle_contains(self):
        """Test CircleObstacle contains check."""
        circle = CircleObstacle(x=50, y=50, radius=10)

        # Inside
        self.assertTrue(circle.contains(np.array([50.0, 50.0])))
        # Outside
        self.assertFalse(circle.contains(np.array([70.0, 70.0])))


class TestForceFields(unittest.TestCase):
    """Test cases for force fields."""

    def test_attractor_field(self):
        """Test AttractorField."""
        field = AttractorField(
            center=np.array([50.0, 50.0]),
            strength=1.0,
        )

        force = field.force_at(np.array([0.0, 0.0]))

        # Force should point toward center
        self.assertGreater(force[0], 0.0)
        self.assertGreater(force[1], 0.0)

    def test_flow_field(self):
        """Test FlowField."""
        field = FlowField(
            direction=np.array([1.0, 0.0]),
            strength=0.5,
        )

        force = field.force_at(np.array([0.0, 0.0]))

        # Force should be in flow direction
        np.testing.assert_array_almost_equal(force, [0.5, 0.0])


if __name__ == "__main__":
    unittest.main()