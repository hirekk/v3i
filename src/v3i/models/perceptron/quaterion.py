"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import quaternion

logger = logging.getLogger(__name__)

ForwardType = Literal["rotated_weight", "rotation", "commutator"]


class QuaternionPerceptron:
    """A biologically-inspired perceptron using a single quaternion weight."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 100,
        random_seed: int = 0,
    ) -> None:
        """Initialize the quaternion perceptron with a single quaternion weight.

        Args:
            learning_rate: Learning rate for weight updates.
            random_seed: Random seed for weight initialization.
        """
        self.learning_rate = learning_rate

        # TODO: Buffered updates idea
        # We will accumulate updates and apply them in one go after a batch of updates.
        self.batch_size = batch_size
        self.update_buffer: list[quaternion.quaternion] = []

        self.rng = np.random.default_rng(seed=random_seed)
        self.weight = self._initialize_weight()

    def _random_unit_vector(self) -> np.ndarray:
        """Generate a random unit vector for rotation axis."""
        v = self.rng.normal(0, 1, 3)
        return v / np.linalg.norm(v)

    def _initialize_weight(self) -> quaternion.quaternion:
        """Initialize weight as a random rotation."""
        # Generate random 4D vector
        components = self.rng.normal(0, 1, 4)
        # Normalize to unit length
        components = components / np.linalg.norm(components)
        return quaternion.quaternion(*components)

    def get_angle(self) -> float:
        """Get the angle of a quaternion."""
        angle = quaternion.rotation_intrinsic_distance(self.weight, quaternion.one)
        if angle < 0 or angle > 2 * np.pi:
            msg = f"Angle is out of range [0, 2π]: {angle}"
            raise ValueError(msg)
        return angle

    def get_axis(self, tolerance: float = 1e-10) -> np.ndarray:
        """Get the axis of the weight quaternion."""
        angle = self.get_angle()
        if angle < tolerance:
            return np.array([1, 0, 0])
        return quaternion.as_rotation_vector(self.weight) / angle

    def _is_rotation_quaternion(self, q: quaternion.quaternion, tolerance: float = 1e-10) -> bool:
        """Check if quaternion represents a valid rotation.

        A rotation quaternion must be unit length.

        Args:
            q: The quaternion to check.
            tolerance: The tolerance for the unit length check.

        Returns:
            True if the quaternion is a valid rotation, False otherwise.
        """
        norm = abs(q)
        return abs(norm - 1.0) < tolerance

    def _assert_rotation_quaternion(self, q: quaternion.quaternion, msg: str = "") -> None:
        """Assert that quaternion represents a valid rotation."""
        if not self._is_rotation_quaternion(q):
            msg = f"Invalid rotation quaternion: {msg}"
            raise ValueError(msg)

    def _compute_geodesic_rotation(
        self,
        start: quaternion.quaternion,
        end: quaternion.quaternion,
    ) -> quaternion.quaternion:
        """Compute the minimal rotation that takes start to end along the geodesic.

        Args:
            start: The starting quaternion.
            end: The ending quaternion.

        Returns:
            A unit quaternion representing this optimal rotation.
        """
        # The rotation we want is: end * start^(-1)
        # This gives us the rotation that takes start to end
        rotation = end * start.conjugate()

        # Normalize to ensure we're on the 3-sphere
        rotation = rotation / abs(rotation)

        # If the rotation angle is > π, we should go the other way around
        # We can check this by looking at the real part (w component)
        # If w < 0, the angle is > π
        if rotation.w < 0:
            rotation = -rotation  # Take the shorter path

        return rotation

    def forward(self, inputs: np.ndarray, tolerance: float = 1e-10) -> quaternion.quaternion:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of quaternions representing the input

        Returns:
            RotationState containing final axis, angle, and quaternion
        """
        # Accumulate inputs
        result = quaternion.one
        for x in inputs:
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x
            if abs(x_q) < tolerance:
                continue
            x_n = x_q / abs(x_q)
            result = x_n * result

        # Apply weight transformation
        return self.weight * result

    def predict(self, rotated: quaternion.quaternion) -> int:
        """Predict class based on final rotation angle."""
        return np.sign(rotated.w)

    def _quaternion_log(self, q: quaternion.quaternion) -> np.ndarray:
        """Convert rotation quaternion to axis-angle representation in log space."""
        if abs(q.w - 1) < 1e-10:  # If close to identity
            return np.zeros(3)

        # Extract vector part
        v = np.array([q.x, q.y, q.z])
        theta = 2 * np.arccos(np.clip(q.w, -1, 1))

        # Normalize and scale by theta
        if np.linalg.norm(v) < 1e-10:
            return np.zeros(3)
        return theta * v / np.linalg.norm(v)

    def _quaternion_exp(self, v: np.ndarray) -> quaternion.quaternion:
        """Convert axis-angle representation back to quaternion."""
        theta = np.linalg.norm(v)
        if theta < 1e-10:
            return quaternion.one

        axis = v / theta
        return quaternion.from_rotation_vector(theta * axis)

    def train(self, inputs: np.ndarray, label: int, tolerance: float = 1e-10) -> None:
        """Update weight using log/exp map for better stability."""
        self._assert_rotation_quaternion(self.weight, "Current weight")

        # Forward pass
        rotated = self.forward(inputs)

        # Compute error as geodesic rotation from current to target
        target = quaternion.quaternion(label, 0, 0, 0)
        error_rotation = self._compute_geodesic_rotation(rotated, target)

        if abs(error_rotation.w - 1) > tolerance:  # if not identity rotation
            # Convert error to log space (axis-angle)
            error_vec = self._quaternion_log(error_rotation)

            # Scale the error by learning rate
            scaled_error = (
                -self.learning_rate * error_vec
            )  # Negative because we want to move opposite to error

            # Add to update buffer in log space
            self.update_buffer.append(scaled_error)

        if len(self.update_buffer) == self.batch_size:
            # Average the updates in log space
            mean_update = np.mean(self.update_buffer, axis=0)

            # Convert back to quaternion
            update = self._quaternion_exp(mean_update)

            # Apply update: rotate weight by update
            self.weight = self.weight * update  # Note: order changed to rotate weight by update
            self.weight = self.weight / abs(self.weight)  # Ensure normalization

            # Reset buffer
            self.update_buffer = []
