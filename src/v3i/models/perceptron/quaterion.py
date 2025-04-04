"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging
from typing import Literal, NamedTuple

import numpy as np
import quaternion

logger = logging.getLogger(__name__)

ForwardType = Literal["rotated_weight", "rotation", "commutator"]


class RotationState(NamedTuple):
    """Current state of the rotation."""

    axis: np.ndarray
    angle: float
    quat: quaternion.quaternion


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
        self.update_buffer: RotationState = RotationState(
            axis=np.array([1, 0, 0]),
            angle=0,
            quat=quaternion.quaternion(1, 0, 0, 0),
        )

        self.rng = np.random.default_rng(seed=random_seed)
        self.weight = self._initialize_weight()

        # Debugging trackers
        self.update_history = []
        self.weight_history = []
        self.prediction_history = []

        # Save initial state
        self._record_weight_state("initialization")

    def _record_weight_state(self, event: str) -> None:
        """Record current weight state with additional context."""
        self.weight_history.append({
            "event": event,
            "axis": self.weight.axis.copy(),
            "angle": self.weight.angle,
            "quaternion": quaternion.as_float_array(self.weight.quat).copy(),
            "update_count": len(self.update_history),
        })

    def _record_update(
        self,
        inputs: np.ndarray,
        label: int,
        prediction: int,
        error_scale: float,
        update_angle: float,
        confidence: float,
    ) -> None:
        """Record details about an update."""
        self.update_history.append({
            "label": label,
            "prediction": prediction,
            "error_scale": error_scale,
            "update_angle": update_angle,
            "confidence": confidence,
            "weight_angle": self.weight.angle,
            "weight_axis": self.weight.axis.copy(),
        })

    def _record_prediction(
        self,
        rotation: RotationState,
        prediction: int,
        confidence: float,
    ) -> None:
        """Record prediction details."""
        self.prediction_history.append({
            "rotation_angle": rotation.angle,
            "rotation_axis": rotation.axis.copy(),
            "prediction": prediction,
            "confidence": confidence,
        })

    def get_update_stats(self, window: int = 100) -> dict:
        """Get statistics about recent updates.

        Args:
            window: Number of recent updates to analyze

        Returns:
            Dictionary containing various statistics about updates
        """
        if not self.update_history:
            return {}

        recent_updates = self.update_history[-window:]

        # Compute basic statistics
        update_angles = [update["update_angle"] for update in recent_updates]
        error_scales = [update["error_scale"] for update in recent_updates]
        confidences = [update["confidence"] for update in recent_updates]

        # Compute weight movement
        weight_angles = [update["weight_angle"] for update in recent_updates]
        weight_axes = [update["weight_axis"] for update in recent_updates]

        # Compute axis stability (dot product between consecutive axes)
        axis_changes = []
        for i in range(1, len(weight_axes)):
            dot_product = np.dot(weight_axes[i], weight_axes[i - 1])
            axis_changes.append(np.arccos(np.clip(dot_product, -1, 1)))

        return {
            "mean_update_angle": np.mean(update_angles),
            "std_update_angle": np.std(update_angles),
            "mean_error_scale": np.mean(error_scales),
            "mean_confidence": np.mean(confidences),
            "weight_angle_change": np.std(weight_angles),
            "mean_axis_change": np.mean(axis_changes) if axis_changes else 0,
            "correct_predictions": sum(1 for u in recent_updates if u["label"] == u["prediction"]),
            "total_updates": len(recent_updates),
        }

    def get_weight_evolution(self) -> dict:
        """Get information about how the weight has evolved."""
        if not self.weight_history:
            return {}

        # Compute trajectory statistics
        angle_trajectory = [w["angle"] for w in self.weight_history]
        axis_trajectory = [w["axis"] for w in self.weight_history]

        # Compute axis stability over time
        axis_changes = []
        for i in range(1, len(axis_trajectory)):
            dot_product = np.dot(axis_trajectory[i], axis_trajectory[i - 1])
            axis_changes.append(np.arccos(np.clip(dot_product, -1, 1)))

        return {
            "initial_state": self.weight_history[0],
            "final_state": self.weight_history[-1],
            "angle_range": (min(angle_trajectory), max(angle_trajectory)),
            "total_axis_change": sum(axis_changes),
            "num_updates": len(self.weight_history) - 1,
        }

    def _initialize_weight(self) -> RotationState:
        """Initialize weight as a random rotation."""
        axis = self._random_unit_vector()
        angle = self.rng.uniform(0, 2 * np.pi)
        quat = quaternion.from_rotation_vector(angle * axis)
        return RotationState(axis=axis, angle=angle, quat=quat)

    def _random_unit_vector(self) -> np.ndarray:
        """Generate a random unit vector for rotation axis."""
        v = self.rng.normal(0, 1, 3)
        return v / np.linalg.norm(v)

    def _rotation_to_quaternion(self, axis: np.ndarray, angle: float) -> quaternion.quaternion:
        """Convert axis-angle representation to quaternion."""
        return quaternion.from_rotation_vector(angle * axis)

    def _quaternion_to_rotation(self, q: quaternion.quaternion) -> tuple[np.ndarray, float]:
        """Extract axis and angle from quaternion."""
        # Ensure unit quaternion
        q = q / abs(q)

        # Extract angle
        angle = 2 * np.arccos(np.clip(q.w, -1, 1))

        # Extract axis
        if abs(angle) < 1e-10:
            # If angle is very small, use previous axis
            return self.weight.axis, angle

        axis = quaternion.as_float_array(q)[1:] / np.sin(angle / 2)
        return axis, angle

    def _compute_minimal_rotation(
        self,
        start: quaternion.quaternion,
        end: quaternion.quaternion,
    ) -> quaternion.quaternion:
        """Compute the minimal rotation quaternion that takes start to end."""
        # Ensure unit quaternions
        start = start / abs(start)
        end = end / abs(end)

        # The rotation quaternion is sqrt(end * start^(-1))
        r = end * start.conjugate()

        # Normalize to get unit quaternion
        r = r / abs(r)

        # Ensure we take the shorter path (rotation angle ≤ π)
        if r.w < 0:
            r = -r

        return r

    def _compute_pulled_back_error(
        self,
        error_rotation: quaternion.quaternion,
        inputs: np.ndarray,
    ) -> quaternion.quaternion:
        """Pull back error through input rotations.

        Args:
            error_rotation: The error represented as a rotation quaternion
            inputs: Array of input quaternions

        Returns:
            Pulled back error rotation
        """
        pulled_back_error = error_rotation

        # Pull back through each input rotation in reverse order
        for x in reversed(inputs):
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x

            if abs(x_q) < 1e-15:
                continue

            # Normalize input
            x_q = x_q / abs(x_q)

            # Pull back through this rotation
            pulled_back_error = x_q * pulled_back_error * x_q.conjugate()

        return pulled_back_error

    def forward(self, inputs: np.ndarray) -> RotationState:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of quaternions representing the input

        Returns:
            RotationState containing final axis, angle, and quaternion
        """
        # Start with current weight quaternion
        result = self.weight.quat

        # Ensure inputs is an array of quaternions
        if isinstance(inputs, list | np.ndarray):
            inputs = [
                x if isinstance(x, quaternion.quaternion) else quaternion.quaternion(*x)
                for x in inputs
            ]

        for x_q in inputs:
            if abs(x_q) < 1e-15:
                continue

            # Normalize input
            x_n = x_q / abs(x_q)

            # Compose rotations
            result = x_n * result

        # Extract final rotation parameters
        axis, angle = self._quaternion_to_rotation(result)
        return RotationState(axis=axis, angle=angle, quat=result)

    def predict(self, rotation: RotationState) -> int:
        """Predict class based on final rotation.

        We use the rotation angle and its direction relative to a reference axis (weight axis).
        If the angle is positive, we predict 1, otherwise -1.

        Args:
            rotation: The final rotation state.

        Returns:
            1 if the rotation is in the positive direction, -1 otherwise.
        """
        # Project rotation axis onto reference axis (using weight axis as reference)
        projection = np.dot(rotation.axis, self.weight.axis)

        # Combine angle and projection for prediction
        # This considers both how much and in what direction we rotated
        decision_value = rotation.angle * projection
        return 1 if decision_value >= 0 else -1

    def _compute_error_rotation(
        self,
        prediction: quaternion.quaternion,
        label: int,
    ) -> quaternion.quaternion:
        """Compute the minimal rotation that would take prediction to target label."""
        # Convert label to target quaternion
        target = quaternion.quaternion(label, 0, 0, 0)
        return self._compute_minimal_rotation(prediction, target)

    def train(self, inputs: np.ndarray, label: int) -> None:
        """Train on a single example using continuous updates.

        Updates are performed based on the residual error, regardless of whether the prediction was correct.
        """
        # Forward pass
        final_rotation = self.forward(inputs)
        prediction = self.predict(final_rotation)

        # Compute confidence
        confidence = abs(np.dot(final_rotation.axis, self.weight.axis))

        # Record prediction
        self._record_prediction(final_rotation, prediction, confidence)

        # Compute error and update
        target = quaternion.quaternion(label, 0, 0, 0)
        prediction_q = quaternion.quaternion(prediction, 0, 0, 0)
        error_rotation = target * prediction_q.conjugate()

        # Scale error by prediction confidence
        error_scale = 1.0 - confidence

        # Pull back error and compute update
        pulled_back_error = self._compute_pulled_back_error(error_rotation, inputs)
        update_axis, update_angle = self._quaternion_to_rotation(pulled_back_error)

        # Record pre-update state
        self._record_weight_state("pre_update")

        # Apply update
        scaled_angle = self.learning_rate * update_angle * error_scale
        update_q = self._rotation_to_quaternion(update_axis, scaled_angle)
        new_q = update_q * self.weight.quat

        # Extract new parameters
        new_axis, new_angle = self._quaternion_to_rotation(new_q)

        # Update weight
        self.weight = RotationState(axis=new_axis, angle=new_angle, quat=new_q)

        # Record update details
        self._record_update(
            inputs=inputs,
            label=label,
            prediction=prediction,
            error_scale=error_scale,
            update_angle=scaled_angle,
            confidence=confidence,
        )

        # Record post-update state
        self._record_weight_state("post_update")
