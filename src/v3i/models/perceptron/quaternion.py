"""Quaternion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses quaternions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging

import numpy as np
import quaternion

from v3i.models.perceptron.utils import ForwardType

logger = logging.getLogger(__name__)


def geodesic_rotation(
    source: quaternion.quaternion,
    target: quaternion.quaternion,
) -> quaternion.quaternion:
    """Minimal rotation (right-multiply) from source to target: source * R = target."""
    r = source.conjugate() * target
    r = r / abs(r)
    if r.w < 0:
        r = -r
    return r


class QuaternionPerceptron:
    """A biologically-inspired perceptron using a single quaternion weight."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        random_seed: int | None = None,
        forward_type: ForwardType = "right_multiplication",
    ) -> None:
        """Initialize the quaternion perceptron with a single quaternion weight."""
        self.forward_type = forward_type
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=random_seed)

        # Single unit quaternion weight: rotation applied on the right (output = reduced * weight).
        self.weight = self._initialize_weight()

    def _random_unit_vector(self) -> np.ndarray:
        """Generate a random unit vector for rotation axis."""
        v = self._rng.normal(0, 1, 3)
        return v / np.linalg.norm(v)

    def _initialize_weight(self) -> quaternion.quaternion:
        """Initialize weight as identity + small random perturbation."""
        components = np.array([1, 0, 0, 0], dtype=np.float64)
        perturbation = self._rng.normal(0, 0.1, 4)
        components += perturbation
        # Normalize to unit length
        components = components / np.linalg.norm(components)
        if components[0] < 0:
            components = -components
        return quaternion.quaternion(*components)

    def _compute_geodesic_rotation(
        self,
        source: quaternion.quaternion,
        target: quaternion.quaternion,
    ) -> quaternion.quaternion:
        """Compute the minimal rotation that takes source to target along the geodesic.

        Args:
            source: The starting quaternion.
            target: The ending quaternion.

        Returns:
            A unit quaternion representing this optimal rotation.
        """
        # The rotation we want is: source^(-1) * target
        # This gives us the rotation that takes source to target via right multiplication:
        # source * rotation = target
        rotation = source.conjugate() * target

        # Normalize to ensure we're on the 3-sphere
        rotation = rotation / abs(rotation)

        # If the rotation angle is > π, we should go the other way around
        # We can check this by looking at the real part (w component)
        # If w < 0, the angle is > π
        if rotation.w < 0:
            rotation = -rotation  # Take the shorter path

        return rotation

    def forward_average(
        self, inputs: np.ndarray
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Computes the average of the inputs using the eigenvector of the sum of
        outer products of the inputs.

        Args:
            inputs: Array of quaternions representing the input orientations.

        Returns:
            Tuple containing incoming quaternion (accumulated input
            orientations) and outgoing quaternion (final orientation).
        """
        # Compute the sum of outer products in a vectorized manner
        M = np.einsum("ij,ik->jk", inputs, inputs)
        eigvals, eigvecs = np.linalg.eig(M)
        max_eigval_index = np.argmax(eigvals)
        reduced = quaternion.quaternion(*eigvecs[:, max_eigval_index])

        result = reduced * self.weight
        return reduced, result

    def forward_left_multiplication(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input
            orientations) and outgoing quaternion (final orientation).
        """
        # Accumulate inputs
        reduced = quaternion.quaternion(1, 0, 0, 0)
        for x in inputs:
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x
            if abs(x_q) < tolerance:
                continue
            x_n = x_q / abs(x_q)
            reduced = x_n * reduced

        result = reduced * self.weight
        return reduced, result

    def forward_right_multiplication(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input
            orientations) and outgoing quaternion (final orientation).
        """
        # Accumulate inputs
        reduced = quaternion.quaternion(1, 0, 0, 0)
        for x in inputs:
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x
            if abs(x_q) < tolerance:
                continue
            x_n = x_q / abs(x_q)
            reduced = reduced * x_n

        result = reduced * self.weight
        return reduced, result

    def forward_algebraic_sum(
        self, inputs: np.ndarray
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Computes the sum of the inputs using the eigenvector of the sum of outer
        products of the inputs.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input
            orientations) and outgoing quaternion (final orientation).
        """
        q_inputs = quaternion.as_quat_array(inputs)
        rot_inputs = quaternion.as_rotation_vector(q_inputs)
        reduced_vector = np.sum(rot_inputs, axis=0)
        reduced = quaternion.from_rotation_vector(reduced_vector)

        result = reduced * self.weight
        return reduced, result

    def forward_algebraic_mean(
        self, inputs: np.ndarray
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Computes the mean of the inputs using the eigenvector of the sum of outer
        products of the inputs.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.
        """
        q_inputs = quaternion.as_quat_array(inputs)
        rot_inputs = quaternion.as_rotation_vector(q_inputs)
        reduced_vector = np.mean(rot_inputs, axis=0)
        reduced = quaternion.from_rotation_vector(reduced_vector)

        result = reduced * self.weight
        return reduced, result

    def predict(self, inputs: np.ndarray) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Predict class based on final rotation angle."""
        # Forward pass
        match self.forward_type:
            case "average":
                q_in, q_out = self.forward_average(inputs=inputs)
            case "left_multiplication":
                q_in, q_out = self.forward_left_multiplication(inputs=inputs)
            case "right_multiplication":
                q_in, q_out = self.forward_right_multiplication(inputs=inputs)
            case "algebraic_mean":
                q_in, q_out = self.forward_algebraic_mean(inputs=inputs)
            case "algebraic_sum":
                q_in, q_out = self.forward_algebraic_sum(inputs=inputs)
            case _:
                msg = (
                    f"Invalid forward type: {self.forward_type}; "
                    f"must be one of {ForwardType.__args__}"
                )
                raise ValueError(
                    msg,
                )

        return q_in, q_out

    def predict_label(self, inputs: np.ndarray) -> int:
        """Predict class: +1 if q_out.w >= 0 else -1 (avoids 0 from np.sign)."""
        _, q_out = self.predict(inputs=inputs)
        return 1 if q_out.w >= 0 else -1

    def compute_update(
        self, inputs: np.ndarray, label: int
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Proposed (u, u_residual): rotation update for weight and residual for stacking."""
        self._ensure_unit_weight()
        _, q_out = self.predict(inputs=inputs)
        q_target = quaternion.quaternion(label, 0, 0, 0)
        q_error = self._compute_geodesic_rotation(source=q_out, target=q_target)
        u = q_error**self.learning_rate
        if u.w < 0:
            u = -u
        return u, u

    def apply_update(self, u: quaternion.quaternion) -> None:
        """Apply rotation update to weight (right multiply) and renormalize."""
        self.weight = self.weight * u
        self.weight = self.weight / abs(self.weight)

    def train(self, inputs: np.ndarray, label: int) -> None:
        """Convenience: compute_update then apply_update (one step). Use optimizer for batching."""
        u, _ = self.compute_update(inputs, label)
        self.apply_update(u)

    def _ensure_unit_weight(self) -> None:
        if abs(abs(self.weight) - 1) > 1e-10:
            self.weight = self.weight / abs(self.weight)


class QuaternionSimpleOptimizer:
    """Apply every update u immediately."""

    def __init__(self, model: QuaternionPerceptron) -> None:
        """Bind the optimizer to a perceptron."""
        self._model = model

    def step(self, u: quaternion.quaternion) -> None:
        """Apply the update immediately."""
        self._model.apply_update(u)
