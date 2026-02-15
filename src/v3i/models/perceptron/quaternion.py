"""Quaternion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses quaternions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import quaternion

logger = logging.getLogger(__name__)

ForwardType = Literal[
    "average",
    "left_multiplication",
    "right_multiplication",
    "algebraic_sum",
    "algebraic_mean",
]


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
        self.error_store: list[quaternion.quaternion] = []
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=random_seed)

        # Bias is used as a left action on the input, applying a "global", input-independent rotation.
        self.bias = self._initialize_weight()

        # Action is used as a right action on the input, applying a "local", input-dependent rotation.
        self.action = self._initialize_weight()

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

    # def get_angle(self) -> float:
    #     """Get the angle of a quaternion."""
    #     angle = quaternion.rotation_intrinsic_distance(self.weight, quaternion.one)
    #     if angle < 0 or angle > 2 * np.pi:
    #         msg = f"Angle is out of range [0, 2π]: {angle}"
    #         raise ValueError(msg)
    #     return angle

    # def get_axis(self, tolerance: float = 1e-10) -> np.ndarray:
    #     """Get the axis of the weight quaternion."""
    #     angle = self.get_angle()
    #     if angle < tolerance:
    #         return np.array([1, 0, 0])
    #     return quaternion.as_rotation_vector(self.weight) / angle

    # def _is_rotation_quaternion(self, q: quaternion.quaternion, tolerance: float = 1e-10) -> bool:
    #     """Check if quaternion represents a valid rotation.

    #     A rotation quaternion must be unit length.

    #     Args:
    #         q: The quaternion to check.
    #         tolerance: The tolerance for the unit length check.

    #     Returns:
    #         True if the quaternion is a valid rotation, False otherwise.
    #     """
    #     norm = abs(q)
    #     return abs(norm - 1.0) < tolerance

    # def _assert_rotation_quaternion(self, q: quaternion.quaternion, msg: str = "") -> None:
    #     """Assert that quaternion represents a valid rotation."""
    #     if not self._is_rotation_quaternion(q):
    #         msg = f"Invalid rotation quaternion: {msg}"
    #         raise ValueError(msg)

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
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Computes the average of the inputs using the eigenvector of the sum of outer products of the inputs.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input orientations) and outgoing quaternion (final orientation).
        """
        # Compute the sum of outer products in a vectorized manner
        M = np.einsum("ij,ik->jk", inputs, inputs)
        eigvals, eigvecs = np.linalg.eig(M)
        max_eigval_index = np.argmax(eigvals)
        reduced = quaternion.quaternion(*eigvecs[:, max_eigval_index])
        # if reduced.w < 0:
        #     reduced = -reduced

        result = self.bias * reduced * self.action
        # if result.w < 0:
        #     result = -result

        return reduced, result

    def forward_left_multiplication(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input orientations) and outgoing quaternion (final orientation).
        """
        # Accumulate inputs
        reduced = quaternion.quaternion(1, 0, 0, 0)
        for x in inputs:
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x
            if abs(x_q) < tolerance:
                continue
            x_n = x_q / abs(x_q)
            reduced = x_n * reduced
        # if reduced.w < 0:
        #     reduced = -reduced

        result = self.bias * reduced * self.action
        # if result.w < 0:
        #     result = -result

        return reduced, result

    def forward_right_multiplication(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input orientations) and outgoing quaternion (final orientation).
        """
        # Accumulate inputs
        reduced = quaternion.quaternion(1, 0, 0, 0)
        for x in inputs:
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x
            if abs(x_q) < tolerance:
                continue
            x_n = x_q / abs(x_q)
            reduced = reduced * x_n
        # if reduced.w < 0:
        #     reduced = -reduced

        # Apply bias and action
        result = self.bias * reduced * self.action
        # if result.w < 0:
        #     result = -result

        return reduced, result

    def forward_algebraic_sum(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Computes the sum of the inputs using the eigenvector of the sum of outer products of the inputs.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.

        Returns:
            Tuple containing incoming quaternion (accumulated input orientations) and outgoing quaternion (final orientation).
        """
        q_inputs = quaternion.as_quat_array(inputs)
        rot_inputs = quaternion.as_rotation_vector(q_inputs)
        reduced_vector = np.sum(rot_inputs, axis=0)
        reduced = quaternion.from_rotation_vector(reduced_vector)
        # if reduced.w < 0:
        #     reduced = -reduced

        result = self.bias * reduced * self.action
        # if result.w < 0:
        #     result = -result

        return reduced, result

    def forward_algebraic_mean(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion, quaternion.quaternion]:
        """Forward pass computing final rotation state.

        Computes the mean of the inputs using the eigenvector of the sum of outer products of the inputs.

        Args:
            inputs: Array of quaternions representing the input orientations.
            tolerance: Tolerance for quaternion normalization.
        """
        q_inputs = quaternion.as_quat_array(inputs)
        rot_inputs = quaternion.as_rotation_vector(q_inputs)
        reduced_vector = np.mean(rot_inputs, axis=0)
        reduced = quaternion.from_rotation_vector(reduced_vector)
        # if reduced.w < 0:
        #     reduced = -reduced

        result = self.bias * reduced * self.action
        # if result.w < 0:
        #     result = -result

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
                msg = f"Invalid forward type: {self.forward_type}; must be one of {ForwardType.__args__}"
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
        """Proposed (u_b, u_a) to move output toward label. Optimizer applies them."""
        self._ensure_unit_weights()
        q_in, q_out = self.predict(inputs=inputs)
        q_target = quaternion.quaternion(label, 0, 0, 0)
        q_error = self._compute_geodesic_rotation(source=q_out, target=q_target)
        q_update = q_error**self.learning_rate
        if q_update.w < 0:
            q_update = -q_update
        u_b, u_residual, u_a = self.decompose_update(q_update=q_update, q_kernel=q_in)
        self.error_store.append(u_residual)
        return u_b, u_a

    def apply_update(self, u_b: quaternion.quaternion, u_a: quaternion.quaternion) -> None:
        """Apply (u_b, u_a) to bias and action and renormalize."""
        self.bias = self.bias * u_b
        self.bias = self.bias / abs(self.bias)
        self.action = self.action * u_a
        self.action = self.action / abs(self.action)

    def train(self, inputs: np.ndarray, label: int) -> None:
        """Convenience: compute_update then apply_update (one step). Use optimizer for batching."""
        u_b, u_a = self.compute_update(inputs, label)
        self.apply_update(u_b, u_a)

    def _ensure_unit_weights(self) -> None:
        if abs(abs(self.bias) - 1) > 1e-10:
            self.bias = self.bias / abs(self.bias)
        if abs(abs(self.action) - 1) > 1e-10:
            self.action = self.action / abs(self.action)

    def decompose_update(
        self,
        q_update: quaternion.quaternion,
        q_kernel: quaternion.quaternion,
    ) -> tuple[quaternion.quaternion, quaternion.quaternion, quaternion.quaternion]:
        """Decompose update into bias, residual, and action components.

        Args:
            update: The update quaternion.
            kernel: The kernel quaternion.

        Returns:
            A tuple containing the bias, residual, and action update components.
        """
        # q_update must be close to the identity.
        # if not quaternion.isclose(q_update, quaternion.quaternion(1, 0, 0, 0), rtol=0.1):
        #     logging.warning("Update must be close to the identity; got %s.", q_update)

        v_b = quaternion.as_rotation_vector(self.bias)
        v_k = quaternion.as_rotation_vector(q_kernel)
        v_a = quaternion.as_rotation_vector(self.action)

        v_u = quaternion.as_rotation_vector(q_update)

        # Form a basis from the vectors v_b, v_k, and v_a.
        basis = np.array([v_b, v_k, v_a])
        det = np.linalg.det(basis)
        if abs(det) < 1e-10:
            err_msg = f"Basis vectors are linearly dependent (det = {det}): {basis}"
            raise ValueError(err_msg)

        # Solve for coefficients
        coefficients = np.linalg.solve(basis, v_u)

        # Project error components back into the quaternion space.
        u_b = (
            q_kernel.conjugate() ** coefficients[1]
            * self.action.conjugate() ** coefficients[2]
            * q_update ** coefficients[0]
        )
        u_b = u_b / abs(u_b)
        if abs(abs(u_b) - 1) > 1e-10:
            logger.warning(
                "Bias error component must be a unit quaternion: ||%s|| = %s.",
                u_b,
                abs(u_b),
            )
        u_residual = (
            self.bias.conjugate() ** coefficients[0]
            * self.action.conjugate() ** coefficients[2]
            * q_update ** coefficients[1]
        )
        u_residual = u_residual / abs(u_residual)
        if abs(abs(u_residual) - 1) > 1e-10:
            logger.warning(
                "Residual error component must be a unit quaternion: ||%s|| = %s.",
                u_residual,
                abs(u_residual),
            )
        u_a = (
            self.bias.conjugate() ** coefficients[0]
            * q_kernel.conjugate() ** coefficients[1]
            * q_update ** coefficients[2]
        )
        u_a = u_a / abs(u_a)
        if abs(abs(u_a) - 1) > 1e-10:
            logger.warning(
                "Action error component must be a unit quaternion: ||%s|| = %s.",
                u_a,
                abs(u_a),
            )
        return u_b, u_residual, u_a


def _tangent_space_avg(
    quat_list: list[quaternion.quaternion], scale: float = 1.0
) -> quaternion.quaternion:
    """Average rotations in tangent space and return a single quaternion (scale shrinks the step)."""
    if not quat_list:
        return quaternion.quaternion(1, 0, 0, 0)
    vecs = np.array([quaternion.as_rotation_vector(q) for q in quat_list])
    avg_vec = np.mean(vecs, axis=0) * scale
    n = np.linalg.norm(avg_vec)
    if n < 1e-12:
        return quaternion.quaternion(1, 0, 0, 0)
    return quaternion.from_rotation_vector(avg_vec)


class SimpleOptimizer:
    """Apply every (u_b, u_a) immediately. Model is bound at construction (PyTorch-like)."""

    def __init__(self, model: QuaternionPerceptron) -> None:
        self._model = model

    def step(self, u_b: quaternion.quaternion, u_a: quaternion.quaternion) -> None:
        self._model.apply_update(u_b, u_a)


class BatchedOptimizer:
    """Accumulate (u_b, u_a) and apply a tangent-space average every batch_size steps. Model bound at construction."""

    def __init__(self, model: QuaternionPerceptron, batch_size: int) -> None:
        self._model = model
        self.batch_size = batch_size
        self._u_b_buf: list[quaternion.quaternion] = []
        self._u_a_buf: list[quaternion.quaternion] = []

    def step(self, u_b: quaternion.quaternion, u_a: quaternion.quaternion) -> None:
        self._u_b_buf.append(u_b)
        self._u_a_buf.append(u_a)
        if len(self._u_b_buf) >= self.batch_size:
            self._apply_batch()

    def flush(self) -> None:
        """Apply any remaining buffered updates (e.g. at end of epoch)."""
        if self._u_b_buf:
            self._apply_batch()

    def _apply_batch(self) -> None:
        n = len(self._u_b_buf)
        scale = 1.0 / n
        u_b_avg = _tangent_space_avg(self._u_b_buf, scale=scale)
        u_a_avg = _tangent_space_avg(self._u_a_buf, scale=scale)
        self._model.apply_update(u_b_avg, u_a_avg)
        self._u_b_buf.clear()
        self._u_a_buf.clear()
