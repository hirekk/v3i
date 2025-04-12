"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import quaternion

from v3i.data import load_mnist
from v3i.data import load_xor

logger = logging.getLogger(__name__)

ForwardType = Literal[
    "average",
    "left_multiplication",
    "right_multiplication",
    "algebraic_sum",
    "algebraic_mean",
]


class QuaternionPerceptron:
    """A biologically-inspired perceptron using a single quaternion weight."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        buffer_size: int | None = None,
        random_seed: int | None = None,
        forward_type: ForwardType = "right_multiplication",
    ) -> None:
        """Initialize the quaternion perceptron with a single quaternion weight.

        Args:
            learning_rate: Learning rate for weight updates.
            random_seed: Random seed for weight initialization.
        """
        self.forward_type = forward_type
        self.learning_rate = learning_rate

        self.buffer_size = buffer_size
        self._buffer_counter = 0
        self.bias_update_buffer: quaternion.quaternion = quaternion.quaternion(1, 0, 0, 0)
        self.action_update_buffer: quaternion.quaternion = quaternion.quaternion(1, 0, 0, 0)

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
        """Initialize weight as a random rotation."""
        # Generate random 4D vector
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
            Tuple containing incoming quaterion (accumulated input orientations) and outgoing quaternion (final orientation).
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
            Tuple containing incoming quaterion (accumulated input orientations) and outgoing quaternion (final orientation).
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
            Tuple containing incoming quaterion (accumulated input orientations) and outgoing quaternion (final orientation).
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
            Tuple containing incoming quaterion (accumulated input orientations) and outgoing quaternion (final orientation).
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
        """Predict class based on final rotation angle."""
        _, q_out = self.predict(inputs=inputs)
        return np.sign(q_out.w)

    def train(self, inputs: np.ndarray, label: int, tolerance: float = 1e-6) -> None:
        """Update weight using log/exp map for better stability."""
        bias_norm = abs(self.bias)
        if abs(bias_norm - 1) > 1e-10:
            logger.warning(
                "Bias is not a unit quaternion: ||%s|| = %s. Normalizing.",
                self.bias,
                bias_norm,
            )
            self.bias = self.bias / bias_norm

        action_norm = abs(self.action)
        if abs(action_norm - 1) > 1e-10:
            logger.warning(
                "Action is not a unit quaternion: ||%s|| = %s. Normalizing.",
                self.action,
                action_norm,
            )
            self.action = self.action / action_norm

        # inputs = np.array([q * [-1, 1, 1, 1] if q[0] < 0 else q for q in inputs])
        q_in, q_out = self.predict(inputs=inputs)  # Orientations.

        # Compute error as geodesic rotation from current to target
        q_target = quaternion.quaternion(label, 0, 0, 0)  # Orientation.
        q_error = self._compute_geodesic_rotation(source=q_out, target=q_target)  # Error rotation.

        # close_to_unity = q_target.conjugate() * q_out * q_error
        # close_to_zero = close_to_unity - quaternion.quaternion(close_to_unity.w, 0, 0, 0)
        # if abs(close_to_zero) > tolerance:
        #     raise ValueError("Bad error term: must take prediction to target quaternion; ||%s|| = %s.", close_to_zero, abs(close_to_zero))

        q_update = q_error.conjugate() ** self.learning_rate  # Must be close to the identity.
        if q_update.w < 0:
            q_update = -q_update
        u_b, u_residual, u_a = self.decompose_update(q_update=q_update, q_kernel=q_in)

        self.error_store.append(u_residual)

        if self.buffer_size is not None:
            self.bias_update_buffer = self.bias_update_buffer * u_b
            self.action_update_buffer = self.action_update_buffer * u_a
            self._buffer_counter += 1
            if self._buffer_counter == self.buffer_size:
                self.bias = self.bias * u_b
                self.bias = self.bias / abs(self.bias)

                self.action = self.action * u_a
                self.action = self.action / abs(self.action)

                self.bias_update_buffer = quaternion.quaternion(1, 0, 0, 0)
                self.action_update_buffer = quaternion.quaternion(1, 0, 0, 0)
                self._buffer_counter = 0
        else:
            self.bias = self.bias * u_b
            self.bias = self.bias / abs(self.bias)

            self.action = self.action * u_a
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
            logging.warning(
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
            logging.warning(
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
            logging.warning(
                "Action error component must be a unit quaternion: ||%s|| = %s.",
                u_a,
                abs(u_a),
            )
        return u_b, u_residual, u_a


def main(dataset: str) -> None:
    """Main function to run the quaternion perceptron."""
    if dataset == "mnist":
        _X_train, _y_train, _X_test, _y_test = load_mnist(model="quaternion")
    elif dataset == "xor":
        _X_train, _y_train, _X_test, _y_test = load_xor(dimensionality=8)
