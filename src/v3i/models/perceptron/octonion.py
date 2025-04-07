"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging

import numpy as np
import quaternion

logger = logging.getLogger(__name__)


class Octonion:
    """An octonion."""

    def __init__(
        self,
        x0: float,
        x1: float,
        x2: float,
        x3: float,
        x4: float,
        x5: float,
        x6: float,
        x7: float,
    ) -> None:
        """Initialize the octonion."""
        # Check for NaN/Inf values
        self.x0 = x0
        self.x1 = x1
        self.x2 = x2
        self.x3 = x3
        self.x4 = x4
        self.x5 = x5
        self.x6 = x6
        self.x7 = x7

        self._components = np.array([x0, x1, x2, x3, x4, x5, x6, x7])

    @classmethod
    def from_components(cls, components: list[float] | np.ndarray) -> Octonion:
        """Create an octonion from a list of components."""
        return cls(*components)

    @classmethod
    def zero(cls) -> Octonion:
        """Create an octonion from an angle."""
        return cls(0, 0, 0, 0, 0, 0, 0, 0)

    @classmethod
    def unit(cls) -> Octonion:
        """Create a unit octonion."""
        return cls(1, 0, 0, 0, 0, 0, 0, 0)

    @property
    def theta(self) -> float:
        """The angle of the octonion."""
        return np.arccos(np.clip(self.x0, -1.0, 1.0))

    @property
    def re(self) -> float:
        """The real part of the octonion."""
        return self.x0

    @property
    def im(self) -> np.ndarray:
        """The imaginary part of the octonion."""
        return np.array([self.x1, self.x2, self.x3, self.x4, self.x5, self.x6, self.x7])

    def __mul__(self, other: float | Octonion) -> Octonion:
        """Octonion multiplication."""
        if isinstance(other, float | int):
            return Octonion.from_components(self._components * other)

        # Extract components
        a0, a1, a2, a3, a4, a5, a6, a7 = self._components
        b0, b1, b2, b3, b4, b5, b6, b7 = other._components
        
        # Direct computation of octonion multiplication
        c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b7 - a5*b6 + a6*b5 - a7*b4
        c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
        c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b5 - a5*b4 + a6*b7 - a7*b6
        c4 = a0*b4 - a1*b7 - a2*b6 - a3*b5 + a4*b0 + a5*b3 + a6*b2 + a7*b1
        c5 = a0*b5 + a1*b6 - a2*b7 + a3*b4 - a4*b3 + a5*b0 - a6*b1 + a7*b2
        c6 = a0*b6 - a1*b5 + a2*b4 - a3*b7 - a4*b2 + a5*b1 + a6*b0 + a7*b3
        c7 = a0*b7 + a1*b4 + a2*b5 + a3*b6 - a4*b1 - a5*b2 - a6*b3 + a7*b0

        return Octonion(c0, c1, c2, c3, c4, c5, c6, c7)

    def __getitem__(self, index: int) -> float:
        """Get the component of the octonion."""
        return self._components[index]

    def __setitem__(self, index: int, value: float) -> None:
        """Set the component of the octonion."""
        self._components[index] = value

    def __rmul__(self, other: float | Octonion) -> Octonion:
        """Octonion multiplication from the right."""
        if isinstance(other, float | int):
            return Octonion.from_components(self._components * other)

        # Extract components
        a0, a1, a2, a3, a4, a5, a6, a7 = other._components
        b0, b1, b2, b3, b4, b5, b6, b7 = self._components
        
        # Direct computation of octonion multiplication
        c0 = a0*b0 - a1*b1 - a2*b2 - a3*b3 - a4*b4 - a5*b5 - a6*b6 - a7*b7
        c1 = a0*b1 + a1*b0 + a2*b3 - a3*b2 + a4*b7 - a5*b6 + a6*b5 - a7*b4
        c2 = a0*b2 - a1*b3 + a2*b0 + a3*b1 + a4*b6 + a5*b7 - a6*b4 - a7*b5
        c3 = a0*b3 + a1*b2 - a2*b1 + a3*b0 + a4*b5 - a5*b4 + a6*b7 - a7*b6
        c4 = a0*b4 - a1*b7 - a2*b6 - a3*b5 + a4*b0 + a5*b3 + a6*b2 + a7*b1
        c5 = a0*b5 + a1*b6 - a2*b7 + a3*b4 - a4*b3 + a5*b0 - a6*b1 + a7*b2
        c6 = a0*b6 - a1*b5 + a2*b4 - a3*b7 - a4*b2 + a5*b1 + a6*b0 + a7*b3
        c7 = a0*b7 + a1*b4 + a2*b5 + a3*b6 - a4*b1 - a5*b2 - a6*b3 + a7*b0

        return Octonion(c0, c1, c2, c3, c4, c5, c6, c7)


    def __truediv__(self, other: float | Octonion) -> Octonion:
        """Octonion division."""
        if isinstance(other, float | int):
            return Octonion.from_components(self._components / other)
        if abs(other) < 1e-15:
            err_msg = f"Dividing by zero octonion: {other}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self * other.inverse()

    def __neg__(self) -> Octonion:
        """Octonion negation."""
        return Octonion.from_components(-self._components)

    def __add__(self, other: Octonion) -> Octonion:
        """Octonion addition."""
        return Octonion.from_components(self._components + other._components)

    def __sub__(self, other: Octonion) -> Octonion:
        """Octonion subtraction."""
        return self + (-other)

    def __abs__(self) -> float:
        """Octonion absolute value."""
        return np.linalg.norm(self._components)

    def __repr__(self) -> str:
        """Representation of the octonion."""
        return f"Octonion(x0={self.x0}, x1={self.x1}, x2={self.x2}, x3={self.x3}, x4={self.x4}, x5={self.x5}, x6={self.x6}, x7={self.x7})"

    def has_nan(self) -> bool:
        """Check if the octonion has any NaN components."""
        return np.any(np.isnan(self._components))

    def conjugate(self) -> Octonion:
        """Octonion conjugate."""
        return Octonion.from_components(
            self._components * np.array([1, -1, -1, -1, -1, -1, -1, -1]),
        )

    def inverse(self) -> Octonion:
        """Octonion inverse."""
        # Check for NaN components
        if self.has_nan():
            err_msg = f"Attempting to invert octonion with NaN components: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        norm_squared = abs(self) ** 2

        if norm_squared < 1e-15:
            err_msg = f"Attempting to invert near-zero octonion: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if abs(norm_squared - 1.0) < 1e-15:
            return self.conjugate()

        return self.conjugate() / norm_squared

    # def conjugate(self, other: Octonion) -> Octonion:
    #     """Conjugate the octonion."""
    #     # Check for zero octonions
    #     if abs(self) < 1e-15:
    #         err_msg = f"Attempting to conjugate a near-zero octonion: {self}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     if abs(other) < 1e-15:
    #         err_msg = f"Attempting to conjugate with a near-zero octonion: {other}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     return (other * self) * other.inverse()

    def dot(self, other: Octonion) -> float:
        """Dot product between two octonions."""
        # Check for zero octonions
        if abs(self) < 1e-15:
            err_msg = f"Attempting to dot with a near-zero octonion: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if abs(other) < 1e-15:
            err_msg = f"Attempting to dot with a near-zero octonion: {other}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return np.dot(self._components, other._components)

    def copy(self) -> Octonion:
        """Copy the octonion."""
        return Octonion.from_components(self._components)

    def exp(self) -> Octonion:
        """Exponential of the octonion."""
        """Exponential map for octonions."""
        # Assume v has zero real part (pure imaginary)
        if self.x0 > 1e-15:
            wrn_msg = f"Attempting to exponentiate octonion with non-zero real part: {self}"
            logger.warning(wrn_msg)
            v_pure = Octonion(*[0, *self._components[1:]])
        else:
            v_pure = self.copy()

        v_norm = abs(v_pure)
        if v_norm < 1e-15:
            return Octonion.unit()

        result = Octonion(*[np.cos(v_norm), 0, 0, 0, 0, 0, 0, 0])
        # Compute sin(|v|)v/|v| with numerical stability
        if v_norm < 1e-6:
            # Use Taylor series approximation for small values
            sin_term = 1.0 - (v_norm**2) / 6.0 + (v_norm**4) / 120.0
        else:
            sin_term = np.sin(v_norm) / v_norm

        imag_part = v_pure.im * sin_term
        result[1:] = imag_part

        return result

    def log(self) -> Octonion:
        """Logarithm of the octonion."""
        self_norm = self / abs(self)

        # Extract real and imaginary parts
        re_component = self_norm.x0
        im_components = self_norm.im
        im_norm = np.sqrt(np.sum(im_components**2))

        # Handle special cases
        if im_norm < 1e-15:
            if re_component >= 0:
                return Octonion.zero()
            # Log of -1 is not uniquely defined in octonions
            err_msg = f"Attempting to log a near-zero octonion with negative real part: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Compute theta with numerical stability
        theta = np.arctan2(im_norm, re_component)

        # Create result
        result = Octonion.zero()
        result[1:] = im_components * (theta / im_norm)

        return result

    # def geodesic_distance(self, other: Octonion) -> float:
    #     """Geodesic distance between two octonions."""
    #     # Check for zero octonions
    #     if abs(self) < 1e-15:
    #         err_msg = f"Attempting to compute geodesic distance with a near-zero octonion: {self}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     if abs(other) < 1e-15:
    #         err_msg = f"Attempting to compute geodesic distance with a near-zero octonion: {other}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     # Ensure both octonions are normalized to avoid numerical issues
    #     self_norm = self / abs(self)
    #     other_norm = other / abs(other)

    #     # Compute dot product with extra safety margin
    #     dot_product = np.clip(self_norm.dot(other_norm), -0.9999, 0.9999)

    #     # Compute arccos
    #     distance = np.arccos(dot_product)

    #     # Safety check for NaN
    #     if np.isnan(distance):
    #         return 0.01  # Small non-zero value instead of 0.0

    #     return distance

    # def geodesic_direction(self, other: Octonion) -> Octonion:
    #     """Geodesic direction between two octonions."""
    #     # Check for zero octonions
    #     if abs(self) < 1e-15:
    #         err_msg = f"Attempting to compute geodesic direction from a near-zero octonion: {self}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     if abs(other) < 1e-15:
    #         err_msg = f"Attempting to compute geodesic direction to a near-zero octonion: {other}"
    #         logger.error(err_msg)
    #         raise ValueError(err_msg)

    #     # Ensure both octonions are normalized
    #     self_norm = self / abs(self)
    #     other_norm = other / abs(other)

    #     # Compute the difference vector
    #     diff = other_norm - self_norm

    #     # Project onto tangent space at self_norm
    #     # (Remove component parallel to self_norm)
    #     dot_product = self_norm.dot(diff)
    #     tangent_vector = diff - self_norm * dot_product

    #     # Check if tangent vector is too small
    #     tangent_norm = abs(tangent_vector)
    #     if tangent_norm < 1e-10:
    #         # Check if points are antipodal (dot product close to -1)
    #         dot = self_norm.dot(other_norm)
    #         if abs(dot + 1) < 1e-10:
    #             # For antipodal points, return a consistent perpendicular direction
    #             # Use a hash of the inputs to get a consistent random seed
    #             seed = hash((tuple(self.components), tuple(other.components))) % 2**32
    #             rng = np.random.RandomState(seed)

    #             # Generate a random direction in the tangent space
    #             random_dir = rng.normal(0, 1, 8)
    #             # Project to make it perpendicular to self_norm
    #             dot_product = np.sum(random_dir * self_norm.components)
    #             random_dir = random_dir - dot_product * self_norm.components
    #             # Normalize
    #             random_dir = random_dir / np.linalg.norm(random_dir)

    #             return Octonion(random_dir)
    #         # Points are very close, return zero
    #         return Octonion.zero()

    #     # Normalize the tangent vector
    #     return tangent_vector / abs(tangent_vector)


class OctonionPerceptron:
    """A biologically-inspired perceptron using a single octonion weight."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        batch_size: int = 100,
        random_seed: int = 0,
    ) -> None:
        """Initialize the octonion perceptron with a single octonion weight.

        Args:
            learning_rate: Learning rate for weight updates.
            batch_size: Batch size for buffered updates.
            random_seed: Random seed for weight initialization.
        """
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.update_buffer: list[Octonion] = []

        self.rng = np.random.default_rng(seed=random_seed)
        self.weight = self._initialize_weight()

    def _initialize_weight(self) -> Octonion:
        """Initialize weight as a random rotation octonion."""
        # Generate random 8D vector
        components = self.rng.normal(0, 1, 8)
        # Normalize to unit length
        components = components / np.linalg.norm(components)
        return Octonion.from_components(components)

    def forward(self, inputs: np.ndarray) -> Octonion:
        """Forward pass computing final rotation state.

        Args:
            inputs: Array of octonions representing the input

        Returns:
            Final rotated state as an octonion
        """
        # Accumulate inputs through multiplication
        result = Octonion.unit()  # Identity octonion
        for x in inputs:
            x_oct = x if isinstance(x, Octonion) else Octonion(*x)
            if abs(x_oct) < 1e-10:
                continue
            x_n = x_oct / abs(x_oct)
            result = x_n * result

        # Apply weight transformation
        return self.weight * result

    def predict(self, rotated: Octonion) -> int:
        """Predict class based on real component."""
        return 1 if rotated[0] >= 0 else -1

    def train(self, inputs: np.ndarray, label: int, tolerance: float = 1e-10) -> None:
        """Update weight using log/exp map for better stability."""
        # Forward pass
        rotated = self.forward(inputs)

        # Compute error as geodesic rotation from current to target
        label * Octonion.unit()

        # Only update if prediction is wrong
        if self.predict(rotated) != label:
            # Convert error to log space (axis-angle)
            error_vec = rotated.log()

            # Scale the error by learning rate
            scaled_error = error_vec * (
                -self.learning_rate
            )  # Negative because we want to move opposite to error

            # Convert to octonion and add to update buffer
            update = scaled_error.exp()
            self.update_buffer.append(update)

            # Apply updates if buffer is full
            if len(self.update_buffer) == self.batch_size:
                # Average the updates in log space
                mean_update = Octonion.unit()  # Identity
                for update in self.update_buffer:
                    mean_update = mean_update * update
                mean_update = mean_update / abs(mean_update)

                # Apply update: rotate weight by update
                self.weight = self.weight * mean_update

                # Normalize weight
                self.weight = self.weight / abs(self.weight)

                # Reset buffer
                self.update_buffer = []
