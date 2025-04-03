"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class Octonion:
    """An octonion."""

    def __init__(self, components: list[float] | np.ndarray) -> None:
        """Initialize the octonion."""
        components_array = np.array(components, dtype=float)

        # Check dimensions
        if len(components_array) != 8:
            msg = f"Octonion must have exactly 8 components, got {len(components_array)}"
            raise ValueError(msg)

        # Check for NaN/Inf values
        if np.any(~np.isfinite(components_array)):
            msg = f"Octonion components must be finite, got {components_array}"
            raise ValueError(msg)

        self.components = components_array

    @classmethod
    def zero(cls) -> Octonion:
        """Create an octonion from an angle."""
        return cls([0] * 8)

    @property
    def theta(self) -> float:
        """The angle of the octonion."""
        return np.arccos(np.clip(self.components[0], -1.0, 1.0))

    @property
    def re(self) -> float:
        """The real part of the octonion."""
        return Octonion(self.components[0] + [0] * 7)

    @property
    def im(self) -> np.ndarray:
        """The imaginary part of the octonion."""
        return Octonion([0] + self.components[1:])

    @staticmethod
    def _quaternion_multiply(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Quaternion multiplication."""
        return np.array([
            x[0] * y[0] - x[1] * y[1] - x[2] * y[2] - x[3] * y[3],
            x[0] * y[1] + x[1] * y[0] + x[2] * y[3] - x[3] * y[2],
            x[0] * y[2] - x[1] * y[3] + x[2] * y[0] + x[3] * y[1],
            x[0] * y[3] + x[1] * y[2] - x[2] * y[1] + x[3] * y[0],
        ])

    @staticmethod
    def _quaternion_conjugate(x: np.ndarray) -> np.ndarray:
        """Quaternion conjugate."""
        return x * np.array([1, -1, -1, -1])

    def __mul__(self, other: float | Octonion) -> Octonion:
        """Octonion multiplication."""
        if isinstance(other, float):
            return Octonion(self.components * other)
        a, b = self.components[:4], self.components[4:]
        c, d = other.components[:4], other.components[4:]

        components = np.zeros(8)
        components[:4] = self._quaternion_multiply(a, c) - self._quaternion_multiply(
            self._quaternion_conjugate(d),
            b,
        )
        components[4:] = self._quaternion_multiply(d, a) + self._quaternion_multiply(
            b,
            self._quaternion_conjugate(c),
        )

        return Octonion(components)

    def __getitem__(self, index: int) -> float:
        """Get the component of the octonion."""
        if not isinstance(index, int):
            msg = f"Octonion indices must be integers, not {type(index).__name__}"
            raise TypeError(msg)
        if index < 0 or index >= 8:
            msg = f"Octonion index out of range; got {index} but must be in [0, 7]"
            raise IndexError(msg)
        return self.components[index]

    def __setitem__(self, index: int, value: float) -> None:
        """Set the component of the octonion."""
        if not isinstance(index, int):
            msg = f"Octonion indices must be integers, not {type(index).__name__}"
            raise TypeError(msg)
        if index < 0 or index >= 8:
            msg = f"Octonion index out of range; got {index} but must be in [0, 7]"
            raise IndexError(msg)
        self.components[index] = value

    def __rmul__(self, other: float | Octonion) -> Octonion:
        """Octonion multiplication from the right."""
        if isinstance(other, float):
            return Octonion(self.components * other)
        return other * self

    def __truediv__(self, other: float | Octonion) -> Octonion:
        """Octonion division."""
        if isinstance(other, float):
            return Octonion(self.components / other)
        if abs(other) < 1e-15:
            err_msg = f"Dividing by zero octonion: {other}"
            logger.error(err_msg)
            raise ValueError(err_msg)
        return self * other.inverse()

    def __neg__(self) -> Octonion:
        """Octonion negation."""
        return Octonion(-self.components)

    def __add__(self, other: Octonion) -> Octonion:
        """Octonion addition."""
        return Octonion(self.components + other.components)

    def __sub__(self, other: Octonion) -> Octonion:
        """Octonion subtraction."""
        return self + (-other)

    def __abs__(self) -> float:
        """Octonion absolute value."""
        return np.linalg.norm(self.components)

    def __repr__(self) -> str:
        """Representation of the octonion."""
        return f"Octonion({self.components.tolist()})"

    def has_nan(self) -> bool:
        """Check if the octonion has any NaN components."""
        return np.any(np.isnan(self.components))

    def star(self) -> Octonion:
        """Octonion star."""
        return Octonion(self.components * np.array([1, -1, -1, -1, -1, -1, -1, -1]))

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
            return self.star()

        return self.star() / norm_squared

    def normalize(self) -> Octonion:
        """Normalize the octonion."""
        # Check for NaN components
        if self.has_nan():
            err_msg = f"Attempting to normalize octonion with NaN components: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        norm = abs(self)
        if norm < 1e-15:
            err_msg = f"Attempting to normalize near-zero octonion: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

            # Handle numerical precision
        if abs(norm - 1.0) < 1e-15:
            # Already normalized within precision
            return self

        return self / norm

    def conjugate(self, other: Octonion) -> Octonion:
        """Conjugate the octonion."""
        # Check for zero octonions
        if abs(self) < 1e-15:
            err_msg = f"Attempting to conjugate a near-zero octonion: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if abs(other) < 1e-15:
            err_msg = f"Attempting to conjugate with a near-zero octonion: {other}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        return (other * self) * other.inverse()

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

        return np.dot(self.components, other.components)

    def copy(self) -> Octonion:
        """Copy the octonion."""
        return Octonion(self.components)

    def exp(self) -> Octonion:
        """Exponential of the octonion."""
        """Exponential map for octonions."""
        # Assume v has zero real part (pure imaginary)
        if self.components[0] > 1e-15:
            wrn_msg = f"Attempting to exponentiate octonion with non-zero real part: {self}"
            logger.warning(wrn_msg)
            v_pure = Octonion([0, *self.components[1:]])
        else:
            v_pure = self.copy()

        v_norm = abs(v_pure)
        if v_norm < 1e-15:
            return Octonion([1, 0, 0, 0, 0, 0, 0, 0])

        result = Octonion([np.cos(v_norm), 0, 0, 0, 0, 0, 0, 0])
        # Compute sin(|v|)v/|v| with numerical stability
        if v_norm < 1e-6:
            # Use Taylor series approximation for small values
            sin_term = 1.0 - (v_norm**2) / 6.0 + (v_norm**4) / 120.0
        else:
            sin_term = np.sin(v_norm) / v_norm

        imag_part = v_pure.components[1:] * sin_term
        result.components[1:] = imag_part

        return result

    def log(self) -> Octonion:
        """Logarithm of the octonion."""
        self_norm = self.normalize()

        # Extract real and imaginary parts
        re_component = self_norm.components[0]
        im_components = self_norm.components[1:]
        im_norm = np.sqrt(np.sum(im_components**2))

        # Handle special cases
        if im_norm < 1e-15:
            if re_component >= 0:
                return Octonion([0, 0, 0, 0, 0, 0, 0, 0])
            # Log of -1 is not uniquely defined in octonions
            err_msg = f"Attempting to log a near-zero octonion with negative real part: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Compute theta with numerical stability
        theta = np.arctan2(im_norm, re_component)

        # Create result
        result = Octonion([0, 0, 0, 0, 0, 0, 0, 0])
        result.components[1:] = im_components * (theta / im_norm)

        return result

    def geodesic_distance(self, other: Octonion) -> float:
        """Geodesic distance between two octonions."""
        # Check for zero octonions
        if abs(self) < 1e-15:
            err_msg = f"Attempting to compute geodesic distance with a near-zero octonion: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if abs(other) < 1e-15:
            err_msg = f"Attempting to compute geodesic distance with a near-zero octonion: {other}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Ensure both octonions are normalized to avoid numerical issues
        self_norm = self.normalize()
        other_norm = other.normalize()

        # Compute dot product with extra safety margin
        dot_product = np.clip(self_norm.dot(other_norm), -0.9999, 0.9999)

        # Compute arccos
        distance = np.arccos(dot_product)

        # Safety check for NaN
        if np.isnan(distance):
            return 0.01  # Small non-zero value instead of 0.0

        return distance

    def geodesic_direction(self, other: Octonion) -> Octonion:
        """Geodesic direction between two octonions."""
        # Check for zero octonions
        if abs(self) < 1e-15:
            err_msg = f"Attempting to compute geodesic direction from a near-zero octonion: {self}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        if abs(other) < 1e-15:
            err_msg = f"Attempting to compute geodesic direction to a near-zero octonion: {other}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Ensure both octonions are normalized
        self_norm = self.normalize()
        other_norm = other.normalize()

        # Compute the difference vector
        diff = other_norm - self_norm

        # Project onto tangent space at self_norm
        # (Remove component parallel to self_norm)
        dot_product = self_norm.dot(diff)
        tangent_vector = diff - self_norm * dot_product

        # Check if tangent vector is too small
        tangent_norm = abs(tangent_vector)
        if tangent_norm < 1e-10:
            # Check if points are antipodal (dot product close to -1)
            dot = self_norm.dot(other_norm)
            if abs(dot + 1) < 1e-10:
                # For antipodal points, return a consistent perpendicular direction
                # Use a hash of the inputs to get a consistent random seed
                seed = hash((tuple(self.components), tuple(other.components))) % 2**32
                rng = np.random.RandomState(seed)

                # Generate a random direction in the tangent space
                random_dir = rng.normal(0, 1, 8)
                # Project to make it perpendicular to self_norm
                dot_product = np.sum(random_dir * self_norm.components)
                random_dir = random_dir - dot_product * self_norm.components
                # Normalize
                random_dir = random_dir / np.linalg.norm(random_dir)

                return Octonion(random_dir)
            # Points are very close, return zero
            return Octonion.zero()

        # Normalize the tangent vector
        return tangent_vector.normalize()


def compute_associator(x: Octonion, y: Octonion, z: Octonion) -> Octonion:
    """Octonion associator."""
    # Check for zero octonions
    if abs(x) < 1e-15:
        err_msg = f"Attempting to compute associator with a near-zero octonion: {x}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if abs(y) < 1e-15:
        err_msg = f"Attempting to compute associator with a near-zero octonion: {y}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    if abs(z) < 1e-15:
        err_msg = f"Attempting to compute associator with a near-zero octonion: {z}"
        logger.error(err_msg)
        raise ValueError(err_msg)

    return x * (y * z) - x * (y * z)


class OctonionPerceptron:
    """A classic perceptron model using octonions for weights and computations."""

    def __init__(self, input_size: int, learning_rate: float = 0.01, random_seed: int = 0) -> None:
        """Initialize the octonion perceptron.

        Args:
            input_size: Number of input features
            learning_rate: Learning rate for weight updates
            random_seed: Random seed for weight initialization
        """
        self.learning_rate: float = learning_rate
        self.random_seed: int = random_seed
        self.rng: np.random.Generator = np.random.default_rng(seed=random_seed)

        self.weights: list[Octonion] = self._initialize_weights(input_size)
        self.updates: list[Octonion] = [Octonion([1] + [0] * 7) for _ in range(input_size)]

    def _initialize_weights(self, input_size: int) -> list[Octonion]:
        """Initialize weights as random octonions on the unit L2 sphere."""
        weights = self.rng.normal(0, 1, (input_size, 8))
        weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
        return [Octonion(w) for w in weights]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass of the perceptron using octonion conjugation.

        Computes f(x) = w * x * w^(-1) where w is the weight and x is the input.

        Args:
            inputs: Numpy array of shape (n_samples, n_features)

        Returns:
            Numpy array of shape (n_samples, 1) representing the result of the octonion conjugation
        """
        if len(inputs) != len(self.weights):
            msg = f"Expected {len(self.weights)} inputs, got {len(inputs)}"
            raise ValueError(msg)

        # Aggregate inputs by element-wise conjugation with weights
        # result = Octonion([0] + [1 / np.sqrt(7)] * 7)  # USE WITH CONJUGATION ACCUMULATION
        result = Octonion([1] + [0] * 7)  # USE WITH MULTIPLICATION ACCUMULATION
        for i, (x, w) in enumerate(zip(inputs, self.weights, strict=False)):
            x_oct = Octonion(x)

            # Skip near-zero inputs
            if abs(x_oct) < 1e-15:
                wrn_msg = f"Near-zero input at index {i}, skipping."
                logger.warning(wrn_msg)
                continue

            # Apply conjugation: (w * x) * w⁻¹
            transformed = x_oct.conjugate(w)
            if abs(transformed) < 1e-15:
                wrn_msg = f"Conjugation resulted in near-zero octonion at index {i}, skipping."
                logger.warning(wrn_msg)
                continue

            transformed = transformed.normalize()

            result = (result * transformed).normalize()

            # Check for numerical issues
            if result.has_nan():
                err_msg = f"NaN detected in forward pass at index {i}"
                logger.error(err_msg)
                raise ValueError(err_msg)

        return result

    def predict(self, result: Octonion) -> int:
        """Predict the class of the result.

        Args:
            result: Octonion output from forward pass

        Returns:
            int: Classification (0 or 1)
        """
        # For binary classification, we can use the sign of the real part
        return 1 if result[0] >= 0 else -1

    def loss(self, prediction: Octonion, target: Octonion) -> Octonion:
        """Compute the loss for a single example.

        Args:
            prediction: Octonion output from forward pass.
            target: Octonion target.

        Returns:
            Loss octonion.
        """
        err = target * prediction.inverse()
        return err.normalize()

    def train(self, inputs: np.ndarray, label: int) -> None:
        """Train the perceptron on a single example.

        Args:
            inputs: Numpy array of shape (n_features,)
            label: Target value (-1 or 1)
        """
        result: Octonion = self.forward(inputs)
        prediction: int = self.predict(result)

        # Only update if prediction is wrong
        if prediction != label:
            target: Octonion = Octonion([label, 0, 0, 0, 0, 0, 0, 0])
            loss: Octonion = self.loss(result, target)

            self.reset_updates()
            for i, x in enumerate(inputs):
                x_oct = Octonion(x)
                if abs(x_oct) < 1e-10:
                    continue

                # Current weight
                self.weights[i]
                update_contrib = self.compute_weight_update_log_exp(
                    loss=loss,
                    x=x_oct,
                )
                # update_contrib = self.compute_weight_update_bidirectional(
                #     loss=loss,
                #     x=x_oct,
                #     w=w,
                # )
                self.updates[i] = self.updates[i].conjugate(update_contrib).normalize()

    def apply_updates(self) -> None:
        """Apply the updates to the weights."""
        for i, (w, u) in enumerate(zip(self.weights, self.updates, strict=False)):
            self.weights[i] = w.inverse().conjugate(u).normalize()
        self.reset_updates()

    def reset_updates(self) -> None:
        """Reset the updates."""
        self.updates = [Octonion([1] + [0] * 7) for _ in range(len(self.weights))]

    def compute_weight_update_log_exp(
        self,
        loss: Octonion,
        x: Octonion,
    ) -> Octonion:
        """Create a rotation octonion that rotates in the given direction.

        Args:
            x: Input octonion.
            loss: Loss octonion.

        Returns:
            Rotation octonion.
        """
        # Handle zero cases
        if abs(loss) < 1e-15 or abs(x) < 1e-15:
            return Octonion([1, 0, 0, 0, 0, 0, 0, 0])  # Identity rotation

        try:
            # Normalize loss to ensure it's on the unit sphere
            loss_norm = loss.normalize()

            # Extract real and imaginary parts
            real_part = loss_norm[0]
            imag_parts = loss_norm.components[1:].copy()
            imag_norm = np.linalg.norm(imag_parts)

            # Balance real and imaginary components
            # If real part is too small compared to imaginary parts, boost it
            if abs(real_part) < 0.1 and imag_norm > 0.1:
                # Boost real part while maintaining unit norm
                boost_factor = 0.3  # Adjust this to control real part influence
                new_real = np.sign(real_part) * boost_factor

                # Scale down imaginary parts to maintain unit norm
                scale_factor = np.sqrt(1 - new_real**2) / imag_norm if imag_norm > 0 else 0
                new_imag = imag_parts * scale_factor

                # Create balanced loss
                balanced_components = np.zeros(8)
                balanced_components[0] = new_real
                balanced_components[1:] = new_imag
                loss_norm = Octonion(balanced_components)

            # Convert loss to a rotation generator
            loss_generator = loss_norm.log()

            # Scale by learning rate and input magnitude
            # Use a smaller scale factor to prevent wild oscillations
            scale_factor = min(self.learning_rate * abs(x), 0.1)  # Cap the scale factor
            scaled_generator = loss_generator * scale_factor

            # Create rotation through exponential map
            rotation = scaled_generator.exp()

            # Ensure result is normalized
            return rotation.normalize()
        except Exception as err:
            err_msg = f"Error in log-exp weight update: {err}"
            logger.exception(err_msg)
            raise ValueError(err_msg) from err

    def compute_weight_update_bidirectional(
        self,
        loss: Octonion,
        x: Octonion,
        w: Octonion,
    ) -> Octonion:
        """Standard rotation for imaginary components.

        Args:
            loss: Loss octonion.
            x: Input octonion.
            w: Weight octonion.

        Returns:
            Rotation octonion.
        """
        try:
            # Normalize inputs
            w_norm = w.normalize()
            loss_norm = loss.normalize()

            # Standard rotation for imaginary components
            direction = w_norm.geodesic_direction(-loss_norm)
            angle1 = self.learning_rate * abs(x) * w_norm.geodesic_distance(loss_norm)

            # Cap angle for stability
            angle1 = min(angle1, np.pi / 2)

            # Create standard rotation
            rotation1 = Octonion.zero()
            rotation1[0] = np.cos(angle1 / 2)
            if abs(direction) > 1e-15:
                rotation1.components[1:] = np.sin(angle1 / 2) * direction.normalize().components[1:]

            # Create a rotation in the real-imaginary plane
            # The angle depends on the real part error
            real_error = loss_norm[0]
            angle2 = self.learning_rate * abs(x) * abs(real_error) * np.sign(real_error)

            # Cap angle for stability
            angle2 = min(angle2, np.pi / 4)

            # Choose which imaginary component to rotate with
            # Use a more robust approach: pick component with largest magnitude in both input and loss
            x_imag_magnitude = np.abs(x.components[1:])
            loss_imag_magnitude = np.abs(loss_norm.components[1:])
            combined_magnitude = x_imag_magnitude * loss_imag_magnitude

            if np.sum(combined_magnitude) < 1e-15:
                # If no clear direction, use first imaginary component
                main_imag_idx = 1
            else:
                # Convert np.int64 to Python int
                main_imag_idx = int(np.argmax(combined_magnitude) + 1)

            # Create real-imaginary rotation
            rotation2 = Octonion([np.cos(angle2 / 2), 0, 0, 0, 0, 0, 0, 0])
            rotation2[main_imag_idx] = np.sin(angle2 / 2)

            # Apply both rotations
            return (rotation1 * rotation2).normalize()

        except Exception as e:
            logger.exception(f"Error in bidirectional weight update: {e}")
            return Octonion([1, 0, 0, 0, 0, 0, 0, 0])  # Return identity on error
