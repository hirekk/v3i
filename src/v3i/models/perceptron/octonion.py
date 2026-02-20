"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging

import numpy as np

from v3i.algebra import Octonion
from v3i.algebra import cross_product_7d

logger = logging.getLogger(__name__)


class OctonionPerceptron:
    """A single-unit non-associative processing element constrained to the 7-sphere ($S^7$)."""

    def __init__(
        self,
        learning_rate: float = 0.1,
        random_seed: int | None = None,
    ) -> None:
        """Initializes the OctonionPerceptron.

        Args:
            learning_rate:
                The scaling factor applied to the torque vector during weight updates.
                Must be between 0 and 1. Defaults to 0.1.
            random_seed:
                Optional seed for the internal pseudo-random number generator to ensure
                deterministic weight initialization. Defaults to None.
        """
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=self.random_seed)

        self.weight = self._initialize_weight()
        self.last_input: Octonion | None = None
        self.last_output: Octonion | None = None

    def _initialize_weight(self) -> Octonion:
        """Unit octonion: identity + small random perturbation."""
        perturbation = self._rng.normal(0, 0.05, 8)
        return (Octonion.unit() + Octonion(perturbation)).normalize()

    def _ensure_unit_weight(self) -> None:
        if abs(abs(self.weight) - 1.0) > 1e-10:
            self.weight = self.weight.normalize()

    def _inputs_to_unit_octonions(
        self, inputs: np.ndarray, tolerance: float = 1e-10
    ) -> list[Octonion]:
        """Convert input array to list of unit octonions (skipping near-zero)."""
        out: list[Octonion] = []
        for x in inputs:
            row = np.atleast_1d(x).ravel()[:8]
            if len(row) < 8:
                row = np.resize(row, 8)
            o = Octonion(row)
            if abs(o) >= tolerance:
                out.append(o / abs(o))
        return out

    def forward(self, x: Octonion) -> Octonion:
        """Signal propagation (The 'Act' step).

        Performs right-multiplication: y = x * w.

        Args:
            x:
                The input signal to be processed.

        Returns:
            The output signal after right-multiplication.
        """
        self.last_input = x
        self.last_output = x * self.weight
        return self.last_output

    def correct(self, incoming_error: np.ndarray) -> np.ndarray:
        """The 'correct' step (forward error propagation).

        Args:
            incoming_error: 8D tangent vector in the identity space.

        Returns:
            The residual error 8D vector transported for the next layer.
        """
        if self.last_input is None:
            error_message = "Must call forward() before correct()."
            raise RuntimeError(error_message)

        # 1. Transport global error to local weight frame
        # r_local = w_inv * r_global * w
        w_inv = self.weight.conjugate()
        r_global = Octonion(incoming_error)
        r_local = w_inv * r_global * self.weight

        # 2. Compute local torque (Commutator Alignment)
        # Torque is the 7D cross product of weight and local error.
        torque_vec = np.zeros(8)
        torque_vec[1:] = cross_product_7d(self.weight.im, r_local.im)
        torque = Octonion(torque_vec)

        # 3. Compute Associator-based scaling factor (Kappa)
        kappa = self._compute_kappa(self.last_input, self.weight, r_local)

        # 4. Apply Geodesic Update (The 'Bite')
        # We move along the geodesic in the direction of the torque.
        update_mag = self.learning_rate * kappa
        # delta_w = exp(update_mag * torque)
        self.weight = (self.weight * (torque * update_mag).exp()).normalize()

        # 5. Debt Accounting: Subtraction in local tangent space
        # r_residual_local = r_local - projection(r_local onto torque)
        absorbed = self._project(r_local.to_array(), torque_vec)
        r_res_local = r_local.to_array() - absorbed

        # 6. Adjoint Transport forward to the next layer
        # r_next = w * r_res_local * w_inv
        r_next = self.weight * Octonion(r_res_local) * self.weight.conjugate()

        return r_next.to_array()

    def _compute_kappa(self, q: Octonion, w: Octonion, r: Octonion) -> float:
        """Measures non-associative drift to scale the update."""
        # [q, w, r] = (qw)r - q(wr)
        assoc = (q * w) * r - q * (w * r)
        denom = abs(q) * abs(w) * abs(r)
        if denom < 1e-15:
            return 1.0
        return 1.0 - np.clip(abs(assoc) / denom, 0.0, 1.0)

    def _project(self, vector: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Orthogonal projection of vector onto target."""
        mag_sq = np.dot(target, target)
        if mag_sq < 1e-15:
            return np.zeros(8)
        return (np.dot(vector, target) / mag_sq) * target

    def predict_label(self, x: Octonion) -> int:
        """Binary classification helper based on the real part."""
        output = self.forward(x)
        return 1 if output.re >= 0 else -1


class OctonionForwardOptimizer:
    """Orchestrates the Forward-Wave correction across a chain."""

    def __init__(self, layers: list[OctonionPerceptron]) -> None:
        self.layers = layers

    def step(self, global_error: np.ndarray) -> None:
        """Propagates the error wave through all layers."""
        current_error = global_error
        for layer in self.layers:
            current_error = layer.correct(current_error)


class OctonionSequential:
    """A sequential container for Octonion Perceptrons.

    Coordinates the Act (forward) and Correct (forward-error) phases.
    The error propagates in the same direction as the data, but transforms
    the manifold state of each layer as it passes through.
    """

    def __init__(self, layers: list[OctonionPerceptron]) -> None:
        """Initializes the model with a list of layers.

        Args:
            layers: Ordered list of OctonionPerceptron instances.
        """
        self.layers = layers

    def forward(self, x: Octonion) -> Octonion:
        """Forward Pass: Signal propagation.

        Args:
            x: Input unit octonion.

        Returns:
            The final prediction octonion on S^7.
        """
        current_val = x
        for layer in self.layers:
            current_val = layer.forward(current_val)
        return current_val

    def correct(self, target: Octonion) -> np.ndarray:
        """Correction Pass: Forward-Wave Error Propagation.

        Args:
            target: The desired output octonion (e.g. Octonion.unit()).

        Returns:
            The final residual 8D error vector after the wave completes.
        """
        # 1. Observe the terminal output
        p = self.layers[-1].last_output
        if p is None:
            raise RuntimeError("Must call forward() before correct().")

        # 2. Compute Global Error r = log(p_inv * target)
        # This defines the "Torque" needed to rotate the output to the target.
        error_oct = (p.conjugate() * target).log()
        current_error_vec = error_oct.to_array()

        # 3. Propagate the correction wave forward through the hierarchy
        for layer in self.layers:
            current_error_vec = layer.correct(current_error_vec)

        return current_error_vec

    def predict_label(self, x: Octonion) -> int:
        """Helper for binary classification."""
        output = self.forward(x)
        return 1 if output.re >= 0 else -1
