"""Quaternion-based Perceptron implementation from scratch.

A classic perceptron whose weight, inputs, and outputs are unit quaternions
instead of real numbers; the forward pass accumulates inputs by
right-multiplication and applies the weight, and the update is a geodesic
rotation of the weight toward the target pole.
"""

from __future__ import annotations

import numpy as np
import quaternion


def geodesic_rotation(
    source: quaternion.quaternion,
    target: quaternion.quaternion,
) -> quaternion.quaternion:
    """Minimal rotation (right-multiply) from source to target: source * R = target."""
    r = source.conjugate() * target
    r = r / abs(r)
    if r.w < 0:  # take the shorter path (angle ≤ π)
        r = -r
    return r


class QuaternionPerceptron:
    """A perceptron using a single unit-quaternion weight, updated along geodesics."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        random_seed: int | None = None,
    ) -> None:
        """Initialize with a near-identity unit-quaternion weight."""
        self.learning_rate = learning_rate
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=random_seed)
        # Unit quaternion weight: rotation applied on the right (output = reduced * weight).
        self.weight = self._initialize_weight()

    def _initialize_weight(self) -> quaternion.quaternion:
        """Identity plus a small random perturbation, normalized to unit length."""
        components = np.array([1, 0, 0, 0], dtype=np.float64)
        components += self._rng.normal(0, 0.1, 4)
        components = components / np.linalg.norm(components)
        if components[0] < 0:
            components = -components
        return quaternion.quaternion(*components)

    def forward(
        self, inputs: np.ndarray, tolerance: float = 1e-10
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Accumulate inputs by right-multiplication, then apply the weight.

        Returns (reduced, result): the accumulated input orientation and the output
        orientation after right-multiplying by the weight.
        """
        reduced = quaternion.quaternion(1, 0, 0, 0)
        for x in inputs:
            x_q = quaternion.quaternion(*x) if isinstance(x, np.ndarray) else x
            if abs(x_q) < tolerance:
                continue
            reduced = reduced * (x_q / abs(x_q))
        return reduced, reduced * self.weight

    def predict(self, inputs: np.ndarray) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Forward pass: (accumulated input, output) orientations."""
        return self.forward(inputs)

    def predict_label(self, inputs: np.ndarray) -> int:
        """Predict class: +1 if q_out.w >= 0 else -1 (avoids 0 from np.sign)."""
        _, q_out = self.predict(inputs=inputs)
        return 1 if q_out.w >= 0 else -1

    def compute_update(
        self, inputs: np.ndarray, label: int
    ) -> tuple[quaternion.quaternion, quaternion.quaternion]:
        """Proposed (u, u_residual): rotation update for the weight and residual for stacking."""
        self._ensure_unit_weight()
        _, q_out = self.predict(inputs=inputs)
        q_target = quaternion.quaternion(label, 0, 0, 0)
        q_error = geodesic_rotation(q_out, q_target)
        u = q_error**self.learning_rate
        if u.w < 0:
            u = -u
        return u, u

    def apply_update(self, u: quaternion.quaternion) -> None:
        """Apply rotation update to the weight (right multiply) and renormalize."""
        self.weight = self.weight * u
        self.weight = self.weight / abs(self.weight)

    def _ensure_unit_weight(self) -> None:
        if abs(abs(self.weight) - 1) > 1e-10:
            self.weight = self.weight / abs(self.weight)
