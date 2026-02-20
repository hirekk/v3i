"""Octonion-based Perceptron implementation from scratch.

This module implements a classic perceptron model that uses octonions
for weights, inputs, and outputs instead of real numbers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from v3i.numbers import Octonion
from v3i.numbers import slerp

if TYPE_CHECKING:
    from typing import Literal

    ForwardType = Literal[
        "left_multiplication",
        "right_multiplication",
        "two_bracketings",
    ]

logger = logging.getLogger(__name__)


def _tangent_space_avg_octonion(oct_list: list[Octonion], scale: float = 1.0) -> Octonion:
    """Average unit octonions in tangent space; scale shrinks the step."""
    if not oct_list:
        return Octonion.unit()
    vecs = np.array([o.to_rotation_vector() for o in oct_list])
    avg_vec = np.mean(vecs, axis=0) * scale
    n = np.linalg.norm(avg_vec)
    if n < 1e-12:
        return Octonion.unit()
    return Octonion.from_rotation_vector(avg_vec)


def left_assoc(oct_list: list[Octonion]) -> Octonion:
    """Left-associated product: ((o0*o1)*o2)*...  Parenthesization matters for octonions."""
    if not oct_list:
        return Octonion.unit()
    acc = oct_list[0]
    for o in oct_list[1:]:
        acc = acc * o  # (acc * o) is the next left-associated product
    return acc


def right_assoc(oct_list: list[Octonion]) -> Octonion:
    """Right-associated product: o0*(o1*(o2*...))  Parenthesization matters for octonions."""
    if not oct_list:
        return Octonion.unit()
    acc = oct_list[-1]
    for o in reversed(oct_list[:-1]):
        acc = o * acc  # (o * acc) is the next right-associated product
    return acc


class OctonionPerceptron:
    """Perceptron on the octonion algebra; non-associativity is central to the forward pass.

    Octonions are non-associative: (a*b)*c != a*(b*c). They are also non-commutative.
    The forward pass is designed so that this property is front and center:
    - left_multiplication / right_multiplication: one fixed parenthesization (left- or
      right-associated product of inputs), then * weight. Different bracketings yield
      different outputs; we pick one.
    - two_bracketings: compute BOTH left-associated and right-associated products of the
      same input sequence, then combine them (tangent-space average). The output explicitly
      depends on the fact that the two bracketings differ; we do not collapse to a single
      "rotation" as with quaternions.
    """

    FORWARD_TYPES = ("left_multiplication", "right_multiplication", "two_bracketings")

    def __init__(
        self,
        learning_rate: float = 0.01,
        random_seed: int | None = None,
        forward_type: ForwardType = "right_multiplication",
    ) -> None:
        self.forward_type = forward_type
        self.learning_rate = learning_rate
        self.error_store: list[Octonion] = []
        self.random_seed = random_seed
        self._rng = np.random.default_rng(seed=random_seed)
        self.weight = self._initialize_weight()

    def _initialize_weight(self) -> Octonion:
        """Unit octonion: identity + small random perturbation."""
        return Octonion.unit() + Octonion(self._rng.normal(0, 0.1, 8)).normalize()

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

    def forward_left_multiplication(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[Octonion, Octonion]:
        """Left-associated product ((x0*x1)*x2)*... then * weight. One fixed bracketing."""
        octs = self._inputs_to_unit_octonions(inputs, tolerance)
        reduced = left_assoc(octs) if octs else Octonion.unit()
        result = reduced * self.weight
        return reduced, result

    def forward_right_multiplication(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[Octonion, Octonion]:
        """Right-associated product x0*(x1*(x2*...)) then * weight. One fixed bracketing."""
        octs = self._inputs_to_unit_octonions(inputs, tolerance)
        reduced = right_assoc(octs) if octs else Octonion.unit()
        result = reduced * self.weight
        return reduced, result

    def forward_two_bracketings(
        self,
        inputs: np.ndarray,
        tolerance: float = 1e-10,
    ) -> tuple[Octonion, Octonion]:
        """Explicitly use non-associativity: compute both bracketings, then combine.

        left = ((x0*x1)*x2)*..., right = x0*(x1*(x2*...)). For octonions left != right.
        We combine them in tangent space and multiply by weight so the output
        depends on both parenthesizations.
        """
        octs = self._inputs_to_unit_octonions(inputs, tolerance)
        if not octs:
            reduced = Octonion.unit()
        else:
            left_r = left_assoc(octs)
            right_r = right_assoc(octs)
            reduced = _tangent_space_avg_octonion([left_r, right_r], scale=1.0)
            reduced = reduced.normalize()
        result = reduced * self.weight
        return reduced, result

    def predict(self, inputs: np.ndarray) -> tuple[Octonion, Octonion]:
        """Return (reduced, output). Output depends on parenthesization (non-associative)."""
        match self.forward_type:
            case "left_multiplication":
                return self.forward_left_multiplication(inputs=inputs)
            case "right_multiplication":
                return self.forward_right_multiplication(inputs=inputs)
            case "two_bracketings":
                return self.forward_two_bracketings(inputs=inputs)
            case _:
                error_message = (
                    f"Invalid forward_type: {self.forward_type}; use one of {self.FORWARD_TYPES}"
                )
                raise ValueError(error_message)

    def predict_label(self, inputs: np.ndarray) -> int:
        """+1 if output.re >= 0 else -1."""
        _, o_out = self.predict(inputs=inputs)
        return 1 if o_out.re >= 0 else -1

    def compute_update(self, inputs: np.ndarray, label: int) -> tuple[Octonion, Octonion]:
        """Proposed (u, u_residual). Same semantics as quaternion: learning_rate scales the weight step.

        We compute the full rotation u_full that would move output to target, then take only
        learning_rate of that step: u = slerp(1, u_full, learning_rate). So learning_rate has
        the same meaning as for quaternions (fraction of the step in weight space).
        """
        self._ensure_unit_weight()
        reduced, o_out = self.predict(inputs=inputs)

        o_target = Octonion.unit() if label >= 0 else -Octonion.unit()

        # Full step: weight_new such that reduced * weight_new = o_target
        weight_new = (reduced.inverse() * o_target).normalize()
        u_full = (self.weight.inverse() * weight_new).normalize()
        u = slerp(Octonion.unit(), u_full, self.learning_rate).normalize()
        self.error_store.append(u)
        return u, u

    def apply_update(self, u: Octonion) -> None:
        """Apply update on the right and renormalize (Correct-step heartbeat per review)."""
        self.weight = (self.weight * u).normalize()

    def train(self, inputs: np.ndarray, label: int) -> None:
        u, _ = self.compute_update(inputs, label)
        self.apply_update(u)


class OctonionSimpleOptimizer:
    """Apply every update u immediately."""

    def __init__(self, model: OctonionPerceptron) -> None:
        self._model = model

    def step(self, u: Octonion) -> None:
        self._model.apply_update(u)


class OctonionBatchedOptimizer:
    """Accumulate updates and apply tangent-space average every batch_size steps."""

    def __init__(self, model: OctonionPerceptron, batch_size: int) -> None:
        self._model = model
        self.batch_size = batch_size
        self._u_buf: list[Octonion] = []

    def step(self, u: Octonion) -> None:
        self._u_buf.append(u)
        if len(self._u_buf) >= self.batch_size:
            self._apply_batch()

    def flush(self) -> None:
        if self._u_buf:
            self._apply_batch()

    def _apply_batch(self) -> None:
        n = len(self._u_buf)
        u_avg = _tangent_space_avg_octonion(self._u_buf, scale=1.0 / n)
        self._model.apply_update(u_avg)
        self._u_buf.clear()
