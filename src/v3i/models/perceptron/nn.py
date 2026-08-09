"""Stacked quaternion perceptrons with act-observe-correct and forward-propagated error.

No backprop: after the forward pass, a LIFO update pass moves each layer's output
toward identity from its recorded input, and the residual becomes the previous
layer's error.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import quaternion

if TYPE_CHECKING:
    from v3i.models.perceptron.quaternion import QuaternionPerceptron


class QuaternionSequential:
    """Stack of QuaternionPerceptron layers, composable like NN layers."""

    def __init__(self, layers: list[QuaternionPerceptron]) -> None:
        """Wrap an ordered list of perceptron layers."""
        self.layers = list(layers)
        self._last_q_out: quaternion.quaternion | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run input through all layers. Returns final output as (1, 4)."""
        x = np.atleast_2d(x)
        for layer in self.layers:
            _, q_out = layer.predict(x)
            x = np.atleast_2d(quaternion.as_float_array(q_out))
        self._last_q_out = quaternion.quaternion(*x[0])
        return x

    def predict_label(self, x: np.ndarray) -> int:
        """Predict class from final layer output: +1 if q_out.w >= 0 else -1."""
        self.forward(x)
        return 1 if self._last_q_out and self._last_q_out.w >= 0 else -1

    def learn_step(self, x: np.ndarray, label: int) -> None:  # noqa: ARG002
        """Act-observe-correct: forward(x) recording activations, then a LIFO update pass.

        Each layer updates toward identity from its recorded input; the residual
        becomes the previous layer's error (no backprop). `label` is currently
        unused — the wave restarts from each layer's stored input with
        target=identity (see the error-wave design).
        """
        x = np.atleast_2d(x)
        hidden_list = [x.copy()]
        for layer in self.layers:
            _, q_out = layer.predict(x)
            x = np.atleast_2d(quaternion.as_float_array(q_out))
            hidden_list.append(x.copy())
        self._last_q_out = quaternion.quaternion(*x[0])
        err = None
        for i in range(len(self.layers) - 1, -1, -1):
            inp = hidden_list[i] if err is None else np.atleast_2d(quaternion.as_float_array(err))
            u, u_residual = self.layers[i].compute_update(inp, 1)  # target = identity
            self.layers[i].apply_update(u)
            err = u_residual
