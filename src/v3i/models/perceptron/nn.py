"""Stacked quaternion perceptrons with act-observe-correct and forward-propagated error.

No backprop: the correction term (error quaternion) is fed in as a normal input in "learn"
mode and flows forward; each layer updates to move its output toward identity (consume error).
"""

from __future__ import annotations

import numpy as np
import quaternion

from v3i.models.perceptron.quaternion import QuaternionPerceptron
from v3i.models.perceptron.quaternion import geodesic_rotation


class QuaternionSequential:
    """Stack of QuaternionPerceptron layers. Composable like NN layers.

    - Act: forward(x) runs input through each layer in order.
    - Observe: compare final output to target, get error quaternion.
    - Correct (learn mode): feed error as input, forward propagate; each layer
      updates with target=identity so the error is "consumed" layer by layer.
    """

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

    def learn_mode(
        self,
        q_error: np.ndarray | quaternion.quaternion,
        optimizers: list,
    ) -> None:
        """Forward-propagate the error and update each layer (no backprop).

        First layer receives the overall error, updates its weights, and returns the
        residual update (the part it did not absorb). That residual is fed to the
        next layer, which updates and passes its residual on, and so on.
        """
        if isinstance(q_error, np.ndarray):
            q_error = quaternion.quaternion(*np.atleast_1d(q_error).ravel()[:4])
        inp = np.atleast_2d(quaternion.as_float_array(q_error))
        for layer, opt in zip(self.layers, optimizers, strict=True):
            u, u_residual = layer.compute_update(inp, 1)  # target = identity
            opt.step(u)
            inp = np.atleast_2d(quaternion.as_float_array(u_residual))

    def _learn_backward(
        self,
        q_error: quaternion.quaternion,  # noqa: ARG002 — currently unused; see note below
        optimizers: list,
        hidden_list: list[np.ndarray],
    ) -> None:
        """LIFO pass: the last layer pairs the output error with its real input.

        Each layer's residual becomes the previous layer's error. Note: q_error
        is currently unused here — the wave restarts from each layer's stored
        input with target=identity (see the error-wave ticket).
        """
        err = None
        for i in range(len(self.layers) - 1, -1, -1):
            inp = hidden_list[i] if err is None else np.atleast_2d(quaternion.as_float_array(err))
            u, u_residual = self.layers[i].compute_update(inp, 1)
            optimizers[i].step(u)
            err = u_residual

    def learn_step(
        self,
        x: np.ndarray,
        label: int,
        optimizers: list,
    ) -> None:
        """Act-observe-correct: forward(x) recording activations, then LIFO learn."""
        x = np.atleast_2d(x)
        hidden_list = [x.copy()]
        for layer in self.layers:
            _, q_out = layer.predict(x)
            x = np.atleast_2d(quaternion.as_float_array(q_out))
            hidden_list.append(x.copy())
        self._last_q_out = quaternion.quaternion(*x[0])
        q_target = quaternion.quaternion(label, 0, 0, 0)
        q_err = geodesic_rotation(self._last_q_out, q_target)
        self._learn_backward(q_err, optimizers, hidden_list)
