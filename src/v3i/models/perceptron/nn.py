"""Stacked quaternion perceptrons with act–observe–correct and forward-propagated error.

No backprop: the correction term (error quaternion) is fed in as a normal input in "learn"
mode and flows forward; each layer updates to move its output toward identity (consume error).
"""

from __future__ import annotations

import numpy as np
import quaternion

from v3i.models.perceptron.quaternion import QuaternionPerceptron
from v3i.models.perceptron.quaternion import geodesic_rotation


class Sequential:
    """Stack of QuaternionPerceptron layers. Composable like NN layers.

    - Act: forward(x) runs input through each layer in order.
    - Observe: compare final output to target, get error quaternion.
    - Correct (learn mode): feed error as input, forward propagate; each layer
      updates with target=identity so the error is "consumed" layer by layer.
    """

    def __init__(self, layers: list[QuaternionPerceptron]) -> None:
        self.layers = list(layers)
        self._last_q_out: quaternion.quaternion | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Run input through all layers. Returns final output as (1, 4). Updates _last_q_out."""
        x = np.atleast_2d(x)
        for layer in self.layers:
            _, q_out = layer.predict(x)
            x = np.atleast_2d(quaternion.as_float_array(q_out))
        self._last_q_out = quaternion.quaternion(*x[0])
        return x

    def predict_label(self, x: np.ndarray) -> int:
        """Predict class from final layer output: +1 if q_out.w >= 0 else -1."""
        self.forward(x)
        return 1 if self._last_q_out.w >= 0 else -1

    def learn_mode(
        self,
        q_error: np.ndarray | quaternion.quaternion,
        optimizers: list,
    ) -> None:
        """Forward-propagate the error and update each layer (no backprop).

        Error is fed as input; each layer does its local act–observe–correct with
        target = identity (1,0,0,0), so the network learns to "consume" the error.
        """
        if isinstance(q_error, np.ndarray):
            q_error = quaternion.quaternion(*np.atleast_1d(q_error).ravel()[:4])
        inp = np.atleast_2d(quaternion.as_float_array(q_error))
        for layer, opt in zip(self.layers, optimizers, strict=True):
            _, q_out = layer.predict(inp)
            u_b, u_a = layer.compute_update(inp, 1)  # target = identity
            opt.step(u_b, u_a)
            inp = np.atleast_2d(quaternion.as_float_array(q_out))

    def learn_step(
        self,
        x: np.ndarray,
        label: int,
        optimizers: list,
    ) -> None:
        """Full act–observe–correct: forward(x), compute error, then learn_mode(error)."""
        self.forward(x)
        q_out = self._last_q_out
        q_target = quaternion.quaternion(label, 0, 0, 0)
        q_err = geodesic_rotation(q_out, q_target)
        self.learn_mode(q_err, optimizers)
