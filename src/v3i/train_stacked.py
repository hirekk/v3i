"""Train a stack of quaternion perceptrons (act–observe–correct, forward-propagated error).

Example: 2 layers
  uv run python -m v3i.train_stacked --layers 2 --epochs 10
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from v3i.models.perceptron.quaterion import QuaternionPerceptron
from v3i.models.perceptron.quaterion import SimpleOptimizer
from v3i.models.perceptron.stack import Sequential

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train stacked quaternion perceptrons (forward error, no backprop)."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/binary_3sphere"))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=2, help="Number of stacked perceptrons.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    train = np.load(args.data_dir / "train.npz")
    test = np.load(args.data_dir / "test.npz")
    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    layers = [
        QuaternionPerceptron(
            learning_rate=args.lr, random_seed=args.seed + i, forward_type="right_multiplication"
        )
        for i in range(args.layers)
    ]
    model = Sequential(layers)
    optimizers = [SimpleOptimizer(layer) for layer in layers]

    n = len(y_train)

    def acc():
        tr = sum(model.predict_label(np.atleast_2d(X_train[i])) == y_train[i] for i in range(n)) / n
        te = sum(
            model.predict_label(np.atleast_2d(X_test[i])) == y_test[i] for i in range(len(y_test))
        ) / len(y_test)
        return tr, te

    train_acc_0, test_acc_0 = acc()
    logger.info(
        "Initial (untrained)  train_acc=%.2f%%  test_acc=%.2f%%",
        train_acc_0 * 100,
        test_acc_0 * 100,
    )

    rng = np.random.default_rng(args.seed)
    for epoch in range(args.epochs):
        for idx in rng.permutation(n):
            x = np.atleast_2d(X_train[idx])
            model.learn_step(x, int(y_train[idx]), optimizers)
        tr, te = acc()
        logger.info(
            "Epoch %d/%d  train_acc=%.2f%%  test_acc=%.2f%%",
            epoch + 1,
            args.epochs,
            tr * 100,
            te * 100,
        )


if __name__ == "__main__":
    main()
