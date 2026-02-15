"""Train a stack of quaternion perceptrons."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from v3i.models.perceptron.nn import Sequential
from v3i.models.perceptron.quaternion import QuaternionPerceptron
from v3i.models.perceptron.quaternion import SimpleOptimizer

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Train stacked quaternion perceptrons (forward error, no backprop)."
    )
    p.add_argument("--data-dir", type=Path, default=Path("data/binary_1d"))
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--layers", type=int, default=2, help="Number of stacked perceptrons.")
    p.add_argument("--random-seed", type=int, default=0)
    args = p.parse_args()

    train = np.load(args.data_dir / "train.npz")
    test = np.load(args.data_dir / "test.npz")
    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    layers = [
        QuaternionPerceptron(
            learning_rate=args.learning_rate,
            random_seed=args.random_seed + i,
            forward_type="right_multiplication",
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

    rng = np.random.default_rng(args.random_seed)
    for epoch in range(args.num_epochs):
        for idx in rng.permutation(n):
            x = np.atleast_2d(X_train[idx])
            label = int(y_train[idx])
            if model.predict_label(x) != label:
                model.learn_step(x, label, optimizers)
        tr, te = acc()
        logger.info(
            "Epoch %d/%d  train_acc=%.2f%%  test_acc=%.2f%%",
            epoch + 1,
            args.num_epochs,
            tr * 100,
            te * 100,
        )


if __name__ == "__main__":
    main()
