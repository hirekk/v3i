"""Train a quaternion perceptron on a binary classification dataset."""

import argparse
import logging
from pathlib import Path

import numpy as np

from v3i.models.perceptron.quaternion import BatchedOptimizer
from v3i.models.perceptron.quaternion import ForwardType
from v3i.models.perceptron.quaternion import QuaternionPerceptron
from v3i.models.perceptron.quaternion import SimpleOptimizer

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def main(
    data_dir: Path,
    num_epochs: int,
    learning_rate: float,
    batch_size: int,
    forward_type: ForwardType = "right_multiplication",
    random_seed: int | None = None,
) -> None:
    train = np.load(data_dir / "train.npz")
    test = np.load(data_dir / "test.npz")
    X_train, y_train = train["X"], train["y"]  # noqa: N806
    X_test, y_test = test["X"], test["y"]  # noqa: N806

    model = QuaternionPerceptron(
        learning_rate=learning_rate,
        random_seed=random_seed,
        forward_type=forward_type,
    )
    optimizer = SimpleOptimizer(model) if batch_size <= 1 else BatchedOptimizer(model, batch_size)
    rng = np.random.default_rng(random_seed)
    n = len(y_train)

    def acc(m, X, y):
        return sum(m.predict_label(np.atleast_2d(X[i])) == y[i] for i in range(len(y))) / len(y)

    train_acc_0 = acc(model, X_train, y_train)
    test_acc_0 = acc(model, X_test, y_test)
    logger.info(
        "Initial (untrained)  train_acc=%.2f%%  test_acc=%.2f%%",
        train_acc_0 * 100.0,
        test_acc_0 * 100.0,
    )

    for epoch in range(num_epochs):
        perm = rng.permutation(n)
        for idx in perm:
            x = np.atleast_2d(X_train[idx])
            u_b, _, u_a = model.compute_update(x, int(y_train[idx]))
            optimizer.step(u_b, u_a)
        if isinstance(optimizer, BatchedOptimizer):
            optimizer.flush()
        train_acc = acc(model, X_train, y_train)
        test_acc = acc(model, X_test, y_test)
        logger.info(
            "Epoch %d/%d  train_acc=%.2f%%  test_acc=%.2f%%",
            epoch + 1,
            num_epochs,
            train_acc * 100.0,
            test_acc * 100.0,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a quaternion perceptron on a binary classification dataset.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default="data/binary_3sphere",
        help="Path to the data directory. Default: data/binary_1d",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs. Default: 10",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate. Default: 0.01",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="1 = update every sample (SimpleOptimizer). >1 = batch N steps (BatchedOptimizer).",
    )
    parser.add_argument(
        "--forward-type",
        type=str,
        default="right_multiplication",
        help="Forward type. Default: right_multiplication.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed. Default: 0",
    )
    args = parser.parse_args()

    main(
        data_dir=args.data_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        forward_type=args.forward_type,
        random_seed=args.random_seed,
    )
