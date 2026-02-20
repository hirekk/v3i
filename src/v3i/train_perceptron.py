"""Train a single quaternion or octonion perceptron on binary classification."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from v3i.models.perceptron import ForwardType
from v3i.models.perceptron import OctonionBatchedOptimizer
from v3i.models.perceptron import OctonionPerceptron
from v3i.models.perceptron import OctonionSimpleOptimizer
from v3i.models.perceptron import QuaternionBatchedOptimizer
from v3i.models.perceptron import QuaternionPerceptron
from v3i.models.perceptron import QuaternionSimpleOptimizer

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
    forward_type: ForwardType,
    random_seed: int | None,
    octonion: bool,
) -> None:
    train = np.load(data_dir / "train.npz")
    test = np.load(data_dir / "test.npz")
    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    expected_dim = 8 if octonion else 4
    if X_train.shape[1] != expected_dim:
        raise ValueError(
            f"Data dimension {X_train.shape[1]} does not match "
            f"{'octonion (8)' if octonion else 'quaternion (4)'}. "
            f"Generate with: python -m v3i.make_data --binary-1d "
            f"{'--octonion' if octonion else '--quaternion'}"
        )

    if octonion:
        model = OctonionPerceptron(
            learning_rate=learning_rate,
            random_seed=random_seed,
            forward_type=forward_type,
        )
        optimizer = (
            OctonionSimpleOptimizer(model)
            if batch_size <= 1
            else OctonionBatchedOptimizer(model, batch_size)
        )
        BatchedCls = OctonionBatchedOptimizer
    else:
        model = QuaternionPerceptron(
            learning_rate=learning_rate,
            random_seed=random_seed,
            forward_type=forward_type,
        )
        optimizer = (
            QuaternionSimpleOptimizer(model)
            if batch_size <= 1
            else QuaternionBatchedOptimizer(model, batch_size)
        )
        BatchedCls = QuaternionBatchedOptimizer

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
            u, _ = model.compute_update(x, int(y_train[idx]))
            optimizer.step(u)
        if isinstance(optimizer, BatchedCls):
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
        description="Train a single quaternion or octonion perceptron. Use --quaternion or --octonion.",
    )
    dataset_group = parser.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--binary-1d",
        action="store_true",
        help="Binary classification on the line (±1 + noise).",
    )
    dataset_group.add_argument(
        "--binary-xor",
        action="store_true",
        help="XOR on the plane (four blobs).",
    )
    parser.add_argument("--num-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="1 = update every sample. >1 = batch N steps.",
    )
    parser.add_argument(
        "--forward-type",
        type=str,
        default="right_multiplication",
        choices=ForwardType.__args__,
    )
    parser.add_argument("--random-seed", type=int, default=0)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--quaternion", action="store_true", help="Train quaternion perceptron (4D data)."
    )
    group.add_argument(
        "--octonion", action="store_true", help="Train octonion perceptron (8D data)."
    )
    args = parser.parse_args()

    data_dir = Path("data")
    data_dir = (
        data_dir
        / ("binary_1d" if args.binary_1d else "binary_xor")
        / ("octonion" if args.octonion else "quaternion")
    )

    main(
        data_dir=data_dir,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        forward_type=args.forward_type,
        random_seed=args.random_seed,
        octonion=args.octonion,
    )
