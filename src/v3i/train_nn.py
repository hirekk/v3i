"""Train a sequential network of quaternion or octonion perceptrons."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from v3i.models.perceptron import ForwardType
from v3i.models.perceptron import OctonionBatchedOptimizer
from v3i.models.perceptron import OctonionPerceptron
from v3i.models.perceptron import OctonionSequential
from v3i.models.perceptron import OctonionSimpleOptimizer
from v3i.models.perceptron import QuaternionBatchedOptimizer
from v3i.models.perceptron import QuaternionPerceptron
from v3i.models.perceptron import QuaternionSequential
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
    num_layers: int,
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
            f"Generate with: python -m v3i.make_data --binary-1d {'--octonion' if octonion else '--quaternion'}"
        )

    if octonion:
        layers = [
            OctonionPerceptron(
                learning_rate=learning_rate,
                random_seed=random_seed if random_seed is not None else i,
                forward_type=forward_type,
            )
            for i in range(num_layers)
        ]
        model = OctonionSequential(layers)
        if batch_size <= 1:
            optimizers = [OctonionSimpleOptimizer(layer) for layer in layers]
        else:
            optimizers = [OctonionBatchedOptimizer(layer, batch_size) for layer in layers]
        BatchedCls = OctonionBatchedOptimizer
    else:
        layers = [
            QuaternionPerceptron(
                learning_rate=learning_rate,
                random_seed=random_seed if random_seed is not None else i,
                forward_type=forward_type,
            )
            for i in range(num_layers)
        ]
        model = QuaternionSequential(layers)
        if batch_size <= 1:
            optimizers = [QuaternionSimpleOptimizer(layer) for layer in layers]
        else:
            optimizers = [QuaternionBatchedOptimizer(layer, batch_size) for layer in layers]
        BatchedCls = QuaternionBatchedOptimizer

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

    rng = np.random.default_rng(random_seed)
    for epoch in range(num_epochs):
        for idx in rng.permutation(n):
            x = np.atleast_2d(X_train[idx])
            label = int(y_train[idx])
            # if model.predict_label(x) != label:
            #     model.learn_step(x, label, optimizers)
            model.learn_step(x, label, optimizers)
        for opt in optimizers:
            if isinstance(opt, BatchedCls):
                opt.flush()
        tr, te = acc()
        logger.info(
            "Epoch %d/%d  train_acc=%.2f%%  test_acc=%.2f%%",
            epoch + 1,
            num_epochs,
            tr * 100,
            te * 100,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Train stacked perceptrons. Use --quaternion or --octonion."
    )
    dataset_group = p.add_mutually_exclusive_group(required=True)
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
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="1 = update every sample. >1 = batch N steps.",
    )
    p.add_argument(
        "--forward-type",
        type=str,
        default="right_multiplication",
        choices=ForwardType.__args__,
    )
    p.add_argument("--random-seed", type=int, default=0)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--quaternion", action="store_true", help="Quaternion stack (4D data).")
    group.add_argument("--octonion", action="store_true", help="Octonion stack (8D data).")
    args = p.parse_args()

    data_dir = Path("data")
    data_dir = (
        data_dir
        / ("binary_1d" if args.binary_1d else "binary_xor")
        / ("octonion" if args.octonion else "quaternion")
    )

    main(
        data_dir=data_dir,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        forward_type=args.forward_type,
        random_seed=args.random_seed,
        octonion=args.octonion,
    )
