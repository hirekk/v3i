"""Train a quaternion perceptron on a binary classification dataset (e.g. binary_3sphere)."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from v3i.models.perceptron.quaterion import QuaternionPerceptron


def load_npz_data(data_dir: Path):
    """Load train and test from data_dir/train.npz and test.npz (keys X, y)."""
    data_dir = Path(data_dir)
    train = np.load(data_dir / "train.npz")
    test = np.load(data_dir / "test.npz")
    return (
        train["X"],
        train["y"],
        test["X"],
        test["y"],
    )


def accuracy(model: QuaternionPerceptron, X: np.ndarray, y: np.ndarray) -> float:
    """Fraction of samples correctly classified. X shape (n, 4) or (n, 1, 4)."""
    n = len(y)
    correct = 0
    for i in range(n):
        xi = X[i]
        if xi.ndim == 1:
            xi = xi.reshape(1, -1)
        if model.predict_label(xi) == y[i]:
            correct += 1
    return correct / n if n else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train quaternion perceptron.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/binary_3sphere"),
        help="Directory with train.npz and test.npz",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--buffer-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    X_train, y_train, X_test, y_test = load_npz_data(args.data_dir)
    model = QuaternionPerceptron(
        learning_rate=args.lr,
        buffer_size=args.buffer_size,
        random_seed=args.seed,
        forward_type="right_multiplication",
    )

    rng = np.random.default_rng(args.seed)
    n = len(y_train)

    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        for i in range(n):
            idx = perm[i]
            xi = X_train[idx]
            if xi.ndim == 1:
                xi = xi.reshape(1, -1)
            model.train(xi, int(y_train[idx]))
        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        print(f"Epoch {epoch + 1}/{args.epochs}  train_acc={train_acc:.4f}  test_acc={test_acc:.4f}")


if __name__ == "__main__":
    main()
