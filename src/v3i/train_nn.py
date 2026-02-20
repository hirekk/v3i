"""Train a sequential network of quaternion or octonion perceptrons."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from v3i.algebra import Octonion
from v3i.models.perceptron import OctonionPerceptron
from v3i.models.perceptron import OctonionSequential

logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger.addHandler(handler)


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the accuracy of the model.

    Args:
        y_true: The true labels.
        y_pred: The predicted labels.

    Returns:
        The accuracy of the model.
    """
    return np.mean(y_pred == y_true)


def main(
    data_dir: Path,
    num_layers: int,
    num_epochs: int,
    learning_rate: float,
    random_seed: int | None,
) -> None:
    """Train a sequential network of octonion perceptrons.

    Args:
        data_dir:
            The directory containing the training and test data.
        num_layers:
            The number of octonion perceptrons in the network.
        num_epochs:
            The number of epochs to train the network.
        learning_rate:
            The learning rate for the network.
        random_seed:
            The random seed for the network.
    """
    # 1. Data Loading
    train = np.load(data_dir / "train.npz")
    test = np.load(data_dir / "test.npz")
    X_train, y_train = train["X"], train["y"]
    X_test, y_test = test["X"], test["y"]

    if X_train.shape[1] != 8:
        error_message = f"Octonion training requires 8D data. Got {X_train.shape[1]}D."
        raise ValueError(error_message)

    # 2. Model Initialization
    # Initializing near identity ensures signal doesn't vanish in deep chains.
    layers = [
        OctonionPerceptron(
            learning_rate=learning_rate,
            random_seed=random_seed if random_seed is not None else i,
        )
        for i in range(num_layers)
    ]
    model = OctonionSequential(layers)

    # 3. Initial Baseline

    train_acc_0 = calculate_accuracy(
        y_train, np.array([model.predict_label(Octonion(x)) for x in X_train])
    )
    test_acc_0 = calculate_accuracy(
        y_test, np.array([model.predict_label(Octonion(x)) for x in X_test])
    )
    logger.info("Initial: train_acc=%.2f%% test_acc=%.2f%%", train_acc_0 * 100, test_acc_0 * 100)

    # 4. Training Loop (Act-Observe-Correct)
    rng = np.random.default_rng(random_seed)
    n = len(y_train)

    for epoch in range(num_epochs):
        indices = rng.permutation(n)
        epoch_residual = 0.0

        for idx in indices:
            # --- ACT ---
            x_oct = Octonion(X_train[idx])
            model.forward(x_oct)

            # --- OBSERVE & CORRECT ---
            # Define target based on label (Identity for +1, -Identity for -1)
            target = Octonion.unit() if y_train[idx] >= 0 else -Octonion.unit()

            # The 'correct' call propagates the Torque Wave through all layers.
            residual_vec = model.correct(target)
            epoch_residual += np.linalg.norm(residual_vec)

        # 5. Renormalization Heartbeat (Essential for Manifold Stability)
        # Prevents floating point drift from pulling weights off the 7-sphere.
        for layer in model.layers:
            layer.weight = layer.weight.normalize()

        # 6. Logging
        train_acc = calculate_accuracy(
            y_train, np.array([model.predict_label(Octonion(x)) for x in X_train])
        )
        test_acc = calculate_accuracy(
            y_test, np.array([model.predict_label(Octonion(x)) for x in X_test])
        )
        logger.info(
            "Epoch %d/%d | Res: %.4f | Train Acc: %.2f%% | Test Acc: %.2f%%",
            epoch + 1,
            num_epochs,
            epoch_residual / n,
            train_acc * 100,
            test_acc * 100,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Octonion Manifold Trainer")
    p.add_argument("--data-type", choices=["binary-1d", "binary-xor"], default="binary-1d")
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--num-epochs", type=int, default=10)
    p.add_argument("--learning-rate", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=42)
    args = p.parse_args()

    data_path = Path("data") / args.data_type / "octonion"
    main(
        data_dir=data_path,
        num_layers=args.num_layers,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )
