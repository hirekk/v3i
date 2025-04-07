"""Example of using QuaternionPerceptron on MNIST to classify even vs odd digits."""

from datetime import datetime
from itertools import starmap
import json
import logging
from pathlib import Path

import numpy as np
import quaternion
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from tqdm.auto import tqdm

from v3i.data.extract import DEFAULT_DATA_DIR
from v3i.models.baseline.decision_tree import DecisionTreeBaseline
from v3i.models.baseline.logistic_regression import LogisticRegressionBaseline
from v3i.models.baseline.random_choice import RandomChoiceBaseline
from v3i.models.perceptron.quaterion import QuaternionPerceptron

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def digit_to_label(digit: int) -> int:
    """Return 1 if digit is even, -1 if odd."""
    return 1 if digit % 2 == 0 else -1


def create_quaternion_from_pixel(
    pixel_value: float,
    x_pos: float,
    y_pos: float,
    img_size: int,
) -> np.ndarray:
    """Create a quaternion representation of a pixel."""
    if pixel_value < -1.0 or pixel_value > 1.0:
        error_msg = f"Pixel value must be between -1.0 and 1.0, got {pixel_value}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Normalize position to [-1, 1]
    normalized_x = (2.0 * x_pos / (img_size - 1)) - 1.0
    normalized_y = (2.0 * y_pos / (img_size - 1)) - 1.0

    # Create components [w,x,y,z]
    w = pixel_value
    x = normalized_x * np.sqrt((1 - w * w) / 3)
    y = normalized_y * np.sqrt((1 - w * w) / 3)
    z = np.sqrt(1 - w * w - x * x - y * y)

    # Create quaternion using the quaternion package
    return quaternion.as_float_array(quaternion.quaternion(w, x, y, z))


def convert_mnist_to_quaternions(images: list[np.ndarray], img_size: int = 28) -> np.ndarray:
    """Convert MNIST images to lists of quaternions."""
    quaternion_images = []

    for img in tqdm(images, desc="Converting images to quaternions"):
        # Reshape to 2D if flattened
        img_2d = img.reshape(img_size, img_size) if img.ndim == 1 else img

        # Convert to pixel values -1 to 1
        img_2d = (img_2d / 127.5) - 1.0

        # Create quaternion for each pixel
        quaternion_pixels = []
        for y in range(img_size):
            for x in range(img_size):
                pixel_value = img_2d[y, x]
                quaternion_pixels.append(
                    create_quaternion_from_pixel(
                        pixel_value=pixel_value,
                        x_pos=x,
                        y_pos=y,
                        img_size=img_size,
                    ),
                )

        quaternion_images.append(quaternion_pixels)

    return np.array(quaternion_images)


def main() -> None:
    """Main function to run the MNIST classification using QuaternionPerceptron."""
    QUATERNION_DATA_DIR = DEFAULT_DATA_DIR / "MNIST" / "quaternion"
    QUATERNION_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_quaternion_train_fpath = QUATERNION_DATA_DIR / "X_quaternion_train.npy"
    X_quaternion_test_fpath = QUATERNION_DATA_DIR / "X_quaternion_test.npy"
    X_baseline_train_fpath = QUATERNION_DATA_DIR / "X_baseline_train.npy"
    X_baseline_test_fpath = QUATERNION_DATA_DIR / "X_baseline_test.npy"
    y_train_fpath = QUATERNION_DATA_DIR / "y_train.npy"
    y_test_fpath = QUATERNION_DATA_DIR / "y_test.npy"

    if all(
        p.exists()
        for p in [
            X_quaternion_train_fpath,
            X_quaternion_test_fpath,
            X_baseline_train_fpath,
            X_baseline_test_fpath,
            y_train_fpath,
            y_test_fpath,
        ]
    ):
        logger.info("Loading existing data from %s", QUATERNION_DATA_DIR)
        X_quaternion_train = np.load(X_quaternion_train_fpath)
        X_quaternion_test = np.load(X_quaternion_test_fpath)
        X_baseline_train = np.load(X_baseline_train_fpath)
        X_baseline_test = np.load(X_baseline_test_fpath)
        y_train = np.load(y_train_fpath)
        y_test = np.load(y_test_fpath)
    else:
        logger.info("Converting MNIST data to quaternions")
        mnist_train = datasets.MNIST(root=DEFAULT_DATA_DIR, train=True, download=True)
        mnist_test = datasets.MNIST(root=DEFAULT_DATA_DIR, train=False, download=True)

        # Use subset of data for faster training
        train_samples, test_samples = 10000, 1000

        X_train = [np.array(mnist_train[i][0]) for i in range(train_samples)]
        y_train = [digit_to_label(mnist_train[i][1]) for i in range(train_samples)]
        X_test = [np.array(mnist_test[i][0]) for i in range(test_samples)]
        y_test = [digit_to_label(mnist_test[i][1]) for i in range(test_samples)]

        # Preprocess data
        scaler = StandardScaler()
        X_quaternion_train = convert_mnist_to_quaternions(X_train)
        X_quaternion_test = convert_mnist_to_quaternions(X_test)
        X_baseline_train = scaler.fit_transform([img.flatten() for img in X_train])
        X_baseline_test = scaler.transform([img.flatten() for img in X_test])

        # Save processed data
        np.save(QUATERNION_DATA_DIR / "X_quaternion_train.npy", X_quaternion_train)
        np.save(QUATERNION_DATA_DIR / "X_quaternion_test.npy", X_quaternion_test)
        np.save(QUATERNION_DATA_DIR / "X_baseline_train.npy", X_baseline_train)
        np.save(QUATERNION_DATA_DIR / "X_baseline_test.npy", X_baseline_test)
        np.save(QUATERNION_DATA_DIR / "y_train.npy", y_train)
        np.save(QUATERNION_DATA_DIR / "y_test.npy", y_test)

    # Setup experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    rng = np.random.default_rng(42)
    q_perceptron = QuaternionPerceptron(learning_rate=0.001, batch_size=10, random_seed=42)
    baselines = {
        "decision_tree": DecisionTreeBaseline(random_seed=42),
        "logistic": LogisticRegressionBaseline(random_seed=42),
        "random": RandomChoiceBaseline(random_seed=42),
    }

    # Training loop
    epochs = 50
    experiment_data = {
        "quaternion": {
            "weight_history": [],
            "train_accuracies": [],
            "test_accuracies": [],
        },
    }

    # Add baseline models to experiment data
    for name in baselines:
        experiment_data[name] = {
            "train_accuracies": [],
            "test_accuracies": [],
        }

    for epoch in range(epochs):
        # Train on shuffled data with progress bar
        perm = rng.permutation(len(X_train))
        pbar = tqdm(perm, desc=f"Epoch {epoch + 1}/{epochs}")

        quaternion_correct = 0
        for i in pbar:
            batch_idx = perm[i : i + 100]

            # Train quaternion model
            for idx in batch_idx:
                quaternion_pred = q_perceptron.predict(
                    q_perceptron.forward(
                        np.array(list(starmap(quaternion.quaternion, X_quaternion_train[idx]))),
                    ),
                )
                q_perceptron.train(X_quaternion_train[idx], y_train[idx])
                quaternion_correct += quaternion_pred == y_train[idx]

            # Train baselines on batch
            baseline_accs = {
                name: model.fit_batch(X_baseline_train[batch_idx], y_train[batch_idx])
                for name, model in baselines.items()
            }

            # Update progress bar
            pbar.set_postfix({
                "quaternion_acc": f"{quaternion_correct / (i + 100):.4f}",
                **{f"{k}_acc": f"{v:.4f}" for k, v in baseline_accs.items()},
            })

            # Record quaternion weights
            if (i // 100) % 10 == 0:
                w = quaternion.as_float_array(q_perceptron.weight)
                experiment_data["quaternion"]["weight_history"].append({
                    "epoch": epoch,
                    "step": i + 100,
                    "w": float(w[0]),
                    "x": float(w[1]),
                    "y": float(w[2]),
                    "z": float(w[3]),
                })

        # Record training accuracies
        experiment_data["quaternion"]["train_accuracies"].append(
            float(quaternion_correct / len(y_train)),
        )
        for name, model in baselines.items():
            train_acc = model.score(X_baseline_train, y_train)
            experiment_data[name]["train_accuracies"].append(float(train_acc))

        # Test accuracies
        quaternion_test_correct = sum(
            q_perceptron.predict(
                q_perceptron.forward(np.array(list(starmap(quaternion.quaternion, x)))),
            )
            == y
            for x, y in zip(X_quaternion_test, y_test, strict=False)
        )
        experiment_data["quaternion"]["test_accuracies"].append(
            float(quaternion_test_correct / len(y_test)),
        )

        for name, model in baselines.items():
            test_acc = model.score(X_baseline_test, y_test)
            experiment_data[name]["test_accuracies"].append(float(test_acc))

        # Log results
        log_msg = f"Epoch {epoch + 1} complete - "
        log_msg += f"Quaternion: {quaternion_test_correct / len(y_test):.4f}, "
        log_msg += ", ".join(
            f"{name}: {model.score(X_baseline_test, y_test):.4f}"
            for name, model in baselines.items()
        )
        logger.info(log_msg)

    # Save experiment data
    save_experiment_data(experiment_data)


def save_experiment_data(experiment_data: dict) -> None:
    """Save experiment data to a JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    with open(experiment_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment_data, f, indent=2)


if __name__ == "__main__":
    main()
