"""Example of using OctonionPerceptron on MNIST to classify even vs odd digits."""

from datetime import datetime
from itertools import starmap
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from torchvision import datasets
from tqdm.auto import tqdm

from v3i.data.extract import DEFAULT_DATA_DIR
from v3i.models.baseline.decision_tree import DecisionTreeBaseline
from v3i.models.baseline.logistic_regression import LogisticRegressionBaseline
from v3i.models.baseline.random_choice import RandomChoiceBaseline
from v3i.models.perceptron.octonion import Octonion
from v3i.models.perceptron.octonion import OctonionPerceptron

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OCTONION_DATA_DIR = DEFAULT_DATA_DIR / "MNIST" / "octonion"
OCTONION_DATA_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42


def digit_to_label(digit: int) -> int:
    """Return 1 if digit is even, -1 if odd."""
    return 1 if digit % 2 == 0 else -1


def create_octonion_from_pixel(
    pixel_value: float,
    x_pos: float,
    y_pos: float,
    img_size: int,
    random_seed: int = 0,
) -> np.ndarray:
    """Create an octonion representation of a pixel.

    Args:
        pixel_value: The pixel value
        x_pos: x-coordinate of the pixel
        y_pos: y-coordinate of the pixel
        img_size: Size of the image (assuming square)
        random_seed: The random seed for the random number generator

    Returns:
        np.ndarray: Octonion representation of the pixel
    """
    if pixel_value < -1.0 or pixel_value > 1.0:
        error_msg = f"Pixel value must be between -1.0 and 1.0, got {pixel_value}"
        logger.error(error_msg)
        raise ValueError(error_msg)

    np.random.RandomState(random_seed)

    # Normalize position to [-1, 1]
    normalized_x = (2.0 * x_pos / (img_size - 1)) - 1.0
    normalized_y = (2.0 * y_pos / (img_size - 1)) - 1.0

    # Generate random values for remaining components
    # Using unit normal distribution
    # random_components = rng.normal(loc=0, scale=1 / np.sqrt(6), size=5)
    random_components = np.zeros(5)

    result = np.array([pixel_value, normalized_x, normalized_y, *random_components])
    return result / np.linalg.norm(result)


def convert_mnist_to_octonions(images: list[np.ndarray], img_size: int = 28) -> np.ndarray:
    """Convert MNIST images to lists of octonions.

    Args:
        images: List of flattened MNIST images
        img_size: Size of the square images

    Returns:
        List of lists of octonions
    """
    octonion_images = []

    for img in tqdm(images, desc="Converting images to octonions"):
        # Reshape to 2D if flattened
        img_2d = img.reshape(img_size, img_size) if img.ndim == 1 else img

        # Convert to pixel values -1 to 1
        img_2d = (img_2d / 127.5) - 1.0

        # Create octonion for each pixel
        octonion_pixels = []
        for y in range(img_size):
            for x in range(img_size):
                pixel_value = img_2d[y, x]
                octonion_pixels.append(
                    create_octonion_from_pixel(
                        pixel_value=pixel_value,
                        x_pos=x,
                        y_pos=y,
                        img_size=img_size,
                    ),
                )

        octonion_images.append(octonion_pixels)

    return np.array(octonion_images)


def main() -> None:
    """Main function to run the MNIST classification using QuaternionPerceptron."""
    X_octonion_train_fpath = OCTONION_DATA_DIR / "X_octonion_train.npy"
    X_octonion_test_fpath = OCTONION_DATA_DIR / "X_octonion_test.npy"
    X_baseline_train_fpath = OCTONION_DATA_DIR / "X_baseline_train.npy"
    X_baseline_test_fpath = OCTONION_DATA_DIR / "X_baseline_test.npy"
    y_train_fpath = OCTONION_DATA_DIR / "y_train.npy"
    y_test_fpath = OCTONION_DATA_DIR / "y_test.npy"

    if all(
        p.exists()
        for p in [
            X_octonion_train_fpath,
            X_octonion_test_fpath,
            X_baseline_train_fpath,
            X_baseline_test_fpath,
            y_train_fpath,
            y_test_fpath,
        ]
    ):
        logger.info("Loading existing data from %s", OCTONION_DATA_DIR)
        X_octonion_train = np.load(X_octonion_train_fpath)
        X_octonion_test = np.load(X_octonion_test_fpath)
        X_baseline_train = np.load(X_baseline_train_fpath)
        X_baseline_test = np.load(X_baseline_test_fpath)
        y_train = np.load(y_train_fpath)
        y_test = np.load(y_test_fpath)
    else:
        logger.info("Converting MNIST data to octonions")
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
        X_octonion_train = convert_mnist_to_octonions(X_train)
        X_octonion_test = convert_mnist_to_octonions(X_test)
        X_baseline_train = scaler.fit_transform([img.flatten() for img in X_train])
        X_baseline_test = scaler.transform([img.flatten() for img in X_test])

        # Save processed data
        np.save(OCTONION_DATA_DIR / "X_octonion_train.npy", X_octonion_train)
        np.save(OCTONION_DATA_DIR / "X_octonion_test.npy", X_octonion_test)
        np.save(OCTONION_DATA_DIR / "X_baseline_train.npy", X_baseline_train)
        np.save(OCTONION_DATA_DIR / "X_baseline_test.npy", X_baseline_test)
        np.save(OCTONION_DATA_DIR / "y_train.npy", y_train)
        np.save(OCTONION_DATA_DIR / "y_test.npy", y_test)

    # Setup experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    rng = np.random.RandomState(RANDOM_SEED)
    octonion_perceptron = OctonionPerceptron(learning_rate=0.01, batch_size=10, random_seed=RANDOM_SEED)
    baselines = {
        "decision_tree": DecisionTreeBaseline(random_seed=RANDOM_SEED),
        "logistic": LogisticRegressionBaseline(random_seed=RANDOM_SEED),
        "random": RandomChoiceBaseline(random_seed=RANDOM_SEED),
    }

    # Convert labels to numpy array if they aren't already
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Training loop
    epochs = 30
    experiment_data = {
        "octonion": {"weight_history": [], "train_accuracies": [], "test_accuracies": []},
    }

    # Add baseline models to experiment data
    for name in baselines:
        experiment_data[name] = {"train_accuracies": [], "test_accuracies": []}

    for epoch in range(epochs):
        n_samples = len(y_train)
        n_batches = n_samples // 100

        perm = rng.permutation(n_samples)
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{epochs}")

        octonion_correct = 0
        for i in pbar:
            # Get batch indices
            start_idx = i * 100
            end_idx = start_idx + 100
            batch_idx = perm[start_idx:end_idx]

            # Train octonion model
            for idx in batch_idx:
                inputs = np.array(list(starmap(Octonion, X_octonion_train[idx])))
                target = y_train[idx]
                octonion_pred = octonion_perceptron.predict(octonion_perceptron.forward(inputs))
                octonion_perceptron.train(inputs, target)
                octonion_correct += octonion_pred == target

            # Train baselines on batch
            baseline_accs = {
                name: model.fit_batch(X_baseline_train[batch_idx], y_train[batch_idx])
                for name, model in baselines.items()
            }

            # Update progress bar
            pbar.set_postfix({
                "octonion_acc": f"{octonion_correct / ((i + 1) * 100):.4f}",
                **{f"{k}_acc": f"{v:.4f}" for k, v in baseline_accs.items()},
            })

            # Record octonion weights
            if i % 10 == 0:
                w = octonion_perceptron.weight
                experiment_data["octonion"]["weight_history"].append({
                    "epoch": epoch,
                    "step": i * 100,
                    "x0": float(w[0]),
                    "x1": float(w[1]),
                    "x2": float(w[2]),
                    "x3": float(w[3]),
                    "x4": float(w[4]),
                    "x5": float(w[5]),
                    "x6": float(w[6]),
                    "x7": float(w[7]),
                })

        # Record training accuracies
        experiment_data["octonion"]["train_accuracies"].append(float(octonion_correct / n_samples))
        for name, model in baselines.items():
            train_acc = model.score(X_baseline_train, y_train)
            experiment_data[name]["train_accuracies"].append(float(train_acc))

        # Test accuracies
        octonion_test_correct = sum(
            octonion_perceptron.predict(
                octonion_perceptron.forward(np.array(list(starmap(Octonion, x)))),
            )
            == y
            for x, y in zip(X_octonion_test, y_test, strict=False)
        )
        experiment_data["octonion"]["test_accuracies"].append(
            float(octonion_test_correct / len(y_test)),
        )

        for name, model in baselines.items():
            test_acc = model.score(X_baseline_test, y_test)
            experiment_data[name]["test_accuracies"].append(float(test_acc))

        # Log results
        log_msg = f"Epoch {epoch + 1} complete - "
        log_msg += f"Octonion: {octonion_test_correct / len(y_test):.4f}, "
        log_msg += ", ".join(
            f"{name}: {model.score(X_baseline_test, y_test):.4f}"
            for name, model in baselines.items()
        )
        logger.info(log_msg)

        # Save experiment data
        with open(experiment_dir / "experiment.json", "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, indent=2)


if __name__ == "__main__":
    main()
