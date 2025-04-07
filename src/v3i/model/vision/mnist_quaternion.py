"""Example of using QuaternionPerceptron on MNIST to classify even vs odd digits."""

from datetime import datetime
from itertools import starmap
import json
import logging
from pathlib import Path

import numpy as np
import quaternion
from torchvision import datasets
from tqdm.auto import tqdm

from v3i.data.extract import DEFAULT_DATA_DIR
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

    X_train_fpath = QUATERNION_DATA_DIR / "X_train.npy"
    X_test_fpath = QUATERNION_DATA_DIR / "X_test.npy"
    y_train_fpath = QUATERNION_DATA_DIR / "y_train.npy"
    y_test_fpath = QUATERNION_DATA_DIR / "y_test.npy"

    if all(p.exists() for p in [X_train_fpath, X_test_fpath, y_train_fpath, y_test_fpath]):
        logger.info("Loading existing data from %s", QUATERNION_DATA_DIR)
        X_train = np.load(X_train_fpath)
        X_test = np.load(X_test_fpath)
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

        X_train = convert_mnist_to_quaternions(X_train)
        X_test = convert_mnist_to_quaternions(X_test)

        # Save processed data
        np.save(QUATERNION_DATA_DIR / "X_train.npy", X_train)
        np.save(QUATERNION_DATA_DIR / "X_test.npy", X_test)
        np.save(QUATERNION_DATA_DIR / "y_train.npy", y_train)
        np.save(QUATERNION_DATA_DIR / "y_test.npy", y_test)

    # Setup experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    experiment_data = {
        "weight_history": [],  # Will store weight components over time
        "train_accuracies": [],
        "test_accuracies": [],
    }

    model = QuaternionPerceptron(learning_rate=1.0, batch_size=2, random_seed=42)
    epochs = 1000

    for epoch in range(epochs):
        correct_train = 0

        # Train on shuffled data with progress bar
        perm = np.random.permutation(len(X_train))
        pbar = tqdm(perm, desc=f"Epoch {epoch + 1}/{epochs}")
        for i, idx in enumerate(pbar):
            inputs = np.array(list(starmap(quaternion.quaternion, X_train[idx])))
            target = y_train[idx]

            model.train(inputs, target)
            pred = model.predict(model.forward(inputs))
            correct_train += pred == target

            # Record weight components every 1000 steps
            if (i + 1) % 1000 == 0:
                train_acc = correct_train / (i + 1)
                pbar.set_postfix({"train_acc": f"{train_acc:.4f}"})

                # Record weight state
                w = quaternion.as_float_array(model.weight)
                experiment_data["weight_history"].append({
                    "epoch": epoch,
                    "step": i + 1,
                    "w": float(w[0]),  # real part
                    "x": float(w[1]),  # i component
                    "y": float(w[2]),  # j component
                    "z": float(w[3]),  # k component
                })

        train_acc = correct_train / len(y_train)
        experiment_data["train_accuracies"].append(float(train_acc))

        # Test accuracy
        correct_test = sum(
            model.predict(model.forward(np.array(list(starmap(quaternion.quaternion, x))))) == y
            for x, y in zip(X_test, y_test, strict=False)
        )
        test_acc = correct_test / len(y_test)
        experiment_data["test_accuracies"].append(float(test_acc))

        logger.info(f"Epoch {epoch + 1} complete - Test accuracy: {test_acc:.4f}")

        # Save experiment data after each epoch
        with open(experiment_dir / "experiment.json", "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, indent=2)


if __name__ == "__main__":
    main()
