"""Example of using QuaternionPerceptron on MNIST to classify even vs odd digits."""

from datetime import datetime
import json
import logging
from pathlib import Path

import numpy as np
import quaternion
from tqdm.auto import tqdm

from v3i.data.extract import DEFAULT_DATA_DIR
from v3i.models.baseline.decision_tree import DecisionTreeBaseline
from v3i.models.baseline.logistic_regression import LogisticRegressionBaseline
from v3i.models.baseline.random_choice import RandomChoiceBaseline
from v3i.models.perceptron.quaterion import QuaternionPerceptron

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

QUATERNION_DATA_DIR = DEFAULT_DATA_DIR / "MNIST" / "quaternion"
QUATERNION_DATA_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_SEED = 42


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


def spiral_iterate():
    directions = [  # counter-clockwise
        (-1, 0),  # up
        (0, -1),  # left
        (1, 0),  # down
        (0, 1),  # right
    ]
    curr_steps = 1
    next_steps = 1
    curr_dir = 0
    x, y = 0, 0
    while True:
        for _ in range(curr_steps):
            yield x, y
            x += directions[curr_dir][0]
            y += directions[curr_dir][1]
        curr_steps = next_steps
        if curr_dir % 2 == 0:
            next_steps += 1
        curr_dir = (curr_dir + 1) % 4


def test_spiral_iterate() -> None:
    spiral = spiral_iterate()
    for i, _coords in enumerate(spiral):
        if i > 10:
            break


def generate_receptor_tuples(
    img_size: int,
    tuple_size: int = 4,
    sigma: float = 2.0,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate tuples of values from nearby locations in a grid.

    Args:
        img_size: Size of the image.
        tuple_size: Size of each tuple (default 4).
        sigma: Standard deviation for the normal distribution (controls spread).
        random_seed: Seed for the random number generator.

    Returns:
        Array of shape (n_tuples, tuple_size) containing the sampled values
    """
    rng = np.random.default_rng(seed=random_seed)
    used = set()
    height, width = img_size, img_size
    n_tuples = height * width // tuple_size
    # Array of tuples of 2d coordinates
    result = np.empty((n_tuples, tuple_size, 2), dtype=int)
    step_x = int(np.sqrt(tuple_size))
    step_y = round(np.sqrt(tuple_size))

    start_x = width % 2
    start_y = 0
    tuple_idx = 0

    centers = [
        (x, y) for x in range(start_x, width, step_x) for y in range(start_y, height, step_y)
    ]

    centers_shuffled = rng.permutation(centers)

    for tuple_idx, (center_x, center_y) in enumerate(centers_shuffled):
        for j in range(tuple_size):
            miss_count = 0
            while True:
                # Sample x, y offsets from normal distribution
                dx = round(rng.normal(0, sigma))
                dy = round(rng.normal(0, sigma))

                # Calculate sample coordinates
                x = center_x + dx
                y = center_y + dy

                # Check if coordinates are within grid bounds
                if 0 <= x < width and 0 <= y < height and (x, y) not in used:
                    result[tuple_idx, j] = (x, y)
                    used.add((x, y))
                    miss_count = 0
                    break

                miss_count += 1
                logger.warning("[%d] Missed (#%d): (%d, %d)", tuple_idx, miss_count, x, y)
                if miss_count == 1000:
                    for dx, dy in spiral_iterate():
                        x = center_x + dx
                        y = center_y + dy
                        if 0 <= x < width and 0 <= y < height and (x, y) not in used:
                            result[tuple_idx, j] = (x, y)
                            used.add((x, y))
                            miss_count = 0
                            break
                    break

    return result


def convert_mnist_to_quaternion_receptors(
    images: list[np.ndarray],
    img_size: int = 28,
    tuple_size: int = 4,
    sigma: float = 2.0,
    random_seed: int = 42,
) -> np.ndarray:
    """Convert MNIST images to lists of quaternion receptors."""
    receptor_coords = generate_receptor_tuples(
        img_size=img_size,
        tuple_size=tuple_size,
        sigma=sigma,
        random_seed=random_seed,
    )
    quaternion_images = []

    for img in tqdm(images, desc="Converting images to quaternion receptors"):
        # Reshape to 2D if flattened
        img_2d = img.reshape(img_size, img_size) if img.ndim == 1 else img

        # Convert to pixel values -1 to 1
        img_2d = (img_2d / 127.5) - 1.0

        # Create quaternion for each pixel
        quaternion_receptors = []
        for receptor in receptor_coords:
            quaternion = np.zeros(tuple_size, dtype=np.float32)
            for j, (x, y) in enumerate(receptor):
                quaternion[j] = img_2d[y, x]
            quaternion_receptors.append(quaternion)
        quaternion_images.append(np.array(quaternion_receptors))

    return np.array(quaternion_images)


def main() -> None:
    """Main function to run the MNIST classification using QuaternionPerceptron."""
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

    # Setup experiment tracking
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = Path("data/experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Initialize models
    rng = np.random.RandomState(RANDOM_SEED)
    quaternion_perceptron = QuaternionPerceptron(
        forward_type="average",
        learning_rate=0.01,
        # buffer_size=100,
        # random_seed=RANDOM_SEED,
    )
    baselines = {
        "decision_tree": DecisionTreeBaseline(random_seed=RANDOM_SEED),
        "logistic": LogisticRegressionBaseline(random_seed=RANDOM_SEED),
        "random": RandomChoiceBaseline(random_seed=RANDOM_SEED),
    }

    # Convert labels to numpy array if they aren't already
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Training loop
    epochs = 10
    experiment_data = {
        "quaternion": {
            "bias_history": [],
            "action_history": [],
            "train_accuracies": [],
            "test_accuracies": [],
        },
    }

    # Add baseline models to experiment data
    for name in baselines:
        experiment_data[name] = {"train_accuracies": [], "test_accuracies": []}

    for epoch in range(epochs):
        n_samples = len(y_train)
        n_batches = n_samples // 100

        perm = rng.permutation(n_samples)
        pbar = tqdm(range(n_batches), desc=f"Epoch {epoch + 1}/{epochs}")

        quat_correct = 0
        for i in pbar:
            # Get batch indices
            start_idx = i * 100
            end_idx = start_idx + 100
            batch_idx = perm[start_idx:end_idx]

            # Train quaternion model
            for idx in batch_idx:
                inputs = X_quaternion_train[idx]
                target = y_train[idx]
                quat_pred = quaternion_perceptron.predict_label(inputs)
                quaternion_perceptron.train(inputs, target)
                quat_correct += quat_pred == target

            # Train baselines on batch
            baseline_accs = {
                name: model.fit_batch(X_baseline_train[batch_idx], y_train[batch_idx])
                for name, model in baselines.items()
            }

            # Update progress bar
            pbar.set_postfix({
                "quat_acc": f"{quat_correct / ((i + 1) * 100):.4f}",
                **{f"{k}_acc": f"{v:.4f}" for k, v in baseline_accs.items()},
            })

            # Record quaternion weights
            if i % 100 == 0:
                b = quaternion.as_float_array(quaternion_perceptron.bias)
                experiment_data["quaternion"]["bias_history"].append({
                    "epoch": epoch,
                    "step": i * 100,
                    "w": float(b[0]),
                    "x": float(b[1]),
                    "y": float(b[2]),
                    "z": float(b[3]),
                })

                a = quaternion.as_float_array(quaternion_perceptron.action)
                experiment_data["quaternion"]["action_history"].append({
                    "epoch": epoch,
                    "step": i * 100,
                    "w": float(a[0]),
                    "x": float(a[1]),
                    "y": float(a[2]),
                    "z": float(a[3]),
                })

        # Record training accuracies
        experiment_data["quaternion"]["train_accuracies"].append(float(quat_correct / n_samples))
        for name, model in baselines.items():
            train_acc = model.score(X_baseline_train, y_train)
            experiment_data[name]["train_accuracies"].append(float(train_acc))

        # Test accuracies
        quat_test_correct = sum(
            quaternion_perceptron.predict_label(x) == y
            for x, y in zip(X_quaternion_test, y_test, strict=False)
        )
        experiment_data["quaternion"]["test_accuracies"].append(
            float(quat_test_correct / len(y_test)),
        )

        for name, model in baselines.items():
            test_acc = model.score(X_baseline_test, y_test)
            experiment_data[name]["test_accuracies"].append(float(test_acc))

        # Log results
        log_msg = f"Epoch {epoch + 1} complete - "
        log_msg += f"Quaternion: {quat_test_correct / len(y_test):.4f}, "
        log_msg += ", ".join(
            f"{name}: {model.score(X_baseline_test, y_test):.4f}"
            for name, model in baselines.items()
        )
        logger.info(log_msg)

        # Save experiment data
        with open(experiment_dir / "experiment.json", "w", encoding="utf-8") as f:
            json.dump(experiment_data, f, indent=2)

    # Record predictions.
    predictions = []
    for x, y in zip(X_quaternion_test, y_test, strict=False):
        q_in, q_out = quaternion_perceptron.predict(x)
        predictions.append({
            "target": int(y),
            "input_reduced": quaternion.as_float_array(q_in).tolist(),
            "prediction": quaternion.as_float_array(q_out).tolist(),
        })
    with open(experiment_dir / "predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    main()
