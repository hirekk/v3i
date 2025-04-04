"""Example of using QuaternionPerceptron on MNIST to classify even vs odd digits."""

from datetime import datetime
from itertools import starmap
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import quaternion
from torchvision import datasets
from tqdm.auto import tqdm

from v3i.data.extract import DEFAULT_DATA_DIR
from v3i.models.perceptron.quaterion import QuaternionPerceptron
from v3i.tracking import ExperimentHistory

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
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    experiment_dir = Path("data/experiments") / timestamp
    experiment_dir.mkdir(parents=True, exist_ok=True)

    model_params = {
        "learning_rate": 0.1,
        "random_seed": 42,
    }

    # Initialize history with custom step interval
    history = ExperimentHistory(
        timestamp=timestamp,
        experiment_dir=experiment_dir,
        model_params=model_params,
        step_interval=1000,  # Record every 100 steps
    )

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
        # Load MNIST dataset
        mnist_train = datasets.MNIST(
            root=DEFAULT_DATA_DIR,
            train=True,
            download=True,
            transform=None,
        )
        mnist_test = datasets.MNIST(
            root=DEFAULT_DATA_DIR,
            train=False,
            download=True,
            transform=None,
        )

        # Limit samples for faster training
        train_samples = 20000
        test_samples = 1000

        # Process training data
        X_train = []
        y_train = []
        for i in tqdm(range(train_samples), desc="Processing training data"):
            img, label = mnist_train[i]
            X_train.append(np.array(img))
            y_train.append(digit_to_label(label))

        # Process test data
        X_test = []
        y_test = []
        for i in tqdm(range(test_samples), desc="Processing test data"):
            img, label = mnist_test[i]
            X_test.append(np.array(img))
            y_test.append(digit_to_label(label))

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Save labels
        np.save(y_train_fpath, y_train)
        np.save(y_test_fpath, y_test)

        # Convert to quaternion representation
        X_train = convert_mnist_to_quaternions(X_train)
        X_test = convert_mnist_to_quaternions(X_test)

        # Save quaternion data
        np.save(X_train_fpath, X_train)
        np.save(X_test_fpath, X_test)
        logger.info("Saved quaternion data to %s", QUATERNION_DATA_DIR)

    # Create and train model
    model = QuaternionPerceptron(**model_params)

    epochs = 50
    train_accuracies = []
    test_accuracies = []
    rng = np.random.default_rng(42)
    X_train_shuffled = X_train[rng.permutation(len(X_train))]
    y_train_shuffled = y_train[rng.permutation(len(y_train))]

    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")

        # Print weight state at start of epoch
        weight_state = model.get_weight_evolution()
        logger.info("Weight state at epoch start:")
        logger.info(f"  Angle: {weight_state['final_state']['angle']:.4f}")
        logger.info(f"  Axis: {weight_state['final_state']['axis']}")

        # Train on each example
        correct_train = 0
        for i, (inputs, target) in enumerate(zip(X_train_shuffled, y_train_shuffled, strict=False)):
            # Convert input array to quaternion array
            inputs_quaternion = np.array(list(starmap(quaternion.quaternion, inputs)))

            # Train and get prediction
            model.train(inputs_quaternion, target)
            pred = model.predict(model.forward(inputs_quaternion))
            if pred == target:
                correct_train += 1

            # Record metrics at specified intervals
            if (i + 1) % history.step_interval == 0:
                stats = model.get_update_stats()
                # Get current predictions on a small test batch for distribution
                test_batch = X_test[:100]  # Use first 100 test samples for quick distribution check
                test_batch_predictions = []
                for test_inputs in test_batch:
                    test_inputs_quaternion = np.array(
                        list(starmap(quaternion.quaternion, test_inputs)),
                    )
                    test_pred = model.predict(model.forward(test_inputs_quaternion))
                    test_batch_predictions.append(test_pred)

                weight_components = quaternion.as_float_array(model.weight.quat)
                weight_norm = abs(model.weight.quat)
                # Calculate prediction statistics from test batch
                test_confidences = []
                error_angles = []
                for test_inputs in test_batch:
                    test_inputs_quaternion = np.array(
                        list(starmap(quaternion.quaternion, test_inputs)),
                    )
                    rotation_state = model.forward(test_inputs_quaternion)
                    pred = model.predict(rotation_state)
                    test_batch_predictions.append(pred)

                    # Record confidence (angle of rotation)
                    test_confidences.append(abs(rotation_state.angle))

                    # Record error angle if we have ground truth
                    test_idx = len(test_batch_predictions) - 1
                    if pred != y_test[test_idx]:
                        error_angles.append(abs(rotation_state.angle))

                history.add_train_step(
                    epoch=epoch,
                    step=i + 1,
                    accuracy=correct_train / (i + 1),
                    weight_angle=float(model.weight.angle),
                    weight_angle_change=float(stats["weight_angle_change"]),
                    weight_axis=float(np.mean(model.weight.axis)),
                    weight_axis_change=float(stats["mean_axis_change"]),
                    update_angle=float(stats["mean_update_angle"]),
                    confidence=float(stats["mean_confidence"]),
                    pos_preds=sum(1 for p in test_batch_predictions if p > 0),
                    neg_preds=sum(1 for p in test_batch_predictions if p < 0),
                    weight_w=float(weight_components[0]),  # real part
                    weight_x=float(weight_components[1]),  # i component
                    weight_y=float(weight_components[2]),  # j component
                    weight_z=float(weight_components[3]),  # k component
                    # Additional diagnostic metrics
                    weight_norm=float(weight_norm),  # Should stay close to 1.0
                    weight_axis_magnitude=float(
                        np.linalg.norm(model.weight.axis),
                    ),  # Should be close to 1.0
                    # Learning dynamics
                    mean_confidence=float(np.mean(test_confidences)),
                    std_confidence=float(np.std(test_confidences)),
                    mean_error_angle=float(np.mean(error_angles)) if error_angles else 0.0,
                    # Class balance metrics
                    class_balance=float(
                        sum(1 for p in test_batch_predictions if p > 0)
                        / len(test_batch_predictions),
                    ),  # Should be ~0.5
                    # Prediction stability
                    prediction_switches=sum(
                        1
                        for i in range(1, len(test_batch_predictions))
                        if test_batch_predictions[i] != test_batch_predictions[i - 1]
                    ),
                    # Update statistics
                    recent_update_angles=stats.get("recent_update_angles", [])[
                        -5:
                    ],  # Last 5 update angles
                    update_angle_std=float(np.std(stats.get("recent_update_angles", [0]))),
                    # Training progress indicators
                    running_accuracy=float(
                        sum(1 for i, p in enumerate(test_batch_predictions) if p == y_test[i])
                        / len(test_batch_predictions),
                    ),
                )

        # Calculate training accuracy
        train_acc = correct_train / len(y_train)
        train_accuracies.append(train_acc)
        logger.info(f"Epoch {epoch + 1} training accuracy: {train_acc:.4f}")

        # Calculate test accuracy
        correct_test = 0
        test_predictions = []
        for inputs in X_test:
            inputs_quaternion = np.array(list(starmap(quaternion.quaternion, inputs)))
            pred = model.predict(model.forward(inputs_quaternion))
            test_predictions.append(pred)
            if pred == y_test[len(test_predictions) - 1]:  # Compare with corresponding test label
                correct_test += 1

        # Add test metrics at end of epoch
        test_acc = correct_test / len(y_test)
        test_accuracies.append(test_acc)
        history.add_test_metrics(epoch=epoch, accuracy=test_acc)

        # DEBUGGING: Print weight and some statistics
        logger.info(
            f"Current weight: {quaternion.as_float_array(model.weight.quat)}",
        )  # Access .quat
        logger.info(f"Weight norm: {abs(model.weight.quat)}")  # Access .quat

        # Check distribution of predictions on test set
        test_predictions = []
        for inputs in X_test:
            inputs_quaternion = np.array(list(starmap(quaternion.quaternion, inputs)))
            pred = model.predict(model.forward(inputs_quaternion))
            test_predictions.append(pred)

        pos_preds = sum(1 for p in test_predictions if p > 0)
        neg_preds = sum(1 for p in test_predictions if p < 0)
        logger.info(f"Test predictions distribution: +1: {pos_preds}, -1: {neg_preds}")

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Quaternion Perceptron on MNIST (Even vs Odd)")
    plt.legend()
    plt.grid(True)
    plt.savefig("quaternion_perceptron_mnist.png")
    plt.show()


if __name__ == "__main__":
    main()
