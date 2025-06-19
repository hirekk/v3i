"""MNIST dataset."""

import enum
import pathlib

import numpy as np
from sklearn.preprocessing import StandardScaler
from torchvision import datasets

from v3i.data import DATA_DIRPATH
from v3i.data.dataset import DatasetType

MAX_TRAIN_SIZE = 60_000
MAX_TEST_SIZE = 10_000

class MnistPreprocessingType(enum.StrEnum):
    """Preprocessing types for MNIST."""

    QUATERNION = "quaternion"
    OCTONION = "octonion"
    BASELINE = "baseline"


def digit_to_label(digit: int) -> int:
    """Return 1 if digit is even, -1 if odd."""
    return digit % 2


def mnist_image_as_quaternion_array(image: np.ndarray) -> np.ndarray:
    """Convert MNIST image to quaternion array."""
    img_size = 28
    img_2d = image.reshape(img_size, img_size) if image.ndim == 1 else image

    # Convert pixel values to range [-1, 1]
    img_2d = (img_2d / 127.5) - 1.0

    # Create quaternion for each pixel
    converted = []
    for pixel_x, pixel_y in np.ndindex(img_size, img_size):
        pixel_value = img_2d[pixel_y, pixel_x]
        if pixel_value < -1.0 or pixel_value > 1.0:
            error_msg = f"Pixel value must be between -1.0 and 1.0, got {pixel_value}"
            raise ValueError(error_msg)

        # Normalize position to [-1, 1]
        scaled_x = (2.0 * pixel_x / (img_size - 1)) - 1.0
        scaled_y = (2.0 * pixel_y / (img_size - 1)) - 1.0

        # Create components [w, x, y, z]
        w = pixel_value
        x = scaled_x
        y = scaled_y
        z = (w + x + y) / 3

        converted.append([w, x, y, z])

    return np.array(converted)


def as_quaternions(images: list[np.ndarray]) -> np.ndarray:
    """Convert MNIST images to quaternions."""
    return np.array([mnist_image_as_quaternion_array(img) for img in images])


def mnist_image_as_octonion_array(image: np.ndarray) -> np.ndarray:
    """Convert MNIST image to octonion array."""
    img_size = 28
    img_2d = image.reshape(img_size, img_size) if image.ndim == 1 else image

    # Convert pixel values to range [-1, 1]
    img_2d = (img_2d / 127.5) - 1.0

    # Create quaternion for each pixel
    converted = []
    for pixel_x, pixel_y in np.ndindex(img_size, img_size):
        pixel_value = img_2d[pixel_y, pixel_x]
        if pixel_value < -1.0 or pixel_value > 1.0:
            error_msg = f"Pixel value must be between -1.0 and 1.0, got {pixel_value}"
            raise ValueError(error_msg)

        # Normalize position to [-1, 1]
        scaled_x = (2.0 * pixel_x / (img_size - 1)) - 1.0
        scaled_y = (2.0 * pixel_y / (img_size - 1)) - 1.0

        # Create components [w, x, y, z]
        x0 = pixel_value
        x1 = scaled_x
        x2 = scaled_y
        x3 = 0
        x4 = 0
        x5 = 0
        x6 = 0
        x7 = 0

        converted.append([x0, x1, x2, x3, x4, x5, x6, x7])

    return np.array(converted)


def as_octonions(images: list[np.ndarray]) -> np.ndarray:
    """Convert MNIST images to octonions."""
    return np.array([mnist_image_as_octonion_array(img) for img in images])


def make_mnist(
    data_dirpath: pathlib.Path,
    train_size: int = MAX_TRAIN_SIZE,
    test_size: int = MAX_TEST_SIZE,
) -> None:
    """Make MNIST dataset.

    Args:
        data_dirpath: Path to save the dataset.
        preprocessing: Preprocessing type.
        train_size: Number of training samples.
        test_size: Number of test samples.
    """
    if train_size > MAX_TRAIN_SIZE:
        err_msg = f"train_size must be less than or equal to {MAX_TRAIN_SIZE}."
        raise ValueError(err_msg)
    if test_size > MAX_TEST_SIZE:
        err_msg = f"test_size must be less than or equal to {MAX_TEST_SIZE}."
        raise ValueError(err_msg)

    train_dataset = datasets.MNIST(
        root=data_dirpath.parent,
        train=True,
        download=True,
    )
    test_dataset = datasets.MNIST(
        root=data_dirpath.parent,
        train=False,
        download=True,
    )

    X_train = [np.array(train_dataset[i][0]) for i in range(train_size)]
    y_train = [train_dataset[i][1] for i in range(train_size)]
    X_test = [np.array(test_dataset[i][0]) for i in range(test_size)]
    y_test = [test_dataset[i][1] for i in range(test_size)]

    for preprocessing in MnistPreprocessingType:
        match preprocessing:
            case MnistPreprocessingType.QUATERNION:
                X_train_preprocessed = as_quaternions(X_train)
                X_test_preprocessed = as_quaternions(X_test)
                y_train_preprocessed = np.array([digit_to_label(digit) for digit in y_train])
                y_test_preprocessed = np.array([digit_to_label(digit) for digit in y_test])
            case MnistPreprocessingType.OCTONION:
                X_train_preprocessed = as_octonions(X_train)
                X_test_preprocessed = as_octonions(X_test)
                y_train_preprocessed = np.array([digit_to_label(digit) for digit in y_train])
                y_test_preprocessed = np.array([digit_to_label(digit) for digit in y_test])
            case MnistPreprocessingType.BASELINE:
                scaler = StandardScaler()
                X_train_preprocessed = scaler.fit_transform([img.flatten() for img in X_train])
                X_test_preprocessed = scaler.transform([img.flatten() for img in X_test])
            case _:
                err_msg = f"Preprocessing type {preprocessing} not implemented."
                raise NotImplementedError(err_msg)

        # Save processed data
        data_subdirpath = data_dirpath / preprocessing
        data_subdirpath.mkdir(parents=True, exist_ok=True)
        np.save(data_subdirpath / "X_train.npy", X_train_preprocessed)
        np.save(data_subdirpath / "X_test.npy", X_test_preprocessed)
        np.save(data_subdirpath / "y_train.npy", y_train_preprocessed)
        np.save(data_subdirpath / "y_test.npy", y_test_preprocessed)


def load_mnist_data(model: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST dataset.

    Args:
        model: Model type.

    Returns:
        X_train: Training data.
        y_train: Training labels.
        X_test: Test data.
        y_test: Test labels.
    """
    if model not in ["baseline", "quaternion", "octonion"]:
        err_msg = f"Invalid model type: {model}"
        raise ValueError(err_msg)

    X_train = np.load(DATA_DIRPATH / DatasetType.MNIST / model / "X_train.npy")
    y_train = np.load(DATA_DIRPATH / DatasetType.MNIST / model / "y_train.npy")
    X_test = np.load(DATA_DIRPATH / DatasetType.MNIST / model / "X_test.npy")
    y_test = np.load(DATA_DIRPATH / DatasetType.MNIST / model / "y_test.npy")

    return X_train, y_train, X_test, y_test
