"""Example of using OctonionPerceptron on MNIST to classify even vs odd digits."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from tqdm.auto import tqdm

from v3i.data.extract import DEFAULT_DATA_DIR
from v3i.models.octonion_perceptron import OctonionPerceptron

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_even(digit: int) -> int:
    """Return 1 if digit is even, 0 if odd."""
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
    """Main function to run the MNIST classification using OctonionPerceptron."""
    OCTONION_DATA_DIR = DEFAULT_DATA_DIR / "MNIST" / "octonion"
    OCTONION_DATA_DIR.mkdir(parents=True, exist_ok=True)

    X_train_fpath = OCTONION_DATA_DIR / "X_train.npy"
    X_test_fpath = OCTONION_DATA_DIR / "X_test.npy"
    y_train_fpath = OCTONION_DATA_DIR / "y_train.npy"
    y_test_fpath = OCTONION_DATA_DIR / "y_test.npy"

    if (
        X_train_fpath.exists()
        and X_test_fpath.exists()
        and y_train_fpath.exists()
        and y_test_fpath.exists()
    ):
        logger.info("Loading existing data from %s", OCTONION_DATA_DIR)
        X_train = np.load(X_train_fpath)
        X_test = np.load(X_test_fpath)
        y_train = np.load(y_train_fpath)
        y_test = np.load(y_test_fpath)
    else:
        # Load MNIST dataset without normalization
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

        # Limit the number of samples to make training faster
        train_samples = 20000
        test_samples = 1000

        # Extract raw images and labels
        X_train = []
        y_train = []
        X_test = []
        y_test = []

        # Process training data
        for i in tqdm(range(train_samples), desc="Processing training data"):
            img, label = mnist_train[i]
            X_train.append(np.array(img))  # Convert PIL image to numpy array
            y_train.append(is_even(label))

        # Process test data
        for i in tqdm(range(test_samples), desc="Processing test data"):
            img, label = mnist_test[i]
            X_test.append(np.array(img))
            y_test.append(is_even(label))

        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)

        # Save labels
        np.save(OCTONION_DATA_DIR / "y_train.npy", y_train)
        logger.info("Saved training labels to %s", OCTONION_DATA_DIR / "y_train.npy")
        np.save(OCTONION_DATA_DIR / "y_test.npy", y_test)
        logger.info("Saved test labels to %s", OCTONION_DATA_DIR / "y_test.npy")

        # Convert to octonion representation
        X_train = convert_mnist_to_octonions(X_train)
        logger.info("Converted training data to octonions")
        X_test = convert_mnist_to_octonions(X_test)
        logger.info("Converted test data to octonions")

        np.save(OCTONION_DATA_DIR / "X_train.npy", X_train)
        logger.info("Saved training data to %s", OCTONION_DATA_DIR / "X_train.npy")
        np.save(OCTONION_DATA_DIR / "X_test.npy", X_test)
        logger.info("Saved test data to %s", OCTONION_DATA_DIR / "X_test.npy")

    # Create and train model
    input_size = 28 * 28  # Full image size
    model = OctonionPerceptron(input_size=input_size, learning_rate=0.01)

    # Training
    epochs = 10
    train_accuracies = []
    test_accuracies = []
    rng = np.random.default_rng()

    for epoch in tqdm(range(epochs), desc="Epochs"):
        # Shuffle training data
        X_train_shuffled = rng.permutation(X_train)
        y_train_shuffled = rng.permutation(y_train)

        # Train on each example
        correct_train = 0

        for i, (inputs, target) in tqdm(
            enumerate(zip(X_train_shuffled, y_train_shuffled, strict=False)),
            desc="Training",
            total=len(y_train),
        ):
            model.train(inputs, target)

            # Check prediction
            result = model.forward(inputs)
            prediction = 1 if float(result[0]) >= 0 else -1
            if prediction == target:
                correct_train += 1

            # Print progress
            if (i + 1) % 100 == 0:
                logger.info("Training accuracy: %f", correct_train / (i + 1))
                model.apply_updates()
                logger.info("First three weights: %s", model.weights[:3])

        # Calculate training accuracy
        train_accuracy = correct_train / len(y_train)
        train_accuracies.append(train_accuracy)

        # Calculate test accuracy
        correct_test = 0
        logger.info("Calculating test accuracy at epoch %d", epoch)
        for inputs, target in zip(X_test, y_test, strict=False):
            result = model.forward(inputs)
            prediction = 1 if float(result[0]) >= 0 else -1
            if prediction == target:
                correct_test += 1

        test_accuracy = correct_test / len(y_test)
        logger.info("Test accuracy at epoch %d: %f", epoch, test_accuracy)
        test_accuracies.append(test_accuracy)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_accuracies, label="Training Accuracy")
    plt.plot(range(1, epochs + 1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Octonion Perceptron on MNIST (Even vs Odd)")
    plt.legend()
    plt.grid(True)
    plt.savefig("octonion_perceptron_mnist_even_odd.png")
    plt.show()

    # Visualize some predictions
    plt.figure(figsize=(15, 8))
    for i in range(10):
        # Get a sample from each class (even and odd)
        even_idx = np.where(y_test == 1)[0][i]
        odd_idx = np.where(y_test == -1)[0][i]

        # Make predictions
        even_result = model.forward(X_test[even_idx])
        even_pred = 1 if float(even_result[0]) >= 0 else -1

        odd_result = model.forward(X_test[odd_idx])
        odd_pred = 1 if float(odd_result[0]) >= 0 else -1

        # Plot even example
        plt.subplot(2, 10, i + 1)
        plt.imshow(X_test[even_idx], cmap="gray")
        plt.title(f"Even\nPred: {'Even' if even_pred == 1 else 'Odd'}")
        plt.axis("off")

        # Plot odd example
        plt.subplot(2, 10, i + 11)
        plt.imshow(X_test[odd_idx], cmap="gray")
        plt.title(f"Odd\nPred: {'Even' if odd_pred == 1 else 'Odd'}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig("octonion_perceptron_mnist_predictions.png")
    plt.show()


if __name__ == "__main__":
    main()
