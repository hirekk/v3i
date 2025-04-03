"""Data extraction."""

from pathlib import Path

from torchvision import datasets
from torchvision import transforms

DEFAULT_DATA_DIR = Path("data")


def download_mnist(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[Path, Path]:
    """Download MNIST dataset to local directory.

    Args:
        data_dir: Directory to store the dataset

    Returns:
        tuple[Path, Path]: Paths to train and test dataset directories
    """
    data_path = Path(data_dir)

    # Download training data
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
    )

    # Download test data
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
    )

    train_path = data_path / "MNIST" / "raw" / "train-images-idx3-ubyte"
    test_path = data_path / "MNIST" / "raw" / "t10k-images-idx3-ubyte"

    return train_path, test_path


def get_mnist_transform() -> transforms.Compose:
    """Get the standard MNIST transform pipeline.

    Returns:
        transforms.Compose: Transform pipeline for MNIST
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # MNIST mean and std
    ])


def create_mnist_datasets(
    data_dir: Path = DEFAULT_DATA_DIR,
) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Create MNIST datasets with transforms.

    Args:
        data_dir: Directory where dataset is stored

    Returns:
        tuple[datasets.MNIST, datasets.MNIST]: Train and test datasets
    """
    transform = get_mnist_transform()

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True,
    )

    return train_dataset, test_dataset
