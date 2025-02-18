"""Data extraction."""

import argparse
import logging
from pathlib import Path

from torchvision import datasets

from v3i.logger import setup_logging

DATA_DIR = Path("data")


def download_mnist(dst_dir: Path = DATA_DIR) -> Path:
    """Download MNIST dataset to local directory.

    Args:
        dst_dir (str): Directory to store the dataset

    Returns:
        tuple[Path, Path]: Paths to train and test dataset directories
    """
    datasets.MNIST(
        root=dst_dir,
        train=False,
        download=True,
    )

    return dst_dir / "MNIST" / "raw" / "t10k-images-idx3-ubyte"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dst-dir", type=Path, required=False, default=DATA_DIR)
    parser.add_argument("--log-level", type=int, required=False, default=logging.INFO)
    args = parser.parse_args()

    logger = setup_logging(log_level=args.log_level)
    logger.info("Downloading MNIST dataset...")
    file_path = download_mnist(dst_dir=args.dst_dir)
    logger.info("MNIST dataset downloaded successfully to %s.", file_path)
