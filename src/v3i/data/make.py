"""Make train and test datasets."""

import argparse

from v3i.data import DATA_DIRPATH
from v3i.data.dataset import DatasetType
from v3i.data.mnist import make_mnist
from v3i.data.xor import make_xor


def make_dataset(
    dataset: DatasetType,
    train_size: int,
    test_size: int,
) -> None:
    """Make train and test datasets.

    Args:
        dataset: Dataset type.
        train_size: Number of training samples.
        test_size: Number of test samples.
    """
    data_dirpath = DATA_DIRPATH / dataset
    data_dirpath.mkdir(parents=True, exist_ok=True)

    match dataset:
        case DatasetType.XOR:
            make_xor(
                data_dirpath=data_dirpath,
                train_size=train_size,
                test_size=test_size,
            )
        case DatasetType.MNIST:
            make_mnist(
                data_dirpath=data_dirpath,
                train_size=train_size,
                test_size=test_size,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=DatasetType, required=True)
    parser.add_argument("--train-size", type=int, required=True)
    parser.add_argument("--test-size", type=int, required=True)
    args = parser.parse_args()

    make_dataset(
        dataset=args.dataset,
        train_size=args.train_size,
        test_size=args.test_size,
    )
