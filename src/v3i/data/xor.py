import numpy as np
import itertools
import pathlib

from v3i.data import DATA_DIRPATH
from v3i.data.dataset import DatasetType


def create_xor_dataset(
    n_samples: int,
    dimensionality: int = 4,
    noise_std: float = 0.3,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a 4D XOR-like dataset with blobs at hypercube vertices.

    Args:
        n_samples: Number of samples to generate.
        noise_std: Standard deviation of Gaussian noise added to each vertex.

    Returns:
        Tuple of:
            - data array of shape (n_samples_total, 4),
            - labels array of shape (n_samples_total,) with 0/1 values.
    """
    rng = np.random.default_rng(42)

    # Generate all vertices of a 4D hypercube
    vertices = np.array(list(itertools.product([-1, 1], repeat=dimensionality)))

    # Initialize arrays for data and labels
    X = np.zeros((n_samples, dimensionality))
    y = np.zeros(n_samples)

    n_samples_per_vertex, remainder = divmod(n_samples, len(vertices))
    for i, vertex in enumerate(vertices):
        start_idx = i * n_samples_per_vertex
        end_idx = (i + 1) * n_samples_per_vertex
        if remainder > 0:
            end_idx += 1
            remainder -= 1

        X[start_idx:end_idx] = vertex + rng.normal(
            loc=0,
            scale=noise_std,
            size=(end_idx - start_idx, dimensionality),
        )

        # Label based on parity of 1s in the vertex coordinates
        label = (vertex == 1).sum() % 2
        y[start_idx:end_idx] = label

    return X, y


def make_xor(
    data_dirpath: pathlib.Path,
    train_size: int = 10_000,
    test_size: int = 1_000,
) -> None:
    """Make XOR dataset.

    Args:
        data_dirpath: Path to save the dataset.
        train_size: Number of training samples.
        test_size: Number of test samples.
    """
    for dim in [2, 4, 8]:
        X_train, y_train = create_xor_dataset(n_samples=train_size, dimensionality=dim, noise_std=0.3)
        X_test, y_test = create_xor_dataset(n_samples=test_size, dimensionality=dim, noise_std=0.3)

        data_subdirpath = data_dirpath / f"{dim}-D"
        data_subdirpath.mkdir(parents=True, exist_ok=True)
        np.save(data_subdirpath / "X_train.npy", X_train)
        np.save(data_subdirpath / "X_test.npy", X_test)
        np.save(data_subdirpath / "y_train.npy", y_train)
        np.save(data_subdirpath / "y_test.npy", y_test)


def load_xor_data(dimensionality: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load XOR dataset.

    Args:
        dimensionality: Dimensionality of the dataset.

    Returns:
        Tuple of train input, train label, test input, and test label datasets.
    """
    if dimensionality not in [2, 4, 8]:
        err_msg = f"Invalid dimensionality: {dimensionality}"
        raise ValueError(err_msg)

    X_train = np.load(DATA_DIRPATH / DatasetType.XOR / f"{dimensionality}-D" / "X_train.npy")
    y_train = np.load(DATA_DIRPATH / DatasetType.XOR / f"{dimensionality}-D" / "y_train.npy")
    X_test = np.load(DATA_DIRPATH / DatasetType.XOR / f"{dimensionality}-D" / "X_test.npy")
    y_test = np.load(DATA_DIRPATH / DatasetType.XOR / f"{dimensionality}-D" / "y_test.npy")

    return X_train, y_train, X_test, y_test
