"""Generate datasets on the 3-sphere or 7-sphere. train.npz, test.npz with X (n, 4) or (n, 8), y (n,) ±1."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np


def inverse_stereographic(u: np.ndarray) -> np.ndarray:
    """Inverse stereographic R^d -> S^d. u is (n, d). Returns (n, d+1) on the unit d-sphere.

    North pole (1, 0, ..., 0); x0 = (1 - r^2)/(1 + r^2), (x1,...,xd) = 2*u/(1 + r^2), r^2 = |u|^2.
    """
    r2 = np.sum(u * u, axis=1, keepdims=True)
    denom = 1 + r2
    x0 = (1 - r2) / denom
    rest = 2 * u / denom
    return np.hstack([x0, rest])


def to_s3_from_1d(x: np.ndarray) -> np.ndarray:
    """Map (n,) or (n, 1) to S^3 via inverse stereographic: embed as (x, 0, 0) in R^3."""
    x = np.atleast_1d(x).ravel()
    u = np.zeros((len(x), 3))
    u[:, 0] = x
    return inverse_stereographic(u)


def to_s3_from_2d(xy: np.ndarray) -> np.ndarray:
    """Map (n, 2) to S^3 via inverse stereographic: embed as (x, y, 0) in R^3."""
    n = xy.shape[0]
    u = np.zeros((n, 3))
    u[:, 0] = xy[:, 0]
    u[:, 1] = xy[:, 1]
    return inverse_stereographic(u)


def to_s7_from_1d(x: np.ndarray) -> np.ndarray:
    """Map (n,) or (n, 1) to S^7 via inverse stereographic: embed as (x, 0, ..., 0) in R^7."""
    x = np.atleast_1d(x).ravel()
    u = np.zeros((len(x), 7))
    u[:, 0] = x
    return inverse_stereographic(u)


def to_s7_from_2d(xy: np.ndarray) -> np.ndarray:
    """Map (n, 2) to S^7 via inverse stereographic: embed as (x, y, 0, ..., 0) in R^7."""
    n = xy.shape[0]
    u = np.zeros((n, 7))
    u[:, 0] = xy[:, 0]
    u[:, 1] = xy[:, 1]
    return inverse_stereographic(u)


def generate_binary_1d(
    train_size: int,
    test_size: int,
    noise: float,
    rng: np.random.Generator,
    to_sphere: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Two blobs at ±1 on the line, labels ±1. to_sphere maps (n,) to (n,d) on sphere."""
    half_train, half_test = train_size // 2, test_size // 2
    x_neg = rng.normal(-1, noise, half_train + half_test)
    x_pos = rng.normal(1, noise, half_train + half_test)
    x = np.concatenate([x_neg, x_pos])
    y = np.concatenate([np.full(half_train + half_test, -1), np.full(half_train + half_test, 1)])
    perm = rng.permutation(len(x))
    x, y = x[perm], y[perm]
    X = to_sphere(x)
    n_train = train_size
    return (
        X[:n_train],
        y[:n_train],
        X[n_train : n_train + test_size],
        y[n_train : n_train + test_size],
    )


def generate_binary_xor(
    train_size: int,
    test_size: int,
    noise: float,
    rng: np.random.Generator,
    to_sphere: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """XOR: four blobs, labels ±1. to_sphere maps (n,2) to (n,d) on sphere."""
    corners = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
    labels = np.array([-1, 1, 1, -1])
    n_total = train_size + test_size
    idx = rng.integers(0, 4, size=n_total)
    xy = corners[idx] + rng.normal(0, noise, (n_total, 2))
    y = labels[idx]
    X = to_sphere(xy)
    return X[:train_size], y[:train_size], X[train_size:], y[train_size:]


def save_dataset(
    out_dir: Path, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez(out_dir / "test.npz", X=X_test, y=y_test)
    print(f"Saved train {X_train.shape[0]}, test {X_test.shape[0]} -> {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate train.npz and test.npz. Use one of --binary-1d / --binary-xor and one of --quaternion / --octonion."
    )
    dataset_group = p.add_mutually_exclusive_group(required=True)
    dataset_group.add_argument(
        "--binary-1d",
        action="store_true",
        help="Binary classification on the line (±1 + noise).",
    )
    dataset_group.add_argument(
        "--binary-xor",
        action="store_true",
        help="XOR on the plane (four blobs).",
    )
    algebra_group = p.add_mutually_exclusive_group(required=True)
    algebra_group.add_argument(
        "--quaternion",
        action="store_true",
        help="Map to S^3 (X n×4) via inverse stereographic.",
    )
    algebra_group.add_argument(
        "--octonion",
        action="store_true",
        help="Map to S^7 (X n×8) via inverse stereographic.",
    )
    p.add_argument("--train-size", type=int, default=800)
    p.add_argument("--test-size", type=int, default=200)
    p.add_argument("--noise", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if args.quaternion:
        to_1d, to_2d = to_s3_from_1d, to_s3_from_2d
    else:
        to_1d, to_2d = to_s7_from_1d, to_s7_from_2d

    rng = np.random.default_rng(args.seed)

    if args.binary_1d:
        out_dir = Path("data") / "binary_1d" / ("octonion" if args.octonion else "quaternion")
        X_tr, y_tr, X_te, y_te = generate_binary_1d(
            args.train_size, args.test_size, args.noise, rng, to_sphere=to_1d
        )
    else:
        out_dir = Path("data") / "binary_xor" / ("octonion" if args.octonion else "quaternion")
        X_tr, y_tr, X_te, y_te = generate_binary_xor(
            args.train_size, args.test_size, args.noise, rng, to_sphere=to_2d
        )
    save_dataset(out_dir, X_tr, y_tr, X_te, y_te)


if __name__ == "__main__":
    main()
