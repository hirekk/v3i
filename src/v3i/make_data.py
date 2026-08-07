"""Generate datasets on the 3-sphere or 7-sphere.

train.npz, test.npz with X (n, 4) or (n, 8), y (n,) labels ±1.
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


def generate_parity(
    train_size: int,
    test_size: int,
    noise: float,
    rng: np.random.Generator,
    bits: int,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """k-bit parity on the centered n-cube, embedded on the (dim-1)-sphere.

    Blobs sit at the 2^k vertices of the cube {-1, +1}^k, labelled ±1 by the
    parity of the bits (the k-ary XOR). The k coordinates go into the *imaginary*
    part (coords 1..k) of a dim-vector — real part zero — which is then
    normalized onto the unit sphere. This embedding is linear in the bits, so it
    preserves polynomial degree: a degree-<k readout provably cannot separate
    k-bit parity (unlike an inverse-stereographic embedding, which is rational
    and smears degree). Requires 1 <= bits <= dim - 1 (the imaginary dimension).
    """
    if not 1 <= bits <= dim - 1:
        error_message = (
            f"bits={bits} must be in 1..{dim - 1} (the imaginary dimensions of dim={dim})."
        )
        raise ValueError(error_message)
    n_total = train_size + test_size
    idx = rng.integers(0, 2, size=(n_total, bits))
    y = np.where(idx.sum(axis=1) % 2 == 1, 1, -1)
    pm = (2.0 * idx - 1.0) + rng.normal(0, noise, size=(n_total, bits))
    x = np.zeros((n_total, dim))
    x[:, 1 : 1 + bits] = pm
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    return x[:train_size], y[:train_size], x[train_size:], y[train_size:]


def save_dataset(
    out_dir: Path, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
) -> None:
    """Write train.npz/test.npz to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "train.npz", X=X_train, y=y_train)
    np.savez(out_dir / "test.npz", X=X_test, y=y_test)
    print(f"Saved train {X_train.shape[0]}, test {X_test.shape[0]} -> {out_dir}")


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(
        description=(
            "Generate train.npz and test.npz. Use one of --binary-1d / --binary-xor "
            "/ --parity BITS and one of --quaternion / --octonion."
        )
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
    dataset_group.add_argument(
        "--parity",
        type=int,
        metavar="BITS",
        help="BITS-bit parity on the centered cube (degree-BITS gate).",
    )
    algebra_group = p.add_mutually_exclusive_group(required=True)
    algebra_group.add_argument(
        "--quaternion",
        action="store_true",
        help="Map to S^3 (X n by 4) via inverse stereographic.",
    )
    algebra_group.add_argument(
        "--octonion",
        action="store_true",
        help="Map to S^7 (X n by 8) via inverse stereographic.",
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
    algebra = "octonion" if args.octonion else "quaternion"

    if args.parity is not None:
        out_dir = Path("data") / f"parity-{args.parity}" / algebra
        X_tr, y_tr, X_te, y_te = generate_parity(
            args.train_size,
            args.test_size,
            args.noise,
            rng,
            bits=args.parity,
            dim=8 if args.octonion else 4,
        )
    elif args.binary_1d:
        out_dir = Path("data") / "binary-1d" / algebra
        X_tr, y_tr, X_te, y_te = generate_binary_1d(
            args.train_size, args.test_size, args.noise, rng, to_sphere=to_1d
        )
    else:
        out_dir = Path("data") / "binary-xor" / algebra
        X_tr, y_tr, X_te, y_te = generate_binary_xor(
            args.train_size, args.test_size, args.noise, rng, to_sphere=to_2d
        )
    save_dataset(out_dir, X_tr, y_tr, X_te, y_te)


if __name__ == "__main__":
    main()
