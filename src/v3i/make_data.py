"""Generate a datasets to train and test V3I models on."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def embed_1d_and_stereographic_to_s3(x: np.ndarray) -> np.ndarray:
    """Embed 1D as (x,0,0), then map to unit 3-sphere via inverse stereographic projection.

    Args:
        x: 1D array of shape (n,) or (n, 1).

    Returns:
        Array of shape (n, 4) with columns (w, a, b, c) on the unit 3-sphere.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    rho = x * x  # u² + v² + s² with (u, v, s) = (x, 0, 0)
    one_plus_rho = 1.0 + rho
    w = (rho - 1.0) / one_plus_rho
    a = (2.0 * x) / one_plus_rho
    b = np.zeros_like(x)
    c = np.zeros_like(x)
    return np.stack([w, a, b, c], axis=1)


def make_binary_1d(
    n_samples_per_class: int = 500,
    train_ratio: float = 0.8,
    noise_std: float = 0.3,
    random_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create train/test splits for 1D binary classification (±1 centers + noise).

    Returns:
        X_train, y_train, X_test, y_test
        X shapes: (n, 1), y shapes: (n,) with values in {-1, 1}.
    """
    rng = np.random.default_rng(random_seed)
    n = n_samples_per_class
    # Class -1 around -1, class +1 around +1
    x_neg = -1.0 + rng.normal(0, noise_std, size=n)
    x_pos = 1.0 + rng.normal(0, noise_std, size=n)
    X = np.concatenate([x_neg, x_pos]).reshape(-1, 1)
    y = np.concatenate([np.full(n, -1), np.full(n, 1)])

    # Shuffle
    idx = rng.permutation(2 * n)
    X, y = X[idx], y[idx]

    # Split
    n_train = int(2 * n * train_ratio)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    return X_train, y_train, X_test, y_test


def process_binary_1d_to_binary_3sphere(
    src_dir: Path,
    out_dir: Path,
) -> None:
    """Load binary_1d train/test from src_dir, map 1D -> S³, save to out_dir."""
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    for split in ("train", "test"):
        data = np.load(src_dir / f"{split}.npz")
        X, y = data["X"], data["y"]
        X_s3 = embed_1d_and_stereographic_to_s3(X)
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez(out_dir / f"{split}.npz", X=X_s3, y=y)
    print(f"Processed {src_dir} -> {out_dir} (X shape: (n, 4) on S³)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create 1D binary dataset or map binary_1d -> binary_3sphere (S³).",
    )
    parser.add_argument(
        "--to-3sphere",
        action="store_true",
        help="Load binary_1d from --src-dir, map to S³, save to data/binary_3sphere.",
    )
    parser.add_argument(
        "--src-dir",
        type=Path,
        default=Path("data/binary_1d"),
        help="Source dir for --to-3sphere (train.npz, test.npz).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/binary_1d"),
        help="Output dir (or data/binary_3sphere when --to-3sphere).",
    )
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--noise-std", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.to_3sphere:
        out = (
            args.out_dir
            if args.out_dir != Path("data/binary_1d")
            else Path("data/binary_3sphere")
        )
        process_binary_1d_to_binary_3sphere(args.src_dir, out)
        return

    X_train, y_train, X_test, y_test = make_binary_1d(
        n_samples_per_class=args.samples_per_class,
        train_ratio=args.train_ratio,
        noise_std=args.noise_std,
        random_seed=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_dir / "train.npz", X=X_train, y=y_train)
    np.savez(args.out_dir / "test.npz", X=X_test, y=y_test)
    print(f"Saved train {X_train.shape}, test {X_test.shape} to {args.out_dir}")


if __name__ == "__main__":
    main()
