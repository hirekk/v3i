"""Run a training experiment and log metrics for the dashboard.

Uniform harness around the quaternion/octonion perceptrons, their Sequential
stacks, and the real-valued baselines. Every run writes one JSON file under
runs/ with config, per-epoch metrics (train/test accuracy, geodesic loss), and
per-epoch weight snapshots — the dashboard (src/v3i/dashboard.py) reads these.

Usage:
    uv run python -m v3i.run_experiment --dataset binary-xor --model octonion --epochs 15
    uv run python -m v3i.run_experiment --dataset binary-xor --model quaternion-stack --layers 2
    uv run python -m v3i.run_experiment --dataset binary-xor --model baselines
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import quaternion as quaternion_np

from v3i.algebra import Octonion
from v3i.make_data import generate_binary_1d
from v3i.make_data import generate_binary_xor
from v3i.make_data import to_s3_from_1d
from v3i.make_data import to_s3_from_2d
from v3i.make_data import to_s7_from_1d
from v3i.make_data import to_s7_from_2d
from v3i.models.baseline.decision_tree import DecisionTreeBaseline
from v3i.models.baseline.logistic_regression import LogisticRegressionBaseline
from v3i.models.perceptron.nn import QuaternionSequential
from v3i.models.perceptron.octonion import OctonionPerceptron
from v3i.models.perceptron.octonion import OctonionSequential
from v3i.models.perceptron.quaternion import QuaternionPerceptron
from v3i.models.perceptron.quaternion import QuaternionSimpleOptimizer

# Best homogeneous linear separator on embedded XOR (docs/research/isometry-ceiling.md)
XOR_LINEAR_CEILING = 0.75


def make_dataset(
    dataset: str, algebra_dim: int, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate train/test split on S^3 (dim 4) or S^7 (dim 8) in memory."""
    rng = np.random.default_rng(seed)
    if dataset == "binary-1d":
        to_sphere = to_s3_from_1d if algebra_dim == 4 else to_s7_from_1d
        return generate_binary_1d(800, 200, 0.1, rng, to_sphere=to_sphere)
    to_sphere = to_s3_from_2d if algebra_dim == 4 else to_s7_from_2d
    return generate_binary_xor(800, 200, 0.1, rng, to_sphere=to_sphere)


def geodesic_loss(outputs: np.ndarray, y: np.ndarray) -> float:
    """Mean geodesic angle on the sphere between outputs and target poles.

    Target for label y is (y, 0, ..., 0); the angle is arccos<out, target>,
    which is 0 for a perfect hit and pi for the antipode. Works for any
    algebra dimension; outputs are normalized defensively.
    """
    outputs = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)
    cos = np.clip(outputs[:, 0] * y, -1.0, 1.0)
    return float(np.mean(np.arccos(cos)))


class Harness:
    """Uniform interface: one training epoch, batch outputs, weight snapshots."""

    name: str

    def epoch(self, X: np.ndarray, y: np.ndarray, order: np.ndarray) -> None:
        """Run one training epoch over X[order]."""
        raise NotImplementedError

    def outputs(self, X: np.ndarray) -> np.ndarray:
        """Final-layer output for each row, as (n, dim) unit vectors."""
        raise NotImplementedError

    def weights(self) -> list[list[float]]:
        """Current weights, one flat list per layer."""
        raise NotImplementedError

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Sign-of-real-part labels for each row."""
        return np.where(self.outputs(X)[:, 0] >= 0, 1, -1)


class QuaternionHarness(Harness):
    """Single QuaternionPerceptron or a Sequential stack of them."""

    def __init__(self, layers: int, learning_rate: float, seed: int) -> None:
        """Build the perceptron(s) and per-layer optimizers."""
        self.name = "quaternion" if layers == 1 else f"quaternion-stack-{layers}"
        self._perceptrons = [
            QuaternionPerceptron(learning_rate=learning_rate, random_seed=seed + i)
            for i in range(layers)
        ]
        self._model = QuaternionSequential(self._perceptrons) if layers > 1 else None
        self._optimizers = [QuaternionSimpleOptimizer(p) for p in self._perceptrons]

    def epoch(self, X: np.ndarray, y: np.ndarray, order: np.ndarray) -> None:
        """One pass of per-sample geodesic updates."""
        for idx in order:
            x = np.atleast_2d(X[idx])
            label = int(y[idx])
            if self._model is None:
                u, _ = self._perceptrons[0].compute_update(x, label)
                self._optimizers[0].step(u)
            else:
                self._model.learn_step(x, label, self._optimizers)

    def outputs(self, X: np.ndarray) -> np.ndarray:
        """Final quaternion outputs as (n, 4)."""
        out = np.empty((len(X), 4))
        for i, row in enumerate(X):
            x = np.atleast_2d(row)
            if self._model is None:
                _, q_out = self._perceptrons[0].predict(x)
                out[i] = quaternion_np.as_float_array(q_out)
            else:
                out[i] = self._model.forward(x)[0]
        return out

    def weights(self) -> list[list[float]]:
        """Weight components per layer."""
        return [list(quaternion_np.as_float_array(p.weight)) for p in self._perceptrons]


class OctonionHarness(Harness):
    """Single OctonionPerceptron or an OctonionSequential stack."""

    def __init__(self, layers: int, learning_rate: float, seed: int) -> None:
        """Build the perceptron chain."""
        self.name = "octonion" if layers == 1 else f"octonion-stack-{layers}"
        self._layers = [
            OctonionPerceptron(learning_rate=learning_rate, random_seed=seed + i)
            for i in range(layers)
        ]
        self._model = OctonionSequential(self._layers)

    def epoch(self, X: np.ndarray, y: np.ndarray, order: np.ndarray) -> None:
        """One act-observe-correct pass; renormalize weights after."""
        for idx in order:
            self._model.forward(Octonion(X[idx].copy()))
            target = Octonion.unit() if y[idx] >= 0 else -Octonion.unit()
            self._model.correct(target)
        for layer in self._layers:  # renormalization heartbeat: keep weights on S^7
            layer.weight = layer.weight.normalize()

    def outputs(self, X: np.ndarray) -> np.ndarray:
        """Final octonion outputs as (n, 8)."""
        out = np.empty((len(X), 8))
        for i, row in enumerate(X):
            out[i] = self._model.forward(Octonion(row.copy())).to_array()
        return out

    def weights(self) -> list[list[float]]:
        """Weight components per layer."""
        return [list(layer.weight.to_array()) for layer in self._layers]


def run_model(
    harness: Harness,
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    epochs: int,
    seed: int,
) -> dict:
    """Train and record per-epoch metrics, including the untrained epoch 0."""
    X_train, y_train, X_test, y_test = data
    rng = np.random.default_rng(seed)
    metrics = []

    def record(epoch: int) -> None:
        train_out, test_out = harness.outputs(X_train), harness.outputs(X_test)
        metrics.append(
            {
                "epoch": epoch,
                "train_acc": float(np.mean(np.where(train_out[:, 0] >= 0, 1, -1) == y_train)),
                "test_acc": float(np.mean(np.where(test_out[:, 0] >= 0, 1, -1) == y_test)),
                "train_loss": geodesic_loss(train_out, y_train),
                "test_loss": geodesic_loss(test_out, y_test),
                "weights": harness.weights(),
            }
        )

    record(0)
    for epoch in range(1, epochs + 1):
        harness.epoch(X_train, y_train, rng.permutation(len(y_train)))
        record(epoch)
    return {"model": harness.name, "metrics": metrics}


def run_baselines(
    data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], seed: int
) -> list[dict]:
    """Fit the real-valued baselines once; reported as flat reference lines."""
    X_train, y_train, X_test, y_test = data
    results = []
    for name, model in [
        ("logistic-regression", LogisticRegressionBaseline(random_seed=seed)),
        ("decision-tree", DecisionTreeBaseline(random_seed=seed)),
    ]:
        model.fit(X_train, y_train)
        results.append(
            {
                "model": name,
                "train_acc": float(model.score(X_train, y_train)),
                "test_acc": float(model.score(X_test, y_test)),
            }
        )
    return results


def main() -> None:
    """CLI entry point."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset", choices=["binary-1d", "binary-xor"], default="binary-xor")
    p.add_argument(
        "--model",
        choices=["quaternion", "quaternion-stack", "octonion", "octonion-stack", "baselines"],
        default="octonion",
    )
    p.add_argument("--layers", type=int, default=2, help="Stack depth for *-stack models.")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0, help="Model/shuffle seed.")
    p.add_argument("--data-seed", type=int, default=42)
    p.add_argument("--tag", type=str, default=None, help="Run name; default model-dataset-seed.")
    p.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = p.parse_args()

    algebra_dim = 4 if args.model.startswith("quaternion") else 8
    data = make_dataset(args.dataset, algebra_dim, args.data_seed)

    config = {
        "dataset": args.dataset,
        "model": args.model,
        "layers": args.layers if args.model.endswith("stack") else 1,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "data_seed": args.data_seed,
        "xor_linear_ceiling": XOR_LINEAR_CEILING if args.dataset == "binary-xor" else None,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if args.model == "baselines":
        result = {"config": config, "baselines": run_baselines(data, args.seed)}
    else:
        layers = args.layers if args.model.endswith("stack") else 1
        cls = QuaternionHarness if algebra_dim == 4 else OctonionHarness
        harness = cls(layers, args.learning_rate, args.seed)
        result = {"config": config} | run_model(harness, data, args.epochs, args.seed)
        config["model"] = result["model"]  # resolved name, e.g. octonion-stack-2

    tag = args.tag or f"{config['model']}-{args.dataset}-s{args.seed}"
    args.out_dir.mkdir(exist_ok=True)
    out_path = args.out_dir / f"{tag}.json"
    out_path.write_text(json.dumps(result, indent=1))

    last = result.get("metrics", [{}])[-1]
    print(f"wrote {out_path}")
    if "train_acc" in last:
        print(
            f"final: train_acc={last['train_acc']:.3f} test_acc={last['test_acc']:.3f} "
            f"train_loss={last['train_loss']:.3f}"
        )
    for b in result.get("baselines", []):
        print(f"{b['model']}: train_acc={b['train_acc']:.3f} test_acc={b['test_acc']:.3f}")


if __name__ == "__main__":
    main()
