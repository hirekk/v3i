from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass
class ExperimentHistory:
    """Experiment history with configurable step intervals."""

    timestamp: str
    experiment_dir: Path
    model_params: Mapping[str, Any]
    metrics: Mapping[int, Mapping[str, Any]]  # Matches the schema
    step_interval: int = 100  # Configurable interval

    def __init__(
        self,
        timestamp: str,
        experiment_dir: Path,
        model_params: Mapping[str, Any],
        step_interval: int = 100,
    ) -> None:
        self.timestamp = timestamp
        self.experiment_dir = experiment_dir
        self.model_params = model_params
        self.metrics = {}
        self.step_interval = step_interval

    def add_train_step(
        self,
        epoch: int,
        step: int,
        accuracy: float,
        weight_angle: float,
        weight_angle_change: float,
        weight_axis: float,
        weight_axis_change: float,
        update_angle: float,
        confidence: float,
        pos_preds: int,
        neg_preds: int,
        weight_w: float,
        weight_x: float,
        weight_y: float,
        weight_z: float,
        weight_norm: float,
        weight_axis_magnitude: float,
        mean_confidence: float,
        std_confidence: float,
        mean_error_angle: float,
        class_balance: float,
        prediction_switches: int,
        recent_update_angles: list[float],
        update_angle_std: float,
        running_accuracy: float,
    ) -> None:
        """Record training metrics at a step."""
        if step % self.step_interval != 0:
            return

        if epoch not in self.metrics:
            self.metrics[epoch] = {"train": {}, "test": {"accuracy": 0.0}}

        self.metrics[epoch]["train"][step] = {
            "accuracy": accuracy,
            "weight_angle": weight_angle,
            "weight_angle_change": weight_angle_change,
            "weight_axis": weight_axis,
            "weight_axis_change": weight_axis_change,
            "update_angle": update_angle,
            "confidence": confidence,
            "prediction_distribution": {
                "pos": pos_preds,
                "neg": neg_preds,
            },
            "weight_w": weight_w,
            "weight_x": weight_x,
            "weight_y": weight_y,
            "weight_z": weight_z,
            "weight_norm": weight_norm,
            "weight_axis_magnitude": weight_axis_magnitude,
            "mean_confidence": mean_confidence,
            "std_confidence": std_confidence,
            "mean_error_angle": mean_error_angle,
            "class_balance": class_balance,
            "prediction_switches": prediction_switches,
            "recent_update_angles": recent_update_angles,
            "update_angle_std": update_angle_std,
            "running_accuracy": running_accuracy,
        }
        self.save()

    def add_test_metrics(self, epoch: int, accuracy: float) -> None:
        """Record test metrics for an epoch."""
        if epoch not in self.metrics:
            self.metrics[epoch] = {"train": {}, "test": {"accuracy": accuracy}}
        else:
            self.metrics[epoch]["test"]["accuracy"] = accuracy
        self.save()

    def save(self) -> None:
        """Save history to JSON file."""
        history_path = self.experiment_dir / "history.json"
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": self.timestamp,
                    "model_params": self.model_params,
                    "metrics": self.metrics,
                    "step_interval": self.step_interval,
                },
                f,
                indent=2,
            )
