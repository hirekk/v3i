import argparse
import enum
from pathlib import Path

import yaml

from v3i.models.perceptron.octonion import run as run_octonion_perceptron
from v3i.models.perceptron.quaterion import run as run_quaternion_perceptron


class ModelType(enum.StrEnum):
    """Model types."""

    QUATERNION_MNIST = "quaternion_mnist"
    OCTONION_MNIST = "octonion_mnist"
    QUATERNION_XOR = "quaternion_xor"
    OCTONION_XOR = "octonion_xor"


def main(model: ModelType, config_path: Path) -> None:
    with config_path.open(mode="r") as file:
        run_config = yaml.safe_load(file)

    match model:
        case ModelType.QUATERNION_MNIST:
            run_quaternion_perceptron(config=run_config)
        case ModelType.OCTONION_MNIST:
            run_octonion_perceptron(config=run_config)
        case ModelType.QUATERNION_XOR:
            run_quaternion_perceptron(config=run_config)
        case ModelType.OCTONION_XOR:
            run_octonion_perceptron(config=run_config)
        case _:
            err_msg = f"Invalid model type: {model}"
            raise ValueError(err_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=ModelType, required=True)
    parser.add_argument("--config-path", type=Path, required=True)
    args = parser.parse_args()

    main(model=args.model, config_path=args.config_path)
