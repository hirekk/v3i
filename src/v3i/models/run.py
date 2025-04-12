import argparse
import enum

from v3i.data import DatasetType
from v3i.models.perceptron.octonion import main as run_octonion_perceptron
from v3i.models.perceptron.quaterion import main as run_quaternion_perceptron


class ModelType(enum.StrEnum):
    """Model types."""

    QUATERNION_MNIST = "quaternion_mnist"
    OCTONION_MNIST = "octonion_mnist"
    QUATERNION_XOR = "quaternion_xor"
    OCTONION_XOR = "octonion_xor"


def main(model: ModelType) -> None:
    match model:
        case ModelType.QUATERNION_MNIST:
            run_quaternion_perceptron(dataset=DatasetType.MNIST)
        case ModelType.OCTONION_MNIST:
            run_octonion_perceptron(dataset=DatasetType.MNIST)
        case ModelType.QUATERNION_XOR:
            run_quaternion_perceptron(dataset=DatasetType.XOR)
        case ModelType.OCTONION_XOR:
            run_octonion_perceptron(dataset=DatasetType.XOR)
        case _:
            err_msg = f"Invalid model type: {model}"
            raise ValueError(err_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=ModelType, required=True)
    args = parser.parse_args()

    main(model=args.model)
