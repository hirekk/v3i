"""Perceptrons."""

from .octonion import OctonionSequential
from .nn import QuaternionSequential
from .octonion import OctonionPerceptron
from .utils import ForwardType

__all__ = [
    "ForwardType",
    "OctonionPerceptron",
    "OctonionSequential",
    "QuaternionSequential",
]
