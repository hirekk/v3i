"""Perceptrons."""

from .nn import QuaternionSequential
from .octonion import OctonionPerceptron
from .octonion import OctonionSequential
from .quaternion import QuaternionPerceptron

__all__ = [
    "OctonionPerceptron",
    "OctonionSequential",
    "QuaternionPerceptron",
    "QuaternionSequential",
]
