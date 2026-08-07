"""Perceptrons."""

from .nn import QuaternionSequential
from .octonion import OctonionPerceptron
from .octonion import OctonionSequential
from .quaternion import QuaternionPerceptron
from .quaternion import QuaternionSimpleOptimizer
from .utils import ForwardType

__all__ = [
    "ForwardType",
    "OctonionPerceptron",
    "OctonionSequential",
    "QuaternionPerceptron",
    "QuaternionSequential",
    "QuaternionSimpleOptimizer",
]
