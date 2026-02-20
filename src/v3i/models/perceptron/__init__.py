"""Perceptrons."""

from .nn import OctonionSequential
from .nn import QuaternionSequential
from .octonion import OctonionBatchedOptimizer
from .octonion import OctonionPerceptron
from .octonion import OctonionSimpleOptimizer
from .quaternion import QuaternionBatchedOptimizer
from .quaternion import QuaternionPerceptron
from .quaternion import QuaternionSimpleOptimizer
from .utils import ForwardType

__all__ = [
    "ForwardType",
    "OctonionBatchedOptimizer",
    "OctonionPerceptron",
    "OctonionSequential",
    "OctonionSimpleOptimizer",
    "QuaternionBatchedOptimizer",
    "QuaternionPerceptron",
    "QuaternionSequential",
    "QuaternionSimpleOptimizer",
]
