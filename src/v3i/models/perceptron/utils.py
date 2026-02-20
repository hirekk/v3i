"""Utility functions for perceptron models."""

from __future__ import annotations

from typing import Literal

ForwardType = Literal[
    "average",
    "left_multiplication",
    "right_multiplication",
    "algebraic_sum",
    "algebraic_mean",
    "two_bracketings",  # Octonion-only: combine left- and right-associated products
]
