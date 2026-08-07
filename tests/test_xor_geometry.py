"""Geometric invariants behind the dashboard's XOR view."""

import numpy as np

from v3i import xor_geometry as xg
from v3i.algebra import Octonion


def test_embedding_populates_only_first_three_coords() -> None:
    """Embedding confined to coords 0,1,2 (the separator-2-sphere depends on it)."""
    data = xg.load_xor()
    assert np.allclose(data["X_all"][:, 3:], 0.0)


def test_batched_mul_matches_octonion_class() -> None:
    """Batched rmul agrees with the reference Octonion product."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(16, 8))
    w = rng.normal(size=8)
    expected = np.array([(Octonion(row) * Octonion(w)).to_array() for row in x])
    assert np.allclose(xg.rmul(x, w), expected)


def test_linear_readout_never_beats_ceiling() -> None:
    """No random linear separator exceeds the proven 75% ceiling (small margin)."""
    data = xg.load_xor()
    sph = xg.sample_linear_separators(data, n_draws=3000, seed=0)
    assert sph["acc"].max() <= xg.XOR_LINEAR_CEILING + 0.02


def test_nonlinear_mechanisms_can_break_ceiling() -> None:
    """Branch product has random draws above the ceiling; linear does not."""
    data = xg.load_xor()
    linear = xg.sample_mechanism_accuracies(data, "linear (current)", n_draws=500, seed=0)
    branch = xg.sample_mechanism_accuracies(data, "branch product", n_draws=500, seed=0)
    assert linear.max() <= xg.XOR_LINEAR_CEILING + 0.02
    assert branch.max() > xg.XOR_LINEAR_CEILING
