"""Tests for v3i.numbers (Octonion and related)."""

import numpy as np

from v3i.numbers import Octonion

_TRIALS = 10_000


def _unit_octonions(n: int, seed: int = 42) -> list[Octonion]:
    rng = np.random.default_rng(seed)
    return [
        Octonion(np.asarray(rng.standard_normal(8), dtype=np.float64)).normalize() for _ in range(n)
    ]


def test_exp_log_symmetry() -> None:
    """exp(log(q)) == q for unit octonions."""
    q_unit = _unit_octonions(_TRIALS)
    for q in q_unit:
        q_recon = q.log().exp()
        assert np.allclose(q_recon.to_array(), q.to_array(), atol=1e-10)


def test_norm_preservation() -> None:
    """|q1 * q2| == |q1| * |q2| (division algebra property)."""
    q_unit = _unit_octonions(_TRIALS)
    for i in range(_TRIALS):
        q1, q2 = q_unit[i], q_unit[(i + 1) % _TRIALS]
        assert np.isclose(abs(q1 * q2), abs(q1) * abs(q2), atol=1e-12)


def test_alternative_property() -> None:
    """(q * q) * q2 == q * (q * q2) (octonions are alternative)."""
    q_unit = _unit_octonions(_TRIALS)
    for i in range(_TRIALS):
        q, q2 = q_unit[i], q_unit[(i + 1) % _TRIALS]
        left = ((q * q) * q2).to_array()
        right = (q * (q * q2)).to_array()
        assert np.allclose(left, right, atol=1e-12)


def test_small_angle_stability() -> None:
    """from_rotation_vector with tiny components does not produce NaNs."""
    rng = np.random.default_rng(42)
    for _ in range(_TRIALS):
        v_tiny = np.zeros(8)
        v_tiny[1:] = rng.standard_normal(7) * 1e-10
        q_tiny = Octonion.from_rotation_vector(v_tiny)
        assert not np.any(np.isnan(q_tiny.to_array()))
