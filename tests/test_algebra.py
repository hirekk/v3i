"""Tests for v3i.algebra (Octonion and related)."""

import numpy as np

from v3i.algebra import Octonion
from v3i.algebra import cross_product_7d

_TRIALS = 10_000


def _unit_octonions(n: int, seed: int = 42) -> list[Octonion]:
    rng = np.random.default_rng(seed)
    return [
        Octonion(np.asarray(rng.standard_normal(8), dtype=np.float64)).normalize() for _ in range(n)
    ]


def _random_7d_pairs(n: int, seed: int = 42) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [(rng.standard_normal(7), rng.standard_normal(7)) for _ in range(n)]


def _imaginary_octonion(v: np.ndarray) -> Octonion:
    return Octonion(np.concatenate([[0.0], v]))


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


def test_cross_product_7d_matches_basis_products() -> None:
    """e_i x e_j == Im(e_i * e_j) for all 42 ordered distinct basis pairs."""
    for i in range(7):
        for j in range(7):
            if i == j:
                continue
            e_i = np.zeros(7)
            e_i[i] = 1.0
            e_j = np.zeros(7)
            e_j[j] = 1.0
            expected = (_imaginary_octonion(e_i) * _imaginary_octonion(e_j)).im
            assert np.allclose(cross_product_7d(e_i, e_j), expected, atol=1e-14)


def test_cross_product_7d_matches_octonion_product() -> None:
    """U x v == Im(o_u * o_v) for random pure-imaginary octonions."""
    for u, v in _random_7d_pairs(_TRIALS):
        expected = (_imaginary_octonion(u) * _imaginary_octonion(v)).im
        assert np.allclose(cross_product_7d(u, v), expected, atol=1e-12)


def test_cross_product_7d_anticommutativity() -> None:
    """U x v == -(v x u)."""
    for u, v in _random_7d_pairs(_TRIALS):
        assert np.allclose(cross_product_7d(u, v), -cross_product_7d(v, u), atol=1e-12)


def test_cross_product_7d_orthogonality() -> None:
    """<u x v, u> == <u x v, v> == 0, including the regression case u=e1+e4, v=e7."""
    # Regression: the old hard-coded table returned u itself here (<u x v, u> = 2).
    u_reg = np.array([1.0, 0, 0, 1.0, 0, 0, 0])  # e1 + e4
    v_reg = np.array([0.0, 0, 0, 0, 0, 0, 1.0])  # e7
    c_reg = cross_product_7d(u_reg, v_reg)
    assert np.isclose(np.dot(c_reg, u_reg), 0.0, atol=1e-14)
    assert np.isclose(np.dot(c_reg, v_reg), 0.0, atol=1e-14)
    for u, v in _random_7d_pairs(_TRIALS):
        c = cross_product_7d(u, v)
        assert np.isclose(np.dot(c, u), 0.0, atol=1e-10)
        assert np.isclose(np.dot(c, v), 0.0, atol=1e-10)


def test_cross_product_7d_norm_identity() -> None:
    """|u x v|^2 == |u|^2 |v|^2 - <u,v>^2."""
    for u, v in _random_7d_pairs(_TRIALS):
        c = cross_product_7d(u, v)
        expected = np.dot(u, u) * np.dot(v, v) - np.dot(u, v) ** 2
        assert np.isclose(np.dot(c, c), expected, atol=1e-8)
