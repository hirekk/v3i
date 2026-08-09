"""Batched hypercomplex algebra agrees with the per-object Octonion reference.

This is the single consistency suite that replaces the per-file preflight
cross-checks the batched screens each carried.
"""

import numpy as np

from v3i import batched_algebra as ba
from v3i.algebra import Octonion
from v3i.algebra import cross_product_7d
from v3i.algebra import slerp


def _rows(rng: np.random.Generator, n: int = 32) -> np.ndarray:
    return rng.normal(size=(n, 8))


def test_bmul_matches_octonion_product() -> None:
    """Batched product matches the per-object Octonion product row by row."""
    rng = np.random.default_rng(0)
    x, y = _rows(rng), _rows(rng)
    expected = np.array([(Octonion(a) * Octonion(b)).to_array() for a, b in zip(x, y, strict=True)])
    assert np.allclose(ba.bmul(x, y), expected)


def test_rmul_and_lmul_match_octonion_product() -> None:
    """Fixed-weight right/left multiply match the reference product."""
    rng = np.random.default_rng(1)
    x = _rows(rng)
    w = rng.normal(size=8)
    assert np.allclose(ba.rmul(x, w), [(Octonion(a) * Octonion(w)).to_array() for a in x])
    assert np.allclose(ba.lmul(w, x), [(Octonion(w) * Octonion(a)).to_array() for a in x])


def test_bconj_matches() -> None:
    """Batched conjugate matches the reference."""
    rng = np.random.default_rng(2)
    x = _rows(rng)
    assert np.allclose(ba.bconj(x), [Octonion(a).conjugate().to_array() for a in x])


def test_norm_multiplicativity() -> None:
    """Product of unit rows stays unit norm."""
    rng = np.random.default_rng(3)
    x, y = ba.bnormalize(_rows(rng)), ba.bnormalize(_rows(rng))
    prod = ba.bmul(x, y)
    assert np.allclose(np.linalg.norm(prod, axis=1), 1.0)


def test_bexp_blog_match_octonion() -> None:
    """Batched exp/log match the reference exp/log."""
    rng = np.random.default_rng(4)
    x = _rows(rng) * 0.5  # keep magnitudes modest for a clean round trip
    assert np.allclose(ba.bexp(x), [Octonion(a).exp().to_array() for a in x])
    unit = ba.bnormalize(_rows(rng))
    assert np.allclose(ba.blog(unit), [Octonion(a).log().to_array() for a in unit])


def test_bexp_blog_round_trip_on_unit_rows() -> None:
    """exp(log(x)) recovers unit rows."""
    rng = np.random.default_rng(5)
    x = ba.bnormalize(_rows(rng))
    assert np.allclose(ba.bexp(ba.blog(x)), x, atol=1e-10)


def test_bcross7_matches_reference() -> None:
    """Batched 7D cross product matches cross_product_7d."""
    rng = np.random.default_rng(6)
    u, v = rng.normal(size=(16, 7)), rng.normal(size=(16, 7))
    assert np.allclose(
        ba.bcross7(u, v), [cross_product_7d(a, b) for a, b in zip(u, v, strict=True)]
    )


def test_bslerp_matches_reference() -> None:
    """Batched slerp matches the reference slerp."""
    rng = np.random.default_rng(7)
    a, b = ba.bnormalize(_rows(rng, 16)), ba.bnormalize(_rows(rng, 16))
    t = rng.uniform(0.1, 0.9, size=16)
    expected = np.array(
        [
            slerp(Octonion(ai), Octonion(bi), ti).to_array()
            for ai, bi, ti in zip(a, b, t, strict=True)
        ]
    )
    assert np.allclose(ba.bslerp(a, b, t), expected, atol=1e-10)


def test_bassoc_matches_reference() -> None:
    """Batched associator matches the reference associator."""
    rng = np.random.default_rng(8)
    a, b, c = _rows(rng, 16), _rows(rng, 16), _rows(rng, 16)
    expected = np.array(
        [
            (
                (Octonion(ai) * Octonion(bi)) * Octonion(ci)
                - Octonion(ai) * (Octonion(bi) * Octonion(ci))
            ).to_array()
            for ai, bi, ci in zip(a, b, c, strict=True)
        ]
    )
    assert np.allclose(ba.bassoc(a, b, c), expected)


def test_quaternion_subalgebra_stays_quaternionic() -> None:
    """Products of quaternion-subalgebra rows (coords 4..7 zero) stay in it."""
    rng = np.random.default_rng(9)
    x, y = _rows(rng), _rows(rng)
    x[:, 4:] = 0.0
    y[:, 4:] = 0.0
    assert np.allclose(ba.bmul(x, y)[:, 4:], 0.0)
