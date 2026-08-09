"""Batched octonion algebra — the vectorized companion to the per-object `Octonion`.

`Octonion` (in `v3i.algebra`) is the single source of truth for the algebra: this
module derives its 8x8x8 structure tensor from it, so the two can never disagree.
Callers that operate on many octonions at once (experiment harnesses, the geometry
helpers, the wide-layer network) use these row-wise primitives instead of looping
over `Octonion` objects or rebuilding the tensor themselves.

Contract: octonion arrays are `(n, 8)` float64; the quaternion subalgebra is the
`coords 4..7 == 0` slice (closed under the product). `bnormalize` and the geodesic
maps (`bexp`/`blog`) preserve unit norm on their intended inputs.
"""

from __future__ import annotations

import numpy as np

from v3i.algebra import Octonion

# Structure tensor T[i, j, k]: e_i * e_j = sum_k T[i, j, k] e_k, built once through
# the Octonion product so this module can never drift from the reference algebra.
_T = np.zeros((8, 8, 8), dtype=np.float64)
for _i in range(8):
    for _j in range(8):
        _ei, _ej = np.zeros(8), np.zeros(8)
        _ei[_i] = _ej[_j] = 1.0
        _T[_i, _j, :] = (Octonion(_ei) * Octonion(_ej)).to_array()

_T64 = _T.reshape(8, 64)
_CONJ = np.array([1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float64)


def bmul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise octonion product of two `(n, 8)` arrays."""
    a = (x @ _T64).reshape(-1, 8, 8)  # a[n,j,k] = sum_i x[n,i] T[i,j,k]
    return np.einsum("nj,njk->nk", y, a)


def rmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """`x·w` for each row of `x` and a fixed octonion `w` (8,)."""
    return x @ np.tensordot(w, _T, axes=([0], [1]))  # M[i,k] = sum_j w_j T[i,j,k]


def lmul(w: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`w·x` for each row of `x` and a fixed octonion `w` (8,)."""
    return x @ np.tensordot(w, _T, axes=([0], [0]))  # M[j,k] = sum_i w_i T[i,j,k]


def bconj(x: np.ndarray) -> np.ndarray:
    """Row-wise conjugate."""
    return x * _CONJ


def bnormalize(x: np.ndarray) -> np.ndarray:
    """Row-wise normalize; near-zero rows map to the unit octonion."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    out = x / np.where(n < 1e-12, 1.0, n)
    bad = n[:, 0] < 1e-12
    if bad.any():
        out[bad] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    return out


def _sinc(norm: np.ndarray) -> np.ndarray:
    """Row-wise sin(norm)/norm, Taylor-stable near 0 (matches `Octonion.exp`)."""
    small = norm < 1e-8
    safe = np.where(small, 1.0, norm)
    return np.where(small, 1.0 - norm**2 / 6.0 + norm**4 / 120.0, np.sin(safe) / safe)


def bexp(x: np.ndarray) -> np.ndarray:
    """Row-wise octonion exponential; matches `Octonion.exp`."""
    a, v = x[:, 0], x[:, 1:]
    vn = np.linalg.norm(v, axis=1)
    out = np.empty_like(x)
    out[:, 0] = np.cos(vn)
    out[:, 1:] = v * _sinc(vn)[:, None]
    return out * np.exp(a)[:, None]


def blog(x: np.ndarray) -> np.ndarray:
    """Row-wise octonion logarithm; matches `Octonion.log` (norm > 0 assumed)."""
    norm = np.linalg.norm(x, axis=1)
    v = x[:, 1:]
    vn = np.linalg.norm(v, axis=1)
    small = vn < 1e-8
    re = x[:, 0]
    # arctan2(vn, re)/vn, small-angle stable (matches `algebra._safe_arctan2_scale`)
    safe_vn = np.where(small, 1.0, vn)
    scale = np.where(small, 1.0 / re - (vn**2) / (3.0 * re**3), np.arctan2(vn, re) / safe_vn)
    out = np.empty_like(x)
    out[:, 0] = np.log(norm)
    out[:, 1:] = v * scale[:, None]
    return out


def bslerp(a: np.ndarray, b: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Row-wise spherical interpolation between unit rows; per-row `t` in [0, 1]."""
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    theta = np.arccos(dot)
    s = np.sin(theta)
    small = s < 1e-12
    safe = np.where(small, 1.0, s)
    ca = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / safe)
    cb = np.where(small, t, np.sin(t * theta) / safe)
    return ca[:, None] * a + cb[:, None] * b


def bassoc(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Row-wise associator `(ab)c - a(bc)`."""
    return bmul(bmul(a, b), c) - bmul(a, bmul(b, c))


def bcross7(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Row-wise 7D cross product of `(n, 7)` imaginary parts; matches `cross_product_7d`."""
    u8 = np.zeros((u.shape[0], 8))
    v8 = np.zeros((v.shape[0], 8))
    u8[:, 1:] = u
    v8[:, 1:] = v
    return bmul(u8, v8)[:, 1:]


# --- quaternion subalgebra (coords 0..3), for the quaternion-native fast path ---

_TQ = _T[:4, :4, :4]
_TQ16 = _TQ.reshape(4, 16)


def bqmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise quaternion product of `(n, 4)` arrays (the coords-0..3 subalgebra)."""
    p = (a @ _TQ16).reshape(-1, 4, 4)
    return np.einsum("nj,njk->nk", b, p)
