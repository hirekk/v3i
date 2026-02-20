"""Octonion algebra (avoids shadowing stdlib numbers)."""

from typing import Union

import numpy as np

# Type alias for scalar multiplication (float, int, np.float64, etc.)
Scalar = float | int | np.number

# Static mask for conjugation; avoids re-allocating in every conjugate/inverse call
_CONJ_MASK = np.array([1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float64)


def _quat_mul(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Quaternion product (4,) arrays. Used for Cayley-Dickson octonion mul."""
    return np.array(
        [
            u[0] * v[0] - u[1] * v[1] - u[2] * v[2] - u[3] * v[3],
            u[0] * v[1] + u[1] * v[0] + u[2] * v[3] - u[3] * v[2],
            u[0] * v[2] - u[1] * v[3] + u[2] * v[0] + u[3] * v[1],
            u[0] * v[3] + u[1] * v[2] - u[2] * v[1] + u[3] * v[0],
        ],
        dtype=np.float64,
    )


def _quat_conj(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate (4,) array."""
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _safe_sinc(norm: float) -> float:
    """Branchless sinc(norm) = sin(norm)/norm using Taylor for small norm.

    Taylor: 1 - norm^2/6 + norm^4/120. Avoids discontinuity at 1e-12 for
    better gradient properties and SIMD.
    """
    n2 = norm * norm
    if n2 < 1e-8:
        return 1.0 - n2 / 6.0 + (n2 * n2) / 120.0
    return np.sin(norm) / norm


def _safe_arctan2_scale(v_norm: float, re: float) -> float:
    """Branchless arctan2(v_norm, re) / v_norm for log/rotation vector.

    Small-angle: 1/re - v_norm^2/(3*re^3). Consistent with Taylor sinc.
    """
    if v_norm < 1e-8:
        if np.abs(re) < 1e-15:
            return 0.0
        return 1.0 / re - (v_norm * v_norm) / (3.0 * re * re * re)
    return np.arctan2(v_norm, re) / v_norm


class Octonion:
    """Octonion class. Single _data (8,) array as source of truth."""

    __slots__ = ["_data"]

    def __init__(self, data: np.ndarray) -> None:
        """Assign (8,) float64 array. Zero overhead. Use classmethods for other inputs."""
        self._data = data

    @property
    def re(self) -> float:
        """Real component."""
        return self._data[0]

    @property
    def im(self) -> np.ndarray:
        """Imaginary part (view, no allocation)."""
        return self._data[1:]

    def to_array(self) -> np.ndarray:
        """Return octonion as (8,) array."""
        return self._data.copy()

    def to_rotation_vector(self) -> np.ndarray:
        """Map octonion to tangent space (8D). out[0]=log(mag) for scale; out[1:]=rotation."""
        mag = float(np.linalg.norm(self._data))
        if mag < 1e-15:
            return np.zeros(8, dtype=np.float64)
        out = np.zeros(8, dtype=np.float64)
        out[0] = np.log(mag)
        v = self._data[1:]
        v_norm = float(np.linalg.norm(v))
        scale = _safe_arctan2_scale(v_norm, self._data[0])
        out[1:] = v * scale
        return out

    @classmethod
    def from_rotation_vector(cls, vector: np.ndarray) -> "Octonion":
        """Tangent vector (8D; out[0]=log scale) to octonion via exp."""
        vector = np.asarray(vector, dtype=np.float64).ravel()
        if vector.size != 8:
            error_message = f"Vector must be of size 8; got {vector.size}."
            raise ValueError(error_message)
        mag_log = vector[0]
        scale = np.exp(mag_log)
        v_im = vector[1:8]
        norm = float(np.linalg.norm(v_im))
        if norm < 1e-15:
            out = np.zeros(8, dtype=np.float64)
            out[0] = scale
            return cls(out)
        sinc_n = _safe_sinc(norm)
        comps = np.empty(8, dtype=np.float64)
        comps[0] = np.cos(norm)
        comps[1:8] = v_im * sinc_n
        comps *= scale
        return cls(comps)

    @classmethod
    def unit(cls) -> "Octonion":
        """Identity unit octonion (1, 0, 0, 0, 0, 0, 0, 0)."""
        return cls(np.array([1.0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64))

    def normalize(self) -> "Octonion":
        """Return unit octonion in same direction. For zero norm returns unit()."""
        n = float(np.linalg.norm(self._data))
        if n < 1e-15:
            return self.unit()
        return Octonion(self._data / n)

    def as_matrix(self, side: str = "left") -> np.ndarray:
        """8x8 real matrix: L @ q = self*q (left) or R @ q = q*self (right)."""
        w = self._data
        if side == "left":
            return np.array(
                [
                    [w[0], -w[1], -w[2], -w[3], -w[4], -w[5], -w[6], -w[7]],
                    [w[1], w[0], -w[3], w[2], -w[5], w[4], w[7], -w[6]],
                    [w[2], w[3], w[0], -w[1], -w[6], -w[7], w[4], w[5]],
                    [w[3], -w[2], w[1], w[0], -w[7], w[6], -w[5], w[4]],
                    [w[4], w[5], w[6], w[7], w[0], -w[1], -w[2], -w[3]],
                    [w[5], -w[4], w[7], -w[6], w[1], w[0], w[3], -w[2]],
                    [w[6], -w[7], -w[4], w[5], w[2], -w[3], w[0], w[1]],
                    [w[7], w[6], -w[5], -w[4], w[3], w[2], -w[1], w[0]],
                ],
                dtype=np.float64,
            )
        # Right: R such that R @ q = q * self (transpose structure from Cayley-Dickson)
        return np.array(
            [
                [w[0], -w[1], -w[2], -w[3], -w[4], -w[5], -w[6], -w[7]],
                [w[1], w[0], w[3], -w[2], w[5], -w[4], -w[7], w[6]],
                [w[2], -w[3], w[0], w[1], w[6], w[7], -w[4], -w[5]],
                [w[3], w[2], -w[1], w[0], w[7], -w[6], w[5], -w[4]],
                [w[4], -w[5], -w[6], -w[7], w[0], w[1], w[2], w[3]],
                [w[5], w[4], -w[7], w[6], -w[1], w[0], -w[3], w[2]],
                [w[6], w[7], w[4], -w[5], -w[2], w[3], w[0], -w[1]],
                [w[7], -w[6], w[5], w[4], -w[3], -w[2], w[1], w[0]],
            ],
            dtype=np.float64,
        )

    def __mul__(self, other: Union[Scalar, "Octonion"]) -> "Octonion":
        """Multiply this octonion by a scalar or another octonion."""
        if isinstance(other, (float, int, np.number)):
            return Octonion(self._data * float(other))
        if not isinstance(other, Octonion):
            return NotImplemented
        # Cayley-Dickson: (a,b)(c,d) = (ac - d*b, da + bc*) with a,b,c,d quaternions
        a, b = self._data[:4], self._data[4:]
        c, d = other._data[:4], other._data[4:]
        res = np.empty(8, dtype=np.float64)
        res[:4] = _quat_mul(a, c) - _quat_mul(_quat_conj(d), b)
        res[4:] = _quat_mul(d, a) + _quat_mul(b, _quat_conj(c))
        return Octonion(res)

    def __rmul__(self, other: Union[Scalar, "Octonion"]) -> "Octonion":
        """Multiply a scalar or another octonion by this octonion."""
        if isinstance(other, (float, int, np.number)):
            return Octonion(self._data * float(other))
        if not isinstance(other, Octonion):
            return NotImplemented
        return other * self

    def __truediv__(self, other: Scalar) -> "Octonion":
        """Divide this octonion by a scalar."""
        if not isinstance(other, (float, int, np.number)):
            return NotImplemented
        return Octonion(self._data / float(other))

    def __abs__(self) -> float:
        """Return the norm of this octonion."""
        return float(np.linalg.norm(self._data))

    def __add__(self, other: "Octonion") -> "Octonion":
        """Add this octonion to another octonion."""
        return Octonion(self._data + other._data)

    def __sub__(self, other: "Octonion") -> "Octonion":
        """Subtract another octonion from this octonion."""
        return Octonion(self._data - other._data)

    def __neg__(self) -> "Octonion":
        """Return the negation of this octonion."""
        return Octonion(-self._data)

    def inverse(self) -> "Octonion":
        """Return the inverse of this octonion."""
        norm_sq = float(np.dot(self._data, self._data))
        if norm_sq < 1e-15:
            error_message = "Zero division"
            raise ValueError(error_message)
        return Octonion(self._data * _CONJ_MASK / norm_sq)

    def conjugate(self) -> "Octonion":
        """Return the conjugate of this octonion. Optimized via static _CONJ_MASK."""
        return Octonion(self._data * _CONJ_MASK)

    def exp(self) -> "Octonion":
        """Return the exponential of this octonion."""
        a = self._data[0]
        v = self._data[1:]
        v_norm = float(np.linalg.norm(v))
        exp_a = np.exp(a)
        res = np.empty(8, dtype=np.float64)
        res[0] = np.cos(v_norm)
        if v_norm < 1e-8:
            sinc = 1.0 - (v_norm**2) / 6.0 + (v_norm**4) / 120.0
        else:
            sinc = np.sin(v_norm) / v_norm
        res[1:8] = v * sinc
        return Octonion(res * exp_a)

    def log(self) -> "Octonion":
        """Return the logarithm of this octonion."""
        norm = float(np.linalg.norm(self._data))
        if norm < 1e-15:
            error_message = "Log of zero"
            raise ValueError(error_message)
        v = self._data[1:]
        v_norm = float(np.linalg.norm(v))
        res = np.empty(8, dtype=np.float64)
        res[0] = np.log(norm)
        scale = _safe_arctan2_scale(v_norm, self._data[0])
        res[1:] = v * scale
        return Octonion(res)

    def copy(self) -> "Octonion":
        """Return a copy of this octonion."""
        return Octonion(self._data.copy())


def slerp(o1: Octonion, o2: Octonion, t: float) -> Octonion:
    """Spherical linear interpolation on S^7. Unit octonions; t in [0,1].

    If angle is very small, degrades to linear interpolation (lerp) so training never crashes.
    """
    dot = np.dot(o1.to_array(), o2.to_array())
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    if np.abs(theta) < 1e-12:
        # Near-identical: lerp instead of crashing
        return Octonion((1.0 - t) * o1.to_array() + t * o2.to_array())
    sin_theta = np.sin(theta)
    a = np.sin((1 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return Octonion(a * o1.to_array() + b * o2.to_array())


def cross_product_7d(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """7D cross product (Fano plane). Hard-coded to avoid triplet loop overhead."""
    a1, a2, a3, a4, a5, a6, a7 = v1[0], v1[1], v1[2], v1[3], v1[4], v1[5], v1[6]
    b1, b2, b3, b4, b5, b6, b7 = v2[0], v2[1], v2[2], v2[3], v2[4], v2[5], v2[6]
    res = np.empty(7, dtype=np.float64)
    res[0] = (a2 * b3 - a3 * b2) + (a4 * b7 - a7 * b4) + (a5 * b6 - a6 * b5)
    res[1] = (a3 * b1 - a1 * b3) + (a4 * b6 - a6 * b4) + (a7 * b5 - a5 * b7)
    res[2] = (a1 * b2 - a2 * b1) + (a4 * b5 - a5 * b4) + (a6 * b7 - a7 * b6)
    res[3] = (a1 * b7 - a7 * b1) + (a6 * b2 - a2 * b6) + (a5 * b3 - a3 * b5)
    res[4] = (a6 * b1 - a1 * b6) + (a7 * b2 - a2 * b7) + (a3 * b4 - a4 * b3)
    res[5] = (a1 * b5 - a5 * b1) + (a2 * b4 - a4 * b2) + (a7 * b3 - a3 * b7)
    res[6] = (a4 * b1 - a1 * b4) + (a2 * b5 - a5 * b2) + (a3 * b6 - a6 * b3)
    return res


def commutator(o1: Octonion, o2: Octonion) -> Octonion:
    """Commutator [o1, o2] = o1*o2 - o2*o1."""
    return o1 * o2 - o2 * o1


def associator(o1: Octonion, o2: Octonion, o3: Octonion) -> Octonion:
    """Associator (o1, o2, o3) = o1*(o2*o3) - (o1*o2)*o3."""
    return o1 * (o2 * o3) - (o1 * o2) * o3
