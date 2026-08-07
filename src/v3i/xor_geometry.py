"""XOR geometry: the dataset and the space of weights that discriminate it.

Two honest pictures behind the dashboard's "XOR geometry" view:

1. The dataset. XOR is four blobs in the plane (labels ±1 on the diagonal),
   embedded onto S⁷ by inverse stereographic projection. The embedding only
   ever populates coordinates 0, 1, 2 — so the whole dataset lives on an
   ordinary 2-sphere inside R⁸.

2. The discriminating weights. For the current linear readout
   `sign(re(x·w)) = sign(⟨x, conj(w)⟩)` and data confined to coords 0,1,2, the
   readout depends only on the separator normal `(w₀, -w₁, -w₂)` — a point on
   an ordinary 2-sphere. Sampling weights and colouring that sphere by XOR
   accuracy shows the isometry ceiling *geometrically*: no direction beats 75%.
   Richer combiners use several weights (not a single 2-sphere), so those are
   compared by their accuracy distribution instead.

The batched octonion algebra is derived from the project's own `Octonion`
class (single source of truth) exactly as the research screen does.
"""

from __future__ import annotations

import numpy as np

from v3i.algebra import Octonion
from v3i.make_data import generate_binary_xor
from v3i.make_data import to_s7_from_2d

XOR_LINEAR_CEILING = 0.75

# Structure tensor T[i,j,k]: e_i * e_j = sum_k T[i,j,k] e_k, from Octonion.
_T = np.zeros((8, 8, 8))
for _i in range(8):
    for _j in range(8):
        _ei, _ej = np.zeros(8), np.zeros(8)
        _ei[_i] = _ej[_j] = 1.0
        _T[_i, _j, :] = (Octonion(_ei) * Octonion(_ej)).to_array()

_CONJ = np.array([1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float64)


def bmul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Row-wise octonion product of (n, 8) arrays."""
    a = (x @ _T.reshape(8, 64)).reshape(-1, 8, 8)
    return np.einsum("nj,njk->nk", y, a)


def rmul(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x·w for each row of x, fixed weight w (8,)."""
    return x @ np.tensordot(w, _T, axes=([0], [1]))


def bconj(x: np.ndarray) -> np.ndarray:
    """Row-wise conjugate."""
    return x * _CONJ


def bnormalize(x: np.ndarray) -> np.ndarray:
    """Row-wise normalize; near-zero rows map to the identity octonion."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    out = x / np.where(n < 1e-12, 1.0, n)
    bad = n[:, 0] < 1e-12
    if bad.any():
        out[bad] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    return out


def load_xor(seed: int = 42, noise: float = 0.1) -> dict[str, np.ndarray]:
    """XOR dataset: the source plane points, labels, and the S⁷ embedding."""
    rng = np.random.default_rng(seed)
    x_train, y_train, x_test, y_test = generate_binary_xor(
        train_size=800, test_size=200, noise=noise, rng=rng, to_sphere=to_s7_from_2d
    )
    x_all = np.vstack([x_train, x_test])
    y_all = np.concatenate([y_train, y_test])
    # Recover the source-plane coordinate from the embedding: for u=(p,q,0..),
    # embedding is [(1-r²)/(1+r²), 2p/(1+r²), 2q/(1+r²), 0..]; invert via
    # (p,q) = (x1, x2) / (1 + x0).
    denom = 1.0 + x_all[:, 0]
    plane = np.column_stack([x_all[:, 1] / denom, x_all[:, 2] / denom])
    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "X_all": x_all,
        "y_all": y_all,
        "plane": plane,
    }


def _best_orientation_acc(readout: np.ndarray, y: np.ndarray) -> float:
    """Accuracy allowing either label orientation (flip-invariant)."""
    acc = float(np.mean(np.where(readout >= 0, 1, -1) == y))
    return max(acc, 1.0 - acc)


def sample_linear_separators(
    data: dict[str, np.ndarray], n_draws: int = 4000, seed: int = 0
) -> dict[str, np.ndarray]:
    """Random linear readouts, each a point on the separator-normal 2-sphere.

    Returns unit normals `(w₀, -w₁, -w₂)` (n, 3) and their best-orientation XOR
    accuracy — the exact geometric picture of the linear readout's reach.
    """
    rng = np.random.default_rng(seed)
    x, y = data["X_all"], data["y_all"]
    w = rng.normal(size=(n_draws, 8))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    normals = np.column_stack([w[:, 0], -w[:, 1], -w[:, 2]])
    nn = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.where(nn < 1e-12, 1.0, nn)
    # readout for draw k is sign(x0 w0 - x1 w1 - x2 w2) = sign(<x[:, :3], normal>)
    readouts = x[:, :3] @ normals.T  # (n_points, n_draws)
    acc = np.array([_best_orientation_acc(readouts[:, k], y) for k in range(n_draws)])
    return {"normals": normals, "acc": acc}


# --- Combiner mechanisms (subset of the research screen), for distribution view ---


def _linear(x: np.ndarray, ws: list[np.ndarray]) -> np.ndarray:
    return rmul(x, ws[0])


def _branch_product(x: np.ndarray, ws: list[np.ndarray]) -> np.ndarray:
    return bmul(rmul(x, ws[0]), rmul(x, ws[1]))


def _kappa_slerp(x: np.ndarray, ws: list[np.ndarray]) -> np.ndarray:
    a, b = rmul(x, ws[0]), rmul(x, ws[1])
    w1w2 = (Octonion(ws[0]) * Octonion(ws[1])).to_array()
    assoc = rmul(rmul(x, ws[0]), ws[1]) - rmul(x, w1w2)
    kappa = 1.0 - np.clip(np.linalg.norm(assoc, axis=1), 0.0, 1.0)
    dot = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    th = np.arccos(dot)
    s = np.sin(th)
    small = s < 1e-9
    safe = np.where(small, 1.0, s)
    ca = np.where(small, 1.0 - kappa, np.sin((1.0 - kappa) * th) / safe)
    cb = np.where(small, kappa, np.sin(kappa * th) / safe)
    return ca[:, None] * a + cb[:, None] * b


def _triple_cross(x: np.ndarray, ws: list[np.ndarray]) -> np.ndarray:
    u, v, w = rmul(x, ws[0]), rmul(x, ws[1]), rmul(x, ws[2])
    vb = bconj(v)
    return bnormalize(0.5 * (bmul(u, bmul(vb, w)) - bmul(w, bmul(vb, u))))


MECHANISMS: dict[str, tuple[int, object]] = {
    "linear (current)": (1, _linear),
    "kappa-gated slerp": (2, _kappa_slerp),
    "branch product": (2, _branch_product),
    "triple cross product": (3, _triple_cross),
}


def sample_mechanism_accuracies(
    data: dict[str, np.ndarray], mechanism: str, n_draws: int = 2000, seed: int = 0
) -> np.ndarray:
    """Best-orientation XOR accuracy over random-weight draws for one mechanism."""
    k, fn = MECHANISMS[mechanism]
    rng = np.random.default_rng(seed)
    x, y = data["X_all"], data["y_all"]
    out = np.empty(n_draws)
    for i in range(n_draws):
        ws = []
        for _ in range(k):
            w = rng.normal(size=8)
            ws.append(w / np.linalg.norm(w))
        out[i] = _best_orientation_acc(fn(x, ws)[:, 0], y)
    return out
