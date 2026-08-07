"""Empirical screen for the wayfinder ticket "Geometric catalogue of nonlinear maps S7 x S7 -> S7".

Companion to s7-combiner-catalogue.md. Run with:
    uv run python docs/research/s7_combiner_screen.py

For each catalogued combiner candidate F(x; W): sample ~2000 random unit-weight
draws, compute the readout sign(re(F(x))) on the embedded XOR dataset
(generate_binary_xor, seed 42, 800 train / 200 test, noise 0.1, to_s7_from_2d),
and record the best train accuracy found, its test accuracy, and the fraction of
draws whose train accuracy reaches 0.80. This is a can-it-possibly-break-75%
screen under random search — no learning rule involved. Both label orientations
are allowed (orientation is chosen on train, applied to test).

Two passes: octonion weights (full S7) and quaternion-control weights
(restricted to the subalgebra span{1,e1,e2,e3}; the embedded data already lies
in span{1,e1,e2}, so all products stay quaternionic). The control pass measures
which candidates owe their nonlinearity to non-associativity.

All batched algebra is derived from the project's own Octonion class via its
structure tensor, and cross-checked against the class in the preflight section.
"""

from __future__ import annotations

import numpy as np

from v3i.algebra import Octonion, cross_product_7d, slerp
from v3i.make_data import generate_binary_xor, to_s7_from_2d

# ---------------------------------------------------------------------------
# Batched octonion algebra, derived from the Octonion class (single source of
# truth: the structure tensor T is built by multiplying basis elements).
# ---------------------------------------------------------------------------

T = np.zeros((8, 8, 8))
for i in range(8):
    for j in range(8):
        ei = np.zeros(8)
        ej = np.zeros(8)
        ei[i] = 1.0
        ej[j] = 1.0
        T[i, j, :] = (Octonion(ei) * Octonion(ej)).to_array()

CONJ = np.array([1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float64)

# Quaternion structure tensor = top-left corner of T (span{1,e1,e2,e3} is a
# subalgebra of the Cayley-Dickson doubling).
TQ = T[:4, :4, :4]


def bmul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rowwise octonion product of (n,8) arrays."""
    A = (X @ T.reshape(8, 64)).reshape(-1, 8, 8)  # A[n,j,k] = sum_i X[n,i] T[i,j,k]
    return np.einsum("nj,njk->nk", Y, A)


def rmul(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x * w for each row of X, fixed octonion w (8,)."""
    M = np.tensordot(w, T, axes=([0], [1]))  # M[i,k] = sum_j w_j T[i,j,k]
    return X @ M


def lmul(w: np.ndarray, X: np.ndarray) -> np.ndarray:
    """w * x for each row of X, fixed octonion w (8,)."""
    M = np.tensordot(w, T, axes=([0], [0]))  # M[j,k] = sum_i w_i T[i,j,k]
    return X @ M


def bconj(X: np.ndarray) -> np.ndarray:
    return X * CONJ


def bnormalize(X: np.ndarray, fallback: np.ndarray | None = None) -> np.ndarray:
    """Rowwise normalize; rows with ~zero norm get `fallback` (default e0)."""
    n = np.linalg.norm(X, axis=1, keepdims=True)
    bad = n[:, 0] < 1e-12
    out = X / np.where(n < 1e-12, 1.0, n)
    if bad.any():
        fb = np.array([1.0, 0, 0, 0, 0, 0, 0, 0]) if fallback is None else fallback
        out[bad] = fb
    return out


def bslerp(A: np.ndarray, B: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Rowwise slerp between unit rows of A and B, per-row t in [0,1]."""
    dot = np.clip(np.sum(A * B, axis=1), -1.0, 1.0)
    th = np.arccos(dot)
    s = np.sin(th)
    small = s < 1e-9
    safe = np.where(small, 1.0, s)
    ca = np.where(small, 1.0 - t, np.sin((1.0 - t) * th) / safe)
    cb = np.where(small, t, np.sin(t * th) / safe)
    return ca[:, None] * A + cb[:, None] * B


def bassoc(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Rowwise associator [a,b,c] = (ab)c - a(bc)."""
    return bmul(bmul(A, B), C) - bmul(A, bmul(B, C))


def bcross7(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Rowwise 7D cross product of (n,7) imaginary parts: Im(o_u o_v)."""
    U8 = np.zeros((U.shape[0], 8))
    V8 = np.zeros((V.shape[0], 8))
    U8[:, 1:] = U
    V8[:, 1:] = V
    return bmul(U8, V8)[:, 1:]


def bqmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Rowwise quaternion product of (n,4) arrays."""
    P = (A @ TQ.reshape(4, 16)).reshape(-1, 4, 4)
    return np.einsum("nj,njk->nk", B, P)


def n7(U: np.ndarray) -> np.ndarray:
    """Rowwise normalize (n,7) with zero-guard (zero rows stay zero)."""
    n = np.linalg.norm(U, axis=1, keepdims=True)
    return U / np.where(n < 1e-12, 1.0, n)


def g2_conjugation_matrix(axis7: np.ndarray) -> np.ndarray:
    """8x8 matrix of y -> g y g^-1 with g = exp((pi/3) n-hat): an automorphism of O.

    Conjugation by g is an automorphism iff g^6 is real (deep-dive note 4);
    g = exp((pi/3) n) gives g^6 = exp(2 pi n) = 1. Diassociativity makes the
    bracketing of g y g^-1 unambiguous.
    """
    nh = axis7 / np.linalg.norm(axis7)
    g = Octonion.from_rotation_vector(np.concatenate([[0.0], (np.pi / 3) * nh]))
    L = g.as_matrix("left")
    R = g.inverse().as_matrix("right")
    return R @ L  # y -> (g y) g^-1


# ---------------------------------------------------------------------------
# Candidate combiners. Each takes (X, ws) with X (n,8) on S7 and ws a list of
# unit-octonion weights (8,), and returns the pre-readout output Y (n,8).
# Readout is always sign(Y[:,0]) with train-chosen orientation.
# ---------------------------------------------------------------------------


def f_renorm_sum(X, ws):
    """Control: normalize(x w1 + x w2) = x normalize(w1+w2). Provably linear."""
    return bnormalize(rmul(X, ws[0]) + rmul(X, ws[1]))


def f_kappa_slerp(X, ws):
    """Shortlist leader: slerp(x w1, x w2, kappa(x)), kappa = 1 - min(|[x,w1,w2]|,1)."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    w1w2 = (Octonion(ws[0]) * Octonion(ws[1])).to_array()
    assoc = rmul(rmul(X, ws[0]), ws[1]) - rmul(X, w1w2)
    kappa = 1.0 - np.clip(np.linalg.norm(assoc, axis=1), 0.0, 1.0)
    return bslerp(a, b, kappa)


def f_branch_product(X, ws):
    """Shortlist leader: (x w1)(x w2). Unit by norm multiplicativity."""
    return bmul(rmul(X, ws[0]), rmul(X, ws[1]))


def f_jordan(X, ws):
    """Jordan product of branch images: normalize((ab+ba)/2), a = x w1, b = x w2."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    return bnormalize(0.5 * (bmul(a, b) + bmul(b, a)))


def f_commutator_raw(X, ws):
    """[x w1, x w2] raw: pure imaginary, readout-blind by construction."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    return bmul(a, b) - bmul(b, a)


def f_commutator_rotor(X, ws):
    """normalize([x w1, x w2]) * w3: commutator made readout-visible."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    return rmul(bnormalize(bmul(a, b) - bmul(b, a)), ws[2])


def f_twisted_branch(X, ws):
    """(x w1) * sigma(x w2), sigma in G2 (conjugation by exp((pi/3) n), n from w3)."""
    axis = ws[2][1:]
    if np.linalg.norm(axis) < 1e-12:
        axis = np.array([1.0, 0, 0, 0, 0, 0, 0])
    M = g2_conjugation_matrix(axis)
    return bmul(rmul(X, ws[0]), rmul(X, ws[1]) @ M.T)


def f_assoc_channel(X, ws):
    """normalize(x w0 + [x w1, c, x w2]), c = w3. Additive associator channel."""
    a1, a2 = rmul(X, ws[1]), rmul(X, ws[2])
    C = np.broadcast_to(ws[3], a1.shape)
    return bnormalize(rmul(X, ws[0]) + bassoc(a1, C, a2))


def f_assoc_rotor(X, ws):
    """normalize([x w1, c, x w2]) * w4, c = w3. Associator made readout-visible."""
    a1, a2 = rmul(X, ws[0]), rmul(X, ws[1])
    C = np.broadcast_to(ws[2], a1.shape)
    return rmul(bnormalize(bassoc(a1, C, a2)), ws[3])


def f_phi_slerp(X, ws):
    """slerp(x w1, x w2, t), t = (1 + phi(Im x, Im(x w1), Im(x w2)) normalized)/2."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    u, v, w = n7(X[:, 1:]), n7(a[:, 1:]), n7(b[:, 1:])
    phi = np.sum(bcross7(u, v) * w, axis=1)  # associative 3-form, |phi| <= 1
    t = np.clip(0.5 * (1.0 + phi), 0.0, 1.0)
    return bslerp(a, b, t)


def f_psi_slerp(X, ws):
    """slerp(x w1, x w2, t), t = (1 + psi)/2 with the coassociative-form gate.

    psi = (1/2) <Im-hat x, [Im-hat(x w1), Im-hat(x w2), Im-hat(x w3)]>;
    |[u,v,w]| <= 2 on unit args, so |psi| <= 1 without clipping. On H the
    associator vanishes, t = 1/2, and slerp midpoint = normalize(a+b): the
    candidate collapses exactly to the provably-linear renormalized sum.
    """
    a, b, c = rmul(X, ws[0]), rmul(X, ws[1]), rmul(X, ws[2])
    im = np.zeros_like(X)
    im[:, 1:] = n7(X[:, 1:])
    A = np.zeros_like(X)
    A[:, 1:] = n7(a[:, 1:])
    B = np.zeros_like(X)
    B[:, 1:] = n7(b[:, 1:])
    C = np.zeros_like(X)
    C[:, 1:] = n7(c[:, 1:])
    psi = 0.5 * np.sum(im * bassoc(A, B, C), axis=1)
    t = np.clip(0.5 * (1.0 + psi), 0.0, 1.0)
    return bslerp(a, b, t)


def f_triple_cross(X, ws):
    """normalize(X3(x w1, x w2, x w3)), X3(u,v,w) = (u(vbar w) - w(vbar u))/2."""
    u, v, w = rmul(X, ws[0]), rmul(X, ws[1]), rmul(X, ws[2])
    vb = bconj(v)
    return bnormalize(0.5 * (bmul(u, bmul(vb, w)) - bmul(w, bmul(vb, u))))


def f_hopf_twist(X, ws):
    """Hopf fiber twist: y1 = x w1 = (a1,b1), q = unit(top half of x w2); y = (a1 q, b1 q).

    Moves x w1 along its own quaternionic Hopf fiber by an amount read from
    x w2; the S4 base point of y equals that of x w1 exactly. |y| = 1 exactly.
    """
    y1, y2 = rmul(X, ws[0]), rmul(X, ws[1])
    q = y2[:, :4]
    qn = np.linalg.norm(q, axis=1, keepdims=True)
    q = np.where(qn < 1e-12, np.array([1.0, 0, 0, 0]), q / np.where(qn < 1e-12, 1.0, qn))
    out = np.empty_like(y1)
    out[:, :4] = bqmul(y1[:, :4], q)
    out[:, 4:] = bqmul(y1[:, 4:], q)
    return out


def f_triality_sandwich(X, ws):
    """(x w1) * (w3 * (x w2)): branch product with an interior fixed rotor."""
    return bmul(rmul(X, ws[0]), lmul(ws[2], rmul(X, ws[1])))


def f_nambu_step(X, ws):
    """Euler-top step: normalize(x + J(x)) * w3, J = cross7(Im(x w1), Im(x w2))."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    J = bcross7(a[:, 1:], b[:, 1:])
    step = X.copy()
    step[:, 1:] = step[:, 1:] + J
    return rmul(bnormalize(step), ws[2])


def f_precession_rotor(X, ws):
    """Precession: rotate x about axis n(x) = unit(cross7(Im(x w1), Im(x w2)))
    by angle theta(x) = (pi/2) min(|cross7|, 1), then * w3 for readout visibility."""
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    c = bcross7(a[:, 1:], b[:, 1:])
    cn = np.linalg.norm(c, axis=1, keepdims=True)
    nh = c / np.where(cn < 1e-12, 1.0, cn)
    th = 0.5 * (np.pi / 2) * np.minimum(cn, 1.0)  # half-angle for the sandwich
    Q = np.concatenate([np.cos(th), np.sin(th) * nh], axis=1)
    return rmul(bmul(bmul(Q, X), bconj(Q)), ws[2])


def f_moebius(X, ws):
    """Moebius: normalize((x w1 + tau w3) * (x w2 + tau w4)^-1), tau = 0.7.

    sign(re) = sign(<x w1 + tau w3, x w2 + tau w4>): an inhomogeneous quadratic.
    """
    tau = 0.7
    num = rmul(X, ws[0]) + tau * ws[2]
    den = rmul(X, ws[1]) + tau * ws[3]
    dn2 = np.sum(den * den, axis=1, keepdims=True)
    inv = bconj(den) / np.where(dn2 < 1e-12, 1.0, dn2)
    return bnormalize(bmul(num, inv))


CANDIDATES = [
    # (name, n_weights, fn)
    ("renormalized sum (control)", 2, f_renorm_sum),
    ("kappa-gated slerp (leader)", 2, f_kappa_slerp),
    ("branch product (leader)", 2, f_branch_product),
    ("Jordan branch product", 2, f_jordan),
    ("commutator raw", 2, f_commutator_raw),
    ("commutator rotor", 3, f_commutator_rotor),
    ("G2-twisted branch product", 3, f_twisted_branch),
    ("additive associator channel", 4, f_assoc_channel),
    ("associator rotor", 4, f_assoc_rotor),
    ("phi-gated slerp", 2, f_phi_slerp),
    ("psi-gated slerp", 3, f_psi_slerp),
    ("triple cross product", 3, f_triple_cross),
    ("Hopf fiber twist", 2, f_hopf_twist),
    ("triality sandwich", 3, f_triality_sandwich),
    ("Nambu/Euler-top step", 3, f_nambu_step),
    ("precession rotor", 3, f_precession_rotor),
    ("Moebius combiner", 4, f_moebius),
]

N_DRAWS = 2000
HIT_THRESHOLD = 0.80


def sample_unit(rng: np.random.Generator, quaternionic: bool) -> np.ndarray:
    w = rng.normal(size=8)
    if quaternionic:
        w[4:] = 0.0
    return w / np.linalg.norm(w)


def accuracy(readout: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Best accuracy over the two label orientations; returns (acc, orientation)."""
    pred = np.where(readout >= 0.0, 1, -1)
    m = float(np.mean(pred == y))
    return (m, 1) if m >= 1.0 - m else (1.0 - m, -1)


def screen(quaternionic: bool) -> list[tuple[str, float, float, float]]:
    rows = []
    for idx, (name, k, fn) in enumerate(CANDIDATES):
        rng = np.random.default_rng(1000 + idx)  # deterministic per candidate
        best_train, best_test, hits = 0.0, 0.0, 0
        for _ in range(N_DRAWS):
            ws = [sample_unit(rng, quaternionic) for _ in range(k)]
            Y_tr = fn(X_train, ws)
            tr, orient = accuracy(Y_tr[:, 0], y_train)
            if tr >= HIT_THRESHOLD:
                hits += 1
            if tr > best_train:
                Y_te = fn(X_test, ws)
                pred = np.where(Y_te[:, 0] >= 0.0, 1, -1) * orient
                best_train, best_test = tr, float(np.mean(pred == y_test))
        rows.append((name, best_train, best_test, hits / N_DRAWS))
    return rows


def print_table(title: str, rows) -> None:
    print(f"\n=== {title} ===")
    print(f"{'candidate':<32} {'best train':>10} {'test @best':>10} {'hit>=0.80':>10}")
    for name, tr, te, hit in rows:
        print(f"{name:<32} {tr:>10.3f} {te:>10.3f} {hit:>10.3f}")


# ---------------------------------------------------------------------------
# Preflight: check the batched algebra and the identities the catalogue cites.
# ---------------------------------------------------------------------------

def preflight() -> None:
    rng = np.random.default_rng(7)
    print("=== Preflight: batched algebra vs Octonion class ===")
    A = rng.normal(size=(64, 8))
    B = rng.normal(size=(64, 8))
    dev = max(
        np.max(np.abs(bmul(A, B)[i] - (Octonion(A[i]) * Octonion(B[i])).to_array()))
        for i in range(64)
    )
    print(f"bmul vs __mul__ max dev: {dev:.2e}")
    assert dev < 1e-12
    w = rng.normal(size=8)
    dev = np.max(np.abs(rmul(A, w) - np.array([(Octonion(a) * Octonion(w)).to_array() for a in A])))
    print(f"rmul vs __mul__ max dev: {dev:.2e}")
    assert dev < 1e-12
    dev = max(
        np.max(np.abs(bcross7(A[i : i + 1, 1:], B[i : i + 1, 1:])[0]
                      - cross_product_7d(A[i, 1:], B[i, 1:])))
        for i in range(64)
    )
    print(f"bcross7 vs cross_product_7d max dev: {dev:.2e}")
    assert dev < 1e-12

    print("\n=== Preflight: identities used by the catalogue ===")
    Au = bnormalize(A)
    Bu = bnormalize(B)
    # (1) re(Jordan) == re(branch product): re(ab) = re(ba).
    dev = np.max(np.abs(0.5 * (bmul(Au, Bu) + bmul(Bu, Au))[:, 0] - bmul(Au, Bu)[:, 0]))
    print(f"(1) re((ab+ba)/2) - re(ab) max: {dev:.2e}  -> Jordan readout == branch-product readout")
    # (2) associator and commutator are pure imaginary.
    Cu = bnormalize(rng.normal(size=(64, 8)))
    dev = max(np.max(np.abs(bassoc(Au, Bu, Cu)[:, 0])),
              np.max(np.abs((bmul(Au, Bu) - bmul(Bu, Au))[:, 0])))
    print(f"(2) |re(associator)|, |re(commutator)| max: {dev:.2e}  -> readout-blind raw")
    # (3) slerp(a,b,1/2) == normalize(a+b) (psi-gate collapse target on H).
    dev = 0.0
    for i in range(64):
        s = slerp(Octonion(Au[i]), Octonion(Bu[i]), 0.5).to_array()
        dev = max(dev, np.max(np.abs(s - bnormalize((Au[i] + Bu[i])[None, :])[0])))
    print(f"(3) slerp(a,b,1/2) vs normalize(a+b) max dev: {dev:.2e}")
    # (4) Hopf twist: exact unit norm, and base point equals base(x w1).
    Xr = bnormalize(rng.normal(size=(256, 8)))
    ws = [sample_unit(np.random.default_rng(3), False) for _ in range(2)]
    Yh = f_hopf_twist(Xr, ws)
    y1 = rmul(Xr, ws[0])

    def hopf_base(Z):
        a, b = Z[:, :4], Z[:, 4:]
        s = np.sum(a * a, axis=1) - np.sum(b * b, axis=1)
        ab = bqmul(a, b * np.array([1, -1, -1, -1]))
        return np.column_stack([s, 2 * ab])

    dev_norm = np.max(np.abs(np.linalg.norm(Yh, axis=1) - 1.0))
    dev_base = np.max(np.abs(hopf_base(Yh) - hopf_base(y1)))
    print(f"(4) Hopf twist: | |y|-1 | max {dev_norm:.2e}; base(y) - base(x w1) max {dev_base:.2e}")
    # (5) calibration bounds: |phi| <= 1 on unit imaginary triples; |psi| <= 1.
    U, V, W = (n7(rng.normal(size=(20000, 7))) for _ in range(3))
    phi = np.sum(bcross7(U, V) * W, axis=1)
    U8, V8, W8, Z = (np.zeros((20000, 8)) for _ in range(4))
    U8[:, 1:] = U
    V8[:, 1:] = V
    W8[:, 1:] = W
    Z[:, 1:] = n7(rng.normal(size=(20000, 7)))
    psi = 0.5 * np.sum(Z * bassoc(U8, V8, W8), axis=1)
    print(f"(5) max |phi| over 20k unit triples: {np.max(np.abs(phi)):.6f} (comass 1);"
          f" max |psi|: {np.max(np.abs(psi)):.6f}")
    # (6) triple cross product X3: norm = parallelepiped volume; orthogonality;
    #     antisymmetry in outer args; behavior under repeated args.
    u, v, w2 = (bnormalize(rng.normal(size=(500, 8))) for _ in range(3))
    X3 = 0.5 * (bmul(u, bmul(bconj(v), w2)) - bmul(w2, bmul(bconj(v), u)))
    G = np.empty((500, 3, 3))
    for a_i, a_v in enumerate((u, v, w2)):
        for b_i, b_v in enumerate((u, v, w2)):
            G[:, a_i, b_i] = np.sum(a_v * b_v, axis=1)
    vol = np.sqrt(np.abs(np.linalg.det(G)))
    dev_norm = np.max(np.abs(np.linalg.norm(X3, axis=1) - vol))
    X3swap = 0.5 * (bmul(w2, bmul(bconj(v), u)) - bmul(u, bmul(bconj(v), w2)))
    dev_anti = np.max(np.abs(X3 + X3swap))
    dot_u = np.max(np.abs(np.sum(X3 * u, axis=1)))
    dot_v = np.max(np.abs(np.sum(X3 * v, axis=1)))
    dot_w = np.max(np.abs(np.sum(X3 * w2, axis=1)))
    print(f"(6) X3: | |X3| - vol(u,v,w) | max {dev_norm:.2e}; antisym(u<->w) {dev_anti:.2e};"
          f" <X3,u> {dot_u:.2e} <X3,v> {dot_v:.2e} <X3,w> {dot_w:.2e}")
    # (7) Moebius readout identity: re(p q^-1) = <p,q>/|q|^2.
    p, q = rng.normal(size=(64, 8)), rng.normal(size=(64, 8))
    qn2 = np.sum(q * q, axis=1, keepdims=True)
    lhs = bmul(p, bconj(q) / qn2)[:, 0]
    rhs = np.sum(p * q, axis=1) / qn2[:, 0]
    print(f"(7) re(p q^-1) - <p,q>/|q|^2 max: {np.max(np.abs(lhs - rhs)):.2e}")
    # (8) G2 conjugation sigma is an automorphism: sigma(ab) = sigma(a)sigma(b).
    M = g2_conjugation_matrix(np.array([0.3, -1.2, 0.5, 0.0, 0.7, -0.2, 1.1]))
    dev = np.max(np.abs(bmul(A @ M.T, B @ M.T) - bmul(A, B) @ M.T))
    print(f"(8) sigma(ab) - sigma(a)sigma(b) max: {dev:.2e}")
    # (9) additive associator channel sign-collapse: sign(re) == sign(re(x w0)).
    ws4 = [sample_unit(np.random.default_rng(5), False) for _ in range(4)]
    Ya = f_assoc_channel(Xr, ws4)
    agree = np.mean(np.sign(Ya[:, 0]) == np.sign(rmul(Xr, ws4[0])[:, 0]))
    print(f"(9) additive associator channel: sign(re) agrees with sign(re(x w0)) on"
          f" {agree:.4f} of 256 random points (expect 1.0)")
    # (10) quaternion-control collapses: kappa == 1 and psi == 0 on H.
    XrQ = Xr.copy()
    XrQ[:, 4:] = 0.0
    XrQ = bnormalize(XrQ)
    wsq = [sample_unit(np.random.default_rng(9), True) for _ in range(3)]
    w1w2 = (Octonion(wsq[0]) * Octonion(wsq[1])).to_array()
    assoc = rmul(rmul(XrQ, wsq[0]), wsq[1]) - rmul(XrQ, w1w2)
    a, b, c = rmul(XrQ, wsq[0]), rmul(XrQ, wsq[1]), rmul(XrQ, wsq[2])
    A8, B8, C8, im = (np.zeros_like(XrQ) for _ in range(4))
    A8[:, 1:] = n7(a[:, 1:])
    B8[:, 1:] = n7(b[:, 1:])
    C8[:, 1:] = n7(c[:, 1:])
    im[:, 1:] = n7(XrQ[:, 1:])
    psiq = 0.5 * np.sum(im * bassoc(A8, B8, C8), axis=1)
    print(f"(10) on H: max |[x,w1,w2]| = {np.max(np.linalg.norm(assoc, axis=1)):.2e} (kappa==1);"
          f" max |psi| = {np.max(np.abs(psiq)):.2e} (t==1/2)")
    # (11) triple-cross collapse on H: X3(xa, xb, xc) = x * (a bbar c - c bbar a)/2
    #      for unit quaternionic x (associativity + |x|^2 = 1 cancels the middle).
    aq, bq, cq = (np.array(sample_unit(np.random.default_rng(s), True)) for s in (11, 12, 13))
    u, v, w3 = rmul(XrQ, aq), rmul(XrQ, bq), rmul(XrQ, cq)
    X3q = 0.5 * (bmul(u, bmul(bconj(v), w3)) - bmul(w3, bmul(bconj(v), u)))
    oa, ob, oc = Octonion(aq), Octonion(bq), Octonion(cq)
    veff = (oa * ob.conjugate() * oc - oc * ob.conjugate() * oa).to_array() * 0.5
    dev = np.max(np.abs(X3q - rmul(XrQ, veff)))
    print(f"(11) on H: X3(xa,xb,xc) - x*((a bbar c - c bbar a)/2) max: {dev:.2e}"
          " -> triple cross collapses to a single linear layer on H")
    # (12) polarization of the norm form: <xa, xb> = |x|^2 <a,b> (kills the
    #      quadratic term of every one-step Moebius readout on unit signals).
    Xu = bnormalize(rng.normal(size=(256, 8)))
    a8 = sample_unit(np.random.default_rng(14), False)
    b8 = sample_unit(np.random.default_rng(15), False)
    dev = np.max(np.abs(np.sum(rmul(Xu, a8) * rmul(Xu, b8), axis=1) - np.dot(a8, b8)))
    print(f"(12) <xa,xb> - <a,b> max over unit x: {dev:.2e}"
          " -> Moebius readout is affine in x, ceiling-bound")


# ---------------------------------------------------------------------------
# Data: the embedded XOR dataset of the prior notes.
# ---------------------------------------------------------------------------

rng_data = np.random.default_rng(42)
X_train, y_train, X_test, y_test = generate_binary_xor(
    train_size=800, test_size=200, noise=0.1, rng=rng_data, to_sphere=to_s7_from_2d
)

if __name__ == "__main__":
    preflight()
    print(f"\nData: XOR seed 42, {len(y_train)} train / {len(y_test)} test, noise 0.1;"
          f" {N_DRAWS} random unit-weight draws per candidate.")
    print("Columns: best train accuracy over draws; test accuracy of the best-train draw;"
          f" fraction of draws with train >= {HIT_THRESHOLD}.")
    oct_rows = screen(quaternionic=False)
    print_table("Screen: octonion weights (full S7)", oct_rows)
    quat_rows = screen(quaternionic=True)
    print_table("Screen: quaternion-control weights (subalgebra span{1,e1,e2,e3})", quat_rows)
    print("\nMarkdown (octonion | quaternion control):")
    print("| candidate | best train | test @ best | hit>=0.80 | H best train | H test | H hit |")
    print("|---|---|---|---|---|---|---|")
    for (name, tr, te, hit), (_, qtr, qte, qhit) in zip(oct_rows, quat_rows, strict=True):
        print(f"| {name} | {tr:.3f} | {te:.3f} | {hit:.3f} | {qtr:.3f} | {qte:.3f} | {qhit:.3f} |")
