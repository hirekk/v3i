"""Empirical screen for the wayfinder ticket "The readout map: beyond sign(re)".

Companion to readout-screen.md. Run with:
    uv run python docs/research/readout_screen.py

The prior combiner screen (s7_combiner_screen.py) held the readout fixed at
sign(re(.)) and varied the *combiner*. This screen inverts that: it varies the
*readout* and asks two questions on the embedded XOR dataset.

(a) Does the CURRENT linear architecture, paired with a quadratic readout,
    already break the 0.75 XOR ceiling? (How much of the problem was ever
    forward-map nonlinearity vs the readout?)
(b) Do the mechanisms the combiner catalogue killed as "readout-invisible"
    (raw associator / commutator channels, additive associator channel) revive
    under a richer readout -- were they killed by the mechanism or the readout?

Framing theorem verified here: G2 = Aut(O) acts transitively on the unit
imaginary sphere S^6, so its orbits on S^7 are exactly the level sets of re(.);
hence any G2-invariant smooth readout is a function of re(x) alone, and the only
G2-invariant quadratic forms are span{|x|^2, re^2}. sign(re(.)) is THE canonical
G2-invariant readout, not a lazy one -- seeing more requires breaking G2.

All batched algebra is derived from the project's own Octonion class via its
structure tensor and cross-checked against __mul__ in the preflight. Deterministic.
"""

from __future__ import annotations

import numpy as np

from v3i.algebra import Octonion
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
E0E0 = np.zeros((8, 8))
E0E0[0, 0] = 1.0
HOPF_SIG = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.float64)  # |a|^2 - |b|^2


def bmul(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Rowwise octonion product of (n,8) arrays."""
    A = (X @ T.reshape(8, 64)).reshape(-1, 8, 8)
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


def bnormalize(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    out = X / np.where(n < 1e-12, 1.0, n)
    bad = n[:, 0] < 1e-12
    if bad.any():
        out[bad] = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    return out


def bassoc(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Rowwise associator [a,b,c] = (ab)c - a(bc)."""
    return bmul(bmul(A, B), C) - bmul(A, bmul(B, C))


# ---------------------------------------------------------------------------
# G2 = Aut(O) automorphisms, built as compositions of conjugations
# y -> (g y) g^-1 with g = exp(theta * n-hat), theta in {pi/3, 2pi/3} so that
# g^6 = 1 is real (the automorphism condition; deep-dive note 4). Products of
# automorphisms are automorphisms.
# ---------------------------------------------------------------------------

def conj_matrix(axis7: np.ndarray, theta: float) -> np.ndarray:
    nh = axis7 / np.linalg.norm(axis7)
    g = Octonion.from_rotation_vector(np.concatenate([[0.0], theta * nh]))
    L = g.as_matrix("left")
    R = g.inverse().as_matrix("right")
    return R @ L  # y -> (g y) g^-1  (unambiguous by diassociativity)


def sample_g2(rng: np.random.Generator, n_factors: int = 8) -> np.ndarray:
    """A random G2 automorphism as an 8x8 matrix: product of n_factors conjugations."""
    M = np.eye(8)
    for _ in range(n_factors):
        axis = rng.normal(size=7)
        theta = rng.choice([np.pi / 3.0, 2.0 * np.pi / 3.0])
        M = conj_matrix(axis, theta) @ M
    return M


# ---------------------------------------------------------------------------
# Readout fitting. Ridge least-squares on a feature map phi(Y) is the closed-form,
# deterministic "best linear-in-features readout"; for phi = identity it is the
# best linear readout <Y,v>+c, for phi = vech(Y Y^T) it is the best quadratic
# form <Y, Q Y> (36 params + bias). We report 0/1 accuracy of sign(fitted score).
# ---------------------------------------------------------------------------

def vech(Y: np.ndarray) -> np.ndarray:
    """Upper-triangular products Y_a Y_b (a<=b): the quadratic-form feature map (36 dims)."""
    cols = [Y[:, a] * Y[:, b] for a in range(8) for b in range(a, 8)]
    return np.stack(cols, axis=1)


def ridge_fit_predict(Ftr, ytr, Fte, lam=1e-3):
    """Fit sign(F w) by ridge LS on standardized features (+bias). Returns (acc_tr, acc_te)."""
    mu = Ftr.mean(axis=0)
    sd = Ftr.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Ztr = np.hstack([(Ftr - mu) / sd, np.ones((Ftr.shape[0], 1))])
    Zte = np.hstack([(Fte - mu) / sd, np.ones((Fte.shape[0], 1))])
    d = Ztr.shape[1]
    R = lam * np.eye(d)
    R[-1, -1] = 0.0  # do not penalize bias
    w = np.linalg.solve(Ztr.T @ Ztr + R, Ztr.T @ ytr.astype(float))
    acc_tr = float(np.mean(np.sign(Ztr @ w) == ytr))
    acc_te = float(np.mean(np.sign(Zte @ w) == yte))
    return acc_tr, acc_te


def best_threshold(score_tr, ytr, score_te):
    """Best-accuracy 1D threshold (both orientations), chosen on train, applied to test."""
    order = np.argsort(score_tr)
    s = score_tr[order]
    yo = ytr[order]
    # candidate thresholds are midpoints; evaluate accuracy of sign(score - tau)*orient
    taus = np.concatenate([[s[0] - 1.0], 0.5 * (s[:-1] + s[1:]), [s[-1] + 1.0]])
    best_acc, best_tau, best_or = -1.0, 0.0, 1
    for tau in taus:
        pred = np.where(score_tr >= tau, 1, -1)
        for orient in (1, -1):
            acc = float(np.mean(pred * orient == ytr))
            if acc > best_acc:
                best_acc, best_tau, best_or = acc, tau, orient
    pred_te = np.where(score_te >= best_tau, 1, -1) * best_or
    return best_acc, float(np.mean(pred_te == yte))


# ---------------------------------------------------------------------------
# Mechanisms: forward maps X (n,8) on S^7 -> pre-readout output Y (n,8).
# Some also expose a pre-normalization magnitude for the FF-goodness readout.
# ---------------------------------------------------------------------------

def m_linear(X, ws):
    return rmul(X, ws[0])


def m_branch(X, ws):
    return bmul(rmul(X, ws[0]), rmul(X, ws[1]))


def m_commutator(X, ws):
    a, b = rmul(X, ws[0]), rmul(X, ws[1])
    return bmul(a, b) - bmul(b, a)


def m_associator(X, ws):
    a1, a2 = rmul(X, ws[0]), rmul(X, ws[1])
    C = np.broadcast_to(ws[2], a1.shape)
    return bassoc(a1, C, a2)


def m_assoc_channel_num(X, ws):
    """Pre-normalization numerator x*w0 + [x*w1, c, x*w2] (additive associator channel)."""
    a1, a2 = rmul(X, ws[1]), rmul(X, ws[2])
    C = np.broadcast_to(ws[3], a1.shape)
    return rmul(X, ws[0]) + bassoc(a1, C, a2)


def m_sandwich(X, ws):
    """xbar * c * x (pure sandwich): readout-invisible to re (re is preserved)."""
    C = np.broadcast_to(ws[0], X.shape)
    return bmul(bmul(bconj(X), C), X)


# name, n_weights, fn, exposes_prenorm_magnitude
MECHANISMS = [
    ("linear x*w (current arch)", 1, m_linear, False),
    ("branch product (x*w1)(x*w2)", 2, m_branch, False),
    ("raw commutator [x*w1,x*w2]", 2, m_commutator, True),
    ("raw associator [x*w1,c,x*w2]", 3, m_associator, True),
    ("additive assoc channel", 4, m_assoc_channel_num, True),
    ("pure sandwich xbar*c*x", 1, m_sandwich, False),
]

N_DRAWS = 300
HIT_THRESHOLD = 0.80


def sample_unit(rng, quaternionic):
    w = rng.normal(size=8)
    if quaternionic:
        w[4:] = 0.0
    return w / np.linalg.norm(w)


def screen_mechanism(name, k, fn, has_prenorm, quaternionic, seed):
    """For one mechanism, best test accuracy over N_DRAWS weight draws for each readout."""
    rng = np.random.default_rng(seed)
    best = {r: (0.0, 0.0, 0) for r in ("re0", "lin", "hopf", "quad", "ff")}
    for _ in range(N_DRAWS):
        ws = [sample_unit(rng, quaternionic) for _ in range(k)]
        Ytr = fn(Xtr, ws)
        Yte = fn(Xte, ws)
        Ytr_n = bnormalize(Ytr)  # on-sphere output for the geometric readouts
        Yte_n = bnormalize(Yte)

        # re at tau=0 (the canonical control readout)
        pred = np.where(Ytr_n[:, 0] >= 0, 1, -1)
        acc = max(np.mean(pred == ytr), np.mean(-pred == ytr))
        orient = 1 if np.mean(pred == ytr) >= 0.5 else -1
        te = float(np.mean(np.where(Yte_n[:, 0] >= 0, 1, -1) * orient == yte))
        _update(best, "re0", float(acc), te)

        # best linear readout <Y,v>+c  (any direction; still linear in Y)
        atr, ate = ridge_fit_predict(Ytr_n, ytr, Yte_n)
        _update(best, "lin", atr, ate)

        # Hopf fiber balance |a|^2 - |b|^2 with learned threshold
        s_tr = (Ytr_n * Ytr_n) @ HOPF_SIG
        s_te = (Yte_n * Yte_n) @ HOPF_SIG
        atr, ate = best_threshold(s_tr, ytr, s_te)
        _update(best, "hopf", atr, ate)

        # general quadratic form <Y, Q Y> (36 params + bias)
        atr, ate = ridge_fit_predict(vech(Ytr_n), ytr, vech(Yte_n))
        _update(best, "quad", atr, ate)

        # FF-style goodness on the pre-normalization magnitude |Y|^2 (learned threshold)
        if has_prenorm:
            g_tr = np.sum(Ytr * Ytr, axis=1)
            g_te = np.sum(Yte * Yte, axis=1)
            atr, ate = best_threshold(g_tr, ytr, g_te)
            _update(best, "ff", atr, ate)
    return best


def _update(best, key, tr, te):
    btr, bte, hits = best[key]
    if tr >= HIT_THRESHOLD:
        hits += 1
    if tr > btr:
        btr, bte = tr, te
    best[key] = (btr, bte, hits)


# ---------------------------------------------------------------------------
# Data: the embedded XOR dataset of the prior notes.
# ---------------------------------------------------------------------------

rng_data = np.random.default_rng(42)
Xtr, ytr, Xte, yte = generate_binary_xor(
    train_size=800, test_size=200, noise=0.1, rng=rng_data, to_sphere=to_s7_from_2d
)


# ---------------------------------------------------------------------------
# Preflight + G2 canonicity verification.
# ---------------------------------------------------------------------------

def preflight():
    rng = np.random.default_rng(7)
    print("=== Preflight: batched algebra vs Octonion class ===")
    A = rng.normal(size=(64, 8))
    B = rng.normal(size=(64, 8))
    dev = max(np.max(np.abs(bmul(A, B)[i] - (Octonion(A[i]) * Octonion(B[i])).to_array()))
              for i in range(64))
    print(f"bmul vs __mul__ max dev: {dev:.2e}")
    assert dev < 1e-12

    print("\n=== G2 canonicity verification (the framing theorem) ===")
    gs = [sample_g2(rng, n_factors=10) for _ in range(2000)]
    # (1) each composed map is a genuine automorphism: g(xy) = g(x)g(y).
    dev_auto = 0.0
    for g in gs[:40]:
        dev_auto = max(dev_auto, np.max(np.abs(bmul(A @ g.T, B @ g.T) - bmul(A, B) @ g.T)))
    print(f"(1) automorphism property g(xy)-g(x)g(y) max: {dev_auto:.2e}")
    # (2) re is G2-invariant: re(g x) = re(x).
    Xu = bnormalize(rng.normal(size=(200, 8)))
    dev_re = max(np.max(np.abs((Xu @ g.T)[:, 0] - Xu[:, 0])) for g in gs[:400])
    print(f"(2) re(g x) - re(x) max: {dev_re:.2e}  -> re is G2-invariant")
    # (3) transitivity on S^6, via orbit isotropy: Im(g x0) covariance ~ (|Im x0|^2/7) I.
    x0 = bnormalize(np.array([[0.5, 0.3, -0.4, 0.2, 0.6, -0.1, 0.25, 0.15]]))[0]
    orbit_im = np.array([(g @ x0)[1:] for g in gs])
    s2 = np.sum(x0[1:] ** 2)
    cov = orbit_im.T @ orbit_im / len(gs)
    off = np.max(np.abs(cov - np.diag(np.diag(cov))))
    eig = np.linalg.eigvalsh(cov)
    print(f"(3) orbit Im-covariance ({len(gs)} g's): max|off-diag| {off:.3f}; eig spread "
          f"[{eig.min():.4f},{eig.max():.4f}] vs isotropic {s2/7:.4f}")
    print("     -> orbit fills S^6 isotropically: numerical evidence for transitivity on S^6")
    # (4) THE canonicity check: the G2 (Haar) average of a random symmetric Q is
    #     a*I + b*e0e0^T -- the only G2-invariant quadratic forms are span{|x|^2, re^2}.
    #     Monte-Carlo estimate; the residual falls off as 1/sqrt(K) toward 0.
    Q = rng.normal(size=(8, 8))
    Q = 0.5 * (Q + Q.T)
    conj = np.array([g.T @ Q @ g for g in gs])
    for K in (400, len(gs)):
        Qavg = conj[:K].mean(axis=0)
        b = Qavg[0, 0] - np.mean(np.diag(Qavg)[1:])
        a = np.mean(np.diag(Qavg)[1:])
        rel = np.linalg.norm(Qavg - (a * np.eye(8) + b * E0E0)) / np.linalg.norm(Qavg)
        print(f"(4) G2-avg of random symmetric Q, K={K:4d}: fit a*I+b*e0e0^T "
              f"(a={a:.3f}, b={b:.3f}); residual/||Qavg|| = {rel:.3f}")
    print("     -> invariant quadratics collapse to span{|x|^2, re^2}; re^2 is the canonical one")
    # (5) Hopf balance is NOT G2-invariant (it breaks the symmetry, by design).
    hb = np.array([np.sum((g @ x0)[:4] ** 2) - np.sum((g @ x0)[4:] ** 2) for g in gs])
    print(f"(5) Hopf balance |a|^2-|b|^2 across the orbit of one x0: "
          f"mean {hb.mean():.3f}, std {hb.std():.3f} (re=const {x0[0]:.3f}) "
          f"-> varies on a re-level set => breaks G2")

    print("\n=== Preflight: the readout-invisibility kills being revisited ===")
    ws2 = [sample_unit(rng, False) for _ in range(2)]
    ws3 = [sample_unit(rng, False) for _ in range(3)]
    ws4 = [sample_unit(rng, False) for _ in range(4)]
    dev = max(np.max(np.abs(m_commutator(Xu, ws2)[:, 0])),
              np.max(np.abs(m_associator(Xu, ws3)[:, 0])))
    print(f"(6) |re(raw commutator)|, |re(raw associator)| max: {dev:.2e}  -> blind to sign(re)")
    Ynum = m_assoc_channel_num(Xu, ws4)
    agree = np.mean(np.sign(Ynum[:, 0]) == np.sign(rmul(Xu, ws4[0])[:, 0]))
    print(f"(7) additive assoc channel: sign(re) == sign(re(x*w0)) on {agree:.4f} of points "
          "(expect 1.0) -> ceiling-bound under re, per the catalogue")
    dev_s = np.max(np.abs(m_sandwich(Xu, [ws2[0]])[:, 0] - Xu[:, 0] * 0 - ws2[0][0]))
    # re(xbar c x) = re(c) for unit x
    dev_s = np.max(np.abs(m_sandwich(Xu, [ws2[0]])[:, 0] - ws2[0][0]))
    print(f"(8) pure sandwich: re(xbar c x) - re(c) max: {dev_s:.2e}  -> constant re, blind")

    print("\n=== Preflight: the shattering escalation on the exact XOR corners ===")
    corners = to_s7_from_2d(np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]]))
    xor = np.array([-1, 1, 1, -1])
    Flin = np.hstack([corners, np.ones((4, 1))])
    Fquad = np.hstack([vech(corners), np.ones((4, 1))])
    rl = np.linalg.matrix_rank(Flin)
    rq = np.linalg.matrix_rank(Fquad)
    wl, *_ = np.linalg.lstsq(Flin, xor.astype(float), rcond=None)
    wq, *_ = np.linalg.lstsq(Fquad, xor.astype(float), rcond=None)
    print(f"(9) corners: max|coord>=3| {np.abs(corners[:,3:]).max():.1e}; "
          f"data spans span{{e0,e1,e2}}")
    print(f"    linear features rank {rl} (<4): fit signs {np.sign(Flin@wl).astype(int)} "
          f"!= XOR {xor}  -> cannot shatter (isometry-ceiling)")
    print(f"    quadratic features rank {rq} (=4): fit signs {np.sign(Fquad@wq).astype(int)} "
          f"== XOR {xor}  -> shatters; a quadratic form realizes XOR exactly")
    # Hopf balance on RAW data is constant (b-half is zero): trivial pre-forward-map.
    hb_raw = (Xtr * Xtr) @ HOPF_SIG
    print(f"(10) Hopf balance on raw embedded X: min {hb_raw.min():.3f} max {hb_raw.max():.3f} "
          "-> constant 1 (b-half zero); Hopf needs a layer to populate it")


def print_table(title, rows, keys, headers):
    print(f"\n=== {title} ===")
    w = 30
    line = f"{'mechanism':<{w}}" + "".join(f"{h:>16}" for h in headers)
    print(line)
    for name, best in rows:
        cells = []
        for kkey in keys:
            tr, te, hits = best[kkey]
            cells.append(f"{te:.3f}({hits/N_DRAWS:.2f})" if tr > 0 else "  --  ")
        print(f"{name:<{w}}" + "".join(f"{c:>16}" for c in cells))


if __name__ == "__main__":
    preflight()
    print(f"\nData: XOR seed 42, {len(ytr)} train / {len(yte)} test, noise 0.1; "
          f"{N_DRAWS} random unit-weight draws per mechanism.")
    print("Cells: best TEST accuracy over draws, with (hit-rate: fraction of draws whose "
          "fitted TRAIN acc >= 0.80).")
    print("Readouts: re0 = sign(re) at tau=0 (canonical control); lin = best <Y,v>+c "
          "(fitted, still linear in Y); hopf = |a|^2-|b|^2 + threshold; quad = <Y,QY> "
          "(fitted, 36 params); ff = goodness |Y_prenorm|^2 + threshold.")

    keys = ["re0", "lin", "hopf", "quad", "ff"]
    headers = ["re (control)", "best linear", "Hopf balance", "quadratic", "FF goodness"]

    oct_rows = [(name, screen_mechanism(name, k, fn, hp, False, 2024 + i))
                for i, (name, k, fn, hp) in enumerate(MECHANISMS)]
    print_table("Octonion weights (full S^7)", oct_rows, keys, headers)

    quat_rows = [(name, screen_mechanism(name, k, fn, hp, True, 5024 + i))
                 for i, (name, k, fn, hp) in enumerate(MECHANISMS)]
    print_table("Quaternion-control weights (subalgebra span{1,e1,e2,e3})",
                quat_rows, keys, headers)

    print("\nMarkdown (octonion pass; TEST acc, hit-rate in parens):")
    print("| mechanism | re (control) | best linear | Hopf balance | quadratic | FF goodness |")
    print("|---|---|---|---|---|---|")
    for name, best in oct_rows:
        cells = []
        for kkey in keys:
            tr, te, hits = best[kkey]
            cells.append(f"{te:.3f} ({hits/N_DRAWS:.2f})" if tr > 0 else "--")
        print(f"| {name} | " + " | ".join(cells) + " |")
