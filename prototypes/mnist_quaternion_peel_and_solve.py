"""PROTOTYPE — throwaway. Does the quaternion peel-and-solve net learn MNIST?

Resolves wayfinder ticket v3i#21 (MNIST prototype). Implements the locked decisions:

  #18 encoding : image -> fixed patch grid -> one unit quaternion per patch
                 (gradient-free, on-sphere); 10 one-vs-rest sign(re) heads,
                 predict argmax_k re(y_k); target +id (true head) / -id (rest).
  #19 arch     : D=1. Each head is a width-L SEQUENCE-AS-BRANCHES bank:
                 branch_i = x_i * w_i, y = b_1 * ... * b_L (branch product,
                 raster order). 10 INDEPENDENT heads. Online per-sample
                 peel-and-solve, eta = 0.9/W (W=L). Contraction invariant
                 ||residual|| <= ||incoming|| logged as a metric.
  #20 protocol : competitors = MLP-matched (DoF) + MLP-best (tuned) + logistic,
                 all on RAW PIXELS. Lean axes: accuracy + sample-efficiency +
                 stability. Pre-registered headline: SUPERIOR iff the quaternion
                 net beats MLP-matched using <= 10% of training data (N <= 6000).

Throwaway: self-contained, no repo abstractions (quaternion ops inlined so the
prototype does not depend on the unmerged batched_algebra). A weak result is a
valid result — it reroutes the encoding/architecture tickets.

Run:  uv run python prototypes/mnist_quaternion_peel_and_solve.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

# ----------------------------------------------------------------------------
# quaternion ops on (..., 4) real arrays  [w, x, y, z], Hamilton product
# ----------------------------------------------------------------------------
IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, broadcasting over leading axes."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def qconj(a: np.ndarray) -> np.ndarray:
    """Conjugate == inverse for unit quaternions."""
    out = a.copy()
    out[..., 1:] *= -1.0
    return out


def qnormalize(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.where(n < 1e-12, 1.0, n)


def qpow(q: np.ndarray, t: float) -> np.ndarray:
    """Unit-quaternion power q^t via axis-angle; t in [0,1] here."""
    w = np.clip(q[..., 0], -1.0, 1.0)
    theta = np.arccos(w)
    s = np.sin(theta)
    small = s < 1e-8
    axis = q[..., 1:] / np.where(small[..., None], 1.0, s[..., None])
    out = np.empty_like(q)
    out[..., 0] = np.cos(t * theta)
    out[..., 1:] = np.sin(t * theta)[..., None] * axis
    out[small] = IDENTITY  # near-identity rotation -> stays identity
    return out


def geodesic_angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle on S^3 between unit quaternions (0 = identical, pi = antipodal)."""
    return np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0))


def _selfcheck() -> None:
    """Sanity: inlined qmul matches numpy-quaternion on random inputs."""
    import quaternion  # noqa: PLC0415  (prototype sanity check only)

    rng = np.random.default_rng(0)
    a, b = rng.normal(size=4), rng.normal(size=4)
    got = qmul(a, b)
    ref = quaternion.as_float_array(quaternion.quaternion(*a) * quaternion.quaternion(*b))
    assert np.allclose(got, ref), (got, ref)


# ----------------------------------------------------------------------------
# encoding (#18): image -> (L, 4) sequence of unit quaternions
# ----------------------------------------------------------------------------
def make_encoder(grid: int, seed: int) -> tuple:
    """Fixed patch->quaternion map. GRID x GRID patches; each patch's pixels are
    projected to R^4 by a fixed random matrix, biased toward identity, normalized.

    Blank (all-zero) patches map to the identity quaternion, so background is
    'transparent' to the branch product — an intentional inductive bias.
    """
    assert 28 % grid == 0, f"grid must divide 28; got {grid}"
    patch = 28 // grid
    rng = np.random.default_rng(seed)
    proj = rng.normal(size=(patch * patch, 4)) / np.sqrt(patch * patch)
    return grid, patch, proj


def encode(images: np.ndarray, enc: tuple) -> np.ndarray:
    """(N, 784) uint8/float -> (N, L, 4) unit quaternions.  L = grid*grid."""
    grid, patch, proj = enc
    n = images.shape[0]
    img = images.reshape(n, 28, 28) / 255.0
    patches = (
        img.reshape(n, grid, patch, grid, patch)  # (n, gy, py, gx, px)
        .transpose(0, 1, 3, 2, 4)  # (n, gy, gx, py, px)
        .reshape(n, grid * grid, patch * patch)  # (n, L, patch^2), raster order
    )
    r = patches @ proj  # (n, L, 4)
    r[..., 0] += 1.0  # bias real part -> blank patch becomes identity
    return qnormalize(r)


# ----------------------------------------------------------------------------
# the quaternion peel-and-solve net (#19): 10 independent sequence-as-branch banks
# ----------------------------------------------------------------------------
class QuatPeelSolve:
    def __init__(self, n_heads: int, length: int, seed: int, lr_scale: float = 0.9) -> None:
        self.h, self.length = n_heads, length
        self.eta = lr_scale / length  # eta = 0.9 / W,  W = L
        rng = np.random.default_rng(seed)
        w = np.tile(IDENTITY, (n_heads, length, 1)) + 0.1 * rng.normal(size=(n_heads, length, 4))
        self.w = qnormalize(w)  # (H, L, 4) unit
        self.inv_violations = 0
        self.inv_count = 0
        self.inv_ratio_sum = 0.0
        self.inv_ratio_max = 0.0

    def _outputs(self, seq: np.ndarray) -> np.ndarray:
        """Batched forward. seq (N,L,4) -> (N,H,4) head outputs y_k."""
        b = qmul(seq[:, None, :, :], self.w[None, :, :, :])  # (N,H,L,4)
        y = b[:, :, 0, :]
        for i in range(1, self.length):
            y = qmul(y, b[:, :, i, :])
        return y  # (N,H,4)

    def predict(self, seq: np.ndarray) -> np.ndarray:
        return np.argmax(self._outputs(seq)[..., 0], axis=1)  # argmax_k re(y_k)

    def _update_one(self, x: np.ndarray, label: int, measure_inv: bool) -> None:
        """Online peel-and-solve on one sample x (L,4). Updates all H heads."""
        y_star = np.tile(-IDENTITY, (self.h, 1))  # rest heads -> -identity
        y_star[label] = IDENTITY  # true head -> +identity

        b = qmul(x[None, :, :], self.w)  # (H,L,4) branches
        # prefix P[:,i] = b0..b_{i-1}, suffix S[:,i] = b_{i+1}..b_{L-1}
        pre = np.empty_like(b)
        suf = np.empty_like(b)
        pre[:, 0] = IDENTITY
        for i in range(1, self.length):
            pre[:, i] = qmul(pre[:, i - 1], b[:, i - 1])
        suf[:, -1] = IDENTITY
        for i in range(self.length - 2, -1, -1):
            suf[:, i] = qmul(b[:, i + 1], suf[:, i + 1])
        y = qmul(pre[:, -1], b[:, -1])  # (H,4) full product

        # peel: b_i* = P_i^-1 y* S_i^-1 ; solve: w_i* = x_i^-1 b_i*
        b_star = qmul(qmul(qconj(pre), y_star[:, None, :]), qconj(suf))
        w_star = qmul(qconj(x)[None, :, :], b_star)
        u = qpow(qmul(qconj(self.w), w_star), self.eta)
        self.w = qnormalize(qmul(self.w, u))

        if measure_inv:
            incoming = geodesic_angle(y, y_star)  # (H,)
            b2 = qmul(x[None, :, :], self.w)
            y2 = b2[:, 0, :]
            for i in range(1, self.length):
                y2 = qmul(y2, b2[:, i, :])
            residual = geodesic_angle(y2, y_star)
            live = incoming > 1e-6
            ratio = residual[live] / incoming[live]
            self.inv_count += int(live.sum())
            self.inv_violations += int((ratio > 1.0 + 1e-6).sum())
            self.inv_ratio_sum += float(ratio.sum())
            self.inv_ratio_max = max(self.inv_ratio_max, float(ratio.max(initial=0.0)))

    def fit(self, seq: np.ndarray, y: np.ndarray, epochs: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        for ep in range(epochs):
            order = rng.permutation(len(y))
            for j, idx in enumerate(order):
                self._update_one(seq[idx], int(y[idx]), measure_inv=(ep == 0 and j < 300))

    def dof(self) -> int:
        return self.h * self.length * 3  # unit quaternion = 3 DoF

    def weight_stats(self) -> tuple[float, float]:
        norms = np.linalg.norm(self.w, axis=-1)
        angles = geodesic_angle(self.w, np.tile(IDENTITY, (self.h, self.length, 1)))
        return float(norms.mean()), float(np.degrees(angles.mean()))


# ----------------------------------------------------------------------------
# data
# ----------------------------------------------------------------------------
def load_mnist() -> tuple:
    cache = Path("/Users/hkubica/.claude/jobs/9bf8b8ba/tmp/mnist.npz")
    if cache.exists():
        d = np.load(cache)
        return d["X"], d["y"]
    from sklearn.datasets import fetch_openml  # noqa: PLC0415

    x, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    x, y = x.astype(np.float32), y.astype(np.int64)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=x, y=y)
    return x, y


def stratified_prefix(y: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    return idx[:n]


# ----------------------------------------------------------------------------
# experiment
# ----------------------------------------------------------------------------
def main() -> None:
    _selfcheck()
    grid = 7  # 7x7 = 49 patches of 4x4 -> L = 49, W = 49, eta = 0.9/49
    epochs = 3
    seed = 0
    ns = [100, 500, 1000, 3000, 6000]  # <= 10% of 60k train
    print(f"GRID={grid} (L={grid * grid})  epochs={epochs}  seed={seed}\n")

    t0 = time.time()
    x_all, y_all = load_mnist()
    # standard split: first 60k train, last 10k test
    x_tr, y_tr, x_te, y_te = x_all[:60000], y_all[:60000], x_all[60000:], y_all[60000:]
    enc = make_encoder(grid, seed=123)
    seq_te = encode(x_te, enc)
    print(f"loaded MNIST + encoded test in {time.time() - t0:.1f}s  (test {len(y_te)})\n")

    quat_ref = QuatPeelSolve(10, grid * grid, seed=seed)
    dof = quat_ref.dof()
    # DoF-matched MLP: 784 -> h -> 10 with params ~ dof
    h_match = max(1, round((dof - 10) / (784 + 1 + 10)))
    match_params = 784 * h_match + h_match + h_match * 10 + 10
    print(f"quaternion net DoF = {dof}   MLP-matched hidden={h_match} (~{match_params} params)")
    print(f"logistic on 784px  = {784 * 10 + 10} params   (linear floor)\n")

    header = f"{'N':>6} | {'quat':>6} {'inv%':>5} | {'MLP-match':>9} | {'MLP-best':>8} | {'logreg':>6}"
    print(header)
    print("-" * len(header))

    results = []
    for n in ns:
        sel = stratified_prefix(y_tr, n, seed=seed)
        xs, ys = x_tr[sel], y_tr[sel]
        seq_tr = encode(xs, enc)

        q = QuatPeelSolve(10, grid * grid, seed=seed)
        q.fit(seq_tr, ys, epochs=epochs, seed=seed)
        q_acc = float(np.mean(q.predict(seq_te) == y_te))
        inv_pct = 100.0 * q.inv_violations / max(1, q.inv_count)

        mlp_m = MLPClassifier(hidden_layer_sizes=(h_match,), max_iter=300, random_state=seed)
        mlp_m.fit(xs / 255.0, ys)
        m_acc = float(mlp_m.score(x_te / 255.0, y_te))

        mlp_b = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=seed)
        mlp_b.fit(xs / 255.0, ys)
        b_acc = float(mlp_b.score(x_te / 255.0, y_te))

        lr = LogisticRegression(max_iter=200, C=1.0)
        lr.fit(xs / 255.0, ys)
        l_acc = float(lr.score(x_te / 255.0, y_te))

        results.append((n, q_acc, m_acc, b_acc, l_acc))
        print(
            f"{n:>6} | {q_acc:>6.3f} {inv_pct:>4.1f}% | {m_acc:>9.3f} | {b_acc:>8.3f} | {l_acc:>6.3f}"
        )

    # stability / interpretation snapshot from the largest run
    mean_norm, mean_angle = q.weight_stats()
    print(
        f"\ninvariant: {q.inv_violations}/{q.inv_count} violations "
        f"(mean ratio {q.inv_ratio_sum / max(1, q.inv_count):.3f}, max {q.inv_ratio_max:.3f})"
    )
    print(f"weights:   mean |w| = {mean_norm:.4f} (should be 1.0), "
          f"mean rotation from identity = {mean_angle:.1f} deg")

    # verdict on the pre-registered headline (results row = (n, q, m, b, l))
    print("\n" + "=" * 60)
    print("PRE-REGISTERED HEADLINE: quat beats MLP-matched at N <= 6000?")
    won = [(n, qa, ma) for (n, qa, ma, _, _) in results if qa > ma]
    if won:
        for n, qa, ma in won:
            print(f"  MET at N={n}: quat {qa:.3f} > MLP-matched {ma:.3f}")
    else:
        best = max(results, key=lambda r: r[1] - r[2])
        print(f"  NOT MET. Closest: N={best[0]} quat {best[1]:.3f} vs MLP-matched {best[2]:.3f}")
    print("=" * 60)
    print(f"\ntotal wall-clock: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
