"""Numerical verification for the wayfinder ticket "Isometry ceiling of the current stack".

Companion to isometry-ceiling.md. Run with: uv run python docs/research/isometry_ceiling_verification.py
"""

import numpy as np
from numpy.linalg import norm
from scipy.optimize import linprog
from sklearn.linear_model import LogisticRegression

from v3i.algebra import Octonion
from v3i.make_data import generate_binary_1d, generate_binary_xor, to_s7_from_1d, to_s7_from_2d

rng = np.random.default_rng(0)


def rand_unit_oct():
    return Octonion(rng.normal(size=8)).normalize()


def rand_oct():
    return Octonion(rng.normal(size=8))


def conj(o):
    return o.conjugate()


print("=== 1. R_w is orthogonal for unit w (composition algebra) ===")
worst = 0.0
for _ in range(200):
    w = rand_unit_oct()
    R = w.as_matrix("right")
    worst = max(worst, norm(R.T @ R - np.eye(8)))
print(f"max ||R^T R - I|| over 200 random unit w: {worst:.3e}")

print("\n=== 1b. as_matrix('right') agrees with __mul__ ===")
worst = 0.0
for _ in range(100):
    w, x = rand_unit_oct(), rand_oct()
    worst = max(worst, norm(w.as_matrix("right") @ x.to_array() - (x * w).to_array()))
print(f"max ||R_w x - (x*w)||: {worst:.3e}")

print("\n=== 2. A depth-5 chain is a single orthogonal matrix ===")
ws = [rand_unit_oct() for _ in range(5)]
M = np.eye(8)
for w in ws:
    M = w.as_matrix("right") @ M
worst = 0.0
for _ in range(100):
    x = rand_oct()
    y = x
    for w in ws:
        y = y * w
    worst = max(worst, norm(M @ x.to_array() - y.to_array()))
print(f"max ||M x - chain(x)||: {worst:.3e}")
print(f"||M^T M - I||: {norm(M.T @ M - np.eye(8)):.3e}")

print("\n=== 3. Readout identity: re(x*w) = <x, conj(w)> ===")
worst = 0.0
for _ in range(200):
    w, x = rand_unit_oct(), rand_oct()
    worst = max(worst, abs((x * w).re - float(np.dot(x.to_array(), conj(w).to_array()))))
print(f"max |re(x*w) - <x, conj(w)>|: {worst:.3e}")

print("\n=== 4. Readout collapse at depth 2 and 3 (octonions!) ===")
worst2 = worst3 = 0.0
for _ in range(200):
    w1, w2, w3, x = rand_unit_oct(), rand_unit_oct(), rand_unit_oct(), rand_oct()
    lhs2 = ((x * w1) * w2).re
    rhs2 = float(np.dot(x.to_array(), conj(w1 * w2).to_array()))
    worst2 = max(worst2, abs(lhs2 - rhs2))
    lhs3 = (((x * w1) * w2) * w3).re
    rhs3 = float(np.dot(x.to_array(), conj(w1 * (w2 * w3)).to_array()))
    worst3 = max(worst3, abs(lhs3 - rhs3))
print(f"max |re((x w1) w2) - <x, conj(w1 w2)>|: {worst2:.3e}")
print(f"max |re(((x w1) w2) w3) - <x, conj(w1 (w2 w3)))>|: {worst3:.3e}")

print("\n=== 5. Quaternion subalgebra: full collapse (associativity) ===")
worst = 0.0
for _ in range(100):
    a, b, c = rng.normal(size=4), rng.normal(size=4), rng.normal(size=4)
    q = lambda v: Octonion(np.concatenate([v, np.zeros(4)]))  # noqa: E731
    lhs = (q(c) * q(a)) * q(b)
    rhs = q(c) * (q(a) * q(b))
    worst = max(worst, norm((lhs - rhs).to_array()))
print(f"max ||((x w1) w2) - (x (w1 w2))|| within quaternion subalgebra: {worst:.3e}")

print("\n=== 6. Octonion full map does NOT collapse (but its readout row does) ===")
gaps, row_gaps = [], []
for _ in range(100):
    w1, w2 = rand_unit_oct(), rand_unit_oct()
    M = w2.as_matrix("right") @ w1.as_matrix("right")
    # First row of any R_v is conj(v), so a single equivalent weight is forced:
    v = Octonion(M[0].copy()).conjugate()
    Rv = v.as_matrix("right")
    gaps.append(norm(M - Rv))
    row_gaps.append(norm(M[0] - conj(w1 * w2).to_array()))
print(f"first row of M equals conj(w1*w2): max gap {max(row_gaps):.3e}")
print(f"||M - R_v|| for the forced single v: min {min(gaps):.3f}, median {np.median(gaps):.3f}")

print("\n=== 7. XOR gate: LP feasibility of sign patterns on embedded corner centers ===")
corners = np.array([[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]])
labels = np.array([-1, 1, 1, -1])  # XOR diagonal labeling (A, B, C, D)
E = to_s7_from_2d(corners)  # (4, 8) on S^7
print(f"embedded corners (first 3 coords):\n{E[:, :3].round(4)}")
print(f"|A|^2+|D|^2 vs |B|^2+|C|^2 in plane: "
      f"{norm(corners[0])**2 + norm(corners[3])**2:.4f} vs "
      f"{norm(corners[1])**2 + norm(corners[2])**2:.4f}")


def separable(signs):
    """Strictly separable by homogeneous hyperplane: exists v with s_i <v, x_i> >= 1."""
    A_ub = -(signs[:, None] * E)  # -s_i <v, x_i> <= -1
    b_ub = -np.ones(len(signs))
    res = linprog(c=np.zeros(8), A_ub=A_ub, b_ub=b_ub, bounds=[(None, None)] * 8,
                  method="highs")
    return res.status == 0


print(f"XOR pattern (-,+,+,-) separable: {separable(labels)}")
for i in range(4):
    flipped = labels.copy()
    flipped[i] = -flipped[i]
    print(f"pattern with corner {i} flipped {tuple(flipped)}: separable: {separable(flipped)}")

print("\n=== 8. Empirical ceiling: homogeneous logistic regression on the actual datasets ===")
rng2 = np.random.default_rng(42)
Xtr, ytr, Xte, yte = generate_binary_xor(800, 200, 0.1, rng2, to_sphere=to_s7_from_2d)
clf = LogisticRegression(fit_intercept=False, max_iter=2000)
clf.fit(Xtr, ytr)
print(f"XOR S^7:      train acc {clf.score(Xtr, ytr):.3f}, test acc {clf.score(Xte, yte):.3f}")

rng3 = np.random.default_rng(42)
Xtr, ytr, Xte, yte = generate_binary_1d(800, 200, 0.1, rng3, to_sphere=to_s7_from_1d)
clf = LogisticRegression(fit_intercept=False, max_iter=2000)
clf.fit(Xtr, ytr)
print(f"binary-1d S^7: train acc {clf.score(Xtr, ytr):.3f}, test acc {clf.score(Xte, yte):.3f}")
print("\n=== 9. Empirical ceiling: best 3-of-4-corner homogeneous separator on noisy XOR ===")
rng4 = np.random.default_rng(42)
Xtr, ytr, Xte, yte = generate_binary_xor(800, 200, 0.1, rng4, to_sphere=to_s7_from_2d)

best = (0.0, None)
for i in range(4):
    s = labels.copy()
    s[i] = -s[i]
    # maximize margin eps: s_j <v, x_j> >= eps, ||v||_1 <= 1 (LP-friendly via v = p - n)
    # variables: v(8), eps(1); maximize eps
    A_ub = np.hstack([-(s[:, None] * E), np.ones((4, 1))])
    b_ub = np.zeros(4)
    # bound v components to [-1, 1] to keep eps finite
    res = linprog(c=[0]*8 + [-1], A_ub=A_ub, b_ub=b_ub,
                  bounds=[(-1, 1)]*8 + [(0, None)], method="highs")
    v = res.x[:8]
    tr = np.mean(np.sign(Xtr @ v) == ytr)
    te = np.mean(np.sign(Xte @ v) == yte)
    print(f"pattern flip corner {i}: margin {res.x[8]:.3f}  train acc {tr:.3f}  test acc {te:.3f}")
    if tr > best[0]:
        best = (tr, te)
print(f"\nbest 3-corner separator: train {best[0]:.3f}, test {best[1]:.3f} (theoretical corner ceiling 0.75)")
