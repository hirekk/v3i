"""Numerical companion to docs/research/octonion-structure-deep-dive.md.

Run from the repo root:

    uv run python docs/research/octonion_structure_verification.py

Every check exercises the project's own implementation (src/v3i/algebra.py).
Output conventions:
  [PASS]/[FAIL] -- the identity under test holds/breaks (tol 1e-10 unless noted)
  [INFO]        -- a quantity expected to be nonzero, reported not judged
  [BUG?]        -- a claimed property of the *implementation* that fails
"""

from __future__ import annotations

import numpy as np

from v3i.algebra import Octonion
from v3i.algebra import associator as assoc_repo
from v3i.algebra import commutator
from v3i.algebra import cross_product_7d

rng = np.random.default_rng(7)
TOL = 1e-10


# ---------------------------------------------------------------- helpers


def rand_oct(unit: bool = False) -> Octonion:
    o = Octonion(rng.standard_normal(8))
    return o.normalize() if unit else o


def rand_imag() -> Octonion:
    v = rng.standard_normal(8)
    v[0] = 0.0
    return Octonion(v).normalize()


def basis(i: int) -> Octonion:
    v = np.zeros(8)
    v[i] = 1.0
    return Octonion(v)


def arr(o: Octonion) -> np.ndarray:
    return o.to_array()


def dev(a: Octonion, b: Octonion) -> float:
    return float(np.max(np.abs(arr(a) - arr(b))))


def dot(a: Octonion, b: Octonion) -> float:
    return float(np.dot(arr(a), arr(b)))


def assoc_std(x: Octonion, y: Octonion, z: Octonion) -> Octonion:
    """Standard associator [x,y,z] = (xy)z - x(yz).

    NOTE: v3i.algebra.associator uses the opposite sign convention,
    o1*(o2*o3) - (o1*o2)*o3.  Only the sign differs; |.| is identical.
    """
    return (x * y) * z - x * (y * z)


def cross_alg7(u7: np.ndarray, v7: np.ndarray) -> np.ndarray:
    """The algebra's own 7D cross product: Im(u*v) for imaginary u, v."""
    u = np.zeros(8)
    u[1:] = u7
    v = np.zeros(8)
    v[1:] = v7
    return arr(Octonion(u) * Octonion(v))[1:]


def check(label: str, value: float, tol: float = TOL) -> bool:
    ok = value <= tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}: max dev {value:.3e}")
    return ok


def info(label: str, value) -> None:
    print(f"  [INFO] {label}: {value}")


def header(title: str) -> None:
    print(f"\n=== {title} ===")


# ---------------------------------------------------------------- section 1


def section_1_composition() -> None:
    header("1. Composition-algebra facts")
    n_mult, adj1, adj2, antih, rmat, lmat = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(1000):
        x, y, z = rand_oct(), rand_oct(), rand_oct()
        n_mult = max(n_mult, abs(abs(x * y) - abs(x) * abs(y)))
        adj1 = max(adj1, abs(dot(x * y, z) - dot(x, z * y.conjugate())))
        adj2 = max(adj2, abs(dot(x * y, z) - dot(y, x.conjugate() * z)))
        antih = max(antih, dev((x * y).conjugate(), y.conjugate() * x.conjugate()))
        rmat = max(rmat, float(np.max(np.abs(y.as_matrix("right") @ arr(x) - arr(x * y)))))
        lmat = max(lmat, float(np.max(np.abs(y.as_matrix("left") @ arr(x) - arr(y * x)))))
    check("norm multiplicativity | |xy|-|x||y| |", n_mult)
    check("adjoint identity <xy,z> = <x, z*conj(y)>", adj1)
    check("adjoint identity <xy,z> = <y, conj(x)*z>", adj2)
    check("conjugation anti-automorphism conj(xy) = conj(y)conj(x)", antih)
    check("as_matrix('right') @ x == x*w", rmat)
    check("as_matrix('left') @ x == w*x", lmat)


# ---------------------------------------------------------------- section 2


def section_2_alternative_moufang() -> None:
    header("2. Alternativity, Artin, Moufang")
    la, ra, fl, artin, inv1, inv2 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    m = [0.0, 0.0, 0.0, 0.0]
    for _ in range(1000):
        x, y, z = rand_oct(), rand_oct(), rand_oct()
        la = max(la, dev(x * (x * y), (x * x) * y))
        ra = max(ra, dev((y * x) * x, y * (x * x)))
        fl = max(fl, dev(x * (y * x), (x * y) * x))
        # Artin/diassociativity: all bracketings of the word xyxy agree
        words = [
            ((x * y) * x) * y,
            (x * (y * x)) * y,
            x * ((y * x) * y),
            x * (y * (x * y)),
            (x * y) * (x * y),
        ]
        for wrd in words[1:]:
            artin = max(artin, dev(words[0], wrd))
        # Moufang identities
        m[0] = max(m[0], dev(z * (x * (z * y)), ((z * x) * z) * y))  # left
        m[1] = max(m[1], dev(x * (z * (y * z)), ((x * z) * y) * z))  # right
        m[2] = max(m[2], dev((z * x) * (y * z), (z * (x * y)) * z))  # middle
        m[3] = max(m[3], dev((z * x) * (y * z), z * ((x * y) * z)))  # middle'
        # exact inversion by conjugate for unit w
        w = rand_oct(unit=True)
        inv1 = max(inv1, dev((x * w) * w.conjugate(), x))
        inv2 = max(inv2, dev(w.conjugate() * (w * x), x))
    check("left alternative x(xy) = (xx)y", la)
    check("right alternative (yx)x = y(xx)", ra)
    check("flexible x(yx) = (xy)x", fl)
    check("Artin: 5 bracketings of xyxy agree", artin)
    check("Moufang left  z(x(zy)) = ((zx)z)y", m[0])
    check("Moufang right x(z(yz)) = ((xz)y)z", m[1])
    check("Moufang middle (zx)(yz) = (z(xy))z", m[2])
    check("Moufang middle (zx)(yz) = z((xy)z)", m[3])
    check("exact layer inversion (xw)w^bar = x (unit w)", inv1)
    check("exact layer inversion w^bar(wx) = x (unit w)", inv2)


# ---------------------------------------------------------------- section 3


def section_3_transport() -> None:
    header("3. Conjugation transport (the error wave's sandwich)")
    dia, rt = 0.0, 0.0
    for _ in range(1000):
        w, r = rand_oct(unit=True), rand_oct()
        dia = max(dia, dev((w.conjugate() * r) * w, w.conjugate() * (r * w)))
        t = (w.conjugate() * r) * w
        rt = max(rt, dev((w * t) * w.conjugate(), r))
    check("diassociativity: (w^bar r)w == w^bar(rw)  [transport unambiguous]", dia)
    check("exact roundtrip w(w^bar r w)w^bar == r", rt)

    # Transport is NOT an algebra automorphism (expected O(1) failure).
    devs = []
    for _ in range(300):
        w = rand_oct(unit=True)
        x, y = rand_oct(unit=True), rand_oct(unit=True)
        phi = lambda t, w=w: (w.conjugate() * t) * w  # noqa: E731
        devs.append(dev(phi(x * y), phi(x) * phi(y)))
    info("conjugation transport as automorphism: |phi(xy)-phi(x)phi(y)|, median/max", f"{np.median(devs):.3f} / {np.max(devs):.3f} (expected nonzero)")

    # Two-layer transport does not collapse to transport by any single product.
    devs12, devs21 = [], []
    for _ in range(300):
        w1, w2, r = rand_oct(unit=True), rand_oct(unit=True), rand_oct()
        two = (w2.conjugate() * ((w1.conjugate() * r) * w1)) * w2
        for prod, out in ((w1 * w2, devs12), (w2 * w1, devs21)):
            out.append(dev(two, (prod.conjugate() * r) * prod))
    info("two-layer sandwich vs sandwich by w1*w2: min/median dev", f"{np.min(devs12):.3f} / {np.median(devs12):.3f} (expected nonzero)")
    info("two-layer sandwich vs sandwich by w2*w1: min/median dev", f"{np.min(devs21):.3f} / {np.median(devs21):.3f} (expected nonzero)")

    # Known special case: conjugation by a IS an automorphism when a^6 is real.
    t = np.zeros(8)
    t[1] = np.pi / 3
    a = Octonion(t).exp()  # cos60 + sin60*e1; a^6 = 1
    aut = 0.0
    for _ in range(300):
        x, y = rand_oct(), rand_oct()
        phi = lambda s, a=a: (a * s) * a.conjugate()  # noqa: E731
        aut = max(aut, dev(phi(x * y), phi(x) * phi(y)))
    check("conj by a = exp(pi/3 e1) (a^6 real) IS an automorphism", aut)
    t2 = np.zeros(8)
    t2[1] = 0.4
    b = Octonion(t2).exp()
    devs_b = []
    for _ in range(100):
        x, y = rand_oct(unit=True), rand_oct(unit=True)
        phi = lambda s, b=b: (b * s) * b.conjugate()  # noqa: E731
        devs_b.append(dev(phi(x * y), phi(x) * phi(y)))
    info("conj by exp(0.4 e1) (a^6 not real): max automorphism dev", f"{np.max(devs_b):.3f} (expected nonzero)")

    # G2-equivariance of associator and cross product under the automorphism a.
    eq_as, eq_cr = 0.0, 0.0
    for _ in range(300):
        x, y, z = rand_oct(), rand_oct(), rand_oct()
        phi = lambda s, a=a: (a * s) * a.conjugate()  # noqa: E731
        eq_as = max(eq_as, dev(phi(assoc_std(x, y, z)), assoc_std(phi(x), phi(y), phi(z))))
        u, v = rand_imag(), rand_imag()
        eq_cr = max(
            eq_cr,
            float(np.max(np.abs(arr(phi(Octonion(np.concatenate([[0.0], cross_alg7(u.im, v.im)]))))[1:] - cross_alg7(arr(phi(u))[1:], arr(phi(v))[1:])))),
        )
    check("automorphism equivariance of the associator", eq_as)
    check("automorphism equivariance of the algebra cross product Im(uv)", eq_cr)

    # Pseudo-automorphism companion search (Moufang-loop folklore): does some
    # power c of w satisfy phi(xy)*c == phi(x)*(phi(y)*c) for phi = conj by w?
    w = rand_oct(unit=True)
    phi = lambda s, w=w: (w.conjugate() * s) * w  # noqa: E731
    candidates = {
        "w": w,
        "w^2": w * w,
        "w^3": (w * w) * w,
        "conj(w)": w.conjugate(),
        "conj(w)^2": w.conjugate() * w.conjugate(),
        "conj(w)^3": (w.conjugate() * w.conjugate()) * w.conjugate(),
    }
    found = []
    for name, c in candidates.items():
        d = 0.0
        for _ in range(50):
            x, y = rand_oct(unit=True), rand_oct(unit=True)
            d = max(d, dev(phi(x * y) * c, phi(x) * (phi(y) * c)))
        if d <= 1e-9:
            found.append(name)
    info("pseudo-automorphism companion for conj-by-w among powers of w", found if found else "none found among w^{1,2,3}, conj(w)^{1,2,3}")


# ---------------------------------------------------------------- section 4


def section_4_associator_structure() -> None:
    header("4. Associator structure")
    perms = [
        ((0, 1, 2), +1), ((1, 2, 0), +1), ((2, 0, 1), +1),
        ((1, 0, 2), -1), ((0, 2, 1), -1), ((2, 1, 0), -1),
    ]
    alt, tri, re_a, orth, real_arg, imdep, sign_conv = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(500):
        t = [rand_oct(), rand_oct(), rand_oct()]
        a0 = assoc_std(*t)
        for p, s in perms:
            alt = max(alt, dev(assoc_std(t[p[0]], t[p[1]], t[p[2]]), s * a0))
        # trilinearity in slot 1
        x2, al, be = rand_oct(), rng.standard_normal(), rng.standard_normal()
        tri = max(tri, dev(assoc_std(al * t[0] + be * x2, t[1], t[2]), al * a0 + be * assoc_std(x2, t[1], t[2])))
        re_a = max(re_a, abs(a0.re))
        orth = max(orth, max(abs(dot(a0, t[0])), abs(dot(a0, t[1])), abs(dot(a0, t[2]))))
        # any real argument kills it
        real_arg = max(real_arg, abs(assoc_std(Octonion(np.array([rng.standard_normal(), 0, 0, 0, 0, 0, 0, 0])), t[1], t[2])))
        # depends only on imaginary parts
        ims = [Octonion(np.concatenate([[0.0], o.im])) for o in t]
        imdep = max(imdep, dev(a0, assoc_std(*ims)))
        sign_conv = max(sign_conv, dev(assoc_repo(*t), -1.0 * a0))
    check("alternating under all 6 permutations (sign = parity)", alt)
    check("trilinear (checked in slot 1; slots 2,3 follow by alternation)", tri)
    check("Re [x,y,z] = 0 (associator is purely imaginary)", re_a)
    check("[x,y,z] orthogonal to each of x, y, z", orth)
    check("vanishes when an argument is real", real_arg)
    check("[x,y,z] = [Im x, Im y, Im z]", imdep)
    check("algebra.associator == -[x,y,z] (opposite sign convention)", sign_conv)

    # vanishes identically on any quaternion subalgebra Q(a,b) = span{1,a,b,ab}
    qsub = 0.0
    for _ in range(200):
        a = rand_imag()
        b0 = rand_imag()
        b = Octonion(arr(b0) - dot(b0, a) * arr(a)).normalize()
        ab = a * b
        span = [Octonion.unit(), a, b, ab]
        elems = []
        for _ in range(3):
            co = rng.standard_normal(4)
            elems.append(Octonion(sum(c * arr(s) for c, s in zip(co, span))))
        qsub = max(qsub, abs(assoc_std(*elems)))
    check("vanishes on random quaternion subalgebras span{1,a,b,ab}", qsub, tol=1e-9)

    # imaginary-argument formula (own derivation):
    # [u,v,w] = (u x v) x w - u x (v x w) + <v,w>u - <u,v>w
    formula = 0.0
    for _ in range(300):
        u, v, w = rand_imag(), rand_imag(), rand_imag()
        lhs = assoc_std(u, v, w).im
        rhs = (
            cross_alg7(cross_alg7(u.im, v.im), w.im)
            - cross_alg7(u.im, cross_alg7(v.im, w.im))
            + dot(v, w) * u.im
            - dot(u, v) * w.im
        )
        formula = max(formula, float(np.max(np.abs(lhs - rhs))))
    check("imaginary-args formula [u,v,w] = (uxv)xw - ux(vxw) + <v,w>u - <u,v>w", formula)


# ---------------------------------------------------------------- section 5


def section_5_associator_bounds() -> None:
    header("5. Associator norm bound, kappa geometry, rank")
    n = 50_000
    best, best_t = 0.0, None
    best_im = 0.0
    for i in range(n):
        if i % 2 == 0:
            t = (rand_oct(unit=True), rand_oct(unit=True), rand_oct(unit=True))
        else:
            t = (rand_imag(), rand_imag(), rand_imag())
        a = abs(assoc_std(*t))
        if i % 2 == 1:
            best_im = max(best_im, a)
        if a > best:
            best, best_t = a, t
    info(f"empirical max |[x,y,z]| over {n} random unit triples", f"{best:.4f}")
    info("empirical max over the imaginary-only half of the samples", f"{best_im:.4f}")
    e1, e2, e4 = basis(1), basis(2), basis(4)
    info("|[e1,e2,e4]| (basis triple off any Fano line)", f"{abs(assoc_std(e1, e2, e4)):.15f}  (theoretical sharp bound = 2)")
    # hill climb toward the bound
    t = list(best_t)
    val = best
    sigma = 0.3
    for _ in range(4000):
        i = rng.integers(3)
        cand = list(t)
        cand[i] = Octonion(arr(t[i]) + sigma * rng.standard_normal(8)).normalize()
        v = abs(assoc_std(*cand))
        if v > val:
            t, val = cand, v
        sigma *= 0.999
    info("hill-climb from best sample (4000 steps)", f"{val:.6f}  (bound 2)")

    # kappa geometry: for unit q,w,r both products are unit and
    # |assoc| = 2 sin(theta/2), theta = angle between (qw)r and q(wr).
    chord, unit_prod = 0.0, 0.0
    for _ in range(500):
        q, w, r = rand_oct(unit=True), rand_oct(unit=True), rand_oct(unit=True)
        p, s = (q * w) * r, q * (w * r)
        unit_prod = max(unit_prod, abs(abs(p) - 1.0), abs(abs(s) - 1.0))
        theta = np.arccos(np.clip(dot(p, s), -1.0, 1.0))
        chord = max(chord, abs(abs(p - s) - 2.0 * np.sin(theta / 2.0)))
    check("|(qw)r| = |q(wr)| = 1 for unit q,w,r", unit_prod)
    check("|[q,w,r]| = 2 sin(theta/2) (chord formula; kappa = 1 - clip(2sin(theta/2)))", chord)

    # x -> [x,a,b] for orthonormal imaginary a,b: rank-4, all nonzero
    # singular values equal to 2 (2 x isometry on the Q(a,b)-complement).
    a = rand_imag()
    b0 = rand_imag()
    b = Octonion(arr(b0) - dot(b0, a) * arr(a)).normalize()
    mat = np.column_stack([arr(assoc_std(basis(i), a, b)) for i in range(8)])
    sv = np.linalg.svd(mat, compute_uv=False)
    info("singular values of x -> [x,a,b], orthonormal imaginary a,b", np.round(sv, 12))
    aa, bb = rand_oct(unit=True), rand_oct(unit=True)
    mat2 = np.column_stack([arr(assoc_std(basis(i), aa, bb)) for i in range(8)])
    info("singular values for generic unit a,b", np.round(np.linalg.svd(mat2, compute_uv=False), 4))

    # the quadratic associator feature: f(x) = [x w1, c, x w2]
    w1, w2, c = rand_oct(unit=True), rand_oct(unit=True), rand_oct(unit=True)
    f = lambda x: assoc_std(x * w1, c, x * w2)  # noqa: E731
    x, xp = rand_oct(unit=True), rand_oct(unit=True)
    check("f(x) = [x w1, c, x w2] is degree-2 homogeneous: f(2x) = 4 f(x)", dev(f(2.0 * x), 4.0 * f(x)))
    info("f is NOT additive: |f(x+x') - f(x) - f(x')|", f"{abs(f(x + xp) - f(x) - f(xp)):.3f} (expected nonzero)")
    info("|f(1)| = |[w1,c,w2]| (nonzero for generic weights)", f"{abs(f(Octonion.unit())):.3f}")
    # alternativity degeneracy: single-slot self-insertions vanish identically
    deg = max(abs(assoc_std(x, w1, x)), abs(assoc_std(x, x, w1)), abs(assoc_std(x, w1, x.conjugate())))
    check("degeneracy: [x,w,x] = [x,x,w] = [x,w,conj(x)] = 0 (alternativity)", deg)


# ---------------------------------------------------------------- section 6


def section_6_cross_product() -> None:
    header("6. 7D cross product: repo implementation vs the algebra's own")

    def cross_repo(u7, v7):
        return cross_product_7d(u7, v7)

    def table_of(fn):
        t = {}
        for i in range(7):
            for j in range(7):
                if i == j:
                    continue
                u = np.zeros(7)
                u[i] = 1.0
                v = np.zeros(7)
                v[j] = 1.0
                w = fn(u, v)
                k = int(np.argmax(np.abs(w)))
                t[(i + 1, j + 1)] = (k + 1, int(np.sign(w[k]))) if np.max(np.abs(w)) > 0.5 else None
        return t

    t_repo, t_alg = table_of(cross_repo), table_of(cross_alg7)

    def lines_of(t):
        return sorted(tuple(sorted((i, j, t[(i, j)][0]))) for (i, j) in t if i < j)

    lines_repo = sorted(set(lines_of(t_repo)))
    lines_alg = sorted(set(lines_of(t_alg)))
    info("Fano lines of Octonion.__mul__ (Cayley-Dickson)", lines_alg)
    info("Fano lines of cross_product_7d", lines_repo)
    swapped = sorted(set(tuple(sorted(7 if x == 5 else 5 if x == 7 else x for x in ln)) for ln in lines_alg))
    info("repo line set == algebra line set with labels e5<->e7 swapped?", swapped == lines_repo)
    mismatch = sum(1 for k in t_repo if t_repo[k] != t_alg[k])
    info("ordered basis pairs (i,j) where cross_product_7d != Im(e_i e_j)", f"{mismatch} of 42")

    anti, ortho, norm_id, vs_alg, vs_comm = 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(2000):
        u7, v7 = rng.standard_normal(7), rng.standard_normal(7)
        c = cross_repo(u7, v7)
        anti = max(anti, float(np.max(np.abs(c + cross_repo(v7, u7)))))
        ortho = max(ortho, abs(float(np.dot(c, u7))), abs(float(np.dot(c, v7))))
        norm_id = max(norm_id, abs(float(np.dot(c, c)) - (np.dot(u7, u7) * np.dot(v7, v7) - np.dot(u7, v7) ** 2)))
        vs_alg = max(vs_alg, float(np.max(np.abs(c - cross_alg7(u7, v7)))))
        uo = Octonion(np.concatenate([[0.0], u7]))
        vo = Octonion(np.concatenate([[0.0], v7]))
        vs_comm = max(vs_comm, float(np.max(np.abs(c - 0.5 * arr(commutator(uo, vo))[1:]))))
    check("repo cross: antisymmetry u x v = -(v x u)", anti)
    ok_o = check("repo cross: orthogonality <u x v, u> = <u x v, v> = 0", ortho)
    ok_n = check("repo cross: norm identity |u x v|^2 = |u|^2|v|^2 - <u,v>^2", norm_id)
    ok_a = check("repo cross == Im(uv) of the implemented algebra", vs_alg)
    ok_c = check("repo cross == (1/2) Im(uv - vu)  ('commutator alignment')", vs_comm)
    if not (ok_o and ok_a and ok_c and ok_n):
        print("  [BUG?] cross_product_7d is NOT the cross product of the implemented algebra.")
        u7 = np.zeros(7)
        u7[0] = 1.0
        u7[3] = 1.0  # u = e1 + e4
        v7 = np.zeros(7)
        v7[6] = 1.0  # v = e7
        c = cross_repo(u7, v7)
        print(f"  [BUG?] counterexample u=e1+e4, v=e7: u x v = {c}, <u x v, u> = {float(np.dot(c, u7)):.1f} (should be 0)")

    # Properties of the algebra's own cross product (the correct one).
    ortho2, norm2, uud = 0.0, 0.0, 0.0
    jac_alg, jac_repo, jac_assoc, tpe = 0.0, 0.0, 0.0, []
    for _ in range(1000):
        u7, v7, w7 = rng.standard_normal(7), rng.standard_normal(7), rng.standard_normal(7)
        c = cross_alg7(u7, v7)
        ortho2 = max(ortho2, abs(float(np.dot(c, u7))), abs(float(np.dot(c, v7))))
        norm2 = max(norm2, abs(float(np.dot(c, c)) - (np.dot(u7, u7) * np.dot(v7, v7) - np.dot(u7, v7) ** 2)))
        uud = max(uud, float(np.max(np.abs(cross_alg7(u7, cross_alg7(u7, v7)) - (np.dot(u7, v7) * u7 - np.dot(u7, u7) * v7)))))

        def jac(fn, u=u7, v=v7, w=w7):
            return fn(u, fn(v, w)) + fn(v, fn(w, u)) + fn(w, fn(u, v))

        j = jac(cross_alg7)
        jac_alg = max(jac_alg, float(np.linalg.norm(j)))
        jac_repo = max(jac_repo, float(np.linalg.norm(jac(cross_repo))))
        uo = Octonion(np.concatenate([[0.0], u7]))
        vo = Octonion(np.concatenate([[0.0], v7]))
        wo = Octonion(np.concatenate([[0.0], w7]))
        jac_assoc = max(jac_assoc, float(np.max(np.abs(j + 1.5 * assoc_std(uo, vo, wo).im))))
        tpe.append(float(np.linalg.norm(cross_alg7(u7, cross_alg7(v7, w7)) - (np.dot(u7, w7) * v7 - np.dot(u7, v7) * w7))))
    check("algebra cross: orthogonality to both arguments", ortho2)
    check("algebra cross: norm identity", norm2)
    check("algebra cross: u x (u x v) = <u,v>u - |u|^2 v (survives from 3D)", uud)
    info("algebra cross: max |Jacobiator| (Jacobi FAILS in 7D)", f"{jac_alg:.3f} (expected nonzero)")
    check("algebra cross: Jacobiator == -(3/2) [u,v,w] (own derivation)", jac_assoc)
    info("repo cross: max |Jacobiator|", f"{jac_repo:.3f}")
    info("algebra cross: triple-product expansion u x (v x w) = <u,w>v - <u,v>w FAILS, median |dev|", f"{np.median(tpe):.3f}")


# ---------------------------------------------------------------- section 7


def section_7_so8_reach() -> None:
    header("7. Right multiplications generate so(8) (SO(8) reach)")
    mats = [basis(i).as_matrix("right") for i in range(1, 8)]
    skew = max(float(np.max(np.abs(m + m.T))) for m in mats)
    check("R_{e_i} are skew-symmetric", skew)
    span = [m.ravel() for m in mats]
    for i in range(7):
        for j in range(i + 1, 7):
            span.append((mats[i] @ mats[j] - mats[j] @ mats[i]).ravel())
    rank = int(np.linalg.matrix_rank(np.array(span), tol=1e-10))
    info("rank of span{R_{e_i}} + first brackets (dim so(8) = 28)", rank)


# ---------------------------------------------------------------- section 8


def section_8_cayley_dickson() -> None:
    header("8. Cayley-Dickson doubling: octonions check out, sedenions break")

    def cd_conj(x):
        out = -x.copy()
        out[0] = x[0]
        return out

    def cd_mul(x, y):
        n = x.size
        if n == 1:
            return x * y
        h = n // 2
        a, b = x[:h], x[h:]
        c, d = y[:h], y[h:]
        return np.concatenate([
            cd_mul(a, c) - cd_mul(cd_conj(d), b),
            cd_mul(d, a) + cd_mul(b, cd_conj(c)),
        ])

    agree = 0.0
    for _ in range(200):
        x, y = rng.standard_normal(8), rng.standard_normal(8)
        agree = max(agree, float(np.max(np.abs(cd_mul(x, y) - arr(Octonion(x.copy()) * Octonion(y.copy()))))))
    check("generic CD doubling reproduces Octonion.__mul__ at dim 8", agree)

    # dim 16: norm multiplicativity fails; explicit basis zero divisors exist.
    ratios = []
    for _ in range(2000):
        x, y = rng.standard_normal(16), rng.standard_normal(16)
        ratios.append(float(np.linalg.norm(cd_mul(x, y))) / (np.linalg.norm(x) * np.linalg.norm(y)))
    info("sedenions: |xy| / (|x||y|) over 2000 random pairs, min/max", f"{np.min(ratios):.4f} / {np.max(ratios):.4f} (octonions: identically 1)")

    zeros = []
    for i in range(1, 16):
        for j in range(i + 1, 16):
            for k in range(1, 16):
                for l in range(k + 1, 16):  # noqa: E741
                    x = np.zeros(16)
                    x[i] = 1.0
                    x[j] = 1.0
                    y = np.zeros(16)
                    y[k] = 1.0
                    y[l] = -1.0
                    if np.linalg.norm(cd_mul(x, y)) < 1e-12:
                        zeros.append(f"(e{i}+e{j})(e{k}-e{l})")
    info("sedenion basis zero divisors (e_i+e_j)(e_k-e_l) = 0 found", f"{len(zeros)}; first three: {zeros[:3]}")


# ---------------------------------------------------------------- main


def main() -> None:
    section_1_composition()
    section_2_alternative_moufang()
    section_3_transport()
    section_4_associator_structure()
    section_5_associator_bounds()
    section_6_cross_product()
    section_7_so8_reach()
    section_8_cayley_dickson()
    print("\nDone.")


if __name__ == "__main__":
    main()
