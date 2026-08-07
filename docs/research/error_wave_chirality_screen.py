"""Numerical companion to error-wave-chirality.md (wayfinder ticket #13).

Run:
    uv run python docs/research/error_wave_chirality_screen.py

Deterministic (fixed seeds throughout); runtime about half a minute. Four parts,
each mapping onto one of the ticket's numbered investigation areas.

  A. Operator identities. The [L_a,R_b]x = a(xb)-(ax)b = -[a,x,b] core fact;
     the update-side GAUGE theorem (single-weight left/right bite is a pure
     reparametrization -- Artin, exact in BOTH algebras); the transport-side
     flip (w-bar r w vs w r w-bar) is commutator-sized and NONzero on H, i.e.
     NOT the associator; the output-rotation gap (right-biting the weight vs
     right-rotating the output) IS the pure associator and vanishes on H;
     kappa's achirality. All checked against the project's Octonion class.
  B. Chain credit assignment. The slot-shift theorem (right-update of layer k
     equals left-update of layer k+1, exact on H, associator gap on O) and the
     readout-weight motion (isometry; endpoint misdirection on O).
  C. Training screen. right-only / left-only / alternating / forward-right-
     correct-left variants of the forward error wave, single unit and
     depth-2/3 chains, binary-1d and XOR (seed 42, 800/200, noise 0.1), model
     seeds 0..4, 10 epochs, octonion and quaternion-control passes. The
     right-only path is preflight-verified against OctonionPerceptron /
     OctonionSequential to machine precision, and a gauge-rule check shows the
     properly transported left-bite reproduces right to machine precision.
  D. Race-set chirality. The natural correction side for each #6 racer
     (kappa-slerp, branch products, triple cross product, commutator rotor).

The quaternion control restricts weights (hence every product) to the
subalgebra span{1,e1,e2,e3}; the embedded data already lies in span{1,e1,e2}.
Every product, exp, log, cross product is the project algebra's own; the
batched ops of Part C are preflight-checked against the Octonion class.
"""

from __future__ import annotations

import numpy as np

from v3i.algebra import Octonion, slerp
from v3i.make_data import (
    generate_binary_1d,
    generate_binary_xor,
    to_s7_from_1d,
    to_s7_from_2d,
)
from v3i.models.perceptron.octonion import OctonionPerceptron, OctonionSequential

E0 = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
CONJ = np.array([1, -1, -1, -1, -1, -1, -1, -1], dtype=np.float64)

# Octonion structure tensor, taken from the Octonion class (single source of
# truth): T[i,j,:] = e_i * e_j. Used only for the batched ops in Part C.
T = np.zeros((8, 8, 8))
for _i in range(8):
    for _j in range(8):
        _a = np.zeros(8)
        _b = np.zeros(8)
        _a[_i] = 1.0
        _b[_j] = 1.0
        T[_i, _j, :] = (Octonion(_a) * Octonion(_b)).to_array()
T2 = T.reshape(8, 64)


# ---------------------------------------------------------------------------
# Batched octonion ops (Part C). Each mirrors the Octonion method it names,
# branch for branch; preflight() checks them against the class.
# ---------------------------------------------------------------------------
def bmul(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Rowwise product x*y of (n,8) arrays; x is the left factor."""
    lhs = (x @ T2).reshape(-1, 8, 8)
    return np.einsum("nj,njk->nk", y, lhs)


def bconj(x: np.ndarray) -> np.ndarray:
    return x * CONJ


def bnormalize(x: np.ndarray) -> np.ndarray:
    """Rowwise normalize; near-zero rows -> e0 (as Octonion.normalize)."""
    n = np.linalg.norm(x, axis=1, keepdims=True)
    out = x / np.where(n < 1e-15, 1.0, n)
    out[n[:, 0] < 1e-15] = E0
    return out


def bexp(x: np.ndarray) -> np.ndarray:
    """Rowwise octonion exp (mirrors Octonion.exp incl. the sinc branch)."""
    a = x[:, 0]
    v = x[:, 1:]
    vn = np.linalg.norm(v, axis=1)
    small = vn < 1e-8
    safe = np.where(small, 1.0, vn)
    sinc = np.where(small, 1.0 - vn**2 / 6.0 + vn**4 / 120.0, np.sin(safe) / safe)
    out = np.empty_like(x)
    out[:, 0] = np.cos(vn)
    out[:, 1:] = v * sinc[:, None]
    return out * np.exp(a)[:, None]


def blog(x: np.ndarray) -> np.ndarray:
    """Rowwise octonion log (mirrors Octonion.log / _safe_arctan2_scale)."""
    n = np.linalg.norm(x, axis=1)
    re = x[:, 0]
    v = x[:, 1:]
    vn = np.linalg.norm(v, axis=1)
    big = vn >= 1e-8
    safe_vn = np.where(big, vn, 1.0)
    re_safe = np.where(np.abs(re) < 1e-15, 1.0, re)
    s_big = np.arctan2(vn, re) / safe_vn
    s_small = np.where(np.abs(re) < 1e-15, 0.0, 1.0 / re_safe - vn**2 / (3.0 * re_safe**3))
    scale = np.where(big, s_big, s_small)
    out = np.empty_like(x)
    out[:, 0] = np.log(np.where(n < 1e-15, 1.0, n))
    out[:, 1:] = v * scale[:, None]
    return out


def bcross_embed(w: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Embedded 7D cross product Im(o_Im(w) . o_Im(r)) as a pure-imaginary (n,8).

    Equals cross_product_7d(w.im, r.im) placed in slots 1..7 -- the #11-fixed
    torque, derived from the algebra's own multiplication.
    """
    u = np.zeros_like(w)
    u[:, 1:] = w[:, 1:]
    v = np.zeros_like(r)
    v[:, 1:] = r[:, 1:]
    tau = bmul(u, v)
    tau[:, 0] = 0.0
    return tau


# ---------------------------------------------------------------------------
# Class-based sampling helpers (Parts A/B/D)
# ---------------------------------------------------------------------------
def runit(rng: np.random.Generator, quat: bool = False) -> Octonion:
    a = rng.normal(size=8)
    if quat:
        a[4:] = 0.0
    return Octonion(a).normalize()


def rimag(rng: np.random.Generator, scale: float = 1.0, quat: bool = False) -> Octonion:
    v = rng.normal(size=8)
    v[0] = 0.0
    if quat:
        v[4:] = 0.0
    v = v / np.linalg.norm(v) * scale
    return Octonion(v)


def sec(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def stat_line(name: str, vals: list[float]) -> None:
    a = np.asarray(vals)
    print(f"  {name:<58s} median {np.median(a):.4f}  max {a.max():.4f}")


def exact_line(name: str, vals: list[float]) -> None:
    a = np.asarray(vals)
    print(f"  {name:<58s} max dev {a.max():.1e}")


# ---------------------------------------------------------------------------
# PART A: operator identities
# ---------------------------------------------------------------------------
def part_a(n: int = 3000) -> None:
    sec("PART A: operator identities (area 1) -- checked vs the Octonion class")
    rng = np.random.default_rng(1)

    # A1. Core fact: [L_a,R_b]x = a(xb) - (ax)b = -[a,x,b], and its H-control.
    d_def, d_mat, d_h = [], [], []
    for _ in range(n):
        a, x, b = runit(rng), runit(rng), runit(rng)
        lhs = a * (x * b) - (a * x) * b
        assoc = (a * x) * b - a * (x * b)  # standard [a,x,b] = (ax)b - a(xb)
        d_def.append(abs(lhs + assoc))
        la = a.as_matrix("left")
        rb = b.as_matrix("right")
        comm = (la @ rb - rb @ la) @ x.to_array()
        d_mat.append(float(np.linalg.norm(comm + assoc.to_array())))
        aq, xq, bq = runit(rng, True), runit(rng, True), runit(rng, True)
        d_h.append(abs(aq * (xq * bq) - (aq * xq) * bq))
    exact_line("[A1] a(xb)-(ax)b = -[a,x,b]  (definition)", d_def)
    exact_line("[A1] matrix [L_a,R_b]x = -[a,x,b]  (as_matrix)", d_mat)
    exact_line("[A1] H-control: a(xb)-(ax)b on quaternions (=0)", d_h)

    # A2. Size of the associator payload for random unit triples.
    g = [abs((runit(rng) * runit(rng)) * runit(rng) - runit(rng) * (runit(rng) * runit(rng)))
         for _ in range(0)]
    g = []
    for _ in range(n):
        a, x, b = runit(rng), runit(rng), runit(rng)
        g.append(abs(a * (x * b) - (a * x) * b))
    stat_line("[A2] |a(xb)-(ax)b|, random unit triples on O", g)

    # A3. Update-side GAUGE theorem: w.exp(t) = exp(w t w-bar).w, exact in both
    # algebras (Artin, two generators). The single-weight bite side is a pure
    # reparametrization, not a degree of freedom.
    d_art, dO, dH = [], [], []
    for _ in range(n):
        w = runit(rng)
        tau = rimag(rng, 0.3)
        d_art.append(abs((w.conjugate() * runit(rng)) * w - w.conjugate() * (runit(rng) * w)))
        dO.append(abs(w * tau.exp() - ((w * tau) * w.conjugate()).exp() * w))
        wq, tq = runit(rng, True), rimag(rng, 0.3, True)
        dH.append(abs(wq * tq.exp() - ((wq * tq) * wq.conjugate()).exp() * wq))
    exact_line("[A3] gauge  w.exp(t) = exp(w t w-bar).w  on O", dO)
    exact_line("[A3] gauge  same identity on H", dH)

    # A4. Transport-side flip is NOT the associator: w-bar r w vs w r w-bar is
    # rotation vs inverse rotation, commutator-sized and NONzero on H.
    d_transp_art, gO, gH = [], [], []
    for _ in range(n):
        w, r = runit(rng), runit(rng)
        d_transp_art.append(abs((w.conjugate() * r) * w - w.conjugate() * (r * w)))
        gO.append(abs((w.conjugate() * r) * w - (w * r) * w.conjugate()))
        wq, rq = runit(rng, True), runit(rng, True)
        gH.append(abs((wq.conjugate() * rq) * wq - (wq * rq) * wq.conjugate()))
    exact_line("[A4] Artin: (w-bar r)w = w-bar(r w)  (transport unambiguous)", d_transp_art)
    stat_line("[A4] |w-bar r w - w r w-bar| on O (transport-flip)", gO)
    stat_line("[A4] |w-bar r w - w r w-bar| on H (NONzero => NOT assoc!)", gH)

    # A5. Output-rotation gap IS the pure associator: right-biting the weight
    # (x(wu)) vs right-rotating the output ((xw)u) differs by -[x,w,u], zero on
    # H. Reported at two step sizes to show the tau-scaling.
    for scale in (0.1, 0.01):
        gR, hR = [], []
        for _ in range(n):
            x, w = runit(rng), runit(rng)
            u = rimag(rng, scale).exp()
            gR.append(abs(x * (w * u) - (x * w) * u))
            xq, wq = runit(rng, True), runit(rng, True)
            uq = rimag(rng, scale, True).exp()
            hR.append(abs(xq * (wq * uq) - (xq * wq) * uq))
        stat_line(f"[A5] |x(wu)-(xw)u| = |[x,w,u]|, |tau|={scale} on O", gR)
        exact_line(f"[A5] H-control at |tau|={scale} (=0)", hR)

    # A6. Kappa is achiral: the associator is alternating, so |[q,w,r]| is
    # invariant under swapping the outer (chirality) slots.
    d = []
    for _ in range(n):
        q, w, r = runit(rng), runit(rng), runit(rng)
        a1 = (q * w) * r - q * (w * r)
        a2 = (r * w) * q - r * (w * q)
        d.append(abs(abs(a1) - abs(a2)))
    exact_line("[A6] | |[q,w,r]| - |[r,w,q]| |  (kappa achirality)", d)


# ---------------------------------------------------------------------------
# PART B: chain credit assignment
# ---------------------------------------------------------------------------
def _nest_right(ws: list[Octonion]) -> Octonion:
    """Right-nested product w0(w1(...wk)) as it appears in the readout weight."""
    out = ws[-1]
    for wi in reversed(ws[:-1]):
        out = wi * out
    return out


def part_b(n: int = 2000, tau: float = 0.1) -> None:
    sec("PART B: chain credit assignment (area 2)")
    rng = np.random.default_rng(2)

    # B1. Slot-shift theorem. In the composed forward output, right-updating
    # layer k by u equals left-updating layer k+1 by u -- exactly on H, with an
    # associator gap on O. This is the chain-level H-control for chirality.
    for depth in (2, 3):
        dH, gO = [], []
        for _ in range(n):
            k = int(rng.integers(0, depth - 1))  # boundary between k and k+1
            wsq = [runit(rng, True) for _ in range(depth)]
            xq = runit(rng, True)
            uq = rimag(rng, tau, True).exp()
            wr = list(wsq)
            wr[k] = wsq[k] * uq  # right-update layer k
            wl = list(wsq)
            wl[k + 1] = uq * wsq[k + 1]  # left-update layer k+1
            dH.append(abs(xq * _nest_right(wr) - xq * _nest_right(wl)))
            wso = [runit(rng) for _ in range(depth)]
            xo = runit(rng)
            uo = rimag(rng, tau).exp()
            wro = list(wso)
            wro[k] = wso[k] * uo
            wlo = list(wso)
            wlo[k + 1] = uo * wso[k + 1]
            gO.append(abs(xo * _nest_right(wro) - xo * _nest_right(wlo)))
        exact_line(f"[B1] depth-{depth}: right@k == left@(k+1) on H", dH)
        stat_line(f"[B1] depth-{depth}: same on O, |tau|={tau} (associator gap)", gO)

    # B2. Readout-weight motion. v = conj(w0(w1(...))). Updating layer k by
    # exp(tau) moves v along a geodesic (isometry: |motion|/|tau| = 1). On H the
    # endpoint is exp(-Ad_A tau).v with A = conj(downstream product); the O
    # endpoint departs from that H-prediction -- the credit misdirection.
    print("  [B2] readout-weight motion  (|motion|/|tau|; endpoint gap to H-pred)")
    for depth in (2, 3):
        for k in range(depth):
            for side in ("R", "L"):
                ratios, gapsO, ratios_h, gaps_h = [], [], [], []
                for _ in range(n // 2):
                    for quat in (False, True):
                        ws = [runit(rng, quat) for _ in range(depth)]
                        t = rimag(rng, tau, quat)
                        u = t.exp()
                        v = _nest_right(ws).conjugate()
                        ws2 = list(ws)
                        ws2[k] = ws[k] * u if side == "R" else u * ws[k]
                        v2 = _nest_right(ws2).conjugate()
                        start = k + 1 if side == "R" else k
                        big = _nest_right(ws[start:]).conjugate() if start < depth \
                            else Octonion(E0.copy())
                        adj = (big * t) * big.conjugate()
                        v2_pred = (-1.0 * adj).exp() * v
                        motion = abs((v.conjugate() * v2).log()) / tau
                        if quat:
                            ratios_h.append(abs(motion - 1.0))
                            gaps_h.append(abs(v2 - v2_pred))
                        else:
                            ratios.append(motion)
                            gapsO.append(abs(v2 - v2_pred))
                print(f"        d{depth} layer {k + 1}/{depth} side {side}: "
                      f"O |motion|/|tau| med {np.median(ratios):.3f} "
                      f"(min {np.min(ratios):.3f} max {np.max(ratios):.3f}); "
                      f"O endpoint gap med {np.median(gapsO):.3f}; "
                      f"H ratio dev {np.max(ratios_h):.0e}, H gap {np.max(gaps_h):.0e}")


# ---------------------------------------------------------------------------
# PART C: training screen
# ---------------------------------------------------------------------------
VARIANTS = ("right", "left", "alt", "fr-cl")
SEEDS = (0, 1, 2, 3, 4)
EPOCHS = 10
LR = 0.1
DATA_SEED = 42
TRAIN_N, TEST_N, NOISE = 800, 200, 0.1


def make_dataset(name: str):
    rng = np.random.default_rng(DATA_SEED)
    if name == "binary-1d":
        return generate_binary_1d(TRAIN_N, TEST_N, NOISE, rng, to_sphere=to_s7_from_1d)
    return generate_binary_xor(TRAIN_N, TEST_N, NOISE, rng, to_sphere=to_s7_from_2d)


def init_weights(seed: int, depth: int, quat: bool) -> np.ndarray:
    """Identity + N(0, 0.05) perturbation per layer (as OctonionPerceptron);
    layer l of model seed s uses rng seed 100*s + l. The quaternion control
    zeroes the e4..e7 perturbation components before normalizing."""
    w = np.zeros((depth, 8))
    for layer in range(depth):
        rng = np.random.default_rng(100 * seed + layer)
        pert = rng.normal(0, 0.05, 8)
        if quat:
            pert[4:] = 0.0
        row = E0 + pert
        w[layer] = row / np.linalg.norm(row)
    return w


def side_flags(variant: int, depth: int, step: int) -> tuple[bool, np.ndarray, np.ndarray]:
    """Per-variant chirality flags: (start_left, transport_left[depth], bite_left[depth]).

      right : all sides right.
      left  : all sides left (full naive mirror of the correction wave).
      alt   : side alternates per layer in a chain, per step for a single unit.
      fr-cl : right transport / left bite (keep the frame, flip only the bite).
    """
    transport_left = np.zeros(depth, dtype=bool)
    bite_left = np.zeros(depth, dtype=bool)
    if variant == 0:  # right
        pass
    elif variant == 1:  # left
        transport_left[:] = True
        bite_left[:] = True
    elif variant == 2:  # alt
        if depth > 1:
            odd = (np.arange(depth) % 2 == 1)
        else:
            odd = np.array([step % 2 == 1])
        transport_left[:] = odd
        bite_left[:] = odd
    else:  # fr-cl
        bite_left[:] = True
    start_left = bool(bite_left[0]) if variant != 3 else False
    if variant == 1:
        start_left = True
    return start_left, transport_left, bite_left


def wave_step(w: np.ndarray, inputs: list[np.ndarray], r: np.ndarray,
              transport_left: np.ndarray, bite_left: np.ndarray) -> tuple[np.ndarray, float]:
    """One correction wave through all layers for a batch of runs. Mutates w in
    place; returns the exiting residual and the max |kappa-1| seen this step."""
    depth = w.shape[1]
    kappa_dev = 0.0
    for layer in range(depth):
        wl = w[:, layer, :]
        wc = bconj(wl)
        tl = transport_left[layer]
        # 1. transport global error into the local weight frame
        r_loc = bmul(bmul(wl, r), wc) if tl else bmul(bmul(wc, r), wl)
        # 2. torque = embedded 7D cross product (fixed #11), same on both sides
        tau = bcross_embed(wl, r_loc)
        # 3. kappa = 1 - clip(|[x_in, w, r_loc]| / (|x_in||w||r_loc|))
        xin = inputs[layer]
        assoc = bmul(bmul(xin, wl), r_loc) - bmul(xin, bmul(wl, r_loc))
        denom = (np.linalg.norm(xin, axis=1) * np.linalg.norm(wl, axis=1)
                 * np.linalg.norm(r_loc, axis=1))
        ratio = np.linalg.norm(assoc, axis=1) / np.where(denom < 1e-15, 1.0, denom)
        kappa = 1.0 - np.clip(ratio, 0.0, 1.0)
        kappa[denom < 1e-15] = 1.0
        kappa_dev = max(kappa_dev, float(np.abs(kappa - 1.0).max()))
        # 4. geodesic bite, right (w.exp) or left (exp.w)
        u = bexp(tau * (LR * kappa)[:, None])
        wl_new = bnormalize(bmul(u, wl) if bite_left[layer] else bmul(wl, u))
        w[:, layer, :] = wl_new
        # 5. debt accounting: subtract the absorbed (torque) component
        tsq = np.sum(tau * tau, axis=1)
        coef = np.where(tsq < 1e-15, 0.0,
                        np.sum(r_loc * tau, axis=1) / np.where(tsq < 1e-15, 1.0, tsq))
        r_res = r_loc - coef[:, None] * tau
        # 6. transport the residual out with the updated weight, same side
        wc2 = bconj(wl_new)
        r = bmul(bmul(wc2, r_res), wl_new) if tl else bmul(bmul(wl_new, r_res), wc2)
    return r, kappa_dev


def eval_runs(w: np.ndarray, x: np.ndarray, y: np.ndarray):
    nruns, depth, _ = w.shape
    m = len(x)
    cur = np.repeat(x[None, :, :], nruns, axis=0).reshape(nruns * m, 8)
    for layer in range(depth):
        cur = bmul(cur, np.repeat(w[:, layer, :], m, axis=0))
    re = cur[:, 0].reshape(nruns, m)
    pred = np.where(re >= 0, 1, -1)
    acc = (pred == y[None, :]).mean(axis=1)
    geo = np.arccos(np.clip(y[None, :] * re, -1.0, 1.0)).mean(axis=1)
    return acc, geo


def train_cell(dataset: str, depth: int, variant: int):
    """All (seed, algebra) runs for one (dataset, depth, variant) cell,
    batched over the 10 runs. Returns per-epoch metric arrays and controls."""
    xtr, ytr, xte, yte = make_dataset(dataset)
    rows = [(s, quat) for s in SEEDS for quat in (False, True)]
    nruns = len(rows)
    quats = np.array([q for _, q in rows])
    w = np.zeros((nruns, depth, 8))
    for i, (s, quat) in enumerate(rows):
        w[i] = init_weights(s, depth, quat)

    off_cols = slice(2, 8) if dataset == "binary-1d" else slice(3, 8)
    train_acc = np.zeros((EPOCHS, nruns))
    test_acc = np.zeros((EPOCHS, nruns))
    geo_tr = np.zeros((EPOCHS, nruns))
    offplane = np.zeros((EPOCHS, nruns))
    w_step1 = None
    slice_dev_h = 0.0

    for ep in range(EPOCHS):
        for i in range(len(xtr)):
            step = ep * len(xtr) + i
            start_left, transport_left, bite_left = side_flags(variant, depth, step)
            cur = np.tile(xtr[i], (nruns, 1))
            inputs = []
            for layer in range(depth):
                inputs.append(cur)
                cur = bmul(cur, w[:, layer, :])
            tgt = np.tile(ytr[i] * E0, (nruns, 1))
            pc = bconj(cur)
            r = blog(bmul(tgt, pc)) if start_left else blog(bmul(pc, tgt))
            wave_step(w, inputs, r, transport_left, bite_left)
            slice_dev_h = max(slice_dev_h, float(np.abs(w[quats][:, :, 4:]).max()))
            if w_step1 is None:
                w_step1 = w.copy()
        acc, geo = eval_runs(w, xtr, ytr)
        tacc, _ = eval_runs(w, xte, yte)
        train_acc[ep] = acc
        test_acc[ep] = tacc
        geo_tr[ep] = geo
        offplane[ep] = np.linalg.norm(w[:, :, off_cols], axis=2).max(axis=1)

    return {
        "quats": quats, "w": w, "w_step1": w_step1,
        "train_acc": train_acc, "test_acc": test_acc, "geo": geo_tr,
        "offplane": offplane, "slice_dev_h": slice_dev_h,
    }


def preflight() -> None:
    sec("PART C preflight: batched ops and the right-only trainer vs the classes")
    rng = np.random.default_rng(3)
    x = rng.normal(size=(500, 8))
    y = rng.normal(size=(500, 8))
    d_mul = max(float(np.linalg.norm((Octonion(a) * Octonion(b)).to_array() - m))
                for a, b, m in zip(x, y, bmul(x, y)))
    d_exp = max(float(np.linalg.norm(Octonion(a).exp().to_array() - e))
                for a, e in zip(x * 0.3, bexp(x * 0.3)))
    xu = x / np.linalg.norm(x, axis=1, keepdims=True)
    d_log = max(float(np.linalg.norm(Octonion(a).log().to_array() - lg))
                for a, lg in zip(xu, blog(xu)))
    print(f"  bmul vs Octonion.__mul__ ................. max dev {d_mul:.1e}")
    print(f"  bexp vs Octonion.exp ..................... max dev {d_exp:.1e}")
    print(f"  blog vs Octonion.log ..................... max dev {d_log:.1e}")

    for depth in (1, 2):
        xtr, ytr, _, _ = make_dataset("binary-1d")
        layers = [OctonionPerceptron(learning_rate=LR, random_seed=layer) for layer in range(depth)]
        model = OctonionSequential(layers)
        w = init_weights(0, depth, False)[None, :, :].copy()
        for layer in range(depth):  # match the class's exact initial weights
            w[0, layer] = model.layers[layer].weight.to_array()
        dev = 0.0
        for i in range(400):
            model.forward(Octonion(xtr[i].copy()))
            model.correct(Octonion(ytr[i] * E0))
            cur = xtr[i][None, :]
            inputs = []
            for layer in range(depth):
                inputs.append(cur)
                cur = bmul(cur, w[:, layer, :])
            r = blog(bmul(bconj(cur), ytr[i] * E0[None, :]))
            wave_step(w, inputs, r, np.zeros(depth, bool), np.zeros(depth, bool))
            for layer in range(depth):
                dev = max(dev, float(np.linalg.norm(
                    w[0, layer] - model.layers[layer].weight.to_array())))
        print(f"  right-only trainer vs OctonionSequential, depth {depth}, "
              f"400 steps: max weight dev {dev:.1e}")


def gauge_check() -> None:
    """Right rule vs a left-bite that transports the torque (exp(w tau w-bar).w).
    By the A3 gauge theorem these are the SAME weight update in both algebras;
    the naive 'left' variant of Part C, which does NOT transport the torque,
    is a different rule -- that contrast is the point. Depth 2, binary-1d."""
    print()
    print("  Gauge check: right vs torque-transported left-bite (should be exact):")
    xtr, ytr, _, _ = make_dataset("binary-1d")
    for quat in (False, True):
        depth = 2
        wr = init_weights(0, depth, quat)[None, :, :].copy()
        wg = wr.copy()
        dev = 0.0
        for _ep in range(2):
            for i in range(len(xtr)):
                # right run
                cur = xtr[i][None, :]
                inputs = []
                for layer in range(depth):
                    inputs.append(cur)
                    cur = bmul(cur, wr[:, layer, :])
                r = blog(bmul(bconj(cur), ytr[i] * E0[None, :]))
                wave_step(wr, inputs, r, np.zeros(depth, bool), np.zeros(depth, bool))
                # gauge run: same wave, but bite LEFT with the transported torque
                # exp(w tau w-bar).w -- by A3 this equals the right bite exactly.
                cur = xtr[i][None, :]
                inputs = []
                for layer in range(depth):
                    inputs.append(cur)
                    cur = bmul(cur, wg[:, layer, :])
                r = blog(bmul(bconj(cur), ytr[i] * E0[None, :]))
                for layer in range(depth):
                    wl = wg[:, layer, :]
                    wc = bconj(wl)
                    r_loc = bmul(bmul(wc, r), wl)
                    tau = bcross_embed(wl, r_loc)
                    xin = inputs[layer]
                    assoc = bmul(bmul(xin, wl), r_loc) - bmul(xin, bmul(wl, r_loc))
                    denom = (np.linalg.norm(xin, axis=1) * np.linalg.norm(wl, axis=1)
                             * np.linalg.norm(r_loc, axis=1))
                    ratio = np.linalg.norm(assoc, axis=1) / np.where(denom < 1e-15, 1.0, denom)
                    kappa = 1.0 - np.clip(ratio, 0.0, 1.0)
                    tau_m = tau * (LR * kappa)[:, None]
                    transported = bmul(bmul(wl, tau_m), wc)  # w tau w-bar
                    wl_new = bnormalize(bmul(bexp(transported), wl))
                    wg[:, layer, :] = wl_new
                    tsq = np.sum(tau * tau, axis=1)
                    coef = np.where(tsq < 1e-15, 0.0,
                                    np.sum(r_loc * tau, axis=1) / np.where(tsq < 1e-15, 1.0, tsq))
                    r_res = r_loc - coef[:, None] * tau
                    wc2 = bconj(wl_new)
                    r = bmul(bmul(wl_new, r_res), wc2)
                dev = max(dev, float(np.abs(wr - wg).max()))
        print(f"    {'H' if quat else 'O'}: max weight deviation over 2 epochs "
              f"(3200 layer-steps): {dev:.1e}")


def part_c() -> None:
    sec("PART C: training screen (area 3) -- 10 epochs, lr 0.1, seeds 0-4")
    print("  variants: right / left / alt / fr-cl; alternation is per-layer in "
          "chains, per-step for the single unit.")
    results = {}
    for dataset in ("binary-1d", "binary-xor"):
        for depth in (1, 2, 3):
            for v in range(4):
                results[(dataset, depth, v)] = train_cell(dataset, depth, v)

    print()
    print("  Final metrics (epoch 10; mean +/- std over seeds 0-4):")
    hdr = (f"  {'dataset':<11s} {'d':>2s} {'variant':<7s} {'alg':<3s} "
           f"{'train':>13s} {'test':>13s} {'geo':>7s} {'off-plane':>10s}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for dataset in ("binary-1d", "binary-xor"):
        for depth in (1, 2, 3):
            for v in range(4):
                rec = results[(dataset, depth, v)]
                for quat in (False, True):
                    m = rec["quats"] == quat
                    tr = rec["train_acc"][-1][m]
                    te = rec["test_acc"][-1][m]
                    ge = rec["geo"][-1][m]
                    op = rec["offplane"][-1][m]
                    print(f"  {dataset:<11s} {depth:>2d} {VARIANTS[v]:<7s} "
                          f"{'H' if quat else 'O':<3s} "
                          f"{tr.mean():.3f} +/- {tr.std():.3f} "
                          f"{te.mean():.3f} +/- {te.std():.3f} "
                          f"{ge.mean():7.4f} {op.mean():10.4f}")

    print()
    print("  Train-accuracy trajectories (octonion pass, mean over seeds):")
    for dataset in ("binary-1d", "binary-xor"):
        for depth in (1, 2, 3):
            for v in range(4):
                rec = results[(dataset, depth, v)]
                tr = rec["train_acc"][:, ~rec["quats"]].mean(axis=1)
                traj = " ".join(f"{a:.3f}" for a in tr)
                print(f"    {dataset:<11s} d{depth} {VARIANTS[v]:<7s} {traj}")

    print()
    print("  Geodesic-loss trajectories (octonion pass, depth 3, mean over seeds):")
    for dataset in ("binary-1d", "binary-xor"):
        for v in range(4):
            rec = results[(dataset, 3, v)]
            ge = rec["geo"][:, ~rec["quats"]].mean(axis=1)
            traj = " ".join(f"{a:.3f}" for a in ge)
            print(f"    {dataset:<11s} d3 {VARIANTS[v]:<7s} {traj}")

    print()
    print("  H-control panel:")
    # kappa deviation restricted to H rows; slice closure over all H rows.
    kd_h = 0.0
    sd = 0.0
    for rec in results.values():
        sd = max(sd, rec["slice_dev_h"])
    for (dataset, depth, v), rec in results.items():
        h = rec["quats"]
        if not h.any():
            continue
        kd_h = max(kd_h, _kappa_dev_h_replay(dataset, depth, v, rec["w"][h]))
    print(f"    max |kappa - 1| on H runs (all cells) .......... {kd_h:.1e}")
    print(f"    max |w[e4..e7]| on H runs (slice closure) ...... {sd:.1e}")

    # right-vs-variant weight gaps on the single unit (binary-1d, seed 0)
    for quat in (False, True):
        alg = "H" if quat else "O"
        base = train_cell_single(quat, "binary-1d", 1, 0)
        for v, name in ((1, "left"), (3, "fr-cl")):
            var = train_cell_single(quat, "binary-1d", 1, v)
            d1 = float(np.linalg.norm(base["w_step1"] - var["w_step1"]))
            df = float(np.linalg.norm(base["w"] - var["w"]))
            print(f"    {alg}: right vs {name:<5s} weight gap  step 1: {d1:.2e}"
                  f"   10 epochs: {df:.2e}")
    gauge_check()


def _kappa_dev_h_replay(dataset: str, depth: int, variant: int, wh: np.ndarray) -> float:
    """Replay 200 forward passes on final H weights; return max |kappa-1| (=0)."""
    xtr, ytr, _, _ = make_dataset(dataset)
    nruns = wh.shape[0]
    dev = 0.0
    for i in range(200):
        start_left, transport_left, bite_left = side_flags(variant, depth, i)
        cur = np.tile(xtr[i], (nruns, 1))
        inputs = []
        for layer in range(depth):
            inputs.append(cur)
            cur = bmul(cur, wh[:, layer, :])
        tgt = np.tile(ytr[i] * E0, (nruns, 1))
        pc = bconj(cur)
        r = blog(bmul(tgt, pc)) if start_left else blog(bmul(pc, tgt))
        _, kdev = wave_step(wh.copy(), inputs, r, transport_left, bite_left)
        dev = max(dev, kdev)
    return dev


def train_cell_single(quat: bool, dataset: str, depth: int, variant: int):
    """One (seed 0, one algebra) run -- for the right-vs-variant weight gaps."""
    xtr, ytr, _, _ = make_dataset(dataset)
    w = init_weights(0, depth, quat)[None, :, :].copy()
    w_step1 = None
    for ep in range(EPOCHS):
        for i in range(len(xtr)):
            step = ep * len(xtr) + i
            start_left, transport_left, bite_left = side_flags(variant, depth, step)
            cur = np.tile(xtr[i], (1, 1))
            inputs = []
            for layer in range(depth):
                inputs.append(cur)
                cur = bmul(cur, w[:, layer, :])
            tgt = np.tile(ytr[i] * E0, (1, 1))
            pc = bconj(cur)
            r = blog(bmul(tgt, pc)) if start_left else blog(bmul(pc, tgt))
            wave_step(w, inputs, r, transport_left, bite_left)
            if w_step1 is None:
                w_step1 = w.copy()
    return {"w": w, "w_step1": w_step1}


# ---------------------------------------------------------------------------
# PART D: race-set chirality
# ---------------------------------------------------------------------------
def part_d(n: int = 1000) -> None:
    sec("PART D: correction chirality for the #6 race set (area 4)")
    rng = np.random.default_rng(4)

    # D1. kappa-slerp. slerp of two branch images is equivariant under ANY
    # right multiplication (an orthogonal map of R^8): the whole combiner
    # commutes with a right-acting correction, in both algebras.
    d = []
    for _ in range(n):
        a, b, q = runit(rng), runit(rng), runit(rng)
        t = float(rng.uniform(0, 1))
        d.append(abs(slerp(a * q, b * q, t) - slerp(a, b, t) * q))
    exact_line("[D1] kappa-slerp: slerp(aq,bq,t) = slerp(a,b,t) q on O", d)

    # D2. Branch product y = (x w1)(x w2). Right-updating the EXTERIOR-right
    # weight w2 right-rotates the output exactly on H (associator gap on O);
    # right-updating the INTERIOR weight w1 does not, even on H.
    g2O, g2H, g1H = [], [], []
    for _ in range(n):
        x, w1, w2 = runit(rng), runit(rng), runit(rng)
        u = rimag(rng, 0.1).exp()
        y = (x * w1) * (x * w2)
        g2O.append(abs((x * w1) * (x * (w2 * u)) - y * u))
        xq, w1q, w2q = runit(rng, True), runit(rng, True), runit(rng, True)
        uq = rimag(rng, 0.1, True).exp()
        yq = (xq * w1q) * (xq * w2q)
        g2H.append(abs((xq * w1q) * (xq * (w2q * uq)) - yq * uq))
        g1H.append(abs((xq * (w1q * uq)) * (xq * w2q) - yq * uq))
    stat_line("[D2] branch: w2(exterior) right-update vs y.u on O", g2O)
    exact_line("[D2] branch: same on H (exterior-right slot is exact)", g2H)
    stat_line("[D2] branch: w1(interior) right-update vs y.u on H", g1H)

    # D3. Commutator rotor core [x w1, x w2]. Two-sided conjugation is an
    # automorphism only on H, so the commutator is sandwich-equivariant there
    # and not on O -- a non-COMMUTATIVITY effect, largely intact under the H
    # ablation.
    dH, gO = [], []
    for _ in range(n):
        aq, bq, gq = runit(rng, True), runit(rng, True), runit(rng, True)
        lh = (gq * (aq * bq - bq * aq)) * gq.conjugate()
        rh = ((gq * aq) * gq.conjugate()) * ((gq * bq) * gq.conjugate()) \
            - ((gq * bq) * gq.conjugate()) * ((gq * aq) * gq.conjugate())
        dH.append(abs(lh - rh))
        a, b, g = runit(rng), runit(rng), runit(rng)
        lo = (g * (a * b - b * a)) * g.conjugate()
        ro = ((g * a) * g.conjugate()) * ((g * b) * g.conjugate()) \
            - ((g * b) * g.conjugate()) * ((g * a) * g.conjugate())
        gO.append(abs(lo - ro))
    exact_line("[D3] commutator sandwich-equivariance on H", dH)
    stat_line("[D3] commutator: same on O (conj not an automorphism)", gO)

    # D4. Triple cross product X3(u,v,w) = (u(v-bar w) - w(v-bar u))/2. Uniformly
    # right-equivariant on H; the gap on O is pure associator -- exactly the
    # clean ablation kappa-slerp and X3 share.
    def x3(u: Octonion, v: Octonion, w: Octonion) -> Octonion:
        return (u * (v.conjugate() * w) - w * (v.conjugate() * u)) * 0.5

    dH, gO = [], []
    for _ in range(n):
        uq, vq, wq, q = (runit(rng, True) for _ in range(4))
        dH.append(abs(x3(uq * q, vq * q, wq * q) - x3(uq, vq, wq) * q))
        u, v, w, qo = (runit(rng) for _ in range(4))
        gO.append(abs(x3(u * qo, v * qo, w * qo) - x3(u, v, w) * qo))
    exact_line("[D4] X3(uq,vq,wq) = X3(u,v,w) q on H", dH)
    stat_line("[D4] X3: same on O (pure associator gap)", gO)


if __name__ == "__main__":
    part_a()
    part_b()
    preflight()
    part_c()
    part_d()
