"""PROTOTYPE — throwaway. Resolves wayfinder issue #9 (findability of the wave).

NOT production code. NOT merged into src/v3i. Answers ONE question: can the
peel-and-solve wide error wave (#8) actually *learn* 3-bit parity (#5), on the
wide layer (#7)? The catalogue already proved the mechanisms can *represent*
parity; this tests whether the gradient-free wave can *find* the weights.

Run:  uv run python prototypes/wide_octonion_parity.py

Architecture (single wide layer, one-sided branches, branch-product combiner):
  branches  b_i = x·w_i           (w_i unit octonion/quaternion)
  output    y = b_1·b_2·…·b_W      (left-fold product; degree-W in x)
  readout   sign(re(y))            (frozen, #5)

Wide error wave (#8), single layer:
  target    y* = label·e0
  error     r = ȳ·y*
  peel      b_i* = L_i⁻¹·y*·R_i⁻¹  (L_i,R_i = left/right sub-products; Jacobi)
  share     s_i = b_i⁻¹·b_i*
  update    w_i ← normalize( w_i · s_i^(η·κ_i) )   (right-chirality; input cancels)
  κ_i       = 1 − clip(|assoc(L_i,b_i,R_i)|)       (per-branch reliability, ≡1 on ℍ)
  residual  ρ = ȳ·y'(new); assert ‖log(ρ̄·r)‖ ≤ ‖log r‖   (recompute-realized)

Controls: quaternion units (assoc inert), W=1 (linearity removed → chance),
renorm-sum combiner (linear aggregation → chance).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from v3i.algebra import Octonion
from v3i.algebra import cross_product_7d
from v3i.make_data import generate_parity

# --- octonion helpers on the real algebra (per-sample; W is small) ---


def oct_pow(s: Octonion, t: float) -> Octonion:
    """Geodesic fraction s^t = exp(t·log s) for a unit octonion."""
    if abs(abs(s) - 1.0) > 1e-9:
        s = s.normalize()
    return (s.log() * t).exp()


def fold_product(bs: list[Octonion]) -> Octonion:
    """Left-fold octonion product b_1·b_2·…·b_W."""
    y = bs[0]
    for b in bs[1:]:
        y = y * b
    return y


def subproducts(bs: list[Octonion], i: int) -> tuple[Octonion, Octonion]:
    """Left sub-product b_1..b_{i-1} and right sub-product b_{i+1}..b_W (unit for i at ends)."""
    left = fold_product(bs[:i]) if i > 0 else Octonion.unit()
    right = fold_product(bs[i + 1 :]) if i < len(bs) - 1 else Octonion.unit()
    return left, right


def associator_mag(a: Octonion, b: Octonion, c: Octonion) -> float:
    """|(ab)c − a(bc)| / (|a||b||c|) ∈ [0, ~2]; 0 on any associative (quaternion) triple."""
    assoc = (a * b) * c - a * (b * c)
    denom = abs(a) * abs(b) * abs(c)
    return 0.0 if denom < 1e-12 else abs(assoc) / denom


class WideLayer:
    """One wide layer: W one-sided branches, branch-product combiner, peel-and-solve wave."""

    def __init__(self, width: int, dim: int, lr: float, rng: np.random.Generator,
                 combiner: str = "product", use_kappa: bool = True, peel: str = "nested",
                 schedule: str = "jacobi") -> None:
        self.W = width
        self.dim = dim
        self.lr = lr
        self.combiner = combiner
        self.use_kappa = use_kappa
        self.peel = peel
        self.schedule = schedule
        self.align_sign = 1.0
        # init: identity + small perturbation, normalized; quaternion => zero the last 4 coords
        self.weights: list[Octonion] = []
        for _ in range(width):
            v = np.zeros(8)
            v[0] = 1.0
            v[: dim] += rng.normal(0, 0.3, dim)
            if dim == 4:
                v[4:] = 0.0
            self.weights.append(Octonion(v).normalize())

    def branches(self, x: Octonion) -> list[Octonion]:
        return [x * w for w in self.weights]

    def forward(self, x: Octonion) -> Octonion:
        bs = self.branches(x)
        if self.combiner == "product":
            return fold_product(bs)
        if self.combiner == "sum":  # renorm-sum control (provably linear)
            acc = bs[0]
            for b in bs[1:]:
                acc = acc + b
            return acc.normalize()
        raise ValueError(self.combiner)

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.empty(len(X))
        for i, row in enumerate(X):
            out[i] = self.forward(Octonion(row.copy())).re
        return np.where(out >= 0, 1, -1)

    def learn(self, x_vec: np.ndarray, label: int) -> float:
        """One act-observe-correct step. Returns residual/incoming norm ratio (should be ≤ 1)."""
        x = Octonion(x_vec.copy())
        bs = self.branches(x)
        y = self.forward(x)
        y_star = Octonion.unit() if label >= 0 else Octonion(-Octonion.unit().to_array())
        r = y.conjugate() * y_star  # rotation: y·r = y*
        incoming = float(np.linalg.norm(r.log().to_array()))
        if self.combiner != "product":  # controls: simple right-nudge toward target on branch 0
            s = y.conjugate() * y_star
            self.weights[0] = (self.weights[0] * oct_pow(s, self.lr)).normalize()
            y2 = self.forward(x)
            rho = y.conjugate() * y2
            return _ratio(rho.conjugate() * r, incoming)
        # HYBRID: exact peel for the outermost branch (it directly controls the
        # output, y = P·b_last), alignment torque for the inner branches.
        if self.peel == "hybrid":
            prefix = [Octonion.unit()]
            for b in bs:
                prefix.append(prefix[-1] * b)  # prefix[k] = b_0·…·b_{k-1}
            delta = (y.conjugate() * y_star).log()
            for i in range(self.W):
                if i == self.W - 1:  # exact update for the outermost branch
                    b_star = prefix[i].inverse() * y_star
                    s = bs[i].inverse() * b_star
                    self.weights[i] = (self.weights[i] * oct_pow(s, self.lr)).normalize()
                else:  # alignment torque for inner branches
                    _left, right = subproducts(bs, i)
                    d_i = right.inverse() * delta * right
                    torque = np.zeros(8)
                    torque[1:] = self.align_sign * cross_product_7d(bs[i].im, d_i.im)
                    self.weights[i] = (self.weights[i] * Octonion(self.lr * torque).exp()).normalize()
            rho = y.conjugate() * self.forward(x)
            return _ratio(rho.conjugate() * r, incoming)
        # ALIGNMENT rule (the project's original mechanism): torque via 7D cross
        # product + kappa, NO exact peel — embraces non-associativity instead of
        # inverting through it. Tests whether this octonion-native rule succeeds
        # where peel-and-solve structurally fails.
        if self.peel == "alignment":
            delta = (y.conjugate() * y_star).log()  # output-error tangent (pure imaginary)
            for i in range(self.W):
                left, right = subproducts(bs, i)
                d_i = right.inverse() * delta * right  # transport error into branch frame
                torque = np.zeros(8)
                torque[1:] = self.align_sign * cross_product_7d(bs[i].im, d_i.im)
                kap = 1.0 - min(associator_mag(left, bs[i], right), 1.0) if self.use_kappa else 1.0
                upd = Octonion(self.lr * kap * torque).exp()
                self.weights[i] = (self.weights[i] * upd).normalize()
            rho = y.conjugate() * self.forward(x)
            return _ratio(rho.conjugate() * r, incoming)
        # Gauss-Seidel: sequential nested peel, fresh neighbours each branch
        if self.schedule == "gauss_seidel" and self.peel == "nested":
            for i in range(self.W):
                cur = self.branches(x)
                prefix_i = fold_product(cur[:i]) if i > 0 else Octonion.unit()
                tgt = y_star
                for k in range(self.W - 1, i, -1):
                    tgt = tgt * cur[k].inverse()
                b_star = prefix_i.inverse() * tgt
                s = cur[i].inverse() * b_star
                self.weights[i] = (self.weights[i] * oct_pow(s, self.lr)).normalize()
            y_new = self.forward(x)
            rho = y.conjugate() * y_new
            return _ratio(rho.conjugate() * r, incoming)
        # peel-and-solve, Jacobi (all shares from current bs)
        shares: list[Octonion] = []
        kappas: list[float] = []
        if self.peel == "nested":
            # EXACT nested peel of the left-fold y = ((b_0·b_1)·b_2)…: unwind the
            # real bracketing. prefix[k] = b_0·…·b_{k-1}; targets T[k] for the
            # partial product P_k, T[W-1]=y*, T[k-1]=T[k]·b_k⁻¹; b_k* = prefix[k]⁻¹·T[k].
            prefix = [Octonion.unit()]
            for b in bs:
                prefix.append(prefix[-1] * b)
            targets: list[Octonion | None] = [None] * self.W
            targets[self.W - 1] = y_star
            for k in range(self.W - 1, 0, -1):
                targets[k - 1] = targets[k] * bs[k].inverse()
            for i in range(self.W):
                b_star = prefix[i].inverse() * targets[i]
                shares.append(bs[i].inverse() * b_star)
                kappas.append(1.0)  # exact peel: no reliability discount needed
        else:  # "assoc": associative-approx peel (the buggy original)
            for i in range(self.W):
                left, right = subproducts(bs, i)
                b_star = left.inverse() * y_star * right.inverse()
                shares.append(bs[i].inverse() * b_star)
                kappas.append(
                    1.0 - min(associator_mag(left, bs[i], right), 1.0) if self.use_kappa else 1.0
                )
        if self.schedule == "adaptive":
            # backtrack η until the residual invariant holds (‖residual‖ ≤ ‖incoming‖)
            saved = [w.copy() for w in self.weights]
            eta = self.lr
            ratio = 2.0
            for _ in range(7):
                for j, w in enumerate(saved):
                    self.weights[j] = w.copy()
                for i in range(self.W):
                    self.weights[i] = (self.weights[i] * oct_pow(shares[i], eta)).normalize()
                rho = y.conjugate() * self.forward(x)
                ratio = _ratio(rho.conjugate() * r, incoming)
                if ratio <= 1.0 + 1e-9:
                    break
                eta *= 0.5
            return ratio
        for i in range(self.W):
            step = self.lr * kappas[i]
            self.weights[i] = (self.weights[i] * oct_pow(shares[i], step)).normalize()
        # recompute-realized residual + invariant
        y_new = self.forward(x)
        rho = y.conjugate() * y_new
        return _ratio(rho.conjugate() * r, incoming)


def _ratio(residual_rot: Octonion, incoming: float) -> float:
    res = float(np.linalg.norm(residual_rot.log().to_array()))
    return res / incoming if incoming > 1e-9 else 0.0


def accuracy(model: WideLayer, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def _pad8(X: np.ndarray) -> np.ndarray:
    """Embed dim-4 (quaternion) rows into 8-vectors; the Octonion class needs 8 coords."""
    if X.shape[1] == 8:
        return X
    out = np.zeros((len(X), 8))
    out[:, : X.shape[1]] = X
    return out


def run(dim: int, width: int, combiner: str, use_kappa: bool, bits: int,
        epochs: int, lr: float, seeds: int, peel: str = "nested", schedule: str = "jacobi") -> dict:
    accs, ratios, wnorms = [], [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        Xtr, ytr, Xte, yte = generate_parity(400, 200, 0.05, np.random.default_rng(1000 + seed),
                                             bits=bits, dim=dim)
        Xtr, Xte = _pad8(Xtr), _pad8(Xte)
        model = WideLayer(width, dim, lr, rng, combiner=combiner, use_kappa=use_kappa, peel=peel,
                          schedule=schedule)
        for _ep in range(epochs):
            order = rng.permutation(len(ytr))
            for idx in order:
                ratios.append(model.learn(Xtr[idx], int(ytr[idx])))
        accs.append(accuracy(model, Xte, yte))
        wnorms.append(max(abs(abs(w) - 1.0) for w in model.weights))
    return {
        "test_accs": accs,
        "median": float(np.median(accs)),
        "invariant_max_ratio": float(np.max(ratios)) if ratios else 0.0,
        "invariant_viol_frac": float(np.mean(np.array(ratios) > 1.0 + 1e-6)) if ratios else 0.0,
        "max_weight_norm_dev": float(np.max(wnorms)),
    }


def geodesic_loss(model: WideLayer, X: np.ndarray, y: np.ndarray) -> float:
    """Mean angle between output and the ±identity target pole (0 = perfect)."""
    res = np.array([model.forward(Octonion(row.copy())).re for row in X])
    return float(np.mean(np.arccos(np.clip(res * y, -1.0, 1.0))))


def emit_run(name: str, dim: int, width: int, combiner: str, bits: int,
             epochs: int, lr: float, seed: int, out_dir: Path) -> str:
    """Train one seed, logging per-epoch metrics + per-branch weight snapshots.

    Writes a run JSON in v3i.run_experiment's format so the existing dashboard
    (uv run streamlit run src/v3i/dashboard.py) renders convergence + weight
    evolution. Each branch is a 'layer' in the weight panel.
    """
    rng = np.random.default_rng(seed)
    Xtr, ytr, Xte, yte = generate_parity(400, 200, 0.05, np.random.default_rng(1000 + seed),
                                         bits=bits, dim=dim)
    Xtr, Xte = _pad8(Xtr), _pad8(Xte)
    model = WideLayer(width, dim, lr, rng, combiner=combiner)
    metrics = []

    def record(epoch: int) -> None:
        metrics.append({
            "epoch": epoch,
            "train_acc": accuracy(model, Xtr, ytr),
            "test_acc": accuracy(model, Xte, yte),
            "train_loss": geodesic_loss(model, Xtr, ytr),
            "test_loss": geodesic_loss(model, Xte, yte),
            "weights": [w.to_array().tolist() for w in model.weights],  # W branches × 8
        })

    record(0)
    for epoch in range(1, epochs + 1):
        for idx in rng.permutation(len(ytr)):
            model.learn(Xtr[idx], int(ytr[idx]))
        record(epoch)

    out_dir.mkdir(exist_ok=True)
    tag = name.replace(" ", "_")
    result = {
        "config": {"dataset": "parity", "bits": bits, "model": name, "layers": width,
                   "epochs": epochs, "seed": seed},
        "model": name,
        "metrics": metrics,
    }
    path = out_dir / f"{tag}.json"
    path.write_text(json.dumps(result, indent=1))
    print(f"wrote {path}  (final test_acc {metrics[-1]['test_acc']:.2f})", flush=True)
    return str(path)


def inspect() -> None:
    """Emit dashboard-viewable runs contrasting the quaternion win vs octonion fail."""
    out = Path("runs")
    print("Emitting convergence runs for the dashboard (per-epoch acc/loss + weight evolution)...\n")
    emit_run("proto-quaternion-product-W3", dim=4, width=3, combiner="product",
             bits=3, epochs=25, lr=0.3, seed=0, out_dir=out)
    emit_run("proto-octonion-product-W3", dim=8, width=3, combiner="product",
             bits=3, epochs=25, lr=0.3, seed=0, out_dir=out)
    emit_run("proto-octonion-product-W2-on-2bit", dim=8, width=2, combiner="product",
             bits=2, epochs=25, lr=0.3, seed=0, out_dir=out)
    print("\nInspect: uv run streamlit run src/v3i/dashboard.py  → Training comparison")
    print("  Accuracy/loss panels = convergence; Weight-evolution panel = per-branch components.")


def main() -> None:
    if "--inspect" in sys.argv:
        inspect()
        return
    epochs, lr, seeds = 25, 0.3, 5
    print(f"3-bit parity gate | {seeds} seeds | {epochs} epochs | lr {lr} | pass = median >= 0.90\n")
    configs = [
        ("octonion product W=3 (the architecture)", dict(dim=8, width=3, combiner="product", use_kappa=True)),
        ("octonion product W=4", dict(dim=8, width=4, combiner="product", use_kappa=True)),
        ("octonion product W=3, kappa OFF", dict(dim=8, width=3, combiner="product", use_kappa=False)),
        ("QUATERNION product W=3 (assoc inert control)", dict(dim=4, width=3, combiner="product", use_kappa=True)),
        ("octonion W=1 LINEAR (nonlinearity removed)", dict(dim=8, width=1, combiner="product", use_kappa=True)),
        ("octonion renorm-SUM W=3 (linear aggregation)", dict(dim=8, width=3, combiner="sum", use_kappa=True)),
    ]
    print(f"{'config':<48} {'median':>7} {'seeds (test acc)':>28} {'inv≤1':>7} {'|w|dev':>8}", flush=True)
    for name, kw in configs:
        r = run(bits=3, epochs=epochs, lr=lr, seeds=seeds, **kw)
        seedstr = " ".join(f"{a:.2f}" for a in r["test_accs"])
        inv = "ok" if r["invariant_viol_frac"] < 0.01 else f"VIOL {r['invariant_viol_frac']:.0%}"
        print(f"{name:<48} {r['median']:>7.2f} {seedstr:>28} {inv:>7} {r['max_weight_norm_dev']:>8.1e}",
              flush=True)


if __name__ == "__main__":
    main()
