# Why quaternion peel-and-solve converges (and octonion does not)

Theory note for the wayfinder ticket
[Theory: why quaternion peel-and-solve converges (#17)](https://github.com/hirekk/v3i/issues/17),
a sub-issue of the quaternion-primary map [#16](https://github.com/hirekk/v3i/issues/16).
It explains, from the algebra up, the central empirical result of the
[octonion effort retrospective](octonion-effort-retrospective.md): the
gradient-free **peel-and-solve** wave learns 3-bit parity to **1.00 in one
epoch on quaternions**, and structurally **fails on octonions (~0.71,
unstable)**. The prototype under study lives on branch
[`prototype/wide-octonion-parity`](https://github.com/hirekk/v3i/tree/prototype/wide-octonion-parity)
(`prototypes/wide_octonion_parity.py`); it is throwaway, never merged.

Every claim is tagged **[derivation]** (my algebra, reproducible by hand),
**[numerical]** (checked against the project's own `Octonion` class), or
**[cited]** (a source in the companion notes). Numerical results are produced
by the self-contained reconstruction in Appendix A (deterministic seeds,
`uv run python`, ~15 s); they were also cross-checked against the actual
prototype. This note leans on the [isometry ceiling](isometry-ceiling.md)
(fixed function class), the [octonion deep dive](octonion-structure-deep-dive.md)
(§2 Artin/alternativity, §3 the associator), the
[error-wave chirality note](error-wave-chirality.md) (right-chirality transport,
the slot-shift/associator gap), and the
[literature survey](hypercomplex-backprop-free-literature.md) (prior art).

## TL;DR

1. **Convergence (H).** One Jacobi peel-and-solve pass multiplies the geodesic
   output error by **exactly `|1 − ηW|` per sample** on the associative algebra
   — not to first order, *exactly* (telescoping proof §3.4, verified 4e-16, and
   the prototype's whole-run `max_ratio = 0.10 = |1 − 0.3·3|`). Contraction
   holds for **η ∈ (0, 2/W)**, is fastest at **η = 1/W** (one pass solves the
   sample exactly), and needs no special initialization. The runtime invariant
   `‖residual‖ ≤ ‖incoming‖` is the contraction certificate; on H it holds with
   ratio `|1 − ηW| < 1` at every step (0 % violations).
2. **Why associativity is required.** "Correct the branch" ≡ "correct the
   output" because, associatively, the output's dependence on branch `bᵢ` is the
   map `bᵢ ↦ Pᵢ·bᵢ·Sᵢ` — a left-and-right multiplication, i.e. an **isometry of
   S³** that carries the branch target `bᵢ*` exactly to `y*`. The exact obstruction
   is one associator: the full-step weight update misses its target by
   **`wᵢ·[w̄ᵢ, x̄, bᵢ*]`** (§4.1, verified 4e-16), ≡ 0 on H, `O(1)` on O. It
   corrupts the update *direction* (not step size): the misdirection is
   η-independent, so backtracking η cannot fix it. This reproduces the
   retrospective's **74 % invariant violations even at η → 0.005** (§4.2,
   reproduced exactly) and the finding that octonion failure grows monotonically
   with weight spread (0 % → ~38 %).
3. **Function class.** A width-`W` branch-product layer computes
   `y = (x·w₁)···(x·w_W)`, whose readout `re(y)` is a **degree-`W` homogeneous
   polynomial in `x`**. On the degree-preserving parity embedding, `k`-bit parity
   needs degree ≥ `k`, so **`W ≥ k` is necessary**; `W = k` is demonstrated
   sufficient *and findable* at `k = 3` on H. Depth composes degrees (up to
   `W^D`) but only the linear pathway is read; findability at depth is untested.
4. **Prior art.** Peel-and-solve is **target propagation on a group manifold
   with the inverse made exact.** Where difference-target-prop learns an
   approximate autoencoder inverse `gᵢ ≈ fᵢ⁻¹`, here the combiner inverse is the
   *exact algebraic inverse* (`b⁻¹ = b̄`), so the DTP "difference" correction is
   unnecessary and the target is reconstructed with zero approximation error —
   but *only* on an associative algebra, which is exactly the §2 requirement.

---

## 1. Setup and conventions

**Forward map (single wide layer, one-sided branches, branch-product combiner).**
Input `x` a unit octonion/quaternion; the parity generator embeds `k` bits on
the imaginary axes and normalizes, so `x ∈ S^{d-1}` with `re(x) = 0`
(**[numerical]** `max|‖x‖−1| = 1.1e-16`, `max|re(x)| = 0`, Appendix Check 0).
Weights `wᵢ` unit. Then

```
branches   bᵢ = x·wᵢ                       (unit, since |x·wᵢ| = |x||wᵢ| = 1)
output     y  = b₁·b₂·…·b_W  (left fold)    (unit)
readout    ŷ  = sign(re(y))                 (frozen; the canonical G₂-readout)
```

The quaternion control restricts every `wᵢ` (hence every `bᵢ`, hence `y`) to the
associative subalgebra `span{1,e₁,e₂,e₃} ≅ ℍ`; the octonion case uses all of
`𝕆`. Write `Pᵢ = b₁…b_{i-1}` (**prefix**, `P₁ = 1`) and `Sᵢ = b_{i+1}…b_W`
(**suffix**, `S_W = 1`); on ℍ, `y = Pᵢ·bᵢ·Sᵢ`.

**Target, error, invariant.** The label `ℓ ∈ {±1}` gives target `y* = ℓ·e₀`
(a real unit). The error is the rotation `r = ȳ·y*` (so `y·r = y*`, using
`|y| = 1`), and the scalar error is the **geodesic distance**
`θ = ‖log r‖ = ∠(y, y*)` on the sphere. A weight update produces `y_new`; the
**residual** is the new error `‖log(ȳ_new·y*)‖`, and the design's runtime
invariant is `residual ≤ incoming`, i.e. `θ_new ≤ θ_old` — the output got no
farther from target. Its per-step *ratio* `θ_new/θ_old` is the object §3–§4
analyze.

**The wave (exact nested peel, Jacobi).** For each branch, reconstruct the value
it *should* have had (holding the others fixed), then take a geodesic weight
step toward it:

```
peel     bᵢ*  = Pᵢ⁻¹ · y* · Sᵢ⁻¹                (branch target)
share    sᵢ   = bᵢ⁻¹ · bᵢ*                       (correction in branch frame)
update   wᵢ  ← normalize( wᵢ · sᵢ^η )            (right-chirality; sᵢ^η = exp(η·log sᵢ))
```

All `W` shares are computed from the *current* branches (Jacobi). This is the
gradient-free analogue of the chain rule: **reconstruct** each branch's target
by inverting the combiner around it, rather than **decompose** a gradient — the
move a division algebra makes possible. The nested implementation unwinds the
real left-fold bracketing, so the peel is *exact in both algebras*
(§3.1). Right-chirality (`wᵢ · sᵢ^η`, forward-consistent) is the convention the
[chirality note](error-wave-chirality.md) established as correct.

---

## 2. The mechanism in one line

On an associative algebra the output's dependence on one branch factors as

```
        Φᵢ : bᵢ  ↦  Pᵢ · bᵢ · Sᵢ  =  L_{Pᵢ} ∘ R_{Sᵢ} (bᵢ),
```

a composition of a left- and a right-multiplication by unit elements — hence an
**isometry of S³** (each of `L_{Pᵢ}, R_{Sᵢ}` is orthogonal;
[isometry-ceiling §1](isometry-ceiling.md)). The peel defines `bᵢ*` precisely so
that `Φᵢ(bᵢ*) = y*`. Because `Φᵢ` is an isometry, it maps *the whole geodesic
from `bᵢ` to `bᵢ*`* onto *the geodesic from `y` to `y*`*, preserving arc length
and fractional position. Therefore:

> **Correct the branch ≡ correct the output** — moving `bᵢ` a fraction `η` of the
> way to `bᵢ*` moves `y` the *same* fraction `η` of the way to `y*`, along the
> minimizing geodesic. (§3.2, **[derivation + numerical]**.)

Everything in §3 (why it converges) and §4 (why octonions break it) is a
consequence of `Φᵢ` being an isometry on ℍ and failing to be one on 𝕆.

---

## 3. Convergence on the associative algebra (Area 1)

### 3.1 The nested peel is exact — in both algebras **[derivation + numerical]**

The left fold `y = (((b₁b₂)b₃)…b_W)` is unwound by right-division. Writing the
partial products `Πₖ = b₁…bₖ` (so `Π_{k+1} = Πₖ·bₖ`), replacing `bᵢ` alone by
`bᵢ*` and demanding the fold equal `y*` gives a chain
`y* = ((Πᵢ·bᵢ*)·b_{i+1})…b_W`, solved one factor at a time:

```
targetᵢ = y* · b_W⁻¹ · … · b_{i+1}⁻¹  ;   bᵢ* = Πᵢ⁻¹ · targetᵢ.
```

Each peel step `(z·b)·b⁻¹ = z` and `Πᵢ⁻¹·(Πᵢ·bᵢ*) = bᵢ*` involves **only two
distinct elements**, so it is exact by Artin's theorem / diassociativity
([deep dive §2](octonion-structure-deep-dive.md)) — *even on the octonions*.

**[numerical]** Replacing `bᵢ` by the nested `bᵢ*` and refolding hits `y*` to
`4.4e-16` on **both** H and O (Appendix Check 1). So the octonion failure is
**not** a wrong branch target; the target is exact. The failure is downstream,
in mapping that target to a weight (§4).

### 3.2 One branch: a fractional weight step is a fractional output step **[derivation + numerical]**

On ℍ the input cancels exactly. The weight target is `wᵢ* = x⁻¹·bᵢ* = x̄·bᵢ*`
(`x` unit), valid because `x·(x̄·z) = z` (two elements — **left alternative
law**, holds on 𝕆 too). The full-step update reaches it:

```
wᵢ·sᵢ = wᵢ·(b̄ᵢ·bᵢ*) = wᵢ·((w̄ᵢx̄)·bᵢ*) —assoc→ (wᵢw̄ᵢ)·(x̄·bᵢ*) = x̄·bᵢ* = wᵢ*.
```

Hence `bᵢ ↦ bᵢ·sᵢ^η` moves `bᵢ` along the geodesic to `bᵢ*`, and
`y = Pᵢ·bᵢ·Sᵢ ↦ Pᵢ·(bᵢ·sᵢ^η)·Sᵢ` moves `y` by the isometry `Φᵢ`. Transporting
the branch tangent into the output frame, the induced output rotation is
`τᵢ(η) = S̄ᵢ·(η·log sᵢ)·Sᵢ`, and because `bᵢ*` was built to send `y → y*`,

```
   S̄ᵢ · log(sᵢ) · Sᵢ  =  log(ȳ·y*)  =  u    for every branch i.        (★)
```

**[numerical]** (★) holds to `1.3e-15` on H (every branch's exact peel induces
the *same* output correction `u`) and is violated by `1.6` on O (Appendix
Check 3). This equality is the engine of §3.4.

### 3.3 The over-counting problem (why η matters)

Each branch, alone, would carry `y` a fraction `η` toward `y*`. Under **Jacobi**
all `W` branches move at once, each attempting the *entire* correction `u` — so
their contributions sum and the naive update over-counts by `W`. This is not a
bug to remove; it is the reason a step size exists.

### 3.4 The exact contraction law: `θ_new = |1 − ηW|·θ_old` **[derivation + numerical]**

By (★), `log sᵢ = Sᵢ·u·S̄ᵢ`, so `sᵢ^η = exp(η·SᵢuS̄ᵢ) = Sᵢ·q·S̄ᵢ` with
`q := exp(ηu)` (conjugation commutes with `exp`). On ℍ the updated branch is
`bᵢ·sᵢ^η = bᵢ·Sᵢ·q·S̄ᵢ`. Write the **suffix-inclusive** products
`Tᵢ = bᵢ·b_{i+1}…b_W` (so `bᵢ·Sᵢ = Tᵢ`, `Sᵢ = T_{i+1}`, `T_{W+1} = 1`,
`T₁ = y`). Then each updated factor is `Tᵢ·q·T̄_{i+1}`, and the whole Jacobi
product **telescopes** because `T̄_{i+1}·T_{i+1} = 1` (units):

```
 y_new = ∏ᵢ (Tᵢ · q · T̄_{i+1})
       = T₁·q·(T̄₂T₂)·q·(T̄₃T₃)·q·…·q·T̄_{W+1}
       = T₁ · q^W  =  y · exp(ηu)^W  =  y · exp(ηW·u).
```

Therefore `ȳ_new·y* = q^{-W}·(ȳy*) = exp(-ηW·u)·exp(u) = exp((1−ηW)·u)`, and

> **`θ_new = |1 − ηW| · θ_old`, exactly, per sample.** **[derivation]**

**[numerical]** Confirmed to `≤ 3.3e-15` per sample across `W ∈ {2,3,4}` and
`η ∈ {0.1, 0.3, 1/W, …}` (Appendix Check 4/A). Independent confirmation from the
*actual prototype*: a full quaternion run (W=3, η=0.3) reports whole-training
`max_ratio = 0.10 = |1 − 0.9|` and **0 % invariant violations**, median test
accuracy **1.00**.

**Consequences.**

- **Contraction window:** `0 < η < 2/W`. Outside it the pass diverges
  (`η ≥ 2/W` ⇒ ratio ≥ 1; the prototype hits ratio → 1 at `η = 2/W`, Check 4).
- **Optimal step `η* = 1/W`:** ratio 0 — **one Jacobi pass solves the sample
  exactly** (Check 4 shows `η = 1/W` drives the ratio to 0 to machine
  precision). The prototype's `η = 0.3, W = 3` sits near `η* = 1/3`, giving
  factor `0.1`: this is *why* one epoch suffices.
- **No initialization condition (on H):** the law is per-sample exact for *any*
  unit weights — Check B finds 0 % violations at weight spread `σ` up to 3.0 and
  for fully random unit quaternions. Associativity buys global well-behavedness,
  not merely local.

### 3.5 The residual invariant is the contraction certificate

`‖residual‖ ≤ ‖incoming‖` is exactly `θ_new ≤ θ_old`: monotone decrease of the
geodesic error, a Lyapunov certificate. On H it holds *strictly* with the known
factor `|1−ηW| < 1` at every step, so the certificate is not just satisfied but
quantitative. Its **violation is the diagnostic** for the octonion failure
(§4.2): when the reconstructed direction is wrong, `θ_new > θ_old` and the
invariant trips — which is precisely what the prototype's adaptive schedule
detects and cannot repair.

### 3.6 Schedule and scope, honestly

- The winning config is **Jacobi**; §3.4 is the Jacobi analysis. **Gauss-Seidel**
  (sequential, refold between branches) is an alternative coordinate-descent
  schedule that also contracts on H — at full step `η = 1` a single branch
  update already reaches `y*` — but was only tabled for O (0.49), where it does
  not help.
- §3.4 proves **per-sample** contraction toward *that sample's* target. Fitting
  the whole dataset (samples pull toward the two poles `±e₀`) is the standard
  stochastic-approximation story on a compact manifold ([cited] Bonnabel,
  Riemannian SGD); the strong fact peel-and-solve contributes is that **every
  per-sample step is an exact, wrong-direction-free contraction**, so the SGD
  has no misdirection noise to average out. The empirical 1.00 is the confirmation;
  a full fixed-point/convergence proof over the dataset is not attempted here and
  is flagged as open.

---

## 4. Why associativity is required (Area 2)

### 4.1 The associator obstruction — an exact identity **[derivation + numerical]**

The peel target `bᵢ*` is exact on 𝕆 (§3.1). The break is in the *weight* step.
Redo §3.2 without associativity: `sᵢ = b̄ᵢ·bᵢ* = (w̄ᵢx̄)·bᵢ*`, so the full-step
update is `wᵢ·sᵢ = wᵢ·((w̄ᵢx̄)·bᵢ*)`, whereas the exact weight target needs the
*other* bracketing `wᵢ·(w̄ᵢ·(x̄·bᵢ*)) = wᵢ*`. Subtracting, with the standard
associator `[a,b,c] = (ab)c − a(bc)`:

```
   wᵢ·sᵢ  −  wᵢ*  =  wᵢ · [ w̄ᵢ, x̄, bᵢ* ].                              (♦)
```

**[numerical]** (♦) is exact: on O the gap `‖wᵢ·sᵢ − wᵢ*‖` is `O(1)` (≈ 1,
sampling-dependent), and it equals `wᵢ·[w̄ᵢ,x̄,bᵢ*]` to `5.6e-16`; on H both sides are `≤ 5.6e-16`
(the associator vanishes, [deep dive §3.1](octonion-structure-deep-dive.md))
(Appendix Check 2). **The input fails to cancel by exactly one associator.**

Two properties of that associator make it fatal, not merely nonzero
([deep dive §3.1](octonion-structure-deep-dive.md), [cited]): it is **orthogonal
to each of its arguments**, so it points in a direction *new* relative to the
intended correction; and it is bounded by 2 with typical size `O(1)`. The update
therefore lands off the target in a genuinely transverse direction.

### 4.2 It is the direction, not the step size **[derivation + numerical]**

Shrinking η cannot rescue this. Expand the realized branch motion of a geodesic
weight step `wᵢ ↦ wᵢ·exp(η g)`, `g = log sᵢ`:

```
  x·(wᵢ·exp(ηg)) − (x·wᵢ)·exp(ηg)  =  −η·[x, wᵢ, g]  +  O(η²).
```

The *intended* motion `(x·wᵢ)·exp(ηg) = bᵢ·sᵢ^η` (the associatively-correct step
toward `bᵢ*`) and the *misdirection* `−η·[x,wᵢ,g]` **both scale as η**. Their
ratio — the angle by which the realized step departs from the intended one — is
therefore **η-independent** to leading order. A wrong step *size* shrinks under
backtracking; a wrong step *direction* does not. When the transverse component
has a positive projection onto the error-increasing direction, the step raises
`θ`, and no η makes it descend.

**[numerical] reproduction of the retrospective's 74 %.** Running the actual
prototype's octonion config with the **adaptive schedule** (backtrack η by
halving up to 7×, down to η ≈ 0.0023, to *try* to satisfy the invariant):
`invariant_viol_frac = 0.74`, `max_ratio = 1.01` — i.e. even the best
backtracked step cannot get the ratio below ~1. This matches the retrospective
exactly. The quaternion control under the same harness: `viol = 0.00`,
`max_ratio = 0.10`.

**[numerical] the failure grows with non-associativity.** The misdirection is
`0` when weights lie in a common quaternion subalgebra and grows as they spread.
Single-branch tiny-step (η = 0.005) error-*increase* fraction, and full-Jacobi
(η = 0.3) invariant-violation fraction, vs weight spread `σ` (Appendix Check B;
percentages are single-seed and shift a few points with sampling order — the
`H ≡ 0 %` column and the monotone O trend are the robust facts):

| weights            | H increase / viol | O increase / viol |
|--------------------|-------------------|-------------------|
| σ = 0.3 (near 1)   | 0 % / 0 %         | 0 % / 0 %         |
| σ = 0.7            | 0 % / 0 %         | 21 % / 17 %       |
| σ = 1.5            | 0 % / 0 %         | 34 % / 21 %       |
| σ = 3.0            | 0 % / 0 %         | 38 % / 25 %       |
| fully random unit  | 0 % / 0 %         | 35 % / 29 %       |

H is **exactly 0 %** at every spread (associativity ⇒ (♦) vanishes ⇒ telescoping
exact); O rises monotonically. This is the honest bridge to the trained-state
74 %: fresh near-identity octonion weights barely fail, but **fitting degree-3
parity forces the weights to spread across all seven imaginary axes**, inflating
the associators in (♦) until the majority of updates point the output the wrong
way. The retrospective's 74 % is that trained regime; the σ-sweep is its
controlled, monotone cause.

### 4.3 The clean statement

> **Peel-and-solve on a multiplicative product combiner requires associativity.**
> On an associative algebra the map "branch value ↦ output" is an isometry
> (`Φᵢ = L_{Pᵢ}R_{Sᵢ}`), so reconstructing the branch target reconstructs the
> output target, and the input cancels out of the weight update exactly (♦ = 0).
> On a non-associative algebra the two decouple by one associator per branch;
> the reconstructed weight step points in a transverse, η-independent wrong
> direction, and the output error increases. Quaternions have it (clean 1.00);
> octonions do not (≤ 0.71, unstable).

This is consistent with, and sharpens, the chirality note's finding that the
only non-associative content in a right-chirality wave is an
**associator-sized cross-slot credit term, ≡ 0 on ℍ**
([chirality §1.4, §2](error-wave-chirality.md)): here that term is named
explicitly — `wᵢ·[w̄ᵢ, x̄, bᵢ*]` — and shown to be the whole obstruction.

---

## 5. Function class: what a width-`W`, depth-`D` layer represents (Area 3)

**Degree.** `y = (x·w₁)·(x·w₂)···(x·w_W)` is a product of `W` factors each linear
in `x`; every component of `y`, and in particular `re(y)`, is a **homogeneous
polynomial of degree `W` in the coordinates of `x`** [derivation]. No
normalization intervenes (products of units are units), so `re(y)` is an exact
degree-`W` form.

**Parity ladder.** The generator embeds the `k` bits linearly on imaginary axes
before normalizing, which **preserves polynomial degree** (`make_data`
docstring; contrast the rational inverse-stereographic embedding). `k`-bit
parity is the `k`-ary XOR, provably requiring a readout of degree ≥ `k`
([readout-screen](readout-screen.md); [retrospective §1](octonion-effort-retrospective.md)).
Hence:

- **`W ≥ k` is necessary.** `W = 1` (linear, degree 1) and the renorm-**sum**
  combiner (linear by distributivity, `Σᵢ x·wᵢ = x·(Σwᵢ)`,
  [lit §4.0](hypercomplex-backprop-free-literature.md)) are both **degree 1** →
  cannot exceed the 75 % XOR/linear ceiling → **chance on 3-bit parity**. This is
  exactly the prototype's control table.
- **`W = k` is sufficient and findable at `k = 3`** (on H): degree-3 `re(y)`
  contains the 3-bit parity form, and peel-and-solve *finds* it (1.00). Whether
  `W = k` suffices to *represent* every `k` is not proved here — it is
  demonstrated at `k = 3`; general representability is flagged open.

**Depth.** A stack of wide product-layers composes degrees: feeding a unit output
into another width-`W` product layer squares the degree, reaching up to `W^D` in
`x` at depth `D`. But the readout sees only `re` of the final unit — and the
[isometry ceiling](isometry-ceiling.md) warns that a *pure* `x·w` stack (no
product combiner) collapses to degree 1 regardless of depth. So depth adds
representational degree **only through the product combiner**, and even then only
the linear pathway is read. **Findability at depth is untested** — the prototype
is a single wide layer. The honest gap: §3 establishes findability for **width**
at **depth 1**; depth is a capacity statement, not yet a findability one.

**Capacity vs findability.** The isometry-ceiling and readout notes settle
*capacity* (what the forward map + readout can represent). This note settles
*findability* for the width-`W`, depth-1 case: the branch product is degree-`W`,
and the gradient-free wave provably (§3.4) and empirically (1.00) reaches the
parity solution on H. The two are distinct and both required; peel-and-solve is
the findability half.

---

## 6. Relation to prior art (Area 4)

Positioned against the families surveyed in the
[literature note](hypercomplex-backprop-free-literature.md) (references reused,
not re-fetched).

**Target propagation / difference-target-propagation** (Lee et al. 2015,
[arXiv:1412.7525]; [lit §2.5]). DTP propagates *targets*, not gradients: each
layer learns an approximate inverse `gᵢ ≈ fᵢ⁻¹` (an autoencoder), and the
**difference correction** `ĥ_{i-1} = h_{i-1} + gᵢ(ĥᵢ) − gᵢ(hᵢ)` exists precisely
to stay stable *despite the inverse being approximate*. Peel-and-solve is DTP's
structural twin with one decisive change: **the combiner inverse is the exact
algebraic inverse** (`b⁻¹ = b̄` on the unit sphere; the peel `bᵢ* = Pᵢ⁻¹y*Sᵢ⁻¹`),
so there is *no* learned autoencoder, *no* approximation error, and the DTP
difference term is **unnecessary** — the reconstructed target is exact (§3.1,
verified 4e-16). This is the sense in which the project's division-algebra bet
pays off: the group/algebra structure hands you `f⁻¹` for free
([lit §2.5 "the inverse network comes for free"]). The catch this note adds:
*exactness of the inverse is not enough* — the inverse must also **compose
isometrically with the output**, which is associativity (§4). DTP over
associative (matrix) layers has this automatically; peel-and-solve loses it on 𝕆.

**Feedback alignment / DFA / IFA** (Lillicrap et al. 2016,
[arXiv:1411.0247]/ncomms13276; Nøkland 2016, [arXiv:1609.01596]; [lit §2.2–2.3]).
FA replaces `Wᵀ` with a fixed random teaching channel and lets the forward
weights *align* to it. Peel-and-solve is the opposite extreme: not a random but
a **principled and exact** teaching channel (the algebraic inverse), needing no
alignment phase. FA's lesson — an imperfect-but-consistent channel can teach —
is the comfort blanket peel-and-solve does *not* need on H (the channel is
exact) and cannot use on 𝕆 (the channel is exact but points the wrong way; the
problem is geometry, not consistency). Nøkland's **IFA** (inject error early,
forward-propagate the update) is the closest *directional* relative: the wave is
IFA with the random loop replaced by log-map error + associative reconstruction.

**Forward-forward / forward-only family** (Hinton 2022, [arXiv:2212.13345];
Kohan et al. EFP 2018, [arXiv:1808.03357]; PEPITA, Dellaferrera & Kreiman 2022,
[arXiv:2201.11665]; [lit §2.1, §2.8]). These keep the teaching signal on the
forward pathway, as the wave does, and FF's **orientation/length split**
(normalize the activity, forward only the direction) is literally the
S³/S⁷-normalization the wave runs on. The distinction: FF and PEPITA use *local
goodness* or a *second forward pass*, contrastive and gradient-flavored;
peel-and-solve carries a **geometric target** (the exact rotation `log(ȳy*)`,
transported by conjugation) and reduces it by an exact reconstruction — the
[lit §2.8] survey found **no prior art** for that transport-and-reconstruct
design. Its novelty (and risk) is the exactness; §3–§4 are the theory of when
that exactness converts to convergence.

**Manifold optimization** (Bonnabel 2013, [arXiv:1111.5280]; [lit §3.1]).
Riemannian SGD guarantees convergence for geodesic steps *along a gradient*.
Peel-and-solve's step is not a gradient; §3.4's `|1−ηW|` contraction is a
*direct* certificate that replaces the missing gradient guarantee on H, and
§4's obstruction is exactly why the Bonnabel comfort does not transfer to 𝕆
(the step is not a descent direction there).

**One-line placement.** Peel-and-solve = **exact-inverse target propagation on a
division-algebra sphere, carried forward** — DTP's targets with FA's
forward-only spirit and FF's orientation-only transport, but with the learned
approximate inverse replaced by the algebra's exact one. It converges when, and
only when, that exact inverse composes with the output isometrically, i.e. on an
associative algebra.

---

## 7. Open threads this note leaves

1. **Dataset-level convergence proof** (not just per-sample contraction, §3.6).
2. **Depth findability** — §3 is width-at-depth-1; the `W^D` capacity is
   unexercised (§5).
3. **General representability** of `k`-bit parity at `W = k` (shown at `k = 3`).
4. **Whether any octonion rule escapes §4** — the retrospective's back-burner
   question. §4 forbids *product-combiner peel-and-solve* on 𝕆; a combiner whose
   credit assignment does not route through a multiplicative inverse (an
   associator-native mechanism) is not covered by this no-go.

---

## Appendix A. Reproducible numerical checks

Self-contained; requires only the repo's `v3i` package. Save and run from the
repo root: `uv run python thisfile.py`. Deterministic (`default_rng`). The same
computations were cross-checked against the actual prototype
(`prototype/wide-octonion-parity`); the adaptive-η 74 % and the quaternion
`max_ratio = 0.10` in §4.2/§3.4 are from that prototype via
`wop.run(..., peel='nested', schedule=...)`.

```python
import numpy as np
from v3i.algebra import Octonion
from v3i.make_data import generate_parity
rng = np.random.default_rng(0)
pad8 = lambda v: np.concatenate([v, np.zeros(8-len(v))]) if len(v) < 8 else v
O   = lambda v: Octonion(pad8(np.asarray(v, float)))
one = Octonion.unit
def opow(s, t):                       # geodesic fraction s^t
    s = s.normalize(); return (s.log()*t).exp()
def fold(bs):
    y = bs[0]
    for b in bs[1:]: y = y*b
    return y
def geo(y, ys): return float(np.linalg.norm((y.conjugate()*ys).log().to_array()))
def peel(bs, ys):                     # exact nested peel -> shares, targets
    W = len(bs); pre = [one()]
    for b in bs: pre.append(pre[-1]*b)
    tg = [None]*W; tg[-1] = ys
    for k in range(W-1, 0, -1): tg[k-1] = tg[k]*bs[k].inverse()
    bstar = [pre[i].inverse()*tg[i] for i in range(W)]
    return [bs[i].inverse()*bstar[i] for i in range(W)], bstar
def wts(W, dim, sig=0.3):
    out = []
    for _ in range(W):
        v = np.zeros(8); v[0] = 1.0; v[:dim] += rng.normal(0, sig, dim)
        if dim == 4: v[4:] = 0.0
        out.append(Octonion(v).normalize())
    return out
def inp(dim, n=120):
    X, y, *_ = generate_parity(400, 200, 0.05, np.random.default_rng(7), bits=3, dim=dim)
    return [(Octonion(pad8(X[k])), 1.0 if y[k] > 0 else -1.0) for k in range(n)]

# Check 1: nested peel is the exact branch target (both algebras) -> 4.4e-16
for dim in (4, 8):
    w = 0.0
    for x, lab in inp(dim):
        bs = [x*wi for wi in wts(3, dim)]; ys = one() if lab > 0 else O([-1.0])
        _, bstar = peel(bs, ys)
        for i in range(3):
            c = list(bs); c[i] = bstar[i]
            w = max(w, abs((fold(c)-ys).to_array()).max())
    print("Check1 dim", dim, "refold-y* =", f"{w:.1e}")

# Check 2: (wi*si) - wi* = wi*[w̄i, x̄, bi*]   (H: 5e-16 ; O: gap O(1), resid 5e-16)
for dim in (4, 8):
    gd = ga = 0.0
    for x, lab in inp(dim):
        ws = wts(3, dim); bs = [x*wi for wi in ws]; ys = one() if lab > 0 else O([-1.0])
        sh, bstar = peel(bs, ys)
        for i in range(3):
            wn = ws[i]*sh[i]; wt = x.conjugate()*bstar[i]
            aw, ax, ab = ws[i].conjugate(), x.conjugate(), bstar[i]
            pred = ws[i]*((aw*ax)*ab - aw*(ax*ab))
            gd = max(gd, abs((wn-wt).to_array()).max())
            ga = max(ga, abs(((wn-wt)-pred).to_array()).max())
    print("Check2 dim", dim, "gap", f"{gd:.2e}", "resid-vs-assoc", f"{ga:.1e}")

# Check 4/A: exact contraction  θ_new/θ_old = |1-ηW|  (H, per sample <=2e-15)
for W in (2, 3, 4):
    for eta in (0.1, 0.3, 1.0/W):
        d = 0.0
        for x, lab in inp(4, 60):
            ws = wts(W, 4); bs = [x*wi for wi in ws]
            y = fold(bs); ys = one() if lab > 0 else O([-1.0]); old = geo(y, ys)
            if old < 1e-6: continue
            sh, _ = peel(bs, ys)
            w2 = [(ws[i]*opow(sh[i], eta)).normalize() for i in range(W)]
            d = max(d, abs(geo(fold([x*wi for wi in w2]), ys)/old - abs(1-eta*W)))
        print("Check4 W", W, "eta", round(eta, 3), "max|ratio-|1-ηW||", f"{d:.1e}")

# Check B: octonion misdirection grows with weight spread; H stays 0
for dim, nm in ((4, "H"), (8, "O")):
    for sig in (0.3, 0.7, 1.5, 3.0):
        inc = tot = viol = pas = 0
        for x, lab in inp(dim, 150):
            ws = wts(3, dim, sig); bs = [x*wi for wi in ws]
            y = fold(bs); ys = one() if lab > 0 else O([-1.0]); old = geo(y, ys)
            if old < 1e-4: continue
            sh, _ = peel(bs, ys)
            for i in range(3):
                w2 = list(ws); w2[i] = (ws[i]*opow(sh[i], 0.005)).normalize()
                tot += 1; inc += geo(fold([x*wi for wi in w2]), ys) > old
            w2 = [(ws[i]*opow(sh[i], 0.3)).normalize() for i in range(3)]
            pas += 1; viol += geo(fold([x*wi for wi in w2]), ys) > old + 1e-9
        print("CheckB", nm, "sig", sig, "inc%", round(100*inc/tot), "viol%", round(100*viol/pas))
```

Observed (seed 0): Check1 `4.4e-16` both; Check2 H `≤ 5.6e-16`, O gap `O(1)`
(≈ 1.1 here) with residual-vs-associator `5.6e-16`; Check4 all `≤ 3.3e-15`;
CheckB `H ≡ 0 %` at every σ, O rising `1 → 25 → 32 → 32 %` (single-seed;
cf. the §4.2 table for another sampling). (Check 0 and Check 3, and the prototype-driven 74 % / `max_ratio`
reproductions, are in the companion scratch scripts and the prototype's own
`run(...)`.)

## References

Reused from the companion notes (each verified against a primary source in the
[literature survey](hypercomplex-backprop-free-literature.md)):

- J. C. Baez, *The Octonions*, Bull. AMS 39 (2002), arXiv:math/0105155 —
  composition algebras, associator, `[L_a,R_b] = −`associator.
- R. D. Schafer, *An Introduction to Nonassociative Algebras*, Dover 1995 —
  **Artin's theorem** and alternativity (the exact two-element identities of §3.1).
- A. Hurwitz (1898) — normed division algebras exist only in dims 1, 2, 4, 8.
- Lee, Zhang, Fischer, Bengio, *Difference Target Propagation*, ECML-PKDD 2015 —
  arXiv:1412.7525.
- Lillicrap, Cownden, Tweed, Akerman, *Feedback alignment*, Nature Comms 7:13276
  (2016) — arXiv:1411.0247.
- Nøkland, *Direct Feedback Alignment* (and IFA), NeurIPS 2016 — arXiv:1609.01596.
- Hinton, *The Forward-Forward Algorithm*, 2022 — arXiv:2212.13345.
- Kohan, Rietman, Siegelmann, *Error Forward-Propagation*, 2018 — arXiv:1808.03357.
- Dellaferrera & Kreiman, *PEPITA*, ICML 2022 — arXiv:2201.11665.
- Bonnabel, *Stochastic Gradient Descent on Riemannian Manifolds*, IEEE TAC 58(9)
  (2013) — arXiv:1111.5280.

Sibling notes: [octonion-effort-retrospective.md](octonion-effort-retrospective.md)
(the arc and the numbers this note explains),
[isometry-ceiling.md](isometry-ceiling.md) (function class / capacity),
[octonion-structure-deep-dive.md](octonion-structure-deep-dive.md) (Artin,
alternativity, the associator),
[error-wave-chirality.md](error-wave-chirality.md) (right-chirality transport,
the associator credit term),
[hypercomplex-backprop-free-literature.md](hypercomplex-backprop-free-literature.md)
(prior art), [readout-screen.md](readout-screen.md) (`sign(re)` and the parity
degree argument).
```
