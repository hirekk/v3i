# The isometry ceiling of the current stack

Research note for the wayfinder ticket
[Isometry ceiling of the current stack](https://github.com/hirekk/v3i/issues/2).
Every numbered claim is verified to machine precision by
[isometry_ceiling_verification.py](isometry_ceiling_verification.py)
(`uv run python docs/research/isometry_ceiling_verification.py`).

## TL;DR

1. Every current layer `y = x·w` (unit `w`) is an orthogonal linear map of R⁸.
   Any stack, any depth, either algebra, is a **single orthogonal matrix** —
   the network is linear.
2. The classifier `sign(re(output))` of a depth-n stack equals `sign⟨x, v⟩` for a
   single unit octonion `v` — and a **single layer already reaches every such
   `v`**. For classification, depth adds exactly nothing, *even for octonions*.
   Non-associativity does prevent the full 8D map from collapsing — but the one
   row of it the readout sees collapses anyway.
3. **XOR is a valid gate.** Homogeneous separators pull back through the inverse
   stereographic embedding to a circle/line family in the data plane, and the
   XOR corner labeling is provably unrealizable in that family. The ceiling is
   ¾ of the corners: best achievable ≈ **75% accuracy**. The current
   architecture can never exceed it, no matter how it is trained.

## 1. The forward map is orthogonal, hence linear

Octonions are a composition algebra: `|x·w| = |x||w|`. For unit `w`,
right-multiplication `R_w` (the 8×8 matrix with `R_w x = x·w`) preserves the
Euclidean norm of R⁸ and is therefore orthogonal. Verified:
`max ‖R_wᵀR_w − I‖ ≈ 1e−15` over 200 random unit `w`, and a depth-5 chain equals
its composed matrix `M = R_{w₅}···R_{w₁}` with `‖MᵀM − I‖ ≈ 6e−16`.

So the entire network — restricted to S⁷, an isometry — is one orthogonal
transformation. No curvature of the sphere rescues expressivity: orthogonal maps
are the *linear* maps that preserve S⁷.

## 2. The readout collapses at every depth

The composition-algebra adjoint identity `⟨xy, z⟩ = ⟨x, z·ȳ⟩` gives:

- **Depth 1:** `re(x·w) = ⟨x, w̄⟩`. The label function is `sign⟨x, w̄⟩`, and since
  `w̄` ranges over all of S⁷, *a single layer already realizes every homogeneous
  linear separator of R⁸*.
- **Depth 2:** `re((x·w₁)·w₂) = ⟨x·w₁, w̄₂⟩ = ⟨x, w̄₂w̄₁⟩ = ⟨x, conj(w₁w₂)⟩`.
- **Depth n:** by induction, `⟨x, conj(w₁(w₂(···wₙ)))⟩` — some parenthesization
  of the weight product, still a single unit octonion.

Verified at depths 2 and 3 to `9e−16`. Consequently the **label function class
is `{sign⟨x, v⟩ : v ∈ S⁷}` for every depth ≥ 1** — depth changes nothing about
what can be classified.

The two algebras differ only *above* the readout:

- **Quaternions:** associativity collapses the full map: `(x·w₁)·w₂ = x·(w₁w₂)`
  exactly (verified, `4e−15`). A quaternion chain *is* a single perceptron.
- **Octonions:** the full 8D map does **not** collapse. The first row of any
  `R_v` is `v̄`, so a single equivalent weight is forced — and
  `‖R_{w₂}R_{w₁} − R_v‖` for that forced `v` is 0.88–3.3 across random pairs
  (never zero). Products of right-multiplications escape the 7-parameter family
  `{R_v}` into the 28-dimensional SO(8). But the *first row* of the product is
  exactly `conj(w₁w₂)` (verified, `2e−16`): the readout-visible row collapses
  even though the other seven don't.

**The one asset depth does buy (octonions only):** reachable 8D transformations
beyond single multiplications — invisible to a scalar readout `re(·)`, but a
richer readout could in principle see them. This feeds the readout question
currently parked in the map's fog.

## 3. XOR is a valid gate — with a 75% ceiling

**What homogeneous separators look like in the data plane.** The embedding
`u ↦ ((1−|u|²), 2u₁, 2u₂, 0, …)/(1+|u|²)` lands on the unit sphere of the
`(x₀,x₁,x₂)` subspace. For any separator normal `v`,
`sign⟨v, x(u)⟩ = sign(α(|u|²−1) + ⟨β, u⟩)` with `α = −v₀`, `β = 2(v₁,v₂)` —
a specific family of circles and lines in the original plane.

**Impossibility.** For any `f(u) = α|u|² + ⟨β,u⟩ + γ` and the four XOR corners
A(¼,¼), B(¼,¾), C(¾,¼), D(¾,¾): since `A + D = B + C` and
`|A|² + |D|² = |B|² + |C|²` (= 1.25), we get identically

```
f(A) + f(D) = f(B) + f(C).
```

The XOR labeling needs `f(A), f(D) < 0 < f(B), f(C)` — left side negative, right
side positive. Contradiction. LP confirms on the embedded corners: the pattern
(−,+,+,−) is **infeasible**, while all four 3-of-4 patterns are feasible.

**Empirical ceiling on the actual noisy dataset** (noise 0.1, seed 42, the
`make_data.py` generator): homogeneous logistic regression reaches 0.57 train /
0.64 test; the best margin-LP 3-corner separator reaches **0.72 train / 0.755
test** — right at the theoretical ¾ corner ceiling. For calibration, binary-1d
is fully linearly separable (1.00/1.00), as expected.

## 4. Consequences for the map

1. **Nonlinearity is mandatory, not optional.** No amount of depth, training
   time, or learning-rule cleverness lifts the current architecture past ~75%
   on XOR. The kappa/associator machinery in `OctonionPerceptron.correct` shapes
   *training dynamics* only; the function class is fixed by the forward map.
2. **"Just stack deeper" is dead**, for both algebras — proven, not suspected.
   Any widened-architecture win on XOR is attributable to the new nonlinearity:
   a clean ablation story for the prototype ticket.
3. **The gate stands.** Benchmark criteria can keep XOR-on-S⁷ and should set the
   bar well above the 75% linear ceiling (e.g. ≥ 90% test) so that clearing it
   demonstrates genuine nonlinearity rather than ceiling-grazing.
4. **Octonion depth is not entirely wasted — it's unread.** Products of right
   multiplications reach SO(8) transformations a single weight cannot; only the
   scalar readout discards this. A readout richer than `sign(re(·))` is the
   cheapest conceivable capacity gain and belongs in the nonlinearity/readout
   discussions.

## References

- J. Baez, *The Octonions*, Bull. AMS 39 (2002) — composition algebra
  identities, Cayley–Dickson.
- J. Conway & D. Smith, *On Quaternions and Octonions* (2003) — multiplication
  maps and rotations of R⁸.
- Hurwitz's theorem (1898) — normed division algebras exist only in dimensions
  1, 2, 4, 8.
