# Chirality of the error wave: correcting from the other side

Research note for the wayfinder ticket
[Chirality of the error wave: correcting from the other side](https://github.com/hirekk/v3i/issues/13),
which blocks [Error wave across a wide layer (#8)](https://github.com/hirekk/v3i/issues/8).
Every claim marked *(verified)* is checked to machine precision — and every
screen number produced — by
[error_wave_chirality_screen.py](error_wave_chirality_screen.py)
(`uv run python docs/research/error_wave_chirality_screen.py`, deterministic
seeds, ~37 s), run against the project's own `Octonion` class and the
post-#11-fix `cross_product_7d`. The batched ops of the training screen are
preflight-checked against `OctonionPerceptron` / `OctonionSequential` to 5e−16.
Claims resting on derivation are marked *(own derivation)*. Leans on the
[isometry ceiling](isometry-ceiling.md) (function class is fixed — everything
here is *training dynamics*, never capacity), the
[octonion deep dive](octonion-structure-deep-dive.md) §2 exact-identity
inventory, and the [physics context](physics-context.md) chirality framing.

## TL;DR

1. **The core fact holds exactly and side-switching is pure associator:**
   `[L_a,R_b]x = a(xb) − (ax)b = −[a,x,b]` (0.0 by definition; the `as_matrix`
   operator form to 6.3e−16), and it **vanishes identically on ℍ** (4.0e−16) —
   the clean non-associativity ablation the ticket asked for *(verified)*.
2. **"Correcting from the other side" is not one operation — it is three,
   and only one of them is the associator.** The current rule has two
   independent side choices, the *transport/frame* side (`w̄rw` vs `wrw̄`) and
   the *update/bite* side (`w·exp(τ)` vs `exp(τ)·w`); they behave completely
   differently.
3. **The bite side is pure gauge.** `w·exp(τ) = exp(wτw̄)·w` exactly in **both**
   algebras (Artin, 4.2e−16 / 3.4e−16); a single weight's left-bite and
   right-bite are the *same* S⁷ motion once the torque is transported. The
   training screen confirms it: a torque-transported left-bite reproduces the
   right rule to 4.7e−15 (𝕆) / 9.4e−16 (ℍ) over 3200 layer-steps *(verified)*.
   **There is no non-associativity to exploit in the update side of a single
   weight.**
4. **The transport side is NOT the associator — it is a direction sign.**
   `|w̄rw − wrw̄|` has median **0.84 on ℍ too** (not just 𝕆): it is a rotation
   vs its inverse, real on quaternions *(verified)*. Flipping it points the
   correction the wrong way. The naive **left** variant (which flips it)
   therefore **breaks the ℍ-control** — its weight diverges from the right rule
   to 1.41 on ℍ — and it trains erratically: catastrophic on the single unit
   (0.389, ran fully off-plane) and, alternated, actively anti-learns
   (binary-1d depth-2 **0.045**, geodesic loss > π/2), yet lands 1.000 on the
   depth-2/3 chains. A different dynamical system, not an ablation *(verified)*.
5. **The genuine, ℍ-vanishing chirality is a credit-assignment effect that
   only exists at depth/width.** It shows up as the output-rotation gap
   `x(wu) − (xw)u = −[x,w,u]` (median 0.12·|τ|/0.1, **zero on ℍ** to 5e−16) and
   as the chain **slot-shift**: right-updating layer `k` equals left-updating
   layer `k+1`, **exactly on ℍ** (5e−16), with an associator gap (median 0.12
   at |τ|=0.1) on 𝕆 *(verified)*. Never present in an isolated single-weight
   update; entirely about how a correction threads through the *input* and
   *other layers*.
6. **Chirality buys nothing on XOR — as it must.** Every variant sits at the
   linear ceiling on XOR (accuracy 0.42–0.60, geodesic loss ≈ π/2 for all four
   sides, both algebras): the function class is fixed by the forward map
   (isometry ceiling), so no correction side can lift it *(verified)*.
7. **Recommendation for #8:** keep the forward-consistent **right** chirality
   as the canonical convention. The side is not a wide-layer design knob — the
   bite side is gauge, the transport side is a sign you must get right, and the
   only non-associative payload is an associator-sized *cross-unit /
   cross-layer* term. #8 should spend its chirality budget on **credit
   distribution among units on the right**, use per-unit **kappa** as its
   achiral (4.4e−16) discount, and adopt the slot-shift equality as the wide
   rule's machine-precision ℍ-correctness test.

---

## 0. Conventions

Basis `e0=1, e1…e7` imaginary; associator `[x,y,z] = (xy)z − x(yz)`; unit
weights throughout. The rule under study is `OctonionPerceptron.correct`:
transport the global error into the local weight frame by conjugation
(`r_loc = w̄·r·w`), torque `τ = Im(w·r_loc)` via the #11-fixed 7D cross product,
kappa-scaled geodesic bite `w ← normalize(w·exp(κτ))`, subtract the absorbed
(torque) component, transport the residual out. "Right" is this status quo;
the forward pass is `y = x·w`, a right action. Data: `make_data` seed 42,
noise 0.1, 800/200; binary-1d lies in `span{1,e1}`, XOR in `span{1,e1,e2}`.
Model seeds 0–4, 10 epochs, lr 0.1. The ℍ-control restricts weights (hence
every product) to `span{1,e1,e2,e3}`.

The two independent side choices, named once and used throughout:

| choice | right (status quo) | left | vanishes on ℍ? |
|---|---|---|---|
| **transport / frame** | `r_loc = w̄·r·w` | `r_loc = w·r·w̄` | **no** — rotation vs inverse |
| **update / bite** | `w ← w·exp(κτ)` | `w ← exp(κτ)·w` | pure gauge (same in both) |

## 1. Operator identities (area 1)

### 1.1 The core fact, and what it is *not*

`[L_a,R_b]x = a(xb) − (ax)b` is, by the definition of the associator, exactly
`−[a,x,b]` — an algebraic identity (screen: 0.0e+00), reproduced by the class's
own `as_matrix("left")`/`as_matrix("right")` operators to 6.3e−16, and
identically zero on the quaternion subalgebra (4.0e−16) *(verified)*. So *any*
difference between applying a correction as a left- vs right-multiplication of
the **same operands** is pure associator, ≡ 0 on ℍ. The subtlety this note
exists to resolve: the rule's two side choices are **not** of that form.

### 1.2 The bite side is pure gauge (Artin), in both algebras

For a single weight, left-bite and right-bite are the same motion —
*(own derivation, verified)*:

```
w·exp(τ) = exp(w·τ·w̄)·w      (max dev  O 4.2e−16,  H 3.4e−16)
```

Because `w` and `τ` generate an associative subalgebra (Artin, deep dive §2),
conjugation commutes with `exp`, and `w·exp(τ)·w̄·w = w·exp(τ)`. Reading:
**a right-update by `τ` equals a left-update by the transported generator
`wτw̄`.** The bite side carries *no* information a single weight can exploit —
in either algebra. The training screen makes this operational: rerunning the
whole depth-2 wave with the bite flipped to the left but the torque transported
(`exp(wτw̄)·w`) reproduces the right rule to **4.7e−15 (𝕆) / 9.4e−16 (ℍ)** over
3200 layer-steps (`gauge_check`). This is the honest sense in which "left and
right coincide" — and it needs no ablation, because it holds on 𝕆 too.

### 1.3 The transport side is a direction, not the associator

Flipping the sandwich is a genuinely different map — and it is different **on
ℍ**, so it is not the non-associativity the ticket is after *(verified)*:

```
|w̄·r·w − w·r·w̄|      median  O 0.842 / H 0.812,   max ≈ 1.995 (both)
```

`r ↦ w̄rw` and `r ↦ wrw̄` are a rotation and its inverse; they agree only when
`r` lies on `w`'s axis. Each bracketing is itself unambiguous (Artin,
`(w̄r)w = w̄(rw)` to 3.9e−16), so this is not a parenthesization artifact — it is
chirality in the parity sense of [physics-context.md](physics-context.md), and
it is real on quaternions. **Flipping the transport side corrects in the wrong
frame direction; §3 shows it destabilizes even the associative control.**

### 1.4 The genuine chirality: the output-rotation gap (associator, ≡ 0 on ℍ)

Where *does* `[L_a,R_b] = −assoc` bite in the rule? In the gap between
right-biting the *weight* and right-rotating the *output*. Right-updating
`w ← w·u` sends the output to `x·(w·u)`; the intended `y·u = (x·w)·u`; the
difference is exactly the associator *(verified)*:

```
x(wu) − (xw)u = −[x,w,u]      O median 0.119 (|τ|=0.1), 0.012 (|τ|=0.01);  H 5.0e−16
```

It scales with the step (linear in `|τ|`) and vanishes on ℍ to machine
precision. This is a **three-element** effect — signal `x`, weight `w`, rotor
`u` — so it cannot appear in an isolated single-weight update; it needs the
input `x` (§2 shows the multi-layer version). The associator's typical size,
the ceiling on any chirality payload, is `|a(xb) − (ax)b|` median **1.10**, max
1.90 for random unit triples (§3.3 of the deep dive: sharp bound 2). And the
kappa discount is **achiral**: `||[q,w,r]| − |[r,w,q]|| = 0` to 4.4e−16 (the
associator is alternating), so swapping the chirality slots leaves the
non-associativity gate untouched *(verified)*.

**Verdict.** "The other side" decomposes cleanly: *bite* = gauge (nothing to
exploit, either algebra), *transport* = a direction sign (real on ℍ, get it
right), *the associator* = a three-element credit term that lives only where a
correction meets the input or another layer. Only the third is the
non-associative chirality, and it is bounded by the associator, ≡ 0 on ℍ.

## 2. Chain credit assignment (area 2)

For a depth-`n` chain the readout weight is `v = conj(w₀(w₁(···w_{n−1})))`
(isometry-ceiling §2: `re(output) = ⟨x, v̄⟩`). How does `v` move when we update
layer `k`?

**Readout-weight motion is an exact isometry, and on ℍ it is an adjoint
rotation** *(own derivation, verified)*. Updating layer `k` by `exp(τ)` moves
`v` along a geodesic with `|motion|/|τ| = 1.000` (min = max = 1.000 over all
depths/sides/algebras). On ℍ the endpoint is

```
v ↦ exp(−Ad_Ā τ)·v ,   A = conj(downstream product) ,
```

with the downstream partition starting at `k+1` for a right-update and at `k`
for a left-update — verified to `H gap ≤ 6e−16`. So **right-update of layer `k`
and left-update of layer `k` rotate the readout weight by adjoints that differ
by conjugation-by-`w_k`**: the side chooses which layer's rotation is "absorbed"
into the transported generator. On 𝕆 the exact adjoint formula breaks by the
associator — the O endpoint departs from the ℍ prediction by the
**credit-misdirection gap**, median 0.057–0.135 across layers.

**The slot-shift theorem** makes the chirality explicit and gives the ticket's
mandated chain-level ℍ-control *(own derivation, verified)*:

```
right-update layer k  ==  left-update layer k+1        (exact on ℍ)
      x·(…(w_k·u)…)   =   x·(…w_k·(u·w_{k+1})…)
depth-2/3  H max dev 5.0e−16 / 5.7e−16 ;  O gap median 0.120 / 0.121 (|τ|=0.1)
```

On ℍ, credit can be slid across a layer boundary by switching the side of the
adjacent layer — the two are indistinguishable. On 𝕆 the slide leaks exactly
the associator. This is the deep dive's "two different weights get no exact
rearrangement" (§2) seen from the correction side, and it is the *only* place a
side choice carries non-associative content.

**Verdict.** Chirality in a chain is a statement about **credit assignment
between adjacent layers**, exact-on-ℍ and associator-bounded on 𝕆 — not about
capacity (the function class is fixed) and not about any single layer's update.

## 3. Empirical screen (area 3)

Four variants of the forward error wave — **right** (status quo), **left**
(both sides flipped), **alt** (side alternates per layer in a chain, per step
for the single unit), **fr-cl** (forward stays right, correct with right
transport but a left bite) — on binary-1d and XOR, single unit and depth-2/3
chains, seeds 0–4, both algebras. Metrics: final accuracy, geodesic loss to
the ±identity pole (`arccos(clip(y·re, ±1))`, lower = more confident-correct),
and off-plane weight-slice magnitude (the #11 confinement diagnostic: `e2..e7`
for binary-1d, `e3..e7` for XOR).

### 3.1 Findings

**The ℍ-control is exactly what §1 predicts.** kappa ≡ 1 on every ℍ run
(`max|κ−1| = 5.6e−16`); ℍ weights never leave the quaternion slice
(`max|w[e4..e7]| = 0.0`). But **left does not coincide with right on ℍ** — its
weight gap to the right rule grows to **1.41** — precisely because it flips the
transport side (§1.3), which is real on ℍ. **fr-cl** (right transport, left
bite, un-transported torque) stays close: a small gauge slippage — the §1.2
identity applied without transporting the torque, so it scales with `w`'s own
rotation angle, tiny at the near-identity initialization — gap
2.55e−3 → 2.26e−2 (𝕆) / 3.20e−4 → 2.95e−2 (ℍ) over 10 epochs. This is the honest
resolution of the ticket's "coincide on ℍ": **the gauge (bite-only) flip
coincides on both algebras; the transport flip coincides on neither.**

**Naive left-flipping is erratic, not a controlled ablation** (report, do not
tune away). On the single unit it is unstable — binary-1d depth-1 collapses to
**0.389 ± 0.475** and runs fully off-plane (off-plane 1.0, geodesic loss π/2:
output driven orthogonal to the target). Alternating sides is worse: binary-1d
depth-2 **alt** reaches accuracy **0.045** with geodesic loss **1.79 > π/2** —
the mixed transport frames drive the output to the *wrong* pole
(confidently-wrong anti-learning). Yet naive **left** on the depth-2/3 chains
lands 1.000 with unusually low geodesic loss (0.25–0.42, tighter than right's
1.05–1.32): a different, strongly-polarizing fixed point that happens to descend
on a linearly separable task. Erratic across depth by construction — it is a
different rule, not the right rule ablated.

**Chirality does not touch XOR.** All four variants sit at the linear ceiling:
accuracy 0.42–0.60, geodesic loss ≈ π/2 (1.55–1.62) for every side and both
algebras — the wave does not even reach the 75% corner ceiling, consistent with
the isometry ceiling (function class fixed) and the s7-catalogue's ≤1%
findability. No side choice can help; this is the negative control that the
capacity framing is wrong.

**Right and fr-cl are the sane variants.** Right: binary-1d 0.92 → 1.00 across
depths 1→3, confined (off-plane 0.0). fr-cl tracks it (0.90/0.99/1.00), as the
gauge slippage predicts. The octonion pass out-trains the ℍ pass on the
single unit (0.924 vs 0.825) — a training-dynamics difference at fixed function
class, not capacity.

### 3.2 Final metrics (epoch 10; mean ± std over seeds 0–4)

Selected rows; full table in the screen output.

| dataset | d | variant | alg | train | test | geo | off-plane |
|---|---|---|---|---|---|---|---|
| binary-1d | 1 | right | O | 0.924 | 0.927 | 1.423 | 0.000 |
| binary-1d | 1 | right | H | 0.825 | 0.832 | 1.476 | 0.000 |
| binary-1d | 1 | left  | O | **0.389** | 0.389 | 1.571 | **1.000** |
| binary-1d | 1 | fr-cl | O | 0.898 | 0.903 | 1.441 | 0.000 |
| binary-1d | 2 | alt   | O | **0.045** | 0.036 | **1.794** | 0.000 |
| binary-1d | 2 | left  | O | 1.000 | 1.000 | 0.285 | 0.000 |
| binary-1d | 3 | right | O | 1.000 | 1.000 | 1.046 | 0.000 |
| binary-1d | 3 | fr-cl | O | 1.000 | 1.000 | 1.117 | 0.000 |
| binary-xor | 1 | right | O | 0.543 | 0.527 | 1.571 | 1.000 |
| binary-xor | 3 | fr-cl | O | 0.594 | 0.616 | 1.563 | 0.936 |
| binary-xor | 3 | right | H | 0.454 | 0.426 | 1.564 | 0.806 |

H-control panel: `max|κ−1| = 5.6e−16`; `max|w[e4..e7]| = 0.0`; right-vs-left ℍ
weight gap → 1.41; gauge-transported left-bite ≡ right to 9.4e−16 (ℍ).

## 4. Interaction with the #6 race set (area 4)

Screen part D checks each racer's equivariance under a right-acting correction,
octonion then ℍ-control. One recommendation per mechanism.

**Kappa-gated slerp** `slerp(x·w₁, x·w₂, κ)`. The combiner is equivariant under
*any* unit right-multiplication — `slerp(aq, bq, t) = slerp(a,b,t)·q` to
1.2e−15 *(verified)* — so it commutes with a right-acting correction in both
algebras, and its κ gate is achiral (§1.4). **Correct on the right** and treat κ
as a locally-constant scalar; a side flip adds nothing and would only invoke the
§1.3 direction error. *(This is the front-runner's natural, zero-friction
chirality.)*

**Branch products** `(x·w₁)·(x·w₂)`. The **exterior-right** weight `w₂`
right-rotates the output *exactly on ℍ* (gap 4.7e−16; associator gap 0.118 on
𝕆), so `w₂` takes a clean **right** correction. The **interior** weight `w₁`
corresponds to no exterior rotation even on ℍ (gap 0.143) — it is the hard
credit slot. **Correct both on the right, but split the credit asymmetrically:**
`w₂` gets the output-aligned share, `w₁` a frame-local share. This is #8's
distribution question in miniature; do not mirror `w₁`, transport into its
frame.

**Triple cross product** `normalize(X₃(x·w₁, x·w₂, x·w₃))`. Uniformly
right-equivariant on ℍ — `X₃(uq,vq,wq) = X₃(u,v,w)·q` to 4.5e−16 — with a
**pure-associator** gap on 𝕆 (median 0.554) *(verified)*: the same clean
non-associativity ablation the combiner already carries in the forward pass.
**Correct on the right;** the chirality payload is exactly its existing
associator content, so a side flip double-counts the same effect. *(Cleanest
match: right-forward, right-correct, ℍ-ablatable end to end.)*

**Commutator rotor** `normalize([x·w₁, x·w₂])·w₃`. Its side-sensitivity is a
non-*commutativity* effect, not the associator: sandwich-transport of the
commutator is exact on ℍ (9.7e−16) but fails on 𝕆 by median **1.56** because
conjugation is not an octonion automorphism — a scrambling that *survives* the
ℍ ablation. The trailing rotor `w₃` is an exterior-right slot (like branch's
`w₂`): **correct `w₃` on the right.** For this racer the chirality axis is a red
herring — its interesting side-dependence is non-commutativity, orthogonal to
this ticket — and its conjugation-transport is the weakest link, so keep the
transport strictly right-framed and do not alternate.

**Across all four: right is natural.** Every racer's combiner commutes with a
right correction on ℍ (slerp, X₃ on both), or has a clean exterior-right slot
(branch `w₂`, rotor `w₃`). None benefits from a transport flip; the only
genuine chirality (X₃, slerp) is already the associator each carries.

## 5. Recommendation for #8 (Error wave across a wide layer)

**Keep the forward-consistent right chirality as the canonical convention, and
do not treat "which side" as a wide-layer design parameter.** The evidence:

1. *The bite side is gauge* (§1.2) — left/right updates of one unit are the same
   S⁷ motion in both algebras; a wide layer gains no expressivity or dynamics
   from per-unit side choices.
2. *The transport side is a direction sign* (§1.3, §3.1) — flipping it corrupts
   even the associative control (ℍ gap → 1.41; alt anti-learns to 0.045).
   Right transport is not a preference, it is correctness.
3. *The only non-associative chirality is an associator-sized cross-unit /
   cross-layer credit term* (§1.4, §2), ≡ 0 on ℍ, bounded by the associator
   (≤ 2; median ~0.12 at lr 0.1). This is what #8 should model — as an explicit
   credit-assignment term among the `W` units, all corrected on the right.

Concrete guidance for the five #8 decisions:

- **Distribution:** distribute one incoming error to the `W` units on the
  **right** (transport each into its own frame by `w_i`-conjugation).
  Responsibility-weighting is the honest analogue of the branch-product credit
  split (§4): the unit whose output the readout actually sees takes the
  output-aligned share.
- **Debt accounting:** the residual stays exactly norm-controlled because every
  per-unit transport is an isometry (`|motion|/|τ| = 1.000`, §2); assert
  *residual norm ≤ incoming norm* as the runtime invariant.
- **Through the nonlinearity:** exact where the algebra permits — for the
  right-equivariant racers (slerp, X₃) the correction passes through the
  combiner by the §4 equivariances; approximate only at the interior/aggregation
  slots, on the right.
- **Kappa at width:** per-unit kappa, as the achiral (§1.4) non-associativity
  discount; a shared kappa loses the per-unit credit signal.
- **Sanity property / ℍ-correctness test:** the **slot-shift equality** (§2)
  must hold to machine precision when weights are quaternionic — right-update of
  a unit ≡ left-update of the downstream aggregation on ℍ. Any wide rule that
  violates it on the ℍ-control has smuggled in a non-associativity-irrelevant
  transport-direction error, exactly the §3 pathology.

The one-line answer to the ticket's question — *can error propagation exploit
non-associativity by acting from the other side?* — is: **for a single unit,
no (the side is gauge); the non-associative content of chirality exists only in
multi-unit / multi-layer credit assignment, is bounded by the associator, and
vanishes on ℍ.** That is what #8 inherits.

## References

- R. D. Schafer, *An Introduction to Nonassociative Algebras*, Academic Press
  1966 / Dover 1995 — alternative algebras, **Artin's theorem** (the gauge
  identity §1.2), associator calculus.
- R. Moufang, *Zur Struktur von Alternativkörpern*, Math. Ann. 110 (1935) — the
  exact same-`w` rearrangements behind the transport bookkeeping.
- J. C. Baez, *The Octonions*, Bull. Amer. Math. Soc. 39 (2002),
  arXiv:math/0105155 — `[L_a,R_b] = −` associator, G₂, triality; the
  left/right/bimultiplication triple this note's chirality permutes.
- R. H. Bruck, *A Survey of Binary Systems*, Springer 1958 — Moufang loops and
  pseudo-automorphisms (the `w̄³` companion behind the credit-slide of §2).
- Sibling notes: [isometry-ceiling.md](isometry-ceiling.md) (fixed function
  class — the framing discipline), [octonion-structure-deep-dive.md](octonion-structure-deep-dive.md)
  §2 (exact-identity inventory), [s7-combiner-catalogue.md](s7-combiner-catalogue.md)
  §4 (the #6 race set), [physics-context.md](physics-context.md) (chirality as
  parity). Hurwitz's theorem (1898) underwrites the isometry of every transport
  step.

All operator identities, the gauge and slot-shift theorems, the transport-flip
size, and every screen number are established by
[error_wave_chirality_screen.py](error_wave_chirality_screen.py) against the
project's `Octonion` class; claims not attributable to the cited sources are
marked *own derivation* in-text.
