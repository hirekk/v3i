# Geometric catalogue of nonlinear maps S⁷×S⁷→S⁷

Research note for the wayfinder ticket
[Geometric catalogue of nonlinear maps S⁷×S⁷→S⁷](https://github.com/hirekk/v3i/issues/12),
which blocks [Where does nonlinearity come from?](https://github.com/hirekk/v3i/issues/6).
Every claim marked *(verified)* is checked to machine precision — and every
screen number produced — by
[s7_combiner_screen.py](s7_combiner_screen.py)
(`uv run python docs/research/s7_combiner_screen.py`, deterministic seeds,
~8 s). This note extends the shortlist of the
[literature note](hypercomplex-backprop-free-literature.md) §4 and leans on the
[isometry ceiling](isometry-ceiling.md) and the
[octonion deep dive](octonion-structure-deep-dive.md) throughout; their content
is linked, not repeated. All batched algebra in the screen is derived from the
project's `Octonion` class via its structure tensor and cross-checked against
`__mul__` and the (fixed, #11) `cross_product_7d` *(verified, ≤1.8e−15)*.

## TL;DR

1. **The scrambling criterion is now formal** (§1): a combiner can beat the 75%
   XOR ceiling only if its cleared readout escapes *projective-affineness* —
   grade ≥ S2 on the S0–S3 scale below. Two families die by theorem before any
   experiment: additive associator channels (sign-readout-invisible,
   *(verified)*) and **every one-step Möbius/conformal combiner** (the
   composition-algebra polarization `⟨xa, xb⟩ = |x|²⟨a,b⟩` makes their readout
   affine — *own derivation, verified*).
2. **The screen** (17 candidates × 2000 random unit-weight draws × two
   algebras, §3) confirms the calibration rows behave (control ≈ 0.72, the two
   shortlist leaders 0.87–0.90 best train) and produces one genuinely new
   leader-class candidate: the **triple cross product**
   `normalize(u×v×w)` of three branch images — 0.90/0.90 with the best hit
   rate in the table, *and* an exact collapse to a single current-architecture
   linear layer on ℍ (*own derivation, verified*): the same clean
   non-associativity ablation that made kappa-slerp the front-runner.
3. **Readout visibility is the graveyard.** Raw cross product, commutator, and
   associator outputs are pure imaginary — `sign(re(·))` sees literally zero.
   The "rotor" trick (normalize, then multiply by one more weight) revives
   commutator and associator forms (0.93, hit 0.008 / 0.93, hit 0.005), at the
   price of a normalization singularity on the vanishing locus.
4. **The quaternion control now has teeth** (screen pass 2): kappa-slerp,
   ψ-gated slerp, and triple cross all collapse to *provably linear* maps on ℍ
   (screen lands them at the 0.72 control level exactly); the associator rotor
   collapses to a *constant* (0.50); branch products, commutator rotor, Hopf
   twist survive at 0.87–0.92 — their nonlinearity owes nothing specific to
   non-associativity.
5. **Amendments to the #6 shortlist** (§4): add the triple cross product as a
   third full entrant; keep the associator rotor as the sharpest
   non-associativity *instrument* rather than an architecture; demote the Hopf
   fiber twist to a designated control arm (its readout is provably identical
   under the ℍ ablation on this embedding — elegant, but algebra-agnostic);
   kill Jordan products, raw commutators/cross products, additive associator
   channels, φ-gated slerp, Möbius combiners, and the physics-flow variants,
   each with a one-line obituary. Recommendation: the head-to-head becomes
   **three-way** (kappa-gated slerp, branch products, triple cross product).

## 0. Setup and conventions

A layer's combiner is a map `F_W : S⁷ → S⁷` built from branch images
`x·w_i` (unit weights `w_i`), read out by `sign(re(F_W(x)))`. The screen's
dataset is the embedded XOR of the prior notes: `generate_binary_xor`, seed 42,
800 train / 200 test, noise 0.1, `to_s7_from_2d` — the data lies in
`span{1, e1, e2}`, inside the quaternion subalgebra `span{1, e1, e2, e3}` of
the Cayley–Dickson doubling. The quaternion-control pass therefore only
restricts the *weights* to that subalgebra; every product then stays
quaternionic. Associator convention `[x,y,z] = (xy)z − x(yz)`; `Im`/`×` as in
the [deep dive](octonion-structure-deep-dive.md) §0/§5.2.

The rubric, per candidate:
**(a)** exactly on-sphere or needs normalization (singularities?);
**(b)** degree of nonlinearity in `x` (the cleared readout's degree);
**(c)** readout visibility under `sign(re(·))`;
**(d)** quaternion-control behavior — collapses to linear (clean ablation),
collapses to constant, survives (weak ablation), or is structurally unchanged;
**(e)** error-wave transport story (exact Moufang/alternativity identities?);
**(f)** the empirical screen (§3).

## 1. "Scrambling", formalized

The literature note's §4.0 collapse says: any chain of unit multiplications,
linear aggregations, and normalizations computes `L(x)/s(x)` with `L` linear
and `s > 0`, so `sign(re(·))` realizes only homogeneous linear separators —
the 75% ceiling. Sharpening this into a usable criterion:

**Obstruction space.** Under the inverse-stereographic embedding
`E(u) = ((1−|u|²), 2u₁, 2u₂, 0, …)/(1+|u|²)`, the pullback of any *affine*
readout `x ↦ ⟨x, v⟩ + c`, after clearing the positive denominator `(1+|u|²)`,
is `(c−α)|u|² + ⟨β, u⟩ + (c+α)` — an element of
`𝒪 = span{1, u₁, u₂, |u|²}` *(own derivation)*. The isometry-ceiling note's
corner identity `g(A)+g(D) = g(B)+g(C)` holds for every `g ∈ 𝒪`, and the XOR
sign pattern violates it: **no readout whose cleared pullback stays in 𝒪 can
exceed ¾ on the corners.** This is the criterion a combiner must beat.

**Grades.** Write `r_W(x) = re(F_W(x))` and clear all positive scalar factors
(norms, `1+|u|²` powers). Then:

| grade | cleared readout | XOR verdict |
|---|---|---|
| **S0** projectively linear | `ℓ(x)`, `ℓ` linear | dead — ceiling (lit. note §4.0) |
| **S1** projectively affine | `ℓ(x) + c` | dead — pullback stays in 𝒪 (above) |
| **S2** quadratic-visible | contains nondegenerate quadratic forms `x^T M x` | capable — quartic pullbacks include products of two affine forms, which realize XOR |
| **S3** gated / higher | degree ≥ 3 forms, or non-monotone gate functions of linear features (slerp gates) | capable in principle; capability ≠ findability — must be screened |

Two sharper notions used below. *Readout degree*: the polynomial degree of the
cleared readout — the primary rank within S2/S3. *Fiber mixing*: whether `F`
moves points along Hopf fibers over a fixed base (`h∘F = h∘(linear image)`,
`h` the quaternionic Hopf map S⁷→S⁴) — the Hopf twist below is *pure* fiber
mixing *(verified: base preserved to 6.7e−16)*. Fiber mixing is neither
necessary nor sufficient for XOR; it classifies *where* on the sphere the
nonlinearity acts. Ergodicity-flavored measures (equidistribution of the
pushed-forward data measure) are the natural next rung but are overkill for
ranking single combiners; not formalized here.

**The screen's role.** Grade ≥ S2 is *necessary*. The screen (§3) measures
*findability*: with 2000 random unit-weight draws, can `sign(re(·))` beat the
ceiling on the actual noisy dataset? A grade-S3 candidate that never beats
~0.75 in 2000 draws is presumptively dead unless a specific argument says its
nonlinearity needs trained weights (said explicitly where claimed).

## 2. The catalogue

Screen numbers quoted per candidate are
`best train / test at that draw / hit-rate (fraction of draws with train ≥ 0.80)`,
octonion pass first, quaternion-control pass second; full table in §3.

### 2.1 Calibration rows

**C0. Renormalized sum** `normalize(x·w₁ + x·w₂) = x·normalize(w₁+w₂)`.
Grade S0 by distributivity (lit. note §4.0). Screen: **0.724**/0.690/0.000 —
the empirical face of the ceiling; every number below is read against this.

**C1. Kappa-gated slerp** (shortlist leader):
`y = slerp(x·w₁, x·w₂, κ(x))`, `κ = 1 − min(|[x,w₁,w₂]|, 1)` (the perceptron's
`_compute_kappa` convention). On-sphere by construction, no singularities
(slerp's small-angle branch covers coincident endpoints). Grade S3: the
readout is a `κ`-weighted non-monotone mix of two linear features. ℍ control:
`[x,w₁,w₂] ≡ 0` ⇒ `κ ≡ 1` ⇒ `y = x·w₂` — **exact collapse to the current
linear layer** *(verified: max |assoc| 3.2e−16 on ℍ)*. Transport: two exactly
invertible branch isometries plus a scalar gate; no cross-weight Moufang
rearrangement exists (deep dive §2), but none is needed if the wave treats
`κ` as locally constant. Note the gate is *non-smooth exactly on its ablation
locus* (`|·|` has a kink at 0 — the associative locus). Screen:
**0.870**/0.840/0.001 ‖ ℍ **0.722** — capable, clean ablation confirmed.

**C2. Branch product** (shortlist leader): `y = (x·w₁)·(x·w₂)`. Exactly
on-sphere (norm multiplicativity), no normalization anywhere. Grade S2:
`re(y) = ⟨x·w₁, conj(x·w₂)⟩` is a genuine quadratic form in `x`. ℍ control:
*survives* — `(x·w₁)(x·w₂)` is quadratic on any algebra (on ℂ it is already
`x²w₁w₂`); its nonlinearity is multiplicativity itself, not
non-associativity. Transport: products of exactly invertible isometries;
credit-splitting between `w₁, w₂` is the open design question (lit. note
row C). Screen: **0.895**/0.920/0.003 ‖ ℍ **0.876**/0.875/0.002 — capable in
both algebras; the weakest ablation story among the leaders.

### 2.2 Family 1 — algebra-native bilinear maps

**A1. Jordan branch product** `normalize(½((x·w₁)(x·w₂) + (x·w₂)(x·w₁)))`.
Since `re(ab) = re(ba)` (adjoint identity), the Jordan symmetrization has
**exactly the branch product's readout**: `re` of the two differ by 0.0
*(verified)*. It needs normalization (the symmetrized product can leave S⁷ and
can vanish when the images anticommute) and buys nothing at `sign(re(·))`.
Screen: 0.900/0.940/0.002 ‖ ℍ 0.884 — the numbers match branch products up to
draw noise, as the identity predicts. **Killed: dominated.**

**A2. Commutator, raw** `[x·w₁, x·w₂]`. Pure imaginary (deep dive §3.1), so
`re ≡ 0`: the screen's 0.504/0.515 is selection noise on an exactly-zero
signal. Same fate for any combiner ending in a raw cross product
(`[a,b] = 2·Im(a)×Im(b)` up to real-part cross terms) or raw associator.
**Killed: invisible.** (Rubric point (a) of the ticket, now with numbers.)

**A3. Commutator rotor** `normalize([x·w₁, x·w₂])·w₃`. The rotor
multiplication moves the pure-imaginary commutator off the imaginary
hyperplane: `re(y) = ⟨normalize([a,b]), w̄₃⟩` — grade S2 (quadratic over a
positive norm). Needs normalization; singular where the images commute
(imaginary parts parallel — measure zero, but approachable, and the
normalize's differential blows up there). ℍ control: quaternion commutators
are generically nonzero — *survives* (0.921): this is non-*commutativity*
made visible, not non-associativity. Transport: the rotor `w₃` is exactly
invertible; the normalize step is the weak link. Screen:
**0.931**/0.935/**0.008** ‖ ℍ 0.921/0.945/0.003 — the best simple
algebra-native screen in the table, but scientifically the wrong tool for the
non-associativity question.

**A4. G₂-twisted branch product** `(x·w₁)·σ(x·w₂)`,
`σ = conjugation by exp((π/3)n̂) ∈ G₂` (an automorphism since `g⁶ = 1`;
*(verified: σ(ab) = σ(a)σ(b) to 5.3e−15)*). Same mechanism class as C2 —
a product of two linear images — with a strictly wider reachable family of
quadratic forms. Exactly on-sphere. Screen: **0.960**/0.965/0.002 ‖ ℍ
0.917/0.970/0.005 — the best single number in the table, but the hit rate
matches C2: the tail draw is luck, not a new mechanism. **Verdict: not a
separate candidate — a free capacity knob on branch products** (one extra
7-parameter axis) if C2 wins the head-to-head.

**A5. Additive associator channel**
`normalize(x·w₀ + [x·w₁, c, x·w₂])` — the deep dive's row-2 candidate
(`§3.4`), taken at its word. **Theorem (own derivation, verified):** the
associator contributes *zero* to the numerator's real part and only a positive
scalar to the denominator, so
`sign(re(y)) = sign(⟨x, w̄₀⟩)` **exactly** — grade S0, ceiling-bound. The
screen agrees on all 256 preflight points and lands at 0.721/0.760/0.000,
octonion and ℍ passes identical. This *proves* the deep dive's warning ("never
feeding `re(·)` directly") at sign level: the quadratic associator channel has
a hard dependency on the richer-readout redesign, and under the current
readout it is **killed: sign-invisible in additive position.**

**A6. Associator rotor** `normalize([x·w₁, c, x·w₂])·w₄` (`c` a fixed unit
weight — the multi-image form, since alternativity kills every repeated-slot
insertion, deep dive §3.4). Grade S2. Needs normalization; the associator can
vanish on associative loci (the deep dive's dead-associator risk — here a hard
singularity). ℍ control: `[·,·,·] ≡ 0` ⇒ output is the *constant* `w₄`
(via the normalize fallback) — screen 0.504 = chance. That is the **maximal
𝕆-vs-ℍ gap in the catalogue** (0.927 vs 0.504): everything this candidate does
is non-associativity. Transport: worst of the table — the wave must pass
through a normalize whose differential is unbounded near the associative
locus. Screen: **0.927**/0.920/0.005 ‖ ℍ 0.504. **Verdict: not an
architecture — the sharpest available *instrument* for the ablation arm of
#6** (see §4).

### 2.3 Family 2 — G₂/calibration structures

**B1. φ-gated slerp** `slerp(x·w₁, x·w₂, ½(1+φ̂))`, where
`φ̂ = φ(Îm x, Îm(x·w₁), Îm(x·w₂))` and `φ(u,v,w) = ⟨u, vw⟩ = ⟨u×v, w⟩` on
Im 𝕆 is the associative 3-form — Harvey–Lawson's associative calibration
(their Eq. (1.1); comass 1 by their Thm. 1.4, equality exactly on associative
3-planes), so `t = ½(1+φ̂) ∈ [0,1]` needs no clipping in principle *(verified:
max |φ̂| 0.9606 over 20k unit triples)*. Grade S3 (cubic gate). Note how close
this gate is to kappa's: H–L Thm. 1.6 gives the exact relation
`⟨u,vw⟩² + ¼|[u,v,w]|² = |u∧v∧w|²` on Im 𝕆 — the φ-gate and the
associator-magnitude gate are Pythagorean complements relative to the volume
of the argument triple. ℍ control: φ restricted to imaginary quaternions is
the 3D determinant — nonzero, so the gate *survives* (weak ablation). Screen:
**0.729**/0.680/0.000 ‖ ℍ 0.772/0.840/0.000 — **screen-dead on octonions**:
in 2000 draws the cubic gate never carves XOR. I do *not* claim trained
weights rescue it: within its own family it is dominated by C1 and B2 (both
of which also own the cleaner ablation). **Killed: dominated within the
gated-slerp family; screen-dead.**

**B2. ψ-gated slerp (coassociative gate)**
`slerp(x·w₁, x·w₂, ½(1+ψ̂))`, with
`ψ̂ = ½⟨Îm x, [Îm(x·w₁), Îm(x·w₂), Îm(x·w₃)]⟩` — exactly Harvey–Lawson's
coassociative calibration `ψ(x,y,z,w) = ½⟨x, [y,z,w]⟩` (their Def. 1.11;
`ψ = *φ`, comass 1) applied to unit imaginary vectors. (Beware the rival
convention `½⟨[x,y,z], w⟩` seen elsewhere: ψ is alternating, so it differs by
a sign; this note and the script use H–L's.) Here `|ψ̂| ≤ 1` also follows
directly from the sharp associator bound `|[u,v,w]| ≤ 2`, deep dive §3.3
*(verified: max |ψ̂| 0.9082)*. Grade S3, quartic gate. ℍ control: the associator dies, so
`t ≡ ½`, and `slerp(a, b, ½) = normalize(a+b)` *(verified, 4.4e−16)* — the
candidate **collapses exactly to the renormalized sum**, i.e. to a provably
linear layer. Same clean ablation as kappa — with a *polynomial* gate, smooth
everywhere, where κ's `|·|` is kinked precisely on the ablation locus.
Transport: as C1. Screen: **0.815**/0.775/0.001 ‖ ℍ **0.723** — capable but
weaker than C1 under random search. **Verdict: catalogue-keep as the smooth
gate variant of C1; test only if κ's kink proves to be a training pathology.**

**B3. Triple cross product (Cayley combiner)** — *the new entrant.*

```
y = normalize(X₃(x·w₁, x·w₂, x·w₃)),   X₃(u,v,w) = ½(u(v̄w) − w(v̄u))
```

`X₃` is the 3-fold vector cross product of ℝ⁸ (Brown–Gray: 3-fold cross
products exist only in dims 4 and 8; Harvey–Lawson build the Cayley 4-form
from it, `Φ(u,v,w,z) = ⟨X₃-type product, z⟩`). Its screen-verified properties
*(all verified)*: `|X₃(u,v,w)| = vol(u,v,w)` — the parallelepiped volume, to
7.2e−16 — so on unit *orthogonal* triples it is exactly on-sphere;
antisymmetric in the outer slots (0.0); output orthogonal to all three
arguments (≤1.8e−16). Normalization is needed only against the degenerate
locus (linearly dependent images — measure zero). Grade S3, cubic readout.

ℍ control — **the collapse theorem** *(own derivation, verified to 2.8e−16)*:
for unit quaternionic `x` and weights `a,b,c`, associativity gives
`(xa)(\overline{xb})(xc) = xa·b̄x̄·xc = x(ab̄c)`, so

```
X₃(x·a, x·b, x·c) = x · ½(ab̄c − cb̄a)
```

— the whole combiner is a *single current-architecture linear layer* on ℍ.
The screen lands it at 0.724, indistinguishable from the control row. The
failure of that rearrangement on 𝕆 is precisely the associator, so — like
kappa-slerp and unlike branch products — everything X₃ adds over linear is
sourced in non-associativity, and the ablation is exact, not just empirical.

Transport: trilinear with each slot a composition of multiplications and a
conjugation — exactly invertible in any single slot on the nondegenerate
locus; alternativity gives `X₃(u,v,u) `-type degeneracies analogous to the
associator's, and the normalization is benign away from dependent triples
(unlike A6, whose singular locus is the *ablation* locus — X₃'s singular locus
is merely coincident images). Screen: **0.899**/0.900/**0.009** ‖ ℍ
**0.724** — leader-class capability with the best hit rate in the table, plus
the cleanest possible ablation. **Verdict: add to the shortlist.**

**B4. Nambu/Euler-top step** `normalize(x + J(x))·w₃`,
`J = Im(x·w₁) ×₇ Im(x·w₂)` — the discrete step of a Nambu-style flow
`ẋ = ∇H₁ × ∇H₂` with image-dependent Hamiltonians (Nambu's generalized
dynamics; the 7D cross product with φ is the standard almost-Nambu structure
on Im 𝕆). The raw step is readout-dead (`J` is pure imaginary:
`sign(re(x+J)) = sign(re x)` — the A5 trap again); the trailing `w₃` revives
it. Needs normalization; `|x+J| ≥ |re x|`, which thins near the embedded unit
circle `|u| = 1`. ℍ control: 3D cross product — survives in principle;
screen finds nothing (0.740). Screen: 0.866/0.830/0.001 ‖ ℍ 0.740.
**Verdict: killed — it is the commutator rotor with extra machinery** (an
additive step and a normalize) and a weaker screen; the conserved-quantity
story it gestures at is not exact for the one-step map anyway.

### 2.4 Family 3 — fibration-based maps

**H1. Hopf fiber twist.** Split `x·w₁ = (a₁, b₁)` into Cayley–Dickson
quaternion halves and twist along the quaternionic Hopf fiber by the top half
of the second image:

```
q = unit(top(x·w₂)),   y = (a₁·q, b₁·q)
```

The Hopf map `h(a,b) = (|a|²−|b|², 2ab̄)` sends S⁷ → S⁴ with fiber
`{(aq, bq) : q ∈ Sp(1)}`; the twist moves `x·w₁` *purely along its own
fiber*: `|y| = 1` exactly and `h(y) = h(x·w₁)` exactly *(both verified,
≤6.7e−16)* — no normalization anywhere. This is the catalogue's exemplar of
**pure fiber mixing** (§1). Note the fiber action is *not* any octonion
multiplication: right-multiplication by a quaternion gives `(aq, bq̄)`, not
`(aq, bq)` — the twist genuinely leaves the `{R_v}` family. Grade S2:
`re(y) = ⟨a₁, q̄⟩` is quadratic over a positive norm. Transport: the best of
the new entrants — both halves are *quaternion* products (associative!), the
twist is exactly invertible (`·q̄`), and there is no normalize.

ℍ control — a structural surprise *(own derivation, confirmed by the screen's
identical numbers)*: for data in `span{1,e1,e2}` (b-half zero),
`top(x·w) = a_x·w[:4]` depends only on the weight's top half, and rescaling a
weight's top half never changes `sign(⟨a₁, q̄⟩)`; hence the octonion and
ℍ-restricted screens are **the same experiment** — 0.870/0.895/0.004 in both
passes, digit for digit. The twist's nonlinearity is quaternionic Hopf
geometry; octonion structure contributes nothing on this embedding.
**Verdict: keep — but as the designated *fibration control* arm, not a
leader**: if it trains as well as the octonion-native candidates, the
octonion-specific bet is not earning its keep.

**H2. Triality sandwich** `(x·w₁)·(w₃·(x·w₂))` — mixing the left- and
right-multiplication representations that triality permutes (deep dive §4).
Exactly on-sphere; grade S2. Screen: 0.914/0.915/0.002 ‖ ℍ 0.865. Same hit
rate as branch products, same mechanism (product of two linear images with an
interior fixed rotor). **Verdict: fold into C2 as a capacity knob, like A4.**

### 2.5 Family 4 — physics-inspired dynamics

**P1. Precession rotor** — rigid-body `ω×L` flavor: rotate `x` about the
data-dependent axis `n̂(x) = unit(Im(x·w₁) ×₇ Im(x·w₂))` by angle
`θ(x) = (π/2)·min(|Im(x·w₁)×₇Im(x·w₂)|, 1)`, then apply `w₃`:
`y = (q(x)·x·q̄(x))·w₃`, `q = exp(θn̂/2)`. The trailing `w₃` is mandatory:
pure sandwiches preserve `re` exactly (lit. note row E), so the bare
precession is readout-dead. Exactly on-sphere (rotation + unit
multiplication); axis singular where the cross product vanishes. ℍ control:
nonzero in principle, screen finds nothing (0.736). Screen: 0.845/0.845/0.001.
**Verdict: killed — a costumed commutator rotor** (same cross-product core,
extra exp/sandwich machinery, weaker screen). The conserved-quantity framing
does not survive discretization: one step of a precession flow conserves
nothing exactly here.

**P2. Möbius combiner** `normalize((x·w₁ + τw₃)·(x·w₂ + τw₄)⁻¹)` (τ = 0.7) —
the conformal/geodesic-translation family on S⁷.
**Kill theorem (own derivation, verified):** `re(pq⁻¹) = ⟨p,q⟩/|q|²`
*(verified, 1.7e−16)*, and the polarization of the norm form in any
composition algebra gives `⟨x·a, x·b⟩ = |x|²⟨a,b⟩` *(verified, 1.7e−16)* —
constant on S⁷. So the cleared readout

```
⟨x·w₁ + τw₃, x·w₂ + τw₄⟩ = ⟨w₁,w₂⟩ + τ⟨x·w₁, w₄⟩ + τ⟨w₃, x·w₂⟩ + τ²⟨w₃,w₄⟩
```

is **affine in `x`** — grade S1, ceiling-bound by §1, for *every* choice of
translations and every one-step Möbius form (`(ax+b)(cx+d)⁻¹` left-versions
die by the same identity). The screen concurs: 0.736/0.785/0.000 ‖ ℍ 0.733.
This also disposes of "geodesic/conformal transformations of S⁷" as a family:
one-step conformal maps cannot scramble the readout, despite being the
textbook "geometrically elegant" candidate. **Killed: provably affine.**

## 3. The empirical screen

Protocol: for each candidate, 2000 deterministic random draws of the unit
weights it needs (seeded per candidate); readout `sign(re(F(x)))` with
orientation chosen on train; columns are best train accuracy, test accuracy of
that draw, and the fraction of draws reaching train ≥ 0.80. Linear ceiling for
reference: ≈ 0.75 (0.72–0.75 empirically on this dataset, isometry-ceiling
§3). This measures *findability by random search*, not trainability.

| candidate | best train | test @ best | hit ≥ 0.80 | ℍ best train | ℍ test | ℍ hit |
|---|---|---|---|---|---|---|
| renormalized sum (control) | 0.724 | 0.690 | 0.000 | 0.722 | 0.680 | 0.000 |
| kappa-gated slerp (leader) | 0.870 | 0.840 | 0.001 | 0.722 | 0.750 | 0.000 |
| branch product (leader) | 0.895 | 0.920 | 0.003 | 0.876 | 0.875 | 0.002 |
| Jordan branch product | 0.900 | 0.940 | 0.002 | 0.884 | 0.935 | 0.002 |
| commutator raw | 0.504 | 0.515 | 0.000 | 0.504 | 0.515 | 0.000 |
| commutator rotor | 0.931 | 0.935 | 0.008 | 0.921 | 0.945 | 0.003 |
| G₂-twisted branch product | 0.960 | 0.965 | 0.002 | 0.917 | 0.970 | 0.005 |
| additive associator channel | 0.721 | 0.760 | 0.000 | 0.721 | 0.760 | 0.000 |
| associator rotor | 0.927 | 0.920 | 0.005 | 0.504 | 0.515 | 0.000 |
| φ-gated slerp | 0.729 | 0.680 | 0.000 | 0.772 | 0.840 | 0.000 |
| ψ-gated slerp | 0.815 | 0.775 | 0.001 | 0.723 | 0.760 | 0.000 |
| **triple cross product** | **0.899** | **0.900** | **0.009** | **0.724** | 0.690 | 0.000 |
| Hopf fiber twist | 0.870 | 0.895 | 0.004 | 0.870 | 0.895 | 0.004 |
| triality sandwich | 0.914 | 0.915 | 0.002 | 0.865 | 0.875 | 0.003 |
| Nambu/Euler-top step | 0.866 | 0.830 | 0.001 | 0.740 | 0.760 | 0.000 |
| precession rotor | 0.845 | 0.845 | 0.001 | 0.736 | 0.695 | 0.000 |
| Möbius combiner | 0.736 | 0.785 | 0.000 | 0.733 | 0.700 | 0.000 |

Readings:

1. **Calibration behaves.** The control and both theorem-killed rows (A5,
   P2) sit at 0.72–0.74 in both algebras; raw commutator sits at chance on an
   exactly-zero signal. The screen and the proofs agree everywhere they meet.
2. **The capable band is 0.85–0.96 best train with hit rates ≤ 1%.** Random
   search finds ceiling-beating weights *rarely* for every candidate — the
   screen certifies capability, and #6's real question (can the error wave
   *find* these weights?) remains squarely open.
3. **The ℍ column is the science.** Exact-collapse-to-linear: kappa-slerp,
   ψ-slerp, triple cross (all land on the control number). Collapse-to-
   constant: associator rotor. Survivors (nonlinearity not octonion-specific):
   branch products and all its variants, commutator rotor, Hopf twist
   (structurally identical numbers).
4. **Jordan = branch product** at readout, per the exact identity; the table's
   small differences are draw noise, as predicted.

## 4. Amendments to the #6 shortlist

**Additions (ranked):**

1. **Triple cross product** `normalize(X₃(x·w₁, x·w₂, x·w₃))` — promote to
   full head-to-head entrant. It is the only candidate that pairs
   leader-class screen numbers (0.90, best hit rate 0.009) with an *exact*
   collapse-to-current-architecture on ℍ. It occupies the quadrant the two
   leaders each half-occupy: branch products screen well but survive the
   ablation; kappa-slerp ablates cleanly but screens weaker and carries a
   hand-designed gate. X₃ does both with no gate at all, and its geometry
   (Cayley calibration, norm = volume) is the most principled in the
   catalogue. Cost: three weights per combiner and a (measure-zero)
   degenerate locus needing a guarded normalize.
2. **Associator rotor** — add to #6 as the designated *non-associativity
   instrument*, not as an architecture: largest 𝕆/ℍ gap in the table
   (0.93 → chance), unusable transport (normalize singular exactly on the
   ablation locus). Run it as an ablation arm to bound how much signal pure
   non-associativity carries; do not ship it.
3. **Hopf fiber twist** — add as the *fibration control* beside the
   renormalized-sum control: exactly on-sphere, no normalization, associative
   per-half transport, pure fiber mixing — and provably algebra-blind on this
   embedding. If it matches the octonion-native entrants in training, the
   octonion bet is not paying.
4. **ψ-gated slerp** — hold in reserve within the gated-slerp family: same
   clean ablation as kappa (collapses to the renormalized sum), smooth
   polynomial gate where κ is kinked on its ablation locus; screens below C1.
   Swap in only if κ's kink shows up as a training pathology.

**Eliminations (with obituaries):**

- **Jordan branch product** — `sign(re(·))` cannot tell a Jordan product from
  the plain product (`re(ab) = re(ba)`); the symmetrization costs the exact
  norm and buys nothing.
- **Raw commutator / raw cross product / raw associator combiners** — pure
  imaginary; the readout sees exactly zero.
- **Additive associator channel** — the associator donates no real part to
  the numerator and only positive scale to the denominator;
  `sign(re) = sign(re(x·w₀))` exactly; ceiling-bound until the readout
  redesign ships (which was its known hard dependency — now a theorem).
- **φ-gated slerp** — grade S3 on paper, 0.729 in practice; dominated by κ-
  and ψ-gates on both screen and ablation; no trained-weights plea entered.
- **Möbius / one-step conformal maps** — composition-algebra polarization
  makes every such readout affine; provably under the ceiling, and the
  translation weights break the unit-weight story besides.
- **Nambu/Euler-top step** — the commutator rotor wearing flow costume:
  weaker screen, extra machinery, no exactly conserved quantity after
  discretization.
- **Precession rotor** — same obituary as the Euler-top step, with an exp.
- **G₂-twisted branch product & triality sandwich** — not killed, *merged*:
  capacity knobs on branch products (their tail draws are luck, their hit
  rates are C2's), not separate mechanisms.
- **Commutator rotor** — the awkward case: best simple screen (0.931, hit
  0.008) but its nonlinearity survives on ℍ nearly intact, so it answers the
  capacity question while dodging the science question. Not a head-to-head
  entrant; revisit only if #6's winner needs a cheap capacity booster and the
  quaternion control is run separately.

**Recommendation for #6.** The head-to-head should go **three-way**:
kappa-gated slerp, branch products, and the triple cross product — with the
renormalized sum and the Hopf fiber twist as controls and the associator rotor
as the ablation instrument. The two-way framing is strictly dominated: X₃
would be the presumptive favorite on the combined capability/ablation/
transport rubric, and leaving it out would leave the "does non-associativity
*itself* buy nonlinearity" question answerable only through kappa's
hand-designed gate. One honest caveat carries over from §3: every hit rate in
the screen is ≤ 1%, so none of the three entrants is presumptively *findable*
by the error wave — that is exactly what #6 must now measure.

## References

- J. C. Baez, *The Octonions*, Bull. Amer. Math. Soc. 39 (2002) 145–205,
  [arXiv:math/0105155](https://arxiv.org/abs/math/0105155) — G₂ = Aut(𝕆),
  cross products, Hopf fibrations from division algebras, triality.
- R. Harvey, H. B. Lawson, *Calibrated Geometries*, Acta Math. 148 (1982)
  47–157, [DOI 10.1007/BF02392726](https://doi.org/10.1007/BF02392726) — the
  associative 3-form φ and coassociative 4-form ψ as comass-1 calibrations;
  the triple cross product and the Cayley 4-form on ℝ⁸ ≅ 𝕆. (Convention note:
  H–L define φ(x,y,z) = ⟨x, y×z⟩ on Im 𝕆 and build the Cayley form from the
  triple cross product; the exact formulas used here are re-verified
  numerically against the project algebra by the companion script, so no
  convention mismatch can silently propagate.)
- R. B. Brown, A. Gray, *Vector cross products*, Comment. Math. Helv. 42
  (1967) 222–236 — r-fold cross products; 2-fold only in dims 3 and 7, 3-fold
  only in dims 4 and 8; the norm-equals-volume axiom.
- Y. Nambu, *Generalized Hamiltonian Dynamics*, Phys. Rev. D 7 (1973)
  2405–2412, [DOI 10.1103/PhysRevD.7.2405](https://doi.org/10.1103/PhysRevD.7.2405)
  — ternary brackets, flows with two Hamiltonians.
- J. A. de Azcárraga, J. M. Izquierdo, *n-ary algebras: a review with
  applications*, J. Phys. A 43 (2010) 293001,
  [arXiv:1005.1028](https://arxiv.org/abs/1005.1028) — Nambu and Filippov
  structures; ternary brackets and the octonionic cross product context.
- H. Hopf, *Über die Abbildungen von Sphären auf Sphären niedrigerer
  Dimension*, Fund. Math. 25 (1935) 427–440 — the S⁷ → S⁴ fibration (the
  1931 Math. Ann. paper gives S³ → S²); D. W. Lyons, *An Elementary
  Introduction to the Hopf Fibration*, Math. Mag. 76 (2003) 87–98 — the
  fiber picture used for H1.
- R. Moufang, *Zur Struktur von Alternativkörpern*, Math. Ann. 110 (1935) —
  the exact identities behind every transport claim (via the
  [deep dive](octonion-structure-deep-dive.md) §2).

Where a claim is not attributable to the above it is marked *own derivation*
and verified by [s7_combiner_screen.py](s7_combiner_screen.py); the additive
associator sign-collapse, the Möbius polarization kill, the triple-cross
collapse on ℍ, and the Hopf twist's embedding-level ablation invariance are
established here.
