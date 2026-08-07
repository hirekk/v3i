# Octonion structure deep dive: what the algebra buys the architecture

Research note for the wayfinder ticket
[Algebraic deep dive: what octonion structure buys us](https://github.com/hirekk/v3i/issues/4).
Every numbered claim marked *(verified)* is checked to machine precision by
[octonion_structure_verification.py](octonion_structure_verification.py)
(`uv run python docs/research/octonion_structure_verification.py`), run against
the project's own `Octonion` implementation. Claims resting on my own derivation
or on the numerics alone (rather than a cited source) are marked as such.

## TL;DR

1. **Implementation bug found.** `cross_product_7d` in `src/v3i/algebra.py` is
   **not** the cross product of the octonion algebra implemented by
   `Octonion.__mul__` — its Fano table is the algebra's with labels e5↔e7
   swapped, *and* two lines carry inconsistent orientations, so it is not a
   valid 7D cross product for *any* octonion structure: it fails orthogonality
   (`⟨u×v, u⟩ = 2` for `u = e1+e4, v = e7`) and the norm identity. The
   perceptron's "commutator alignment" torque is therefore not what its
   docstring claims. Details in §5.3.
2. **Composition-algebra guarantees are theorems, not hopes**: `|xy| = |x||y|`
   means no signal collapse/explosion at any depth, and every transport step is
   an exactly invertible isometry. Hurwitz says this holds *only* in dims
   1, 2, 4, 8.
3. **Everything the error wave currently does is algebraically exact.** The
   sandwich `w̄·r·w` is unambiguous (Artin/diassociativity), invertible, and
   Moufang identities license same-`w` rearrangements. But nothing involving
   two *different* weights rearranges: two-layer transport is not transport by
   any product (min deviation 0.61 over 300 trials).
4. **The associator is the octonion-native nonlinearity — but only in
   multi-slot form.** For fixed weights, `x ↦ [x,a,b]` is *linear* (it is
   exactly `const × (isometry ∘ projection)` onto a 4-dim subspace);
   alternativity kills every naive self-insertion (`[x,w,x] ≡ 0`). Genuine
   nonlinearity needs the signal in ≥2 slots via different images — e.g.
   `[x·w₁, c, x·w₂]`, verified nonzero and degree-2 homogeneous. It vanishes
   identically for quaternions (built-in ablation) — and its output has
   **zero real part**, so the current `sign(re(·))` readout is blind to it.
5. **G₂-equivariance of the whole pipeline is a mild asset** (canonical
   initialization, symmetry ablations, known 14-dim flat solution orbits), not
   a nonlinearity source. Triality/Spin(8) context: right multiplications
   generate all of so(8) *(verified: rank 28)* — the reach the isometry-ceiling
   note said the readout discards.
6. **S⁷ parallelizability is already load-bearing** — it is what lets the wave
   pass an 8D error vector between layers with no holonomy bookkeeping. The
   `exp`/`log` in `algebra.py` are the exact octonion exp/log (geodesic normal
   coordinates); what S⁷ ≠ Lie group costs is BCH, adjoints, and composable
   translations — none currently needed.
7. **Stopping at octonions is principled**: sedenions lose norm
   multiplicativity and division — 84 exact basis zero divisors found by
   exhaustive search, e.g. `(e1+e10)(e4−e15) = 0`; random `|xy|/(|x||y|)`
   ranges 0.56–1.27 *(verified)*. Every triad guarantee dies at dim 16.

---

## 0. Conventions

Basis `e0 = 1, e1…e7` imaginary. `Octonion.__mul__` implements Cayley–Dickson
doubling of Hamilton quaternions; its Fano lines are *(verified, §6 of script)*

```
{123} {145} {167} {246} {257} {347} {356}
```

Standard associator convention used throughout this note:
`[x,y,z] = (xy)z − x(yz)`. Note two internal sign conventions *(verified)*:
`algebra.associator` computes the *negative*, `x(yz) − (xy)z`, while
`OctonionPerceptron._compute_kappa` computes the standard sign. Only magnitudes
matter anywhere in the code today, so this is a documentation nit, not a bug.

## 1. Composition and division: what Hurwitz guarantees

**Facts.** (Hurwitz 1898; Baez §1.1; Conway–Smith ch. 6.) The normed division
algebras over ℝ are exactly ℝ, ℂ, ℍ, 𝕆 — dims 1, 2, 4, 8, nothing else, ever.
In all of them:

- `|xy| = |x||y|` *(verified, 3.6e−15 over 1000 random non-unit pairs)*;
- no zero divisors, and every `x ≠ 0` has the two-sided inverse `x̄/|x|²`;
- the adjoint identities `⟨xy, z⟩ = ⟨x, z·ȳ⟩ = ⟨y, x̄·z⟩` *(verified, 1.1e−14)*
  — the workhorse of the isometry-ceiling note's readout collapse;
- conjugation is an anti-automorphism, `conj(xy) = ȳ·x̄` *(verified)*.

**What this buys the triad.**

- *No signal collapse or explosion, at any depth.* Unit weights make every
  forward step an isometry of ℝ⁸; a depth-n chain moves the signal on S⁷ and
  never off it. There is no octonion analogue of vanishing/exploding
  activations for the multiplicative pathway — that pathology can only be
  *introduced*, by whatever nonlinearity we add.
- *Invertible error transport.* The sandwich `r ↦ w̄rw` is an isometry with
  exact inverse `t ↦ wtw̄` *(verified roundtrip, 2.2e−15)*. Information in the
  error wave is destroyed only where the design says so (the projection/
  absorption step), never by transport.
- *Every layer is exactly undoable*: `(x·w)·w̄ = x` for unit `w` *(verified,
  1.3e−15)* — despite non-associativity, because only two distinct elements are
  involved (§2).

**Verdict.** These are the licenses the triad already runs on. The
architectural constraint they impose on ticket 005: any nonlinearity candidate
should come with a norm story (bounded, renormalized, or slerp-interpolated),
because the linear pathway's perfect conditioning is the algebra's single most
valuable gift and is easy to squander.

## 2. Alternativity, Artin, Moufang: the exact-identity inventory

**Facts.** Octonions are *alternative* (Schafer ch. III): the left/right
alternative and flexible laws hold,

```
x(xy) = (xx)y      (yx)x = y(xx)      x(yx) = (xy)x
```

*(all verified, ≤1.4e−14)*. **Artin's theorem** (Schafer III.1): the subalgebra
generated by any *two* elements is associative. Verified concretely: all five
bracketings of the word `xyxy` agree to 8.5e−14. The three **Moufang
identities** (Moufang 1935; the middle one in its two equivalent forms):

```
left:    z(x(zy)) = ((zx)z)y
right:   x(z(yz)) = ((xz)y)z
middle:  (zx)(yz) = (z(xy))z = z((xy)z)
```

*(all four forms verified, ≤8.5e−14)*. Unit octonions are closed under
multiplication (norm multiplicativity) with inverses `w̄` — S⁷ is a **Moufang
loop**, and Moufang's theorem says any three elements that associate generate a
group.

**What the error wave can and cannot exploit — exact, not approximate:**

| pattern | status | why |
|---|---|---|
| `w̄·r·w` without parentheses | **exact** — `(w̄r)w = w̄(rw)`, dev 1.3e−15 | `w̄ ∈ span{1,w}`, so all three factors live in the associative subalgebra ⟨w,r⟩ (Artin). Python's left-to-right evaluation in `correct()` is mathematically canonical. |
| transport roundtrip `w(w̄rw)w̄ = r` | **exact** (2.2e−15) | same subalgebra |
| layer inversion `(xw)w̄ = x` | **exact** (1.3e−15) | alternativity |
| sandwich of a *product*: `(wx)(yw) = w((xy)w)` | **exact** (Moufang middle) | licenses same-`w` bimultiplication layers `x ↦ wxw` with exact factorization rules |
| transport as automorphism: `w̄(xy)w =? (w̄xw)(w̄yw)` | **false** — median dev 0.505, max 1.5 *(verified)* | needs three distinct elements |
| two-layer transport = one-product transport: `w̄₂(w̄₁rw₁)w₂ =? v̄rv` for `v ∈ {w₁w₂, w₂w₁}` | **false** — min dev 0.61/0.62 over 300 trials *(verified)* | ditto |
| transport as *pseudo*-automorphism | **exact with companion `c = w̄³`**: `φ(xy)·c = φ(x)·(φ(y)·c)` for `φ(x) = w̄xw` *(numerically established here, ≤1e−9; consistent with the general theory of Moufang-loop pseudo-automorphisms, Bruck 1958)* | the *entire* failure of transport to respect products is captured by one companion element |

Implementation remark (design observation, not a bug): step 6 of
`OctonionPerceptron.correct` transports the residual out with the *updated*
weight while the residual was computed in the *old* weight's frame. Each step
is still an exact isometry; but the wave's "frame" is a moving target, which
any multi-layer analysis must model.

**Verdict.** Everything the current wave does is exact — the triad's transport
is not an approximation that non-associativity slowly poisons. But the
quaternion intuition "compose all transports into one sandwich" is dead
(that is precisely the isometry-ceiling result seen from the transport side).
The only *new* exact structures available to a future architecture are
same-element patterns: bimultiplication `x ↦ wxw` (Moufang) and the
`w̄³`-companion law. Anything mixing two different weights gets no exact
rearrangement — by design of the universe, not of the code.

## 3. The associator: geometry of kappa, and a serious audit as a nonlinearity

### 3.1 Structure *(all verified)*

`[x,y,z] = (xy)z − x(yz)` is, over the octonions:

- **alternating trilinear**: linear in each slot (2.8e−14); changes sign under
  any transposition, so all six permutations carry the parity sign (1.8e−14);
  consequently `[x,y,z] = 0` whenever two arguments are equal — and also
  `[x,w,x̄] = 0`, since `x̄ = 2Re(x) − x`;
- **purely imaginary**: `Re[x,y,z] = 0` (7.1e−15). Two-line proof via the
  adjoint identity: `⟨(xy)z, 1⟩ = ⟨xy, z̄⟩ = ⟨x, z̄ȳ⟩ = ⟨x, conj(yz)⟩ =
  ⟨x(yz), 1⟩` *(own derivation)*;
- **orthogonal to each of its arguments**: `⟨[x,y,z], x⟩ = ⟨·,y⟩ = ⟨·,z⟩ = 0`
  (3.6e−14) — associator features point in directions *new* relative to their
  inputs;
- depends only on imaginary parts: `[x,y,z] = [Im x, Im y, Im z]` (1.4e−14);
- **vanishes identically on every quaternion subalgebra** `span{1,a,b,ab}`
  (1.1e−14 over 200 random subalgebras) — this is the built-in ablation: every
  associator-based mechanism switches itself off exactly for the associative
  control;
- relation to commutator/cross product, for imaginary `u,v,w` *(own
  derivation, verified to 4.4e−16)*:

  ```
  [u,v,w] = (u×v)×w − u×(v×w) + ⟨v,w⟩u − ⟨u,v⟩w
  ```

  where `u×v = Im(uv) = ½(uv − vu)` — the associator is exactly the failure of
  the 7D cross product to satisfy the 3D triple-product rearrangements (§5.2).

### 3.2 What kappa measures

For unit `q, w, r`, both `(qw)r` and `q(wr)` are *unit* octonions (norm
multiplicativity), so the associator is a chord between two points of S⁷:

```
|[q,w,r]| = 2 sin(θ/2),   θ = angle between (qw)r and q(wr)
```

*(own derivation, verified to 1.9e−15)*. So
`_compute_kappa = 1 − clip(2 sin(θ/2), 0, 1)` is a purely geometric,
scale-invariant, G₂-invariant measure of **how much the two parenthesizations
of the (input, weight, error) triple disagree** — 1 when the triple lies in an
associative subalgebra, hitting 0 at θ = 60° and clipped there (the ratio can
reach 2, see §3.3, so the clip is doing real work: maximally non-associative
triples currently get *zero* update, "trust nothing when transport is
ambiguous").

### 3.3 Sharp norm bound

For unit arguments: `|[x,y,z]| ≤ |(xy)z| + |x(yz)| = 2` by norm
multiplicativity plus the triangle inequality, and the bound is **attained**:
any basis triple not lying on a Fano line anti-associates, e.g.
`|[e1,e2,e4]| = 2.000000000000000` exactly *(elementary own derivation;
verified)*. So the sharp bound is **2**. Numerics: empirical max over 50 000
random unit triples 1.9937; hill-climb reaches 1.999991. Nothing blows up:
associator features of unit inputs live in the closed ball of radius 2.

### 3.4 Audit as a nonlinearity source

The honest structural finding first: **for fixed weights the associator is
linear in the signal.** Trilinear means linear per slot. Stronger *(numerics +
derivation)*: for orthonormal imaginary `a,b`, the map `x ↦ [x,a,b]` has
singular values exactly `(2,2,2,2,0,0,0,0)` — it is `2 × (isometry ∘
orthogonal projection)` killing the quaternion subalgebra `Q(a,b)` and
isometrically mapping its 4-dim complement; for generic unit `a,b` the same
shape holds with the four nonzero values equal (e.g. 1.6824 quadruple)
*(verified)*. A clean "project-and-rotate" — useful, but linear.

Alternativity is the degeneracy trap: every attempt to get nonlinearity by
reusing the signal in a second slot *directly* dies exactly —
`[x,w,x] = [x,x,w] = [x,w,x̄] = 0` *(verified, 2e−16)*. Genuine input-
nonlinearity requires the signal entering ≥2 slots via *different linear
images*:

- `f(x) = [x·w₁, c, x·w₂]` is degree-2 homogeneous (`f(2x) = 4f(x)` exactly),
  non-additive (`|f(x+x′) − f(x) − f(x′)| ≈ 0.91`), and generically nonzero
  (`|f(1)| ≈ 0.82`) *(all verified)* — a bona fide quadratic map of the signal;
- across *width*: `[y_i, c, y_j]` for two different neurons' outputs is
  bilinear in the pair — nonlinear in the shared input upstream.

**Degeneracy risks, stated plainly:**

1. *Blind readout*: the output is purely imaginary — `re([·,·,·]) = 0`
   **always**. Fed directly to `sign(re(·))` the associator produces the
   constant 0. Any associator mechanism forces the readout redesign the map
   already lists as fog.
2. *Half the signal is discarded*: the rank-4 kernel means a lone associator
   channel annihilates the `Q`-component of the signal — it must ride alongside
   a linear branch, never replace it.
3. *Dead-associator risk*: if weights/signals drift into a common quaternion
   subalgebra the mechanism switches off (the flip side of the built-in
   ablation). Norm ∈ [0,2] with 0 attainable ⇒ renormalizing raw associator
   outputs is unstable near the associative locus; prefer bounded gates
   (κ-style) or additive mixing before normalization.

**Verdict.** The associator is the *only* octonion-native, G₂-equivariant,
norm-bounded, quaternion-ablating nonlinearity on offer — and it is genuinely
nonlinear **only in multi-image or cross-width configurations**. It should be
ticket 005's lead candidate, deployed as a gate or as an auxiliary feature
channel, never as the sole pathway, and never feeding `re(·)` directly.

## 4. G₂, triality, Spin(8)

**Facts.** Aut(𝕆) = G₂, the 14-dimensional compact exceptional Lie group
(Baez §4.1); it fixes 1, acts irreducibly on Im 𝕆 = ℝ⁷, and inside SO(7) it is
exactly the stabilizer of the 7D cross product (equivalently of the associative
3-form `⟨u×v, w⟩`). Hence **everything built from multiplication, conjugation,
inner product, cross product, associator, exp/log, slerp is G₂-equivariant** —
i.e. the *entire* forward pass, torque, kappa, geodesic update, and readout
(`re(·)` is G₂-*invariant*: automorphisms fix the real line). Verified with a
concrete nontrivial automorphism, conjugation by `a = exp(π/3 · e1)`:
automorphism property 7.1e−15, associator equivariance 3.0e−14, cross-product
equivariance 6.7e−16. (Conjugation by `a` is an automorphism iff `a⁶` is real —
a known characterization, numerically confirmed here: the `a⁶ = 1` case passes
at 7e−15 while conjugation by `exp(0.4·e1)` deviates by 1.5.)

G₂ acts **freely and transitively on basic triples** (ordered orthonormal
`(e₁', e₂', e₄')` with the third off the subalgebra of the first two) — Baez
§4.1; dim G₂ = 14 = dim of the space of such triples.

**Triality context** (Baez §2.4; Conway–Smith ch. 8): Spin(8) has outer
automorphism group S₃ permuting its three 8-dim representations (vector, left
spinor, right spinor); octonion multiplication is the intertwiner, and triality
cycles the roles of `L_w`, `R_w`, and the bimultiplication. The concrete cash
value for this program: products of right multiplications do not stay in any
small subgroup — the seven `R_{e_i}` together with their first commutators
already span the full 28-dimensional so(8) *(verified: rank exactly 28)*. This
is the rigorous footing under the isometry-ceiling note's observation that
octonion depth reaches SO(8) transformations a single weight cannot.

**Asset or obstacle?**

- *Asset (weight-tying/canonicalization)*: since G₂ is simply transitive on
  basic triples, initialization and analysis can fix a gauge WLOG (e.g. rotate
  the first weight's imaginary part to `e1`); hyperparameter and ablation
  sweeps can quotient out 14 dimensions of pure symmetry.
- *Asset (diagnostics)*: G₂-equivariance predicts that conjugating data and
  initial weights by any automorphism yields bit-identical training curves — a
  free correctness test for any future implementation.
- *Neutral-to-obstacle*: every optimum comes in a 14-dimensional flat orbit
  (loss is G₂-invariant along it). Harmless degeneracy, but flat directions the
  optimizer will wander.
- *Not an obstacle for the readout*: `re(·)` is already G₂-invariant; no
  symmetry needs breaking. The data embedding (which picks out specific
  coordinates) already breaks G₂ externally.

**Verdict.** Equivariance is an asset of the bookkeeping kind — canonical
gauges, free invariance tests, orbit-aware analysis — not a source of capacity.
Triality matters to this program only through its corollary: the SO(8) reach
that a richer readout (ticket 005's sibling question) could finally see.

## 5. The geometry of S⁷

### 5.1 Parallelizability

S¹, S³, S⁷ are the **only** parallelizable spheres (Bott–Milnor 1958, Kervaire
1958; equivalently Adams 1960 on Hopf invariant one) — precisely the unit
spheres of ℂ, ℍ, 𝕆. The parallelization is the algebra: `v ↦ w·v` (an isometry
of ℝ⁸) carries a fixed frame of `T₁S⁷` to a frame of `T_wS⁷`, globally,
smoothly, with no exceptional points. **This is what makes the error wave
well-posed**: an 8D error vector can be handed between layers as raw
coordinates, and re-framed by multiplication (`w̄rw`), with zero holonomy
bookkeeping — on S⁶ or any non-parallelizable manifold, no such global scheme
exists and a connection with path-dependent transport would be mandatory. The
triad quietly assumes this; it is worth knowing it is a theorem with a
three-sphere-long list of cases.

### 5.2 The 7D cross product

Bilinear vector cross products exist **only in dims 3 and 7** (Eckmann 1943;
Brown–Gray 1967; Massey 1983), the 7D one coming from `u×v = Im(uv) =
½(uv−vu)` on imaginary octonions, with the Fano-plane multiplication table
(§0). What survives from 3D and what fails, for the *algebra's own* cross
product *(all verified)*:

| identity | 3D | 7D |
|---|---|---|
| bilinear, antisymmetric | yes | yes (0.0e+00) |
| `⟨u×v, u⟩ = ⟨u×v, v⟩ = 0` | yes | **yes** (1.1e−14) |
| `\|u×v\|² = \|u\|²\|v\|² − ⟨u,v⟩²` | yes | **yes** (5.7e−14) |
| `u×(u×v) = ⟨u,v⟩u − \|u\|²v` | yes | **yes** (1.4e−14) — two distinct vectors only, so alternativity suffices |
| scalar triple `⟨u×v, w⟩` alternating | yes | yes (implied by the above) |
| Jacobi identity | yes | **FAILS** — Jacobiator `= −(3/2)[u,v,w]` exactly *(own derivation, verified 2.8e−14)*; ℝ⁷ with × is not a Lie algebra |
| triple-product expansion `u×(v×w) = ⟨u,w⟩v − ⟨u,v⟩w` | yes | **FAILS** (median deviation 9.7 on random triples) |
| invariance group | all of SO(3) | only G₂ ⊊ SO(7) |

The failed rows are not defects; they are the associator wearing its cross-
product costume — the same non-associativity the architecture wants to mine.

### 5.3 The implementation bug (major finding)

`cross_product_7d` in `src/v3i/algebra.py` does **not** implement the cross
product of the algebra implemented ten lines above it. *(All verified.)*

- Its Fano line set is `{123} {147} {156} {246} {257} {345} {367}` — exactly
  the algebra's line set with labels **e5 ↔ e7 swapped** (script confirms the
  relabeling reproduces it). It disagrees with `Im(e_i e_j)` on **26 of 42**
  ordered basis pairs.
- Worse, its orientations are internally inconsistent (e.g. on line {1,4,7} it
  has both `e4×e1 = e7` and `e4×e7 = e1`), so it is not the cross product of
  *any* octonion structure: **orthogonality fails** — for `u = e1+e4, v = e7`
  it returns `u×v = e1+e4 = u` itself, with `⟨u×v, u⟩ = 2`; max orthogonality
  violation 4.1e+01 and max norm-identity violation 1.5e+02 over 2000 random
  pairs. (Antisymmetry alone survives, by construction.)
- Consequently, in `OctonionPerceptron.correct`, the torque is **not** the
  "commutator alignment" `½ Im(uv − vu)` the comments describe (deviation
  O(10) on random inputs), it can have a component *along* `w.im`, and it is
  not G₂-equivariant with respect to the implemented algebra — so the
  absorption step projects the error onto a direction with no invariant
  meaning.
- Blast radius: **training dynamics only.** The forward map never calls it, so
  the isometry-ceiling results and the function-class analysis are unaffected.
  But any empirical conclusion about the *learning rule* drawn while this
  mismatch stands is measuring an unintended algorithm.
- Fix (one line, not applied here per ticket scope): compute the torque as
  `Im(u·v)` with the class's own multiplication (embed the two imaginary
  7-vectors, multiply, take `.im`), or re-derive the hard-coded table from the
  algebra's lines `{123} {145} {167} {246} {257} {347} {356}` with consistent
  orientations, adding the regression test `cross_product_7d(u,v) ==
  (Octonion(0,u)*Octonion(0,v)).im` to `tests/`.

### 5.4 Moufang loop, not Lie group — and what exp/log really compute

S³ is a Lie group (≅ Sp(1) ≅ SU(2)); S⁷ is a Moufang loop and **cannot** be
made a Lie group (the only sphere groups are S⁰, S¹, S³). What that costs:

- no Baker–Campbell–Hausdorff: `exp(u)exp(v)` has no `exp(BCH(u,v))` form; no
  adjoint representation; no composing of left translations into a group.
- what it does **not** cost: `Octonion.exp/log` remain exact and well-defined —
  any *single* octonion generates a commutative associative slice
  `ℝ ⊕ ℝn̂ ≅ ℂ`, and `exp(a + θn̂) = e^a(cos θ + n̂ sin θ)` is the honest
  exponential there. On unit octonions `log` is angle-axis form — geodesic
  normal coordinates at 1 — and since `L_w` is an isometry of ℝ⁸,
  `t ↦ w·exp(t·n̂)` is a genuine great-circle geodesic through `w`. The
  update `weight * (torque * mag).exp()` in the perceptron and `slerp` are
  therefore exactly the triad's "geodesic updates", no group structure needed.
- diassociativity (§2) is the safety net: any formula involving only powers of
  one or two elements behaves associatively, which is why every exp/log/sandwich
  expression in the codebase is unambiguous.

**Verdict.** Parallelizability: already load-bearing, keep. exp/log: correct
as-is. Cross product: **fix before running any further torque-based
experiments** — it is the one place the code and the algebra disagree.

## 6. Cayley–Dickson doubling: what dies at each step

Each doubling `A → A⊕A`, `(a,b)(c,d) = (ac − d̄b, da + bc̄)` (the exact
convention in `Octonion.__mul__`; the script's generic doubling reproduces it
at dim 8 to 1.8e−15) trades away one property *(Baez §1.1; Schafer)*:

| step | dim | property lost | consequence for the triad |
|---|---|---|---|
| ℝ → ℂ | 2 | trivial conjugation / linear order | none — phase begins |
| ℂ → ℍ | 4 | commutativity | `w̄rw` transport becomes *non-trivial* — the error wave's frame machinery starts existing |
| ℍ → 𝕆 | 8 | associativity (alternativity survives) | associator ≠ 0: kappa, the nonlinearity candidates of §3, SO(8) reach — everything this program mines |
| 𝕆 → 𝕊 (sedenions) | 16 | alternativity, composition `\|xy\|=\|x\|\|y\|`, **division** | fatal, see below |

Sedenion failure, made concrete *(verified with the generic doubling)*:
random pairs give `|xy|/(|x||y|)` from **0.5620 to 1.2728** — isometric forward
maps are gone; and exhaustive search over basis expressions finds **84 exact
zero divisors** of the form `(e_i+e_j)(e_k−e_l) = 0`, the first being
`(e1+e10)(e4−e15) = 0`. A unit-norm sedenion weight can silently annihilate a
nonzero signal; transport is no longer invertible; "geodesic update" loses its
norm control. (Sedenions remain power-associative and flexible, and their zero
divisors have beautiful structure — the unit ones form a manifold homeomorphic
to G₂ (Moreno 1998) — none of which rescues the triad.)

**Verdict.** "Stop at octonions" is not taste: Hurwitz's theorem says dims
1, 2, 4, 8 are *all* the normed division algebras there are, and the script
exhibits the corpse at dim 16. Octonions are the terminal object of the triad —
which also means the map's out-of-scope line on sedenion generalization is
permanently safe to enforce.

## 7. Property → leverage → mechanism (feeding ticket 005)

Ranked by promise for **triad-compatible nonlinearity** (rows 1–3 are the
candidates; rows 4–8 are enabling structure, not nonlinearity).

| # | property | architectural leverage | concrete candidate mechanism |
|---|---|---|---|
| 1 | **Kappa's chord geometry** — `\|[q,w,r]\|/(\|q\|\|w\|\|r\|) = 2sin(θ/2)`: a scale- & G₂-invariant scalar in [0,2], ≡ 0 on ℍ | A data-dependent *gate* that is bounded, geometric, and degenerates to a constant for the quaternion control — nonlinearity with the ablation built in | **Associator-gated slerp**: `y = slerp(x·w₁, x·w₂, κ(x,w₁,w₂))`. Output on S⁷ by construction (norm story free); forward map genuinely nonlinear in `x`; for ℍ, κ ≡ 1 collapses to a *fixed* linear map — the cleanest XOR-gate ablation available |
| 2 | **Associator trilinearity** + sharp bound `\|[·,·,·]\| ≤ 2` + exact vanishing on ℍ + output ⊥ inputs | A quadratic-in-signal feature channel that manufactures *new* directions (orthogonal to its inputs), bounded, quaternion-ablating | **Quadratic associator channel**: `y = normalize(α·(x·w) + β·[x·w₁, c, x·w₂])`. Degree-2 term verified nonzero/non-additive; must be mixed with the linear branch (rank-4 kernel; zero real part) and β kept moderate (associator can vanish on associative loci) |
| 3 | **Norm structure + width** — sums of unit octonions leave S⁷; renormalization is a radial nonlinearity | The cheapest S⁷-compatible nonlinearity, and the one width forces anyway; composes with rows 1–2 | **Renormalized aggregation**: `y = (Σᵢ yᵢ)/\|Σᵢ yᵢ\|` (or tangent-space average + exp). Not octonion-specific — which makes it the *control* nonlinearity to run against rows 1–2 in the XOR gate |
| 4 | **Cross-width associators** — `[y_i, c, y_j]` bilinear across neurons | Octonion-native "interaction features" between channels; vanishes for ℍ | Pairwise associator features feeding the aggregation of row 3; needs width ≥ 2 (which the map mandates anyway) |
| 5 | **Moufang exactness + pseudo-automorphism companion `w̄³`** | Correctness guarantees for new transport patterns; the only exact multi-step rearrangements available | Bimultiplication layers `x ↦ w·x·w` (still linear — SO(8)-reaching, exactly composable via Moufang); companion-corrected product transport if the wave ever transports factored errors |
| 6 | **so(8) reach of multiplicative chains** (triality; rank-28 verified) | Depth/width already generates all of SO(8) — capacity the scalar readout throws away | **Richer readout**: replace `sign(re(y))` with `k` inner products `⟨y, v_k⟩` (or the full 8-vector) into the decision; zero real part of associator features *requires* this anyway — pairs with every row above |
| 7 | **G₂ simply transitive on basic triples** | Gauge-fixing and free invariance tests | Canonical initialization (fix a basic triple); conjugated-run equality as an implementation correctness test; quotient symmetry out of sweeps |
| 8 | **Parallelizability of S⁷** + Hurwitz/sedenion boundary | Keeps the error wave well-posed at any width/depth; forbids dimension escalation | Keep transport multiplicative (no connection machinery); never generalize the doubling upward — dim 16 provably breaks the triad |

Reading of the ranking for ticket 005: row 1 is the best-shaped candidate
(nonlinear, on-sphere by construction, bounded, ablation-ready); row 2 is the
highest-capacity one but drags the readout redesign (row 6) with it as a hard
dependency; row 3 is the mandatory baseline any octonion-native mechanism must
beat to justify itself. Before any of them run: fix §5.3.

## References

- J. C. Baez, *The Octonions*, Bull. Amer. Math. Soc. 39 (2002) 145–205,
  arXiv:math/0105155 — composition algebras, Cayley–Dickson, G₂, basic
  triples, triality, cross products.
- J. H. Conway, D. A. Smith, *On Quaternions and Octonions*, A K Peters 2003 —
  Moufang identities, multiplications as rotations of ℝ⁸, triality.
- R. D. Schafer, *An Introduction to Nonassociative Algebras*, Academic Press
  1966 / Dover 1995 — alternative algebras, Artin's theorem, Moufang
  identities, associator calculus.
- A. Hurwitz, *Über die Composition der quadratischen Formen von beliebig
  vielen Variablen*, Nachr. Ges. Wiss. Göttingen (1898) — the 1, 2, 4, 8
  theorem.
- R. Moufang, *Zur Struktur von Alternativkörpern*, Math. Ann. 110 (1935) —
  Moufang identities and Moufang's theorem.
- R. H. Bruck, *A Survey of Binary Systems*, Springer 1958 — Moufang loops,
  pseudo-automorphisms and companions.
- R. Bott, J. Milnor, *On the parallelizability of the spheres*, Bull. AMS 64
  (1958); M. Kervaire, PNAS 44 (1958); J. F. Adams, Ann. of Math. 72 (1960) —
  S¹, S³, S⁷ only; Hopf invariant one.
- B. Eckmann, Comment. Math. Helv. 15 (1943); R. B. Brown, A. Gray, *Vector
  cross products*, Comment. Math. Helv. 42 (1967); W. S. Massey, Amer. Math.
  Monthly 90 (1983) — cross products exist only in dims 3 and 7.
- G. Moreno, *The zero divisors of the Cayley–Dickson algebras over the real
  numbers*, Bol. Soc. Mat. Mexicana 4 (1998) — sedenion zero-divisor manifold.

Where a claim is not attributable to the above it is marked in-text as *own
derivation* and/or *(verified)* by the companion script; the Moufang-loop
companion `w̄³` and the singular-value structure of `x ↦ [x,a,b]` are
numerically established here.
