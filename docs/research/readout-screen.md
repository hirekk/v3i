# The readout map: beyond sign(re)

Research note for the wayfinder ticket
[The readout map: beyond sign(re)](https://github.com/hirekk/v3i/issues/14),
which blocks [Shape of a wide octonion layer (#7)](https://github.com/hirekk/v3i/issues/7).
Every claim marked *(verified)* is checked to machine precision — and every
screen number produced — by
[readout_screen.py](readout_screen.py)
(`uv run python docs/research/readout_screen.py`, deterministic, ~27 s). This
note builds directly on the [isometry ceiling](isometry-ceiling.md) (§2 readout
collapse, §3 the 75% XOR ceiling), the
[octonion deep dive](octonion-structure-deep-dive.md) (§4 G₂, the rank-28 SO(8)
reach) and the [combiner catalogue](s7-combiner-catalogue.md) (which killed a
family of mechanisms as "readout-invisible"); their content is linked, not
repeated. Derivations, numerical checks, and cited facts are marked as such.

## TL;DR

1. **`sign(re(·))` is canonical, not lazy — a theorem** (§1, *derivation +
   verified*). G₂ = Aut(𝕆) acts transitively on the unit imaginary sphere S⁶,
   so its orbits on S⁷ are **exactly the level sets of `re`**; any G₂-invariant
   smooth readout is therefore a function of `re(x)` alone, and the only
   G₂-invariant *quadratic* forms are `span{|x|², re²}`. `re` is invariant to
   `7.8e−16`; the G₂-average of a random symmetric `Q` collapses onto
   `a·I + b·e₀e₀ᵀ` (residual `0.247 → 0.097` as `K: 400 → 2000`, falling like
   `1/√K`). **Seeing more than `re` REQUIRES breaking G₂ by choosing extra
   structure** — a direction, a Cayley–Dickson split, a learned form.
2. **Most of "the XOR problem" was the readout, not the forward map** (§3,
   question (a)). The **current linear (orthogonal) architecture with a learned
   quadratic readout `⟨x,Qx⟩` reaches 0.960 test on XOR** (hit-rate 1.00), while
   every *linear-family* readout — `sign(re)`, best `⟨x,v⟩+c` — sits in the
   `0.64–0.75` ceiling band. The escalation is exact on the four embedded
   corners: their linear-feature matrix has **rank 3** (cannot shatter → XOR
   unrealizable, the isometry-ceiling result) but their quadratic-feature matrix
   has **rank 4** (shatters → a quadratic form realizes XOR *exactly*). XOR
   never required forward-map nonlinearity; it required a readout of **degree
   ≥ 2**.
3. **Every mechanism the catalogue killed as "readout-invisible" revives under a
   non-`re` readout** (§4, question (b)). Raw associator `0.51 (re) → 0.98`
   (best linear); raw commutator `0.52 → 0.98`; pure sandwich `0.52 → 0.96`;
   additive associator channel `0.76 → 0.99` (and `0.96` via Forward-Forward
   goodness on the pre-normalization magnitude). They were killed by the
   **readout**, not the mechanism.
4. **The ℍ-control is the science.** The raw associator's revival is *entirely*
   non-associativity — on ℍ it dies in **every** readout (`0.98 → 0.515`, the
   maximal 𝕆/ℍ gap in the table). Commutator (non-commutativity), branch
   product and sandwich (multiplicativity) survive on ℍ; the quadratic-readout
   XOR break is algebra-agnostic — a property of the readout, not the algebra.
5. **Recommendation** (§5). The readout is part of the layer's shape: give the
   wide octonion layer a **degree-≥2 readout** (learned quadratic form, or a
   small bank of linear projections `⟨y,v_k⟩`) — that is where to place the
   deliberate G₂-breaking that finally reads the SO(8) reach `re` discards. But
   a quadratic readout on a *linear* forward map already solves XOR, so **XOR
   can no longer certify forward-map nonlinearity once the readout is
   quadratic**: fix the readout when comparing combiners, and keep `sign(re)` as
   the certifying readout for any "genuine nonlinearity" claim.

## 1. The canonicity theorem: `re` is the canonical G₂-invariant readout

**Facts.** (Baez §4.1; deep dive §4.) G₂ = Aut(𝕆) is the 14-dimensional compact
group that fixes `1`, acts on `Im 𝕆 = ℝ⁷` as an *irreducible* subgroup of
SO(7), and acts **transitively on the unit imaginary sphere S⁶** (with
stabilizer SU(3)).

**Theorem (own derivation).** *The G₂-orbits on S⁷ are exactly the level sets of
`re`; hence a G₂-invariant readout `S⁷ → ℝ` is a function of `re(x)` alone.*

*Proof.* Write `x = re(x)·1 + Im(x)` with `re(x)² + |Im(x)|² = 1` on S⁷. Take
`x, y ∈ S⁷` with `re(x) = re(y)` (so `|Im x| = |Im y|`). Transitivity on S⁶
gives `g ∈ G₂` with `g(Im x/|Im x|) = Im y/|Im y|`; since `g` fixes `1` and is
linear, `g(x) = re(x)·1 + |Im x|·g(Im x/|Im x|) = re(y)·1 + Im(y) = y`. So the
orbit of `x` is `{y : re(y) = re(x)}` (the poles `±1` are fixed points), a level
set of `re`. A G₂-invariant function is constant on orbits, hence a function of
`re`; conversely every function of `re` is invariant because `re` is. ∎

**Quadratic specialization (own derivation, via Schur).** Block-decompose a
symmetric form as `⟨x,Qx⟩ = q₀₀ re² + 2 re(x)⟨v, Im x⟩ + ⟨Im x, Q_im Im x⟩`.
G₂ acts irreducibly on `ℝ⁷` with no invariant vector, forcing `v = 0`; Schur's
lemma forces the only invariant symmetric form `Q_im = λI`. Then
`⟨x,Qx⟩ = q₀₀ re² + λ|Im x|² = λ + (q₀₀−λ) re²` on S⁷ — affine in `re²`.
**The only G₂-invariant quadratic forms are `span{|x|², re²}`,** and on S⁷
(`|x|² ≡ 1`) that is affine in `re²`. `re²` is the canonical invariant quadratic;
`sign(re)` is the canonical invariant *decision*.

**Verification** (script §"G2 canonicity", *all verified*):

| check | number | meaning |
|---|---|---|
| composed automorphism `g(xy) − g(x)g(y)` | `4.7e−14` | the sampled maps are genuine G₂ automorphisms |
| `re(g x) − re(x)` | `7.8e−16` | `re` is G₂-invariant (machine precision) |
| orbit `Im`-covariance (2000 g's) | max off-diag `0.005`; eigs `[0.098, 0.113]` vs isotropic `0.107` | orbit fills S⁶ isotropically — numerical evidence for **transitivity** |
| G₂-average of random symmetric `Q` → `a·I + b·e₀e₀ᵀ` | residual/‖·‖ `0.247 (K=400) → 0.097 (K=2000)` | invariant quadratics collapse to `span{\|x\|², re²}`, residual `~1/√K → 0` |
| Hopf balance `\|a\|²−\|b\|²` across one orbit | mean `0.14`, **std `0.35`** at fixed `re = 0.50` | varies on a `re`-level set ⇒ **not** G₂-invariant |

The automorphisms are built as products of ten conjugations `y ↦ (g y)g⁻¹`,
`g = exp(θ n̂)`, `θ ∈ {π/3, 2π/3}` (so `g⁶ = 1` is real — the automorphism
condition, deep dive §4); bracketing is unambiguous by diassociativity.

**Consequence.** Octonion depth reaches SO(8) transformations a single weight
cannot (isometry-ceiling §2; deep dive §4, rank-28 *verified*), and `re`
provably discards all but one row of them. By the theorem, **any readout that
reads more is not G₂-invariant** — it has *chosen structure*. The rest of the
note screens those choices.

## 2. The candidate readouts

A readout is a map `S⁷ → ℝ` (thresholded to a label). Screened here, with the
symmetry it breaks and its parameter count:

| readout | formula | breaks G₂ by | params | grade (§1 of catalogue) |
|---|---|---|---|---|
| **`sign(re)`** (control) | `re(y)` at `τ=0` | nothing — *invariant* | 0 | S1 (affine pullback) — 75% ceiling |
| **best linear** | `⟨y,v⟩ + c` | choosing a direction `v` | 8 + 1 | S1 if `y` linear in `x`; S2 if `y` quadratic |
| **Hopf fiber balance** | `\|a\|² − \|b\|² = 2\|a\|²−1`, `y=(a,b)` | a Cayley–Dickson split (stabilizer SO(4) ⊂ G₂, *cited: issue #14 / Baez §3.1*) | 0 + 1 (threshold) | S2 (signature-(4,4) form) |
| **general quadratic** | `⟨y, Q y⟩`, `Q = Qᵀ` | a learned form (`re²` is the invariant special case `Q=e₀e₀ᵀ`) | 36 + 1 | S2 |
| **FF goodness** | `\|ỹ\|²` on the pre-norm magnitude `ỹ` | a chosen aggregation / objective (*cited: Hinton, Forward-Forward*) | 0 + 1 | ≥ S2 when `ỹ` is a nonlinear aggregation |

Two facts fixed by the algebra before any screen (*verified*):

- **Hopf balance is trivial on the raw embedding.** `to_s7_from_2d` populates
  only coords `0,1,2` (the data lies in `span{1,e₁,e₂}`, *verified* max coord
  `≥3` is `0.0`), so the Cayley–Dickson `b`-half is zero and `|a|²−|b|² ≡ 1`
  (screen §preflight(10): min = max = `1.000`). **The Hopf readout needs a
  forward layer to populate the second quaternion half before it carries any
  signal.**
- **FF goodness on a linear aggregation is constant.** `Σᵢ x·wᵢ = x·(Σwᵢ)` by
  distributivity, so `|Σᵢ x·wᵢ|² = |Σwᵢ|²` is constant on S⁷ — dead. FF goodness
  is informative only downstream of a *nonlinear* pre-normalization aggregation.

## 3. Question (a): the shattering escalation — readout degree, not forward nonlinearity

**The exact statement on the corners** (*derivation + verified*, screen
§preflight(9)). Embed the four XOR corners
`A(¼,¼), B(¼,¾), C(¾,¼), D(¾,¾)` with labels `(−,+,+,−)`. A readout linear in a
feature map `φ` realizes a labeling iff the sign pattern is linearly separable
in `φ`-space.

- **Linear features** `φ(x) = x` (with bias): the four corner vectors have
  **rank 3** (the data is 3-dimensional). Four points in a 3-dim space with a
  bias cannot be shattered; the XOR pattern lands in the unrealizable coset —
  the least-squares linear fit returns signs `[−1, 1, 1, +1] ≠ XOR`. This is
  the isometry-ceiling corner identity `f(A)+f(D) = f(B)+f(C)` seen as a rank
  deficiency.
- **Quadratic features** `φ(x) = vech(x xᵀ)` (with bias): **rank 4**. Four
  general-position points are shattered by a quadratic form; the fit returns
  signs `[−1, 1, 1, −1] = XOR` **exactly**.

So the escalation is precisely `3 → 4`: a linear readout has too few degrees of
freedom on 3-dim data to shatter four points in the XOR pattern; a quadratic
readout has enough (6 dof) to shatter *any* four general-position points. Note
the forward map is irrelevant here — an orthogonal `M` only rotates the same
3-dim data; the obstruction is the readout's degree.

**On the noisy dataset** (800/200, noise 0.1, seed 42), current linear
architecture, per readout (best TEST accuracy over draws, hit-rate = fraction of
draws with fitted train ≥ 0.80):

| readout on `x·w` | `sign(re)` | best linear | Hopf balance | **quadratic** |
|---|---|---|---|---|
| test acc (hit) | `0.750 (0.00)` | `0.640 (0.00)` | `0.600 (0.00)` | **`0.960 (1.00)`** |

Every linear-family readout sits at or below the `~0.72–0.75` linear ceiling
(isometry-ceiling §3); the learned quadratic form breaks it on **every** draw.
The single-`w` Hopf balance is *weak* on a bare linear layer (`0.600`): it is
one specific signature-(4,4) form, and a random `w` rarely aligns it to the XOR
hyperbola — the structured readout earns its keep only downstream of a
fiber-mixing combiner (§4, where it reaches `0.93–0.95`).

**Answer to (a): the forward map's linearity was never the obstruction to XOR —
the readout's degree was.** A purely orthogonal forward pass plus a quadratic
readout solves XOR. Nonlinearity in the forward map matters for tasks that are
*not* quadratic-separable-in-a-linear-image; XOR is not such a task.

## 4. Question (b): the readout-invisible mechanisms revive

For each mechanism the catalogue killed, the pre-readout output `Y(x)` is fed to
the five readouts. Cells are best TEST accuracy over 300 weight draws, with
(hit-rate). The methodological difference from the combiner screen: there the
readout was *fixed* and only weights were random (hit-rates ≤ 1%); here the
readout is *fitted per draw*, so a hit-rate of `1.00` means the mechanism
**reliably injects readout-visible XOR structure**, not that a rare weight was
found.

**Octonion weights (full S⁷):**

| mechanism | `sign(re)` | best linear | Hopf balance | quadratic | FF goodness |
|---|---|---|---|---|---|
| linear `x·w` (current) | 0.750 (0.00) | 0.640 (0.00) | 0.600 (0.00) | 0.960 (1.00) | — |
| branch product `(x·w₁)(x·w₂)` | 0.770 (0.00) | 0.960 (1.00) | 0.930 (0.10) | 0.970 (1.00) | — |
| **raw commutator** `[x·w₁,x·w₂]` | **0.515** (0.00) | **0.975** (1.00) | 0.940 (0.13) | 0.965 (1.00) | 0.900 (0.06) |
| **raw associator** `[x·w₁,c,x·w₂]` | **0.510** (0.00) | **0.980** (1.00) | 0.950 (0.08) | 0.985 (1.00) | 0.935 (0.07) |
| **additive assoc channel** | **0.755** (0.00) | **0.985** (1.00) | 0.895 (0.08) | 0.980 (1.00) | **0.960** (0.07) |
| **pure sandwich** `x̄·c·x` | **0.515** (0.00) | **0.960** (1.00) | 0.535 (0.00) | 0.970 (1.00) | — |

**Quaternion-control weights (subalgebra `span{1,e₁,e₂,e₃}`) — the ablation:**

| mechanism | `sign(re)` | best linear | quadratic | FF goodness |
|---|---|---|---|---|
| linear `x·w` | 0.760 | 0.640 | 0.960 | — |
| branch product | 0.715 | 0.970 | 0.975 | — |
| raw commutator | 0.515 | **0.965** (survives) | 0.980 | 0.950 |
| **raw associator** | 0.515 | **0.515** (dies) | **0.515** (dies) | 0.570 |
| additive assoc channel | 0.750 | **0.640** (collapses) | 0.960 | **0.455** (dies) |
| pure sandwich | 0.515 | 0.975 (survives) | 0.975 | — |

**Readings.**

- **Killed by the readout, not the mechanism.** Every raw imaginary-valued
  channel — commutator, associator, sandwich — reads `≈ 0.51` under `sign(re)`
  (pure imaginary / constant `re`, *verified* `re ≈ 2.8e−16`; the catalogue's
  kill) and `0.96–0.98` the instant a *linear* readout is allowed a non-`re`
  direction. The signal was always there; `re` is one blind direction. The
  additive associator channel reads `sign(re) = sign(re(x·w₀))` exactly
  (*verified*, agree `1.0000`; the catalogue's kill) yet revives to `0.985`
  under a linear readout and to `0.960` under **FF goodness on the
  pre-normalization magnitude** — the magnitude sees the associator through the
  cross term `2⟨x·w₀, [x·w₁,c,x·w₂]⟩` that `re` cancels.
- **The ℍ-control separates readout-revival from non-associativity.** The raw
  **associator dies on ℍ in every readout** (`0.98 → 0.515`, the maximal 𝕆/ℍ
  gap): its entire XOR signal is non-associativity, invisible to `re` but fully
  visible to a plain linear readout. The additive channel's linear/FF revival
  *also* collapses on ℍ (`0.985 → 0.640` ceiling; FF `0.960 → 0.455` dead),
  because on ℍ the channel reduces to the linear map `x·w₀` — confirming the
  revived signal is the associator, while its quadratic-readout number
  (`0.960`, unchanged on ℍ) is the *readout* doing the work, not the algebra.
  The **commutator** (non-commutativity) and **sandwich** (quadratic-in-`x` on
  any algebra) survive on ℍ — their readout-revival owes nothing to
  non-associativity.
- **Branch product** was already S2-visible to `re` (`0.770`; catalogue leader),
  and richer readouts only sharpen it — nothing new here, included as the "was
  never killed" calibration row.

**Answer to (b): the catalogue's "readout-invisible" verdicts were verdicts
about `sign(re)`, not about the mechanisms.** Under a degree-≥2 readout (or even
a linear readout in a chosen imaginary direction) the raw associator and
commutator channels break `0.75` decisively — and the associator does so purely
by non-associativity. The readout redesign the catalogue named as a *hard
dependency* for the associator channel (deep dive §3.4; catalogue A5) is
exactly what unlocks it.

## 5. Recommendation for #7 (shape of a wide octonion layer) and the prototype

**Where the readout sits is part of the layer's shape.** `sign(re)` is a
one-dimensional, symmetry-*forced* bottleneck: it is the canonical G₂-invariant
readout (§1) and it provably discards the SO(8) reach the layer's depth/width
generate. Reading that capacity requires deliberately breaking G₂.

**For #7, place a degree-≥2 readout at the head of the wide layer.** Ranked:

1. **Learned quadratic form `⟨y, Q y⟩`** (36 params), or equivalently a small
   **bank of `k` linear projections `⟨y, v_k⟩`** feeding a linear head. Maximal
   capability (breaks XOR at `0.96–0.98` on *any* mechanism, hit-rate `1.00`),
   cheap, and it is the direct way to read the SO(8) reach. This is the default
   recommendation. `re²` is its G₂-invariant special case — a principled
   initialization.
2. **Cayley–Dickson / Hopf fiber balance `|a|²−|b|²`** — the canonical
   *parameter-free* symmetry-breaking readout (SO(4) stabilizer, interpretable
   as the Hopf base coordinate). But it is **weak on a bare linear layer**
   (`0.600`) and only competitive downstream of a combiner that populates/mixes
   the second quaternion half (`0.93–0.95`). Use it as a structured, low-capacity
   option, not as a drop-in for a learned quadratic.
3. **FF-style goodness** on a *nonlinear* pre-normalization aggregation — a
   natural local (backprop-free) objective that pairs with the error-wave
   program, and the one readout that revives the additive associator channel
   (`0.960`). Dead on linear aggregations; only viable with combiners that leave
   S⁷ before renormalizing.

**Ablation discipline for the prototype (the load-bearing caveat).** Because a
quadratic readout on a *linear* forward map already solves XOR (§3), **XOR
success no longer certifies forward-map nonlinearity once the readout is
degree ≥ 2.** Concretely:

- The isometry-ceiling's "genuine nonlinearity" bar (§4.3: set XOR ≥ 90% to
  beat the 0.75 linear ceiling) was defined *relative to `sign(re)`*. Under a
  quadratic readout the linear-map XOR ceiling is not `0.75` but `~0.96`; the
  bar must move with the readout.
- Run the prototype as a **2×2 ablation**: `{linear forward, octonion combiner}
  × {sign(re), quadratic readout}`. The combiner earns its keep only if it beats
  the linear-forward baseline **under the same readout**. A combiner that only
  wins by being paired with a richer readout has bought a readout, not a
  nonlinearity.
- To certify forward-map nonlinearity specifically, either keep `sign(re)` as
  the certifying readout, or move to a task that is *not*
  quadratic-separable-in-a-linear-image (XOR is; most interesting tasks are
  not).

The cleanest one-line takeaway for the map: **the old fog line "is `sign(re)`
the flaw?" resolves to — `sign(re)` is the canonical G₂-invariant readout, and
that canonicity is exactly the flaw for a task that needs symmetry-breaking. A
degree-≥2 readout is the cheapest capacity gain in the whole program, and it
retroactively revives half the killed combiner catalogue.**

## References

- J. C. Baez, *The Octonions*, Bull. Amer. Math. Soc. 39 (2002) 145–205,
  [arXiv:math/0105155](https://arxiv.org/abs/math/0105155) — §4.1: G₂ = Aut(𝕆),
  transitive on S⁶ (stabilizer SU(3)), irreducible on `Im 𝕆 = ℝ⁷`; §3.1: the
  quaternionic Hopf fibration S³ → S⁷ → S⁴ and the split `x = (a,b)`.
- I. Schur — Schur's lemma: an irreducible real representation admits a
  one-dimensional space of invariant symmetric bilinear forms (used for the
  quadratic specialization in §1).
- G. E. Hinton, *The Forward-Forward Algorithm* (2022),
  [arXiv:2212.13345](https://arxiv.org/abs/2212.13345) — "goodness" as the sum
  of squared activations; the FF-goodness readout of §2/§4.
- Prior notes, all leaned on here: [isometry-ceiling.md](isometry-ceiling.md)
  (readout collapse, the 75% ceiling, the corner identity),
  [octonion-structure-deep-dive.md](octonion-structure-deep-dive.md) (§3 the
  associator, §4 G₂ / triality / rank-28 SO(8) reach),
  [s7-combiner-catalogue.md](s7-combiner-catalogue.md) (the "readout-invisible"
  kills — A2 raw commutator, A5 additive associator channel, A6 associator
  rotor, pure sandwiches).

Where a claim is not attributable to the above it is marked *own derivation*
and/or *(verified)* by [readout_screen.py](readout_screen.py); the orbit =
`re`-level-set theorem, the quadratic-invariant collapse to `span{|x|², re²}`,
the `3 → 4` shattering escalation, and the non-associativity attribution of the
associator's revival are established here.
