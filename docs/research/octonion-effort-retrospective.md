# The octonion architecture effort: design, negative result, and pivot

Consolidated retrospective of the widened-octonion-architecture wayfinder effort
(old map [#1](https://github.com/hirekk/v3i/issues/1), superseded). It records the
design that was decided, the prototype result that falsified it, *why* it failed,
and the pivot to quaternions — so the back-burnered octonion problem has one place
to start from. Companion research notes under `docs/research/` remain valid and are
linked throughout.

## TL;DR

- We designed a widened octonion architecture — a three-slot layer, a
  branch-product combiner, and a **no-backprop forward-error-wave** learning rule
  (**peel-and-solve**) — plus a degree-graded **parity-ladder** benchmark.
- A throwaway prototype (branch
  [`prototype/wide-octonion-parity`](https://github.com/hirekk/v3i/tree/prototype/wide-octonion-parity))
  proved the central bet: **gradient-free peel-and-solve learning works** —
  3-bit parity to **1.00**, one epoch, no backprop — **but only in the associative
  (quaternion) case.** Octonions structurally cap at a partial, unstable **~0.71**.
- **Root cause:** for a multiplicative product combiner, "correct a branch" ≡
  "correct the output" *only under associativity*. Non-associativity decouples them;
  the update direction becomes wrong (not just the step size).
- **Outcome:** pivoted to **quaternion-primary** (new map
  [#16](https://github.com/hirekk/v3i/issues/16)). Octonions remain the primary open
  problem, **on the back-burner** — deferred, not abandoned.

## 1. What was designed (the octonion architecture)

The effort converged, decision by decision, on a clean architecture:

- **Layer** ([#7](https://github.com/hirekk/v3i/issues/7)): a three-slot composition
  `y = M(T(x,w₁),…,T(x,w_W))` — a branch **transform** T (the "neural computation"
  axis: one-sided `x·w`, affine bias, two-sided, and exotic linear branches), a
  **combiner** M, a frozen **readout** R = `sign(re)`. One unit octonion in, one out;
  width and depth free.
- **Combiner / nonlinearity** ([#6](https://github.com/hirekk/v3i/issues/6)): a race
  set — branch product, kappa-gated slerp, triple cross, commutator rotor — since a
  stack of `y = x·w` layers is provably a single orthogonal map (the **isometry
  ceiling**, [docs/research/isometry-ceiling.md](isometry-ceiling.md)).
- **Error wave** ([#8](https://github.com/hirekk/v3i/issues/8)): a forward-propagated,
  right-chirality, **peel-and-solve** rule — the design's most elegant idea (below).
- **Benchmark** ([#5](https://github.com/hirekk/v3i/issues/5)): a **parity ladder** on
  the centered cube (k-bit parity needs a degree-≥k readout, closing the loophole
  that a quadratic readout alone solves XOR — [readout-screen.md](readout-screen.md)),
  with `sign(re)` frozen as the certifying readout and a two-tier failure criterion.

**The elegant core — peel-and-solve.** Because octonions are a division algebra,
every combiner is locally invertible, so credit assignment is **reconstruct, not
decompose**: tell each branch what it *should have been* by inverting the combiner
around it (`bᵢ* = Lᵢ⁻¹·y*·Rᵢ⁻¹`), then take a geodesic weight step toward it. This is
the gradient-free analog of the chain rule, and the deepest reason the project
insisted on a division algebra. It was the right idea — for the wrong algebra.

## 2. What the prototype found ([#9](https://github.com/hirekk/v3i/issues/9), [#15](https://github.com/hirekk/v3i/issues/15))

3-bit parity, single wide layer, branch product, 5 seeds:

| learning rule / variant | octonion | quaternion |
|---|---|---|
| **peel-and-solve** (exact nested peel) | 0.71 (partial, unstable; one seed 1.00) | **1.00, clean, one epoch** |
| nested peel + Gauss-Seidel | 0.49 | — |
| nested peel + adaptive-η (enforce invariant) | 0.52 | — |
| alignment (torque + kappa, ±sign) | 0.43–0.53 | 0.54–0.76 (unstable) |
| hybrid (exact-outer + alignment-inner) | 0.57 | 0.75 |
| linear (W=1) / renorm-sum | chance | chance |

- **Findability confirmed:** the no-backprop wave *learns a genuinely nonlinear
  function* (parity is degree-k; no linear method exceeds chance). This is the real,
  positive result of the whole project.
- **Octonions fail structurally:** seven rule variants, none reaching the clean
  quaternion 1.00. The peel's associative-approximation bug (only the outermost
  factor of a W≥3 product peels exactly) was found *and fixed* (exact nested peel),
  and octonions still didn't stabilize.

## 3. Why octonions fail (the core finding)

The exact nested peel gives each octonion branch the **correct target**, yet moving
its weight toward that target moves the *output* the wrong way most of the time.
The decisive evidence was **adaptive-η**: backtracking the step toward zero should
drive the residual ratio to 1, but it stayed `> 1` for **74%** of updates even at
η ≈ 0.005. A wrong step size can be shrunk; a wrong *direction* cannot. So the
peel-and-solve update points the output in an error-*increasing* direction under
non-associativity.

Stated cleanly: **peel-and-solve requires associativity.** In an associative
algebra, "fix the branch" and "fix the output" are the same operation; in a
non-associative one the product's dependence on each branch is tangled, and the two
decouple. Quaternions have it (clean 1.00); octonions don't (≤0.71).

The project's *original* rule — the torque/7D-cross-product/kappa **alignment**
heuristic, which was built *around* non-associativity — was also tested and also
failed (unstable, ≤0.76 even for quaternions in a rough wide-layer adaptation). So
neither the exact rule nor the alignment rule unlocks octonions here.

## 4. What survives (still-valid research)

The pivot invalidates the octonion *destination*, not the research. All of the
following remain sound and are the foundation for both the quaternion effort and any
future octonion attempt:

- [isometry-ceiling.md](isometry-ceiling.md) — the linear ceiling and why nonlinearity is mandatory.
- [octonion-structure-deep-dive.md](octonion-structure-deep-dive.md) — Moufang/associator/G₂/S⁷ facts (and the `cross_product_7d` bug fix).
- [hypercomplex-backprop-free-literature.md](hypercomplex-backprop-free-literature.md) — prior art; the triad's intersection is unoccupied.
- [readout-screen.md](readout-screen.md) — `sign(re)` is the canonical G₂-invariant readout; the parity-ladder rationale.
- [error-wave-chirality.md](error-wave-chirality.md), [s7-combiner-catalogue.md](s7-combiner-catalogue.md), [physics-context.md](physics-context.md).

**The negative result is itself a contribution:** "gradient-free peel-and-solve
credit assignment requires associativity" is a crisp, defensible claim that scopes
where this class of learning rule can work.

## 5. The pivot, and what's open for octonions

The effort pivoted to **quaternion-primary** — bank the demonstrated no-backprop
learning, understand it, validate on MNIST, and benchmark it against backprop (new
map [#16](https://github.com/hirekk/v3i/issues/16)).

**Open leads for a future octonion effort** (the back-burner problem):

1. **The 0.71 is partial, not chance** — one seed hit 1.00. There may be a
   stabilization (initialization, per-branch damping, a curvature-aware step) that
   the seven quick experiments missed.
2. **A genuinely non-associative combiner.** Peel-and-solve on a *product* needs
   associativity; a combiner whose credit assignment *doesn't* route through a
   multiplicative inverse (the associator-native mechanisms, or a rule matched to
   the Moufang identities) might make non-associativity an asset instead of an
   obstruction.
3. **A different learning rule entirely** — the whole point of the quaternion map is
   to understand peel-and-solve well enough to form an *evidence-based* hypothesis on
   whether, and how, octonions could ever pay off.
4. **Cache forward intermediates to correct the associator defect** (driver idea,
   sharpened by the #17 theory). The theory note pins the obstruction to *one*
   associator, `[x, wᵢ, sᵢ]` — the input, the weight, and the update rotation — which
   is irreducible because Artin only makes *two*-generator subalgebras associative.
   Its three ingredients are all available at forward time. So the "octonion neurons
   cache values as the signal propagates and reuse them for error contribution"
   idea has a concrete target: cache the forward intermediates (`x`, `wᵢ`, `bᵢ`, the
   bracketing) and, at error time, evaluate the associator defect and **correct the
   weight update to cancel it**, so the realized branch lands on its peeled target
   despite non-associativity — a Newton/fixed-point correction seeded by cached
   forward state, not the plain peel that failed. Caveats: the correction is
   *circular* (`sᵢ` sits inside the defect → needs iteration, convergence unproven,
   and the plain rule's direction was wrong ~74% of the time); it edges toward
   backprop-style activation caching (still gradient-free, but a deliberate departure
   from strict forward-only). A more radical variant: cache to feed a
   *non-associative-native* credit rule where the cached partial products *are* the
   signal, rather than patching peel-and-solve. Driver wants to explore this.

The bar a future octonion attempt must clear is now concrete: **beat the clean
quaternion 1.00 on the parity ladder, stably, across seeds.**
