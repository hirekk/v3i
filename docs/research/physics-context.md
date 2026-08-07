# Physics context: hypercomplex algebra, gauge geometry, and this project

Recorded at the driver's request during the resolution of
[Where does nonlinearity come from? (#6)](https://github.com/hirekk/v3i/issues/6).
Not a work item — standing context connecting the project's mathematical
objects to established mathematical physics, and a source of candidate
mechanisms. Sibling notes:
[octonion-structure-deep-dive.md](octonion-structure-deep-dive.md),
[s7-combiner-catalogue.md](s7-combiner-catalogue.md).

## Maxwell and the algebra

Maxwell wrote parts of the 1873 *Treatise* in quaternion notation; Gibbs and
Heaviside's vector calculus is quaternion algebra split apart. In spacetime
(Clifford) algebra all four Maxwell equations are one: `∇F = J` (Hestenes);
the Riemann–Silberstein vector `F = E + icB` gives the biquaternion form.
"Electromagnetism is simple in the right algebra" is a theorem, not a hope.

## Hopf fibrations are gauge geometry

- **S³ → S² (complex Hopf)** *is* the Dirac magnetic monopole bundle: the
  monopole field is the fibration's geometry seen from the base.
- **S⁷ → S⁴ (quaternionic Hopf)** *is* the BPST instanton bundle of SU(2)
  Yang–Mills over compactified Euclidean spacetime: a trivially elegant
  structure upstairs whose projection is a topologically nontrivial field
  downstairs.
- Adams' Hopf-invariant-one theorem restricts such fibrations to dimensions
  1, 2, 4, 8 — the Hurwitz boundary again.

## Extra dimensions: Kaluza–Klein, S⁷, and G₂

Kaluza–Klein (1921): electromagnetism emerges from 5D gravity with a compact
circle fiber; gauge symmetry = isometries of the fiber. Witten (1981): the
minimal number of extra dimensions carrying the Standard Model's
U(1)×SU(2)×SU(3) is **seven** — hence 11D supergravity, whose most famous
compactifications are on the round and squashed **S⁷** (round S⁷ gives SO(8)
gauge fields — the same SO(8) whose generation by right multiplications the
isometry-ceiling note verified). The holonomy group of M-theory's 7D
compactification manifolds is **G₂**, the automorphism group of the
octonions. There is also a serious (unfinished) literature deriving Standard
Model structure from the division algebras: Günaydin–Gürsey (SU(3) as the G₂
stabilizer of one imaginary unit), Dixon, Furey, Baez.

## What this buys the project

1. **Fibration-aware combiners** — the catalogue's Hopf fiber twist (kept as
   the fibration control) samples exactly the monopole/instanton geometry.
   A future candidate left in fog: gating by the canonical connection 1-form
   of the quaternionic Hopf bundle (the BPST potential itself).
2. **Chirality of the error wave** — physics runs on parity; the algebra
   makes side-choice meaningful: `[L_a, R_b]x = a(xb) − (ax)b = −[a, x, b]`,
   so a left-acting correction against a right-acting forward pass differs
   from same-side accounting by pure associator terms — identically zero on
   ℍ. Explored in the ticket "Chirality of the error wave".
3. **G₂-equivariance** (deep-dive note §4) — the symmetry the physics
   compactifications gauge is the same one available here for canonical
   initialization and invariance tests.

## References

- J. C. Maxwell, *A Treatise on Electricity and Magnetism* (1873).
- D. Hestenes, *Space-Time Algebra* (1966).
- P. A. M. Dirac, *Quantised singularities in the electromagnetic field*,
  Proc. R. Soc. A 133 (1931) — monopole ↔ Hopf bundle.
- Belavin, Polyakov, Schwartz, Tyupkin, *Pseudoparticle solutions of the
  Yang–Mills equations*, Phys. Lett. B 59 (1975) — the BPST instanton.
- J. F. Adams, *On the non-existence of elements of Hopf invariant one*,
  Ann. Math. 72 (1960).
- E. Witten, *Search for a realistic Kaluza–Klein theory*, Nucl. Phys. B 186
  (1981).
- M. J. Duff, B. E. W. Nilsson, C. N. Pope, *Kaluza–Klein supergravity*,
  Phys. Rep. 130 (1986) — round and squashed S⁷.
- M. Günaydin, F. Gürsey, *Quark structure and octonions*, J. Math. Phys. 14
  (1973); C. Furey, *Standard model physics from an algebra?* (2016),
  arXiv:1611.09182; J. Baez, *The Octonions*, Bull. AMS 39 (2002).
