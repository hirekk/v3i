# Hypercomplex networks, backprop-free learning, manifold training: a survey

Research note for the wayfinder ticket
[Literature scan: hypercomplex networks and backprop-free learning](https://github.com/hirekk/v3i/issues/3).
The lens throughout is the project triad: **(1)** division-algebra signals and
weights, **(2)** unit-norm weights on S⁷ with geodesic updates, **(3)** a
forward-propagated error wave instead of backprop. Every citation below was
checked against a primary source (arXiv abstract or full text, publisher page,
or proceedings record) during this survey; where a claim could not be verified
against a source, that is flagged inline. Benchmark numbers relayed from
full-text extractions (ar5iv/PDF) rather than abstracts are marked as such.

## TL;DR

1. **No prior work holds all three commitments at once.** Hypercomplex deep
   nets keep the algebra but backprop through componentwise real activations
   with unconstrained norms. Unitary/orthogonal RNNs keep manifold constraints
   and even own a norm-compatible nonlinearity (modReLU) — but backprop.
   Forward-only learning keeps the signal direction — but in real algebra with
   no manifold. The triad's combination appears to be unoccupied territory.
2. **Activation verdict from Area 1:** every verified quaternion/octonion deep
   net uses *split* (componentwise real) activations, which destroy unit norm
   and ignore the algebra product; Parcollet et al. explicitly avoid "pure"
   quaternion activations because of singularities. The one norm-compatible
   activation family in the wild is **modReLU** — but it acts on the modulus,
   which is *constant* on S⁷ signals, so a literal port is inert. The usable
   adaptation is to act on the **polar angle** instead (§4, row A).
3. **Closest relatives of the error wave** (Area 2): Kohan et al.'s error
   forward-propagation (error re-enters the input side and travels through the
   *same* forward weights), Nøkland's IFA (update direction forward-propagated
   from the first hidden layer), and PEPITA (error perturbs the input for a
   second forward pass). This repo's own stacked-perceptron design doc is an
   independent member of that family. No verified prior art exists for the
   current wave's conjugation-transport + projection-subtraction absorption.
4. **Scaling honesty:** every forward-only/local scheme is demonstrated at
   MNIST/CIFAR scale only; Bartunov et al. documented near-chance ImageNet
   failures for FA/target-prop, and Launay et al. state DFA cannot train
   conv layers. Only greedy layerwise auxiliary training reached ImageNet.
   For v3i's synthetic-only evidence bar this is tolerable, not encouraging.
5. **Manifold training is mature** (Area 3): geodesic/retraction SGD with
   Robbins–Monro steps is proven (Bonnabel); Cayley and exponential-map
   parameterizations are standard. Two warnings transfer directly: hard
   orthogonality constraints can *slow and hurt* training (Vorontsov et al.),
   and low-parameter unitary families are capacity-restricted (Wisdom et al. —
   the exact analogue of the 7-parameter `{R_v}` family inside 28-dim SO(8)
   from the isometry-ceiling note).
6. **A structural no-go for the widened architecture** (own derivation, §4.0):
   linear width-aggregation collapses. By distributivity `Σᵢ x·wᵢ = x·(Σᵢwᵢ)`,
   so sum-then-normalize over branches sharing an input *is* a single current
   layer; more generally any sum/normalize chain stays projectively linear and
   the `sign(re(·))` classifier stays under the 75% XOR ceiling. Nonlinearity
   must enter the aggregation or the activation itself — see the shortlist.

## 1. Hypercomplex neural networks

### 1.1 Parcollet et al. — quaternion RNNs and CNNs for speech

**Idea.** The QRNN/QLSTM paper (Parcollet, Ravanelli, Morchid, Linarès,
Trabelsi, De Mori, Bengio, ICLR 2019, [arXiv:1806.04418](https://arxiv.org/abs/1806.04418))
replaces real-valued dense maps with quaternion Hamilton products, arguing the
algebra encodes internal dependencies among grouped input features "similarly
to capsules". On TIMIT phoneme recognition the QLSTM reaches 15.1% PER vs
15.3% for the real LSTM at 14.4M vs 46.2M parameters — the abstract's "up to
3.3× fewer parameters" claim checks out (table numbers via ar5iv extraction).
Companion papers apply the same machinery convolutionally: end-to-end QCNN+CTC
(Interspeech 2018, [arXiv:1806.07789](https://arxiv.org/abs/1806.07789)):
19.64% PER at 8.1M params vs 20.57% at 32.1M for the real CNN (~4×); and the
NeurIPS 2018 IRASL workshop paper
([arXiv:1811.09678](https://arxiv.org/abs/1811.09678)) with QRNN 18.5%/3.8M vs
RNN 19.0%/9.4M. Activations everywhere are **split**: a real nonlinearity
applied per component, `α(Q) = f(r) + f(x)i + f(y)j + f(z)k` — tanh in the
RNNs, PReLU in the CNNs. They explicitly reject "pure quaternion activation
functions" because they "contain singularities". The group's survey (Parcollet,
Morchid, Linarès, *A survey of quaternion neural networks*, Artif. Intell.
Review 53:2957–2982, 2020, [DOI 10.1007/s10462-019-09752-1](https://doi.org/10.1007/s10462-019-09752-1))
is the field's reference; note: only its bibliographic metadata was verified
here, not its content, since the publisher page blocked fetching.

**Triad fit.** Preserves commitment (1) in the forward *linear* map only —
the Hamilton product is real quaternion algebra — but the split activation
breaks (1) between layers (componentwise ops are not algebra maps) and breaks
(2) outright: no norm constraint anywhere, and split ReLU does not even
preserve norms, let alone the sphere. Training is plain backprop through the
real components, violating (3). What is worth borrowing is the *skepticism
calibration*: the parameter-efficiency wins are real but modest in accuracy
(0.2–0.5 PER points), on one task family (speech), against baselines the
authors sized themselves; and the singularity warning about "pure" hypercomplex
activations is a direct caution for any S⁷ activation with a polar
decomposition (§4 row A has the same singular set at `y = ±1`).

### 1.2 Gaudet & Maida — Deep Quaternion Networks

**Idea.** Gaudet & Maida (IJCNN 2018,
[arXiv:1712.04604](https://arxiv.org/abs/1712.04604)) supply the
infrastructure pieces: quaternion convolution, a quaternion weight
initialization derived Glorot/He-style, and quaternion batch normalization
(whitening the 4D covariance). Verified from the PDF: the nonlinearity is
plain **real ReLU** on the component representation inside standard
`BN → ReLU → Conv` residual blocks — the paper never defines a
quaternion-specific activation. Results: CIFAR-10 error 5.44% (quaternion) vs
5.60% (complex, Trabelsi et al.) vs 6.37% (real); CIFAR-100 26.01/27.09/28.07;
KITTI road segmentation IOU 0.827 vs 0.769/0.747, with fewer parameters
(numbers via ar5iv extraction).

**Triad fit.** Same profile as 1.1: algebra in the linear map only, no norm
constraint, backprop — (1) partial, (2) and (3) violated. The borrowable
assets are indirect: the initialization lesson (they derive variance rules per
algebra; v3i's identity-plus-perturbation init on S⁷ has no analogous scale
analysis yet, which the map already lists as unspecified), and the observation
that the *whole* quaternion-CNN literature gets its nonlinearity from ordinary
real activations — evidence that nobody has solved, or even posed, the
"activation on the unit sphere of the algebra" problem the triad forces.

### 1.3 Zhu et al. — rotation-based quaternion convolution

**Idea.** Zhu, Xu, Xu, Chen (ECCV 2018,
[DOI 10.1007/978-3-030-01237-3_39](https://doi.org/10.1007/978-3-030-01237-3_39),
[arXiv:1903.00658](https://arxiv.org/abs/1903.00658)) design quaternion
convolution as **rotation**: RGB lives in the imaginary part, and each kernel
weight acts by rotating color vectors (a sandwich-style transform) rather than
by arbitrary linear mixing. Activations are componentwise ReLU per channel,
plus a color-space projection trick (invalid colors reset to the nearest valid
point). Verified caveat for the efficiency narrative: this design uses ~2×
parameters per kernel versus a real kernel — it wins on representation
(CIFAR-10 shallow 77.78% vs 75.46%; Flowers VGG-S 76.95% vs 73.08%), not on
compression, though it still won at a matched budget (76.03% vs 73.08%).

**Triad fit.** The most triad-resonant idea in Area 1: weights *as rotations*
is exactly the S⁷ picture (unit octonion ↦ orthogonal map), and their
"rotation, not arbitrary mixing" argument is a philosophical ally. But norms
are unconstrained, activations are split, training is backprop — (2) and (3)
violated, (1) partial. Borrowable: the demonstration that constraining weights
to act as rotations *costs accuracy nothing* on small vision tasks; and the
honest lesson that hypercomplex parameter "savings" are design-dependent, not
automatic (their QCNN spends parameters instead of saving them).

### 1.4 Octonion networks — Wu et al. and the sparse rest

**Idea.** The only substantial deep octonion paper found: Wu, Xu, Kong,
Senhadji, Shu, *Deep Octonion Networks* (Neurocomputing 397:179–191, 2020,
[DOI 10.1016/j.neucom.2020.02.053](https://doi.org/10.1016/j.neucom.2020.02.053),
[arXiv:1903.08478](https://arxiv.org/abs/1903.08478)): octonion convolution,
octonion batch norm, octonion init, with layers `OctonionConv → OctonionBN →
ReLU` — again a **real split ReLU** on the 8-component representation
(verified from the PDF). Reported Table-3 progression (params / CIFAR-10
error): real 3.62M/6.37, complex 1.82M/5.60, quaternion 0.93M/5.44, octonion
0.48M/5.35, and CIFAR-100 24.60 for the octonion net — but note the
real/complex/quaternion rows are numbers *quoted from the earlier papers*, not
re-runs at matched budget. Prior/adjacent art: Popa's feedforward
octonion-valued networks (ICANN 2016,
[DOI 10.1007/978-3-319-44778-0_51](https://doi.org/10.1007/978-3-319-44778-0_51);
metadata verified, abstract inaccessible), Cariow & Cariowa's fast octonion
layer algorithms (IEEE TNNLS 34(1):543–548, 2023; found via DBLP, not further
verified), and a stability-analysis line on octonion-valued recurrent dynamics
(not deep learning). **No octonion analogue of the QRNN exists** as far as
this search found.

**Triad fit.** The nearest published neighbor in algebra choice, and the
furthest in spirit: non-associativity is treated as an implementation obstacle
to route around (their convolution is defined to avoid bracketing issues),
whereas v3i's kappa machinery treats the associator as *signal*. Split ReLU
breaks (1)/(2); backprop breaks (3). Two harvests: first, the multi-task-style
argument for why weight-sharing across the 8 components helps is a usable
narrative for octonion width; second, the field's silence on non-associative
structure means the associator-as-mechanism direction (§4 rows D, F) is
genuinely unclaimed — with the corresponding risk that it is unclaimed because
nobody has made it work.

### 1.5 Zhang et al. — parameterized hypercomplex multiplication (PHM)

**Idea.** Zhang, Tay, Zhang, Chan, Luu, Hui, Fu (ICLR 2021,
[arXiv:2102.08597](https://arxiv.org/abs/2102.08597)) generalize the Hamilton
product: a PHM layer *learns* the multiplication rule as a sum of Kronecker
products, subsuming quaternion multiplication as a special case and giving
`1/n` parameters versus a real dense layer for arbitrary `n` (not just 4 or
8). Demonstrated inside LSTMs and Transformers on NLI, machine translation,
style transfer, and subject–verb agreement (abstract-level verification;
per-dataset numbers not verified here). Grassucci, Zhang, Comminiello extend
this to convolutions — PHNNs (IEEE TNNLS,
[DOI 10.1109/TNNLS.2022.3226772](https://doi.org/10.1109/TNNLS.2022.3226772),
[arXiv:2110.04176](https://arxiv.org/abs/2110.04176)) — "1/n free parameters",
outperforming real and quaternion counterparts on image and audio tasks
(abstract; specifics unverified).

**Triad fit.** PHM deliberately *abandons* commitment (1): the learned bilinear
map is generically not a division algebra — no norm composition, no
inverses — so `|x·w| = |x||w|` fails and the whole S⁷ geometry underneath
commitments (2)–(3) evaporates. It is the field's answer to "why these
algebras?" — *don't privilege them, learn the product* — and is therefore the
principal rival hypothesis to v3i's bet that the division-algebra structure
itself (norm composition, associator) carries value. Borrowable: the framing
of hypercomplex layers as structured Kronecker factorizations is the right
formal language for arguing what octonion width does and does not span; and
PHM is the natural ablation *outside* the triad if the project ever needs to
show the division property matters.

### 1.6 Trabelsi et al. — Deep Complex Networks (the activation evidence)

**Idea.** Trabelsi et al. (ICLR 2018,
[arXiv:1705.09792](https://arxiv.org/abs/1705.09792)) built the complex-valued
toolkit (complex conv, BN, init) and — most relevantly — ran the direct
activation bake-off: **modReLU** `= ReLU(|z|+b)·e^{iθ_z}` (magnitude
thresholded, phase preserved), **CReLU** `= ReLU(ℜz)+i·ReLU(ℑz)` (split), and
**zReLU** (pass iff phase ∈ [0, π/2]). Verified from the full text: CReLU won
decisively — CIFAR-10 errors 6.17% (CReLU) vs 11.71% (zReLU) vs 23.42%
(modReLU) on their wide-shallow architecture, with modReLU and zReLU "largely
outperformed" across CIFAR-10/100 and SVHN (numbers via ar5iv extraction; some
instability attributable to their naive complex-BN variant, which failed in 5
of 6 experiments).

**Triad fit.** This is the sharpest empirical warning in the survey. The one
activation family that respects norm structure (modReLU — the only candidate
compatible in spirit with sphere-valued signals) *lost badly* to the
structure-breaking split activation in deep feedforward nets, in the only
controlled comparison found. The triad cannot take the CReLU escape route:
split activations destroy S⁷. So the project must either make an
angle-domain modReLU work where magnitude-domain modReLU underperformed
(§4 row A), or get nonlinearity from multiplicative/algebraic interactions
instead of pointwise activations (§4 rows C–F). Either way, this citation is
the null hypothesis the XOR gate has to beat.

## 2. Backprop-free learning

### 2.1 Hinton — the Forward-Forward algorithm

**Idea.** Hinton (2022, [arXiv:2212.13345](https://arxiv.org/abs/2212.13345),
preprint only) replaces backprop's forward+backward passes with two forward
passes — positive (real) data and negative data — where each layer maximizes a
local **goodness** (sum of squared ReLU activities) on positive data and
minimizes it on negative data; no derivatives cross layer boundaries. The
detail that matters here (verified): between layers the activity vector is
**length-normalized** — "the length is used to define the goodness for that
layer and only the orientation is passed to the next layer" — precisely so a
layer cannot inherit its predecessor's goodness. Results are honest and
modest: MNIST ~1.36–1.37% error vs ~1.4% backprop; CIFAR-10 41–46% vs 37–39%
backprop; Hinton states it is slower, generalizes somewhat worse, and that
large-scale behavior "remains to be seen".

**Triad fit.** Violates (1) and (2) — real algebra, no weight manifold — but
its signal geometry is strikingly triad-shaped: FF factors every layer's
output into *orientation* (forwarded) and *length* (consumed locally as the
teaching scalar), which is exactly the S⁷-normalization decomposition. The
transferable mechanism: in a widened v3i layer, the pre-normalization
magnitude of an aggregated branch sum (`|Σᵢ hᵢ|`, discarded when projecting
back to S⁷) is a free, local, FF-style goodness scalar — usable as a
confidence gate or a local objective without touching the wave (§4 rows B, G).
Caution transfers too: FF's layer-local objectives needed engineered negative
data, and its CIFAR gap shows local goodness alone underperforms global
credit assignment.

### 2.2 Feedback alignment — Lillicrap et al.

**Idea.** Lillicrap, Cownden, Tweed, Akerman (arXiv preprint "Random feedback
weights support learning in deep neural networks",
[arXiv:1411.0247](https://arxiv.org/abs/1411.0247), 2014; journal version
"Random synaptic feedback weights support error backpropagation for deep
learning", Nature Communications 7:13276, 2016,
[DOI 10.1038/ncomms13276](https://doi.org/10.1038/ncomms13276) — two different
titles for the two versions) showed that replacing the transpose `Wᵀ` in
backprop with a *fixed random matrix* `B` still trains networks: the forward
weights drift into alignment with `Bᵀ`, so the random feedback becomes a
useful teacher — "the network learns to learn". Demonstrated scope is linear
problems and MNIST-scale (including spiking nets); no CIFAR or ImageNet in the
paper.

**Triad fit.** Still a backward pass — the teaching signal travels against the
data direction through dedicated feedback weights — so (3) is violated in
letter, though not in the way backprop is: FA proves that *exact adjoint
transport is unnecessary*; a fixed, wrong, but consistent channel suffices
because the forward weights adapt to it. That is the best available
theoretical comfort for v3i's wave, whose conjugation transport
`w̄·r·w` is likewise a principled-but-not-gradient channel: FA suggests layers
can learn to make an imperfect teaching channel useful. (1) and (2) are simply
out of scope (real algebra, unconstrained). Borrow the *alignment diagnostic*:
measure the angle between the wave's per-layer update and the true gradient of
the output loss during training — FA's alignment curves are the template for
showing the wave carries real credit information.

### 2.3 Direct feedback alignment and its scaling study

**Idea.** Nøkland (NeurIPS 2016,
[arXiv:1609.01596](https://arxiv.org/abs/1609.01596)) simplified FA: project
the *output* error directly to every hidden layer through its own fixed random
matrix (`δaᵢ = (Bᵢe) ⊙ f′(aᵢ)`) — no sequential backward chain at all. The
same paper contains **IFA**: the error enters once at the *first* hidden layer
through a single random loop, and the update direction is then
**forward-propagated** up the stack — the closest published relative of a
forward wave in the FA family. Verified honesty: parity with backprop holds on
fully-connected MNIST/CIFAR nets; on a convnet "BP is clearly the best
performer". Launay, Poli, Boniface, Krzakala (NeurIPS 2020,
[arXiv:2006.12878](https://arxiv.org/abs/2006.12878)) scaled DFA to modern
tasks: it trains NeRF-style MLPs, recommender models (DeepFM, DCN, AFN), and
graph nets near backprop level, but Transformers lag badly (perplexity 52–93
vs ~30 for BP on WikiText-103) and — verbatim — "DFA is unable to train
convolutional layers".

**Triad fit.** DFA violates (3)'s letter (a global broadcast, not a wave) but
offers two mechanisms worth stealing. First, IFA is direct precedent that
"inject the teaching signal early, let it ride the forward direction" can
train deep stacks — the wave is IFA with the random loop replaced by principled
algebra (log-map error, conjugation transport). Second, DFA's per-layer random
projections of one global error suggest a *hybrid wave*: instead of each layer
absorbing from the residual left by its predecessor (which risks
starving late layers), every layer could receive the same output tangent
error, transported into its own frame by its own weights — same locality, no
sequential depletion. Scaling caution transfers: DFA's failure on conv layers
and Transformers shows forward/broadcast teaching signals struggle exactly
where geometry-heavy credit assignment is needed.

### 2.4 Bartunov et al. — the scalability audit

**Idea.** Bartunov, Santoro, Richards, Marris, Hinton, Lillicrap (NeurIPS
2018, [arXiv:1807.04587](https://arxiv.org/abs/1807.04587)) stress-tested
FA/DFA/target-prop variants on MNIST, CIFAR, and ImageNet with fully- and
locally-connected architectures. Verified result: on ImageNet "all
biologically motivated algorithms performed very poorly relative to BP" —
DTP/SDTP ~98–99% top-1 error (essentially chance at 1000 classes), FA ~93%,
vs ~71% for BP on the same locally-connected net (numbers via extraction;
DFA was not even runnable at that scale for memory reasons). Gaps already
widen on CIFAR-10, and locally-connected architectures are consistently the
hardest for the biologically motivated methods.

**Triad fit.** Not a mechanism source — a calibration instrument. It sets the
honest prior for commitment (3): *no* local or feedback-free scheme audited
here survived contact with large-scale structured tasks, and the failures
concentrate where architectures get geometric. v3i's synthetic-only evidence
bar (XOR gate) is on the right side of this evidence; the note-worthy borrow
is methodological — Bartunov et al.'s design (same architecture, swap only the
credit-assignment rule) is exactly the wave-vs-oracle ablation the prototype
tickets should run, with the quaternion/associative control playing the role
of their architecture sweeps.

### 2.5 Target propagation — Lee et al.

**Idea.** Difference target propagation (Lee, Zhang, Fischer, Bengio,
ECML-PKDD 2015, [arXiv:1412.7525](https://arxiv.org/abs/1412.7525),
[DOI 10.1007/978-3-319-23528-8_31](https://doi.org/10.1007/978-3-319-23528-8_31))
propagates **targets** (desired activations), not gradients: each layer learns
an approximate inverse `gᵢ ≈ fᵢ⁻¹` (autoencoder-style), and targets flow down
via the difference correction `ĥᵢ₋₁ = hᵢ₋₁ + gᵢ(ĥᵢ) − gᵢ(hᵢ)`, which
guarantees a fixed point under imperfect inverses. The top target still comes
from a gradient step at the output layer. Scope: fully-connected MNIST/CIFAR
(7-layer MNIST 1.94% vs 1.86% BP; CIFAR-10 FC 50.71% accuracy vs 53.72% —
conditions not fully verified); the SDTP variant that removes the last
gradient step is what collapsed on ImageNet in §2.4.

**Triad fit.** Targets flow backward, so (3) is violated directionally — but
DTP is the scheme whose *content* most resembles the wave's: v3i's error
`log(p̄·t)` is literally a target expressed as a geodesic ("the rotation that
carries output to target"), i.e. the wave already propagates DTP-style targets
rather than gradients, just forward. Two borrows: on a group manifold the
inverse network `gᵢ` comes for free (`fᵢ⁻¹(y) = y·w̄` exactly — no autoencoder
training, no approximation error), so a triad-compatible "difference target"
correction is implementable *exactly*; and DTP's fixed-point lemma is the
right formal template for proving the wave's absorption step is stable (each
layer stops updating when the residual hits zero). The ImageNet failure of its
gradient-free variant is the standing caution.

### 2.6 Predictive coding — Rao & Ballard; Millidge et al.

**Idea.** Rao & Ballard (Nature Neuroscience 2(1):79–87, 1999,
[DOI 10.1038/4580](https://doi.org/10.1038/4580)) modeled visual cortex with
predictions flowing down and **residual errors flowing up the feedforward
direction** — error-carrying units even reproduce endstopping. Millidge,
Tschantz, Buckley (Neural Computation 34(6):1329–1368, 2022,
[DOI 10.1162/neco_a_01497](https://doi.org/10.1162/neco_a_01497),
[arXiv:2006.04182](https://arxiv.org/abs/2006.04182); beware — one automated
index falsely lists a NeurIPS venue) proved predictive coding converges to
*exact backprop gradients* on arbitrary computation graphs using only local
Hebbian-style updates, under a "fixed prediction assumption", verified at
CIFAR scale on CNNs/RNNs/LSTMs. The review (Millidge, Seth, Buckley,
[arXiv:2107.12979](https://arxiv.org/abs/2107.12979), arXiv-only) frames the
field.

**Triad fit.** Predictive coding is the *biologically canonical* forward-error
scheme — errors ride the feedforward pathway, as in the wave — but its
mechanics differ: PC needs an iterative relaxation to equilibrium per input
and per-layer error units, whereas the wave is single-pass. It violates (1)
and (2) (real algebra, no manifold) and its backprop-equivalence result cuts
both ways for (3): local error-passing *can* recover exact gradients, but only
in the relaxation limit — a single-pass wave is strictly an approximation, and
PC quantifies what is being given up. Borrowable: the prediction/error-unit
split suggests a wave variant where each layer maintains a running local
target (its expected output) and absorbs only the discrepancy — a smoother,
recurrent version of the current one-shot projection-subtraction accounting.

### 2.7 Synthetic gradients; greedy and local layerwise training

**Idea.** Jaderberg et al. (ICML 2017, PMLR 70:1627–1635,
[arXiv:1608.05343](https://arxiv.org/abs/1608.05343)) decouple layers by
*predicting* gradients with small local models — solving update locking, but
still targeting backprop's gradient (an approximation of it, learned). At the
fully-local end: Belilovsky, Eickenberg, Oyallon (ICML 2019, PMLR 97:583–593,
[arXiv:1812.11446](https://arxiv.org/abs/1812.11446)) showed greedy layerwise
training with shallow auxiliary classifiers scales to ImageNet (exceeding
AlexNet, approaching VGG) — the **only** non-end-to-end scheme in this survey
verified at that scale; and Nøkland & Eidnes (ICML 2019, PMLR 97:4839–4850,
[arXiv:1901.06656](https://arxiv.org/abs/1901.06656)) train each layer with a
detached local loss — a similarity-matching term plus a local classifier —
reaching ~3.97% CIFAR-10 error, 9.02% in the fully backprop-free variant.

**Triad fit.** All three violate (1)/(2); their relation to (3) is
instructive. Synthetic gradients concede that the *content* of backprop is
worth approximating even when its *mechanics* are abandoned — the wave, by
contrast, replaces the content too, which is the riskier bet. The layerwise
results are the strongest evidence that purely local objectives can train
deep stacks, and they port cleanly: a per-layer goodness or
similarity-matching objective on S⁷ (e.g. match the Gram matrix of layer
outputs to the label Gram matrix, with cosine similarity being *native* on the
sphere) is triad-compatible as an auxiliary signal alongside the wave, since
it needs no gradient to cross layers. Nøkland & Eidnes' cosine-similarity
matching is nearly begging for the S⁷ setting (§4 row G).

### 2.8 The forward-only family: error forward-propagation, PEPITA, DRTP

**Idea.** Three verified schemes move the teaching signal *forward*. Kohan,
Rietman, Siegelmann (arXiv-only, v1 2018,
[arXiv:1808.03357](https://arxiv.org/abs/1808.03357)) — **error
forward-propagation**: a recurrent energy-based net, feedforward except for
one loop from output back to the input-receiving layer; the output is nudged
toward the target, and the correction re-enters through the loop and
propagates **through the same feedforward weights**, with contrastive-Hebbian
(equilibrium-propagation-style) local updates; MNIST ~1.85–1.90%. Their later
"Signal Propagation" framework (IEEE TNNLS 35(6):8585–8596, 2024,
[DOI 10.1109/TNNLS.2022.3230914](https://doi.org/10.1109/TNNLS.2022.3230914),
[arXiv:2204.01723](https://arxiv.org/abs/2204.01723)) generalizes to learning
entirely in a forward pass — a successor, not a journal reprint of the 2018
preprint. Dellaferrera & Kreiman's **PEPITA** (ICML 2022, PMLR 162:4937–4955,
[arXiv:2201.11665](https://arxiv.org/abs/2201.11665)): a second forward pass
on the error-modulated input `x + Fe` (`F` fixed random); updates compare the
two passes' activations; CIFAR-10 FC 52.57% vs 55.27% BP. Frenkel, Lefebvre,
Bol's **DRTP** (Frontiers in Neuroscience 15:629892, 2021,
[DOI 10.3389/fnins.2021.629892](https://doi.org/10.3389/fnins.2021.629892),
[arXiv:1909.01311](https://arxiv.org/abs/1909.01311); arXiv v1 carries a
different title): projects the one-hot *target* (a proxy for the error sign)
forward onto each layer through fixed random matrices — no error ever
computed for hidden updates. All are MNIST/CIFAR-scale; PEPITA beats DRTP
throughout and approaches FA.

**Triad fit.** This family is the wave's genus. The repo's own
`STACKED_PERCEPTRON_DESIGN.md` — inject the error rotation as a *regular
input* in a learn pass, each layer updating toward identity — is an
independent rediscovery of the EFP/PEPITA pattern, and the current
`octonion.py` wave is a more structured sibling: where PEPITA uses a random
projection `F` and EFP reuses raw forward weights, the wave transports a
geometrically meaningful tangent error by conjugation and accounts for
absorption explicitly (projection-subtraction) — for which **no prior art was
found in this survey**; that specific transport-and-absorb design appears
novel, with novelty risk to match. All three relatives are real-valued and
unconstrained ((1), (2) violated), and none exceeds CIFAR scale — the honest
reading is that forward-only teaching signals are *proven possible and
unproven at scale*. Concrete borrows: PEPITA's two-pass structure (compare
clean vs error-modulated forward activations) is implementable on S⁷ without
any transport machinery at all — a simpler wave ablation; DRTP's "sign of the
target is enough" result hints the wave may only need to carry the *axis* of
the error rotation, not its magnitude, to teach early layers.

## 3. Manifold-constrained training

### 3.1 Bonnabel — Riemannian SGD

**Idea.** Bonnabel (IEEE Trans. Automatic Control 58(9):2217–2229, 2013,
[arXiv:1111.5280](https://arxiv.org/abs/1111.5280); the commonly cited DOI
10.1109/TAC.2013.2254619 could not be confirmed against IEEE directly) proves
almost-sure convergence of SGD on Riemannian manifolds to critical points,
under classical Robbins–Monro step sizes (`Σγₜ = ∞`, `Σγₜ² < ∞`), bounded
gradients, and trajectories in a compact set (automatic on S⁷, which is
compact with injectivity radius π). The update is a geodesic step via the
exponential map, with **retractions** (first-order approximations of exp)
explicitly sanctioned as cheaper substitutes.

**Triad fit.** Fully compatible with (2) — this *is* the theory of v3i's
update `w ← w·exp(η·κ·τ)` — and silent on (1) and (3). The catch is exactly
(3): Bonnabel's theorem covers steps along the (minus) Riemannian *gradient*
of a cost; the wave's torque `Im(w) × Im(r_local)` is not derived as any
cost's gradient, so convergence is not inherited — the theorem tells you what
you'd need to prove (descent-direction property in expectation, Robbins–Monro
schedule) rather than proving it. Directly borrowable: the step-size schedule
discipline (the map lists the geodesic learning-rate schedule as unspecified —
`γₜ ∝ 1/t` satisfying Robbins–Monro is the theory-backed default), and the
license to use cheap retractions (normalize-after-step) instead of exact
`exp` when width makes exact geodesics expensive.

### 3.2 The reference books — Absil et al.; Boumal

**Idea.** Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix
Manifolds* (Princeton UP, 2008,
[publisher page](https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds))
codified the retraction-based framework for first- and second-order
optimization on manifolds (its standard status is well established, though the
publisher blurb itself does not name "retraction" — flagged per rules). Boumal,
*An Introduction to Optimization on Smooth Manifolds* (Cambridge UP, 2023,
[DOI 10.1017/9781009166164](https://doi.org/10.1017/9781009166164), free PDF at
[nicolasboumal.net/book](https://www.nicolasboumal.net/book/)) is the modern
treatment: retractions, Riemannian gradient/Newton/trust-region methods, with
spheres and Stiefel manifolds as running examples.

**Triad fit.** Compatible with (2) by construction; neutral on (1);
orthogonal to (3) (both books assume gradients exist). Their value to v3i is
vocabulary and hygiene: the wave's `correct()` is, in this language, "a
retraction along a tangent field that is not a gradient" — and the books
supply the checklists (is the tangent field smooth? is the retraction
second-order? does parallel transport of momentum matter?) that the widened
architecture's optimizer story (map: "does tangent-space averaging
generalize?") will need. Tangent-space averaging of per-branch updates
followed by a single retraction is the textbook-recommended pattern.

### 3.3 Arjovsky, Shah, Bengio — uRNN and the origin of modReLU

**Idea.** The unitary-evolution RNN (ICML 2016, PMLR 48:1120–1128,
[arXiv:1511.06464](https://arxiv.org/abs/1511.06464)) keeps the recurrent
matrix exactly unitary via a fixed product parameterization (diagonal phases,
Householder reflections, permutation, FFTs — ~7N parameters, plain SGD, no
projection step), killing vanishing/exploding gradients. Verified: this paper
*introduces* **modReLU** — `σ(z) = ReLU(|z| + b)·z/|z|` — a phase-preserving,
modulus-thresholded nonlinearity with learned bias `b`, placed *between*
unitary maps. This is the field's canonical answer to "nonlinearity under a
norm constraint": let the isometries be exactly isometric and concentrate all
nonlinearity in a radial function that leaves direction untouched.

**Triad fit.** The closest architectural cousin of commitment (2) — but with a
decisive subtlety for v3i: modReLU is nonlinear only through the **modulus**,
and on S⁷-valued signals the modulus is identically 1, so a literal port of
modReLU to the triad is *the identity map* (with `b > −1`). The adaptation
that survives is to transfer the trick from the norm to the **polar angle**:
write a unit octonion as `y = exp(θu) = cosθ + u·sinθ` and threshold/bias θ
while preserving the rotation axis `u` — the same "nonlinear in the radial
coordinate, equivariant in the rest" design, one level down (§4 row A). The
uRNN also violates (3) (backprop through the parameterization), and its
restricted parameterization is the subject of the next entry's warning.

### 3.4 Wisdom et al. — full-capacity unitary RNNs

**Idea.** Wisdom, Powers, Hershey, Le Roux, Atlas (NeurIPS 2016,
[arXiv:1611.00035](https://arxiv.org/abs/1611.00035)) proved the uRNN's 7N
parameters cannot cover the N²-dimensional unitary group for N > 7
(restricted capacity), and instead optimize over the *full* group with a
Cayley-style multiplicative update
`W ← (I + (λ/2)A)⁻¹(I − (λ/2)A)W`, `A = GᴴW − WᴴG` skew-Hermitian — a
Stiefel-manifold gradient step (they cite Tagare's tutorial, which builds on
Wen–Yin; they do not cite Wen–Yin directly). Fixed step size, no gradient
clipping needed; modReLU between maps, as in the uRNN.

**Triad fit.** The capacity theorem is a near-perfect echo of the
isometry-ceiling note: v3i's `{R_v}` family is a 7-parameter subfamily of the
28-dimensional SO(8), just as the uRNN's 7N parameters sit inside N²-dim
U(N) — and the octonion finding that *products* of right-multiplications
escape `{R_v}` into SO(8) is exactly Wisdom's "full capacity" attained by
composition instead of parameterization. Lesson for the widened architecture:
depth/width of octonion multiplications is a capacity mechanism (reaching more
of SO(8)) *independent* of nonlinearity, but only a richer readout can cash it
in (as the ceiling note already argues). The Cayley update itself is
compatible with (2) as an alternative retraction, though v3i's `w·exp(τ)` on
S⁷ is already exact and cheap; (3) is violated (gradients drive `G`).

### 3.5 Parameterization school — scoRNN, expRNN, trivializations

**Idea.** Helfrich, Willmott, Ye (ICML 2018, PMLR 80:1969–1978,
[arXiv:1707.09520](https://arxiv.org/abs/1707.09520)) parameterize orthogonal
matrices by the scaled Cayley transform `W = (I+A)⁻¹(I−A)D` (A
skew-symmetric, D fixed ±1 diagonal), so plain Adam/RMSprop on A keeps W
exactly orthogonal; nonlinearity is again modReLU (real form). Lezcano-Casado
& Martínez-Rubio (ICML 2019, PMLR 97:3794–3803,
[arXiv:1901.08428](https://arxiv.org/abs/1901.08428)) do the same via the Lie
exponential (`W = exp(A)`, expRNN), for any connected compact Lie group; and
Lezcano-Casado (NeurIPS 2019,
[arXiv:1909.09501](https://arxiv.org/abs/1909.09501)) unifies both as
**trivializations** — surjective parameterizations turning constrained into
unconstrained optimization — proving the static families have a performance
issue that his *dynamic trivializations* (periodically re-basing the
parameterization at the current point) fix, landing between parameterization
and Riemannian gradient descent.

**Triad fit.** These solve commitment (2)'s problem by *dissolving* it —
optimize in a flat space, let the map enforce the constraint — which is
tempting but subtly anti-triad: the wave's error is a tangent quantity *at
the weight's current position on S⁷*, and the whole learning rule (transport,
torque, absorption) is written in the manifold's own geometry; flattening the
weight space would leave the wave with nothing to be a wave *in*. The genuinely
transferable result is dynamic trivialization's lesson in reverse: re-basing
at the current point ≈ v3i's existing practice (updates are always expressed
in the local frame via conjugation transport), which this literature suggests
is the right side of the static-vs-dynamic divide. S⁷'s known parallelizability
(it admits a global frame — the one sphere besides S¹ and S³ that does) makes
the local-frame bookkeeping legitimate at any width. Note: whether expRNN's
experiments use modReLU was not verified.

### 3.6 Bécigneul & Ganea — Riemannian adaptive optimizers

**Idea.** Bécigneul & Ganea (ICLR 2019,
[arXiv:1810.00760](https://arxiv.org/abs/1810.00760)) generalize
Adagrad/Adam/AMSGrad to manifolds: exponential-map steps with momentum carried
by **parallel transport**, retraction substitutes allowed, and O(√T) regret
bounds for geodesically convex costs. Verified key restriction: per-coordinate
adaptivity is meaningless on a general manifold (no canonical coordinates), so
adaptivity is defined *across factors of a product manifold* only.

**Triad fit.** Directly relevant to the unspecified optimizer story for wide
layers: a width-k octonion layer's weight space is exactly a product manifold
(S⁷)ᵏ, which is the one setting where their adaptive machinery is
well-defined — per-*weight* (not per-coordinate) adaptive step sizes are
theoretically licensed, and momentum, if ever wanted, must be
parallel-transported along each weight's geodesic (for S⁷, a closed-form
rotation in the plane of motion). Violates (3) as stated (gradient-based), but
the recipe reads verbatim onto wave torques: accumulate per-weight torque
magnitudes, adapt each weight's κ·η accordingly. This is the cheapest
off-the-shelf upgrade to the current fixed learning rate.

### 3.7 Liu et al. — deep hyperspherical learning (SphereNet)

**Idea.** Liu, Zhang, Li, Yu, Dai, Zhao, Song (NeurIPS 2017,
[arXiv:1711.03189](https://arxiv.org/abs/1711.03189)) replace the inner
product in convolution with **SphereConv**, a function of the *angle* between
kernel and input patch — `F(w,x) = g(θ)` with linear, cosine, or learnable
sigmoid `g` — plus an angular-margin softmax. Nuance verified: weights are not
hard-projected onto the sphere; the operator simply ignores magnitudes, making
weight norm irrelevant. Claimed benefits: outputs bounded in [−1,1], improved
conditioning, faster convergence, and a normalization variant (SphereNorm)
that behaves well at small batch sizes.

**Triad fit.** The strongest Area-3 endorsement of sphere geometry as a
*feature* rather than a constraint: making the network angle-native improved
optimization, which is v3i's (2) restated as an empirical claim. The
mechanism to harvest is the **learnable angular function** `g(θ)`: the wave's
world already runs on angles (geodesic distances on S⁷), and an
input-dependent angular reweighting of branch contributions — SphereConv's
`g` applied to `θ(x·wᵢ, 1)` — is a nonlinearity that never leaves the sphere
(§4 rows B, E). Training is backprop ((3) violated) and the algebra is real
((1) violated), so only the geometry transfers — but it transfers cleanly.

### 3.8 Salimans & Kingma; Vorontsov et al. — the two warnings

**Idea.** Weight normalization (Salimans & Kingma, NeurIPS 2016,
[arXiv:1602.07868](https://arxiv.org/abs/1602.07868)) decouples each weight's
length from its direction (`w = g·v/|v|`) and shows the *decoupling itself*
improves conditioning and speeds SGD — the unconstrained cousin of S⁷
weights, with the magnitude kept as a free scalar rather than deleted.
Vorontsov, Trabelsi, Kadoury, Pal (ICML 2017, PMLR 70:3570–3578,
[arXiv:1702.00071](https://arxiv.org/abs/1702.00071)) then measured the cost
of the hard version: verified verbatim, "hard constraints on orthogonality can
negatively affect the speed of convergence and model performance"; letting
singular values drift in a margin [1−m, 1+m] (m ≈ 0.01–0.1) trains faster and
better than exact orthogonality on real tasks, while extreme long-memory
synthetic tasks still favor tight constraints.

**Triad fit.** Together these are the sharpest known pathology for commitment
(2): the literature's evidence says *direction-magnitude decoupling helps, but
deleting the magnitude entirely can hurt optimization*. v3i deletes it twice —
unit weights and unit signals. Two mitigations are triad-compatible: keep a
per-weight or per-layer *scalar* gain that multiplies angles (not norms) so
the algebra stays on S⁷ while the optimizer regains a radial degree of
freedom (the exact analogue of Vorontsov's margin, applied to rotation angle
rather than singular value); or FF-style, let the discarded pre-normalization
magnitude of aggregated sums act as the free scalar (§2.1). The quaternion
control inherits the same medicine, keeping the ablation clean.

## 4. Shortlist: triad-compatible nonlinearity mechanisms

### 4.0 A structural constraint the shortlist must respect (own derivation)

Two facts, provable in two lines each, prune the naive candidates. (i) For
branches sharing an input, **sum-aggregation collapses by distributivity**:
`Σᵢ x·wᵢ = x·(Σᵢ wᵢ)`, and since `|x·c| = |c|` for unit `x`, sum-then-
normalize is *exactly* `x·(Σwᵢ/|Σwᵢ|)` — a single current-architecture layer.
(ii) More generally, any chain of linear aggregations, normalizations, and
unit multiplications computes `L(x)/s(x)` with `L` linear and `s > 0` scalar,
so `sign(re(·))` still realizes only homogeneous linear separators — the
isometry ceiling survives normalization untouched. Every shortlist row must
therefore break linearity-up-to-positive-scaling *before* the readout: through
the angle coordinate (A, B, F), through products of x-dependent terms (C, D,
E), or through a nonlinear readout (G). Geodesic-midpoint/Karcher aggregation
fails this test to first order (the two-point Karcher mean *is* the normalized
sum) and is deliberately not a row.

| # | Mechanism | What it is | Motivating work | Why it preserves the triad | Main risk |
|---|-----------|------------|-----------------|---------------------------|-----------|
| A | Angular modReLU (tangent-space activation) | Polar-decompose `y = exp(θu)` (θ = angle to the real axis, `u ∈ S⁶`); output `exp(f(θ)·u)` with `f` a biased ReLU-like map, e.g. `f(θ) = ReLU(θ − b)`. Equivalently modReLU on `log(y)` at identity. | modReLU (Arjovsky [1511.06464](https://arxiv.org/abs/1511.06464)); reused by Wisdom, scoRNN; SphereNet's `g(θ)` | Output is `exp` of a tangent vector — exactly unit norm; algebra-native (octonion polar form); pointwise, so the wave passes through it as a rotation quantity; bias `b` breaks scale-linearity (§4.0) | modReLU lost badly to split activations in deep nets (Trabelsi: 23.4% vs 6.2% error); singular axis at `y = ±1` (`u` undefined) — the same singularity Parcollet et al. cite against pure hypercomplex activations |
| B | Nonlinear width-aggregation on S⁷ | Combine branch outputs `hᵢ = x·wᵢ` with *angle-dependent* weights — e.g. `Σᵢ g(θᵢ)·hᵢ` then normalize, `g` learnable (softmax over angles, sigmoid gate) — instead of plain sum. | Hinton FF's orientation/length split ([2212.13345](https://arxiv.org/abs/2212.13345)); SphereConv `g(θ)` ([1711.03189](https://arxiv.org/abs/1711.03189)) | All inputs/outputs unit octonions; aggregation weights are scalars, so the algebra is untouched; input-dependent `g` breaks the distributivity collapse of §4.0; normalization discards magnitude the way FF prescribes | If `g` is too flat it degenerates to the provably-linear collapse; per-branch credit assignment for the wave (how much residual does each branch absorb?) is unspecified |
| C | Branch products (input self-products) | Outputs of two branches multiplied: `h = (x·w₁)·(x·w₂)` — quadratic in `x`; deeper variants give higher-degree forms. | PHM's learned bilinear maps ([2102.08597](https://arxiv.org/abs/2102.08597)); Zhu et al.'s rotation view of quaternion kernels ([1903.00658](https://arxiv.org/abs/1903.00658)); algebra-native | Product of unit octonions is unit — stays on S⁷ with *no* activation function at all; purely multiplicative, so the wave's conjugation transport extends naturally; `re(h)` is a quadratic form in `x`, escaping the homogeneous-linear class (XOR-plausible) | Non-associativity makes bracketing a design decision at every width/depth; two weights share credit for one output — the projection-subtraction accounting must be split; no direct prior art for training such products without gradients |
| D | Associator terms across branches | Use `[h₁, w, h₂] = (h₁w)h₂ − h₁(wh₂)` between *distinct* branch outputs, normalized, as an extra branch or perturbation. | The repo's own kappa machinery; absence in Wu et al. ([1903.08478](https://arxiv.org/abs/1903.08478)), who route around non-associativity | Maximally octonion-native; identically zero in the quaternion control — the cleanest possible ablation that nonlinearity comes from non-associativity; quadratic in `x` via the two branches | **Alternativity trap:** any associator with a repeated slot vanishes (`[x,w,x] ≡ 0`), so single-input forms are dead on arrival; `|[·,·,·]|` can hit zero (normalization singularity); magnitude is typically small — may be a weak signal |
| E | Input-dependent rotations / gating | Let the applied rotation depend on `x`: e.g. conjugation sandwich `h = x·(v·(x̄·y))`-style terms, or rotate by angle `g(⟨x, w⟩)` about a learned axis. | Conjugation transport already in `correct()`; SphereNet's learnable angular gate; attention-style modulation (no hypercomplex prior art found — flagged) | Conjugation and rotation by unit octonions preserve S⁷ exactly; gating parameters are weights on S⁷ or scalars; nonlinear in `x` (sandwich terms are quadratic) | Verified pitfall (own derivation): `re(x·v·x̄) = re(v)` — pure sandwiches are *invisible* to the current readout and must be composed with further multiplications; gate scalars need their own gradient-free update rule |
| F | Kappa as forward gate (slerp gating) | Promote the associator scalar from training-only to the forward pass: `h = exp(κ(x)·log(x·w))` with `κ(x) = 1 − |[x, w₁, w₂]|/(|x||w₁||w₂|)` — input-dependent geodesic interpolation between identity and `x·w`. | The repo's `_compute_kappa`; DTP's targets-as-geodesics ([1412.7525](https://arxiv.org/abs/1412.7525)); no external prior art found (flagged) | Stays on S⁷ (exp of a scaled log); in the quaternion control `κ ≡ 1` and the layer reduces *exactly* to the current linear layer — nonlinearity provably sourced from non-associativity alone; `sign(cos(κ(x)θ(x)))` readout escapes §4.0 | κ's empirical range may be narrow (concentration near 1), making the nonlinearity weak — unverified, needs a measurement prototype before design commitment; interacts with κ's training role (same scalar doing two jobs) |
| G | Goodness readout / local sphere objectives | Richer readout than `sign(re(·))`: FF-style goodness on *pre-normalization* aggregate magnitude `|Σᵢhᵢ|` (nonconstant, unlike `|y| ≡ 1`), or Gram-matrix similarity matching of layer outputs to label structure (cosine similarity is native on S⁷). | Hinton FF goodness ([2212.13345](https://arxiv.org/abs/2212.13345)); Nøkland & Eidnes similarity matching ([1901.06656](https://arxiv.org/abs/1901.06656)); isometry-ceiling §4 (readout is the cheapest capacity gain) | Purely a readout/objective change — touches none of the three commitments; quadratic in the layer outputs, so it can see the SO(8) reach that `re(·)` discards; local objectives need no cross-layer gradients (wave-compatible by construction) | FF's own CIFAR gap shows local goodness underperforms global credit assignment; a readout change alone cannot fix hidden-layer linearity — must be paired with A–F, or the function class stays quadratic-in-a-linear-image |

Recommended triage for ticket 005: **C and F first** (pure algebra, no
activation function to invent, quaternion control degenerates exactly to the
current architecture — the cleanest science), **A and B second** (small,
composable, but each carries a known failure mode from the literature),
**G alongside whichever wins** (readout is orthogonal and cheap), **D and E
as instruments** rather than backbones (D as the sharpest
non-associativity probe, E only inside compositions that the readout can see).

## References

Area 1 — hypercomplex networks
- Parcollet et al., *Quaternion Recurrent Neural Networks*, ICLR 2019 — [arXiv:1806.04418](https://arxiv.org/abs/1806.04418)
- Parcollet et al., *Quaternion CNNs for End-to-End ASR*, Interspeech 2018 — [arXiv:1806.07789](https://arxiv.org/abs/1806.07789)
- Parcollet et al., *Speech recognition with quaternion neural networks*, NeurIPS 2018 IRASL — [arXiv:1811.09678](https://arxiv.org/abs/1811.09678)
- Parcollet, Morchid, Linarès, *A survey of quaternion neural networks*, Artif. Intell. Rev. 53 (2020) — [DOI 10.1007/s10462-019-09752-1](https://doi.org/10.1007/s10462-019-09752-1) (metadata verified only)
- Gaudet & Maida, *Deep Quaternion Networks*, IJCNN 2018 — [arXiv:1712.04604](https://arxiv.org/abs/1712.04604)
- Zhu et al., *Quaternion Convolutional Neural Networks*, ECCV 2018 — [DOI 10.1007/978-3-030-01237-3_39](https://doi.org/10.1007/978-3-030-01237-3_39), [arXiv:1903.00658](https://arxiv.org/abs/1903.00658)
- Wu et al., *Deep Octonion Networks*, Neurocomputing 397 (2020) — [DOI 10.1016/j.neucom.2020.02.053](https://doi.org/10.1016/j.neucom.2020.02.053), [arXiv:1903.08478](https://arxiv.org/abs/1903.08478)
- Popa, *Octonion-Valued Neural Networks*, ICANN 2016 — [DOI 10.1007/978-3-319-44778-0_51](https://doi.org/10.1007/978-3-319-44778-0_51) (metadata verified only)
- Cariow & Cariowa, *Fast Algorithms for Deep Octonion Networks*, IEEE TNNLS 34(1) (2023) — via DBLP (not further verified)
- Zhang et al., *Parameterization of Hypercomplex Multiplications*, ICLR 2021 — [arXiv:2102.08597](https://arxiv.org/abs/2102.08597)
- Grassucci, Zhang, Comminiello, *PHNNs*, IEEE TNNLS — [DOI 10.1109/TNNLS.2022.3226772](https://doi.org/10.1109/TNNLS.2022.3226772), [arXiv:2110.04176](https://arxiv.org/abs/2110.04176)
- Trabelsi et al., *Deep Complex Networks*, ICLR 2018 — [arXiv:1705.09792](https://arxiv.org/abs/1705.09792)

Area 2 — backprop-free learning
- Hinton, *The Forward-Forward Algorithm*, 2022 (preprint only) — [arXiv:2212.13345](https://arxiv.org/abs/2212.13345)
- Lillicrap et al., feedback alignment — [arXiv:1411.0247](https://arxiv.org/abs/1411.0247); Nature Comms 7:13276 (2016) — [DOI 10.1038/ncomms13276](https://doi.org/10.1038/ncomms13276)
- Nøkland, *Direct Feedback Alignment*, NeurIPS 2016 — [arXiv:1609.01596](https://arxiv.org/abs/1609.01596)
- Launay et al., *DFA Scales to Modern Deep Learning*, NeurIPS 2020 — [arXiv:2006.12878](https://arxiv.org/abs/2006.12878)
- Bartunov et al., *Assessing the Scalability of Biologically-Motivated Deep Learning*, NeurIPS 2018 — [arXiv:1807.04587](https://arxiv.org/abs/1807.04587)
- Lee et al., *Difference Target Propagation*, ECML-PKDD 2015 — [arXiv:1412.7525](https://arxiv.org/abs/1412.7525), [DOI 10.1007/978-3-319-23528-8_31](https://doi.org/10.1007/978-3-319-23528-8_31)
- Rao & Ballard, *Predictive coding in the visual cortex*, Nat. Neurosci. 2(1):79–87 (1999) — [DOI 10.1038/4580](https://doi.org/10.1038/4580)
- Millidge, Tschantz, Buckley, *PC Approximates Backprop*, Neural Comp. 34(6) (2022) — [DOI 10.1162/neco_a_01497](https://doi.org/10.1162/neco_a_01497), [arXiv:2006.04182](https://arxiv.org/abs/2006.04182)
- Millidge, Seth, Buckley, *Predictive Coding: a Review* — [arXiv:2107.12979](https://arxiv.org/abs/2107.12979) (arXiv only)
- Jaderberg et al., *Decoupled Neural Interfaces using Synthetic Gradients*, ICML 2017 — [arXiv:1608.05343](https://arxiv.org/abs/1608.05343)
- Belilovsky, Eickenberg, Oyallon, *Greedy Layerwise Learning Can Scale to ImageNet*, ICML 2019 — [arXiv:1812.11446](https://arxiv.org/abs/1812.11446)
- Nøkland & Eidnes, *Training Neural Networks with Local Error Signals*, ICML 2019 — [arXiv:1901.06656](https://arxiv.org/abs/1901.06656)
- Kohan, Rietman, Siegelmann, *Error Forward-Propagation*, 2018 (preprint only) — [arXiv:1808.03357](https://arxiv.org/abs/1808.03357)
- Kohan, Rietman, Siegelmann, *Signal Propagation*, IEEE TNNLS 35(6) (2024) — [DOI 10.1109/TNNLS.2022.3230914](https://doi.org/10.1109/TNNLS.2022.3230914), [arXiv:2204.01723](https://arxiv.org/abs/2204.01723)
- Dellaferrera & Kreiman, *PEPITA*, ICML 2022 — [arXiv:2201.11665](https://arxiv.org/abs/2201.11665)
- Frenkel, Lefebvre, Bol, *DRTP*, Front. Neurosci. 15:629892 (2021) — [DOI 10.3389/fnins.2021.629892](https://doi.org/10.3389/fnins.2021.629892), [arXiv:1909.01311](https://arxiv.org/abs/1909.01311)

Area 3 — manifold-constrained training
- Bonnabel, *SGD on Riemannian Manifolds*, IEEE TAC 58(9) (2013) — [arXiv:1111.5280](https://arxiv.org/abs/1111.5280) (DOI unconfirmed, see §3.1)
- Absil, Mahony, Sepulchre, *Optimization Algorithms on Matrix Manifolds*, Princeton UP 2008 — [publisher](https://press.princeton.edu/books/hardcover/9780691132983/optimization-algorithms-on-matrix-manifolds)
- Boumal, *An Introduction to Optimization on Smooth Manifolds*, CUP 2023 — [DOI 10.1017/9781009166164](https://doi.org/10.1017/9781009166164)
- Arjovsky, Shah, Bengio, *Unitary Evolution RNNs*, ICML 2016 — [arXiv:1511.06464](https://arxiv.org/abs/1511.06464)
- Wisdom et al., *Full-Capacity Unitary RNNs*, NeurIPS 2016 — [arXiv:1611.00035](https://arxiv.org/abs/1611.00035)
- Helfrich, Willmott, Ye, *scoRNN*, ICML 2018 — [arXiv:1707.09520](https://arxiv.org/abs/1707.09520)
- Lezcano-Casado & Martínez-Rubio, *Cheap Orthogonal Constraints (expRNN)*, ICML 2019 — [arXiv:1901.08428](https://arxiv.org/abs/1901.08428)
- Lezcano-Casado, *Trivializations*, NeurIPS 2019 — [arXiv:1909.09501](https://arxiv.org/abs/1909.09501)
- Bécigneul & Ganea, *Riemannian Adaptive Optimization Methods*, ICLR 2019 — [arXiv:1810.00760](https://arxiv.org/abs/1810.00760)
- Liu et al., *Deep Hyperspherical Learning*, NeurIPS 2017 — [arXiv:1711.03189](https://arxiv.org/abs/1711.03189)
- Salimans & Kingma, *Weight Normalization*, NeurIPS 2016 — [arXiv:1602.07868](https://arxiv.org/abs/1602.07868)
- Vorontsov et al., *On orthogonality and learning RNNs*, ICML 2017 — [arXiv:1702.00071](https://arxiv.org/abs/1702.00071)
