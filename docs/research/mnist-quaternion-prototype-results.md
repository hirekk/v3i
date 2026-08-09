# MNIST prototype results — quaternion peel-and-solve does not (yet) learn MNIST

Results write-up for wayfinder ticket
[MNIST prototype of the quaternion peel-and-solve network (#21)](https://github.com/hirekk/v3i/issues/21),
a sub-issue of the quaternion-primary map [#16](https://github.com/hirekk/v3i/issues/16).
Prototype code: branch
[`prototype/mnist-quaternion-peel-solve`](https://github.com/hirekk/v3i/tree/prototype/mnist-quaternion-peel-solve)
(`prototypes/mnist_quaternion_peel_and_solve.py`) — throwaway, never merged.

**A weak result is a valid result.** The D=1 architecture as specced does not learn
MNIST competitively; this note records the evidence and the two decisions it
reroutes.

## Setup (the locked decisions, built faithfully)

- **Encoding (#18):** 7×7 grid of 4×4-pixel patches → `L=49` unit quaternions per
  image (fixed random projection of the patch pixels to ℝ⁴, biased toward identity
  so blank patches map to the identity and are transparent to the product).
- **Architecture (#19):** D=1, 10 independent one-vs-rest heads, each a width-49
  sequence-as-branches bank `y = ∏ᵢ (xᵢ·wᵢ)`; online per-sample peel-and-solve;
  `η = 0.9/W`; readout `argmax_k re(y_k)`.
- **Protocol (#20):** MLP-matched (DoF) + MLP-best(128) + logistic, all on raw
  pixels; sample-efficiency sweep `N ∈ {100,500,1k,3k,6k}` (≤10% of 60k train);
  10k test. Seed 0.

DoF: quaternion net = 10·49·3 = **1470**; MLP-matched = 784→2→10 (~1600 params);
logistic = 7850 params.

## Headline results (η=0.9/W, 3 epochs)

| N | quaternion | inv-viol | MLP-matched | MLP-best | logistic |
|---:|---:|---:|---:|---:|---:|
| 100  | 0.200 | 0.0 % | 0.172 | 0.794 | 0.768 |
| 500  | 0.184 | 0.0 % | 0.418 | 0.864 | 0.843 |
| 1000 | 0.194 | 0.0 % | 0.315 | 0.893 | 0.872 |
| 3000 | 0.149 | 0.0 % | 0.632 | 0.929 | 0.895 |
| 6000 | 0.142 | 0.0 % | 0.671 | 0.945 | 0.898 |

The pre-registered headline ("beats MLP-matched at N≤6000") is technically met only at
N=100 (0.200 vs 0.172) — where **both models are at chance** and the tiny MLP hasn't
trained. At every larger N the quaternion net loses to *everything*, including the
linear floor, and its accuracy **decreases as data grows** (0.20 → 0.14). **Honest
verdict: the headline is not genuinely met; the net does not learn MNIST.**

What *did* hold perfectly: the **contraction invariant** — 0/3000 violations, mean
ratio **0.100 = |1−ηW| = |1−0.9|**, exactly the theory (#17). The mechanism is
implemented correctly. Weights stay unit (|w|=1.0000), mean rotation 89° from
identity. The failure is not a bug.

## Diagnosis — two probes

**D1. Is the encoding the bottleneck? No.** Logistic regression *on the flattened
quaternion encoding* scores **0.884** (raw-pixel logistic = 0.898). The encoding
retains essentially all the class signal; a *linear* function of it is strong.

**D2. Is it the learning rule? Yes — catastrophic forgetting.** The `η=0.9/W` rule
contracts each sample's error to 0.1, i.e. **near-exactly solves each sample**, so in
online mode the net ends up fitted to the tail of the stream. Dialing η down trades
per-sample solving for accumulation across samples:

| lr_scale (η·W) | inv ratio | acc @N=300 | acc @N=3000 |
|---:|---:|---:|---:|
| 0.9 | 0.100 | 0.132 | 0.149 |
| 0.3 | 0.700 | 0.367 | 0.510 |
| 0.1 | 0.900 | 0.314 | 0.554 |
| 0.03 | 0.970 | 0.274 | 0.469 |
| 0.01 | 0.990 | 0.215 | 0.393 |

At the specced 0.9 the net does not improve with data (forgetting dominates); around
0.1–0.3 it learns (accuracy rises with N). Best case found: **lr_scale=0.1, N=6000,
15 epochs → 0.660** (3ep 0.578, 8ep 0.631). Still far below logistic (0.88) and
MLP-best (0.945).

## Why it's capped well below linear — function-class mismatch

A linear readout of the encoding gets 0.88, but the peel-and-solve net does **not**
compute linear functions of the encoding — it computes `re(∏ᵢ xᵢwᵢ)`, a degree-49
*multiplicative* form. That class is ideal for **parity** (inherently
degree-structured, where it hit 1.00) but appears unable to represent the
**linearly-separable** structure MNIST has in this encoding. High polynomial degree
≠ useful features. **The proven-on-parity mechanism does not transfer to MNIST
because MNIST is not parity-shaped.** This is the central finding.

## Decisions this reroutes

1. **The online learning rule (#19 Q4).** `η=0.9/W` — tuned for parity's one-epoch
   findability — causes catastrophic forgetting on multi-class online data. The
   real-data rule needs a much smaller η, the deferred **minibatch tangent-averaging**
   (average over samples ⇒ no forgetting), or a decaying schedule. → new ticket.
2. **The combiner/readout (#18 readout + #19 combiner).** The multiplicative
   branch-product read via `re` is mismatched to MNIST's linear structure. Options:
   an additive/mean combiner, a **linear readout head over branch features**, or
   accepting the mismatch as a characterization result (peel-and-solve is
   degree-structured-task-native). → new ticket; the deeper of the two.

## Honest status vs the destination

This is a genuine benchmark data point toward the map's destination: **as specced,
the quaternion peel-and-solve net is not competitive with backprop — or even linear
regression — on MNIST.** Whether that's fixable (rules 1–2) or intrinsic (the
function class is parity-native, not image-native) is the next fork.
