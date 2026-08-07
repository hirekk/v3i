# In (geodesic) pursuit of a more powerful knowledge representation

A research project on quaternion- and octonion-valued perceptrons built around
an inviolable triad:

1. **Division-algebra signals and weights** — octonions primary, quaternions as
   the associative control.
2. **Unit-sphere geometry** — weights live on S³/S⁷ and updates move along
   geodesics.
3. **No backprop** — the error is computed at the output as a rotation quantity
   and propagated *forward* through the layers as a wave, each layer absorbing
   part of it.

## Status

The research phase established (proofs and machine verification in
[docs/research/](docs/research/)):

- Chains of `y = x·w` are a **single orthogonal map** at any depth — the
  current architecture is linear, with a proven **75% ceiling on XOR**.
  Genuine nonlinearity is the next architectural step.
- The forward error wave's algebra is **exact** (Moufang/alternativity
  identities, true geodesic exp/log, parallelizable S⁷).
- The triad's combination — hypercomplex algebra + manifold constraint +
  forward-only credit assignment — is **unoccupied in the literature**.

Planning lives on the wayfinder map:
[issue #1](https://github.com/hirekk/v3i/issues/1) and its sub-issues.

## Layout

```
src/v3i/
  algebra.py                 # Octonion class: Cayley–Dickson mul, exp/log, slerp,
                             # 7D cross product (derived from the product itself)
  make_data.py               # synthetic datasets on S³/S⁷ (binary-1d, binary-xor)
  run_experiment.py          # uniform training harness -> runs/*.json
  dashboard.py               # streamlit comparison dashboard
  models/
    perceptron/quaternion.py # QuaternionPerceptron + simple optimizer
    perceptron/nn.py         # QuaternionSequential (act–observe–correct stack)
    perceptron/octonion.py   # OctonionPerceptron + OctonionSequential (error wave)
    baseline/                # logistic regression, decision tree, random
docs/
  research/                  # verified research notes + verification scripts
  STACKED_PERCEPTRON_DESIGN.md
tests/                       # algebra tests (uv run pytest)
```

## Quickstart

```bash
# Train models on XOR (data is generated in memory; results land in runs/)
uv run python -m v3i.run_experiment --dataset binary-xor --model octonion
uv run python -m v3i.run_experiment --dataset binary-xor --model quaternion
uv run python -m v3i.run_experiment --dataset binary-xor --model octonion-stack --layers 2
uv run python -m v3i.run_experiment --dataset binary-xor --model baselines

# Compare runs: accuracy vs the proven ceiling, geodesic loss, weight evolution
uv run streamlit run src/v3i/dashboard.py

# Verify the research notes' claims
uv run pytest
uv run python docs/research/isometry_ceiling_verification.py
uv run python docs/research/octonion_structure_verification.py
```
