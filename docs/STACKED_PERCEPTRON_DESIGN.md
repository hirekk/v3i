# Stacked Quaternion Perceptrons: Act–Observe–Correct with Forward-Propagated Error

This note describes how to stack two (or more) quaternion perceptrons without backpropagation, using an **act–observe–correct** loop where the **correction term is fed into the network as a regular input** in a distinct **learn** mode and the **error is forward propagated**.

---

## 1. Goal

- **Stack** multiple quaternion perceptrons (layers) so they are **composable** like layers in a classical NN.
- **No backprop**: no gradients, no chain rule. Instead: **act → observe → correct**.
- **Learn mode**: the correction (error quaternion) is injected as a **normal input** and flows **forward** through the stack; each layer updates so that, in learn mode, it tries to move its output toward **identity** (consume the error).

---

## 2. Single-Layer Recap

For one perceptron:

- **Act**: \(q_{\mathrm{out}} = B \cdot q_{\mathrm{in}} \cdot A\).
- **Observe**: given target label \(y \in \{\pm 1\}\), set \(q_{\mathrm{target}} = (y, 0, 0, 0)\) and  
  \(R = q_{\mathrm{out}}^{-1} q_{\mathrm{target}}\) (geodesic from \(q_{\mathrm{out}}\) to \(q_{\mathrm{target}}\)).
- **Correct**: take a fraction of that rotation (\(R^{\,\eta}\)), decompose into bias/action updates, apply.

So the “error” is the rotation \(R\); the layer updates to move \(q_{\mathrm{out}}\) toward \(q_{\mathrm{target}}\).

---

## 3. Two (or More) Stacked Layers

**Forward (act)**  
Input \(x\) (quaternion, shape e.g. \((1,4)\)):

- Layer 1: \(h = L_1(x)\)
- Layer 2: \(y = L_2(h)\)

Prediction: \(\hat{y} = \mathrm{sign}(y.w)\).

**Observe**  
With true label \(y^*\), set \(q_{\mathrm{target}} = (y^*, 0, 0, 0)\) and define the **output-space error** as the rotation that moves the current output to the target:

\[
q_{\mathrm{error}} = q_{\mathrm{out}}^{-1} \, q_{\mathrm{target}}
\]

(normalized, shorter path). So \(q_{\mathrm{error}}\) is the “mistake” in quaternion form.

**Correct (learn mode)**  
We do **not** backpropagate. Instead:

1. **Feed the error as input**: in a separate, **learn** pass, the input to the stack is \(q_{\mathrm{error}}\) (as a quaternion, same shape as a normal input).
2. **Forward propagate** \(q_{\mathrm{error}}\) through the stack:
   - Layer 1 receives \(q_{\mathrm{error}}\), outputs \(h_{\mathrm{learn}}\).
   - Layer 2 receives \(h_{\mathrm{learn}}\), outputs \(y_{\mathrm{learn}}\).
3. **Local updates**: each layer does its usual perceptron update with **target = identity** \((1, 0, 0, 0)\):
   - Layer 1: update so that, for input \(q_{\mathrm{error}}\), its output moves toward identity.
   - Layer 2: update so that, for input \(h_{\mathrm{learn}}\), its output moves toward identity.

So in learn mode, the “correct” output is **identity** for every layer. The error is treated as a **normal input** and is **forward propagated**; each layer only sees its input and tries to rotate its output toward identity. No gradient, no backward pass.

**Interpretation**  
The stack is taught to “consume” the error: when we inject \(q_{\mathrm{error}}\) at the bottom, each layer contributes to rotating that error toward identity. Over time, the same data input \(x\) should produce an output closer to \(q_{\mathrm{target}}\) because the layers have learned to cancel typical errors when they are presented as inputs in learn mode.

---

## 4. Implementation: Layers and Sequential

**Layer**  
A “layer” is a single `QuaternionPerceptron`: it has a forward pass (e.g. `predict`) and an update rule (`compute_update` with a target, then `apply_update`). No new class is strictly required; the perceptron already behaves as one layer.

**Sequential (stack)**  
- **`Sequential(layers)`**: holds an ordered list of `QuaternionPerceptron` instances.
- **`forward(x)`**: runs \(x\) through each layer in order; returns the final output (and stores the last \(q_{\mathrm{out}}\) for the observe step).
- **`predict_label(x)`**: `forward(x)` then \(\mathrm{sign}(\mathrm{last}\,q_{\mathrm{out}}.w)\).
- **`learn_mode(q_error, optimizers)`**:  
  - Set input = \(q_{\mathrm{error}}\).  
  - For each layer in order: run forward, then `compute_update(input, label=1)` (target = identity), then `optimizer.step(u_b, u_a)`; set input = this layer’s output for the next.
- **`learn_step(x, label, optimizers)`**:  
  - Act: `forward(x)`.  
  - Observe: \(q_{\mathrm{error}} = \mathrm{geodesic\_rotation}(q_{\mathrm{out}}, q_{\mathrm{target}})\).  
  - Correct: `learn_mode(q_error, optimizers)`.

Each layer has its own optimizer (e.g. `SimpleOptimizer(layer)`); the same list is passed to `learn_mode` / `learn_step`.

---

## 5. Summary

| Step   | What happens |
|--------|----------------|
| Act    | \(x \to L_1 \to h \to L_2 \to y\); predict from \(y\). |
| Observe| \(q_{\mathrm{error}} = q_{\mathrm{out}}^{-1} q_{\mathrm{target}}\). |
| Correct| Input = \(q_{\mathrm{error}}\); forward through \(L_1, L_2\); each layer updates with target = identity. |

Error is **forward propagated** as a normal input; no backprop. Layers are **composable** via `Sequential` and the same act–observe–correct loop.
