# 2. Write the two-tower model in NumPy with hand-derived gradients

- Status: accepted
- Date: 2026-09-27

## Context

Stage 1 of the cascade needs a learned embedding model: a user tower and an item tower that map into
one space so that a nearest-neighbour search over item embeddings is a retrieval step. The obvious
choice is PyTorch — twenty lines, autograd, batteries included.

Three constraints push the other way.

The repo has to run on a CPU-only machine with about 2 GB of free RAM, and the whole test suite,
demo and benchmark set has to finish in minutes. A PyTorch wheel is roughly 200 MB installed plus its
CUDA-less runtime; importing it costs a second before anything happens. For a model whose largest
tensor at the specification's scale is a 100k x 32 item embedding table, that is a lot of dependency
for very little compute.

The second constraint is what the repo is trying to show. A cascade's interesting parts are the
recall ceiling, the latency budget and the bias correction — but a two-tower model behind
`nn.Linear` is a black box in exactly the way the rest of the repo refuses to be. Sampled softmax
with logQ correction has a specific derivation (which logits the correction touches, which candidates
are masked as duplicates rather than treated as negatives, what the gradient through an L2
normalization is), and writing it out is the difference between using the technique and understanding
it.

The third is reproducibility. The bundle manifest pins a metric hash, and a test trains twice and
compares artifacts byte for byte. Getting that from a framework means pinning thread counts, disabling
non-deterministic kernels and hoping a minor version does not change a reduction order.

## Decision

Implement both towers in NumPy with hand-derived gradients:

- forward pass `x = [E[id], side]`, `h = relu(x W1 + b1)`, `z = h W2 + b2`, `y = z / ||z||`;
- sampled softmax over in-batch positives plus uniformly sampled negatives, with logQ correction
  (Yi et al., "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations",
  RecSys 2019);
- the backward pass written out: `dZ = (softmax(Z) - I) / B`, then through the normalization
  `dz = (dy - y <y, dy>) / ||z||`, then the dense/ReLU chain rule, with id-embedding rows receiving
  the first `id_dim` columns of `dx` summed over repeats;
- Adam (Kingma and Ba 2015), lazy row-wise for the embedding tables so a step costs O(batch) rather
  than O(table size);
- a `gradient_check` that perturbs **every** parameter entry and compares the analytic gradient with
  a central finite difference, run as a property test over a range of shapes and both logQ settings.

Two numerical choices inside this are deliberate and would be easy to get wrong with a framework
hiding them. The normalization is `||z|| = sqrt(sum(z^2) + eps^2)` rather than `max(||z||, eps)`:
both avoid dividing by zero when every ReLU unit is dead for a given input, but only the first is
smooth. With the clamped form the derivative is discontinuous at the clamp and a dead layer produces a
gradient of magnitude `1/eps`; with epsilon inside the square root the map is differentiable
everywhere, a dead layer yields the zero embedding and a finite gradient, and the backward formula is
unchanged. And `b1` starts at a small positive value so that no hidden unit is dead at initialization,
while id embeddings start at exactly zero so an entity never seen in training keeps a content-only
representation instead of random noise.

## Consequences

- No torch, no CUDA, no model download. `uv sync` installs numpy, faiss-cpu, LightGBM, DuckDB and the
  web stack, and training the demo world takes about 14 s on six cores.
- The gradient check is a real test rather than a formality, and it is the reason the noise-floor
  reasoning in `gradient_check`'s docstring exists: a central difference of a loss of order 1 with
  step 1e-6 carries absolute noise of about `machine_eps / eps`, so an entry has to stand well above
  that before a *relative* error means anything. The first version of the check used a floor 1000x too
  low and failed on floating-point cancellation rather than on a wrong derivation.
- Determinism comes for free. Two training runs produce byte-identical `tower.npz`.
- **The cost is capacity.** Two 64-unit hidden layers with 32-dimensional output is a small model. A
  deeper tower, attention over user history, or multi-task heads would all need more than this
  implementation is worth extending to. The measured consequence is visible in the corpus evaluation:
  the tower's score is already a strong relevance proxy, and a LambdaMART reranker fitted on the
  logging policy's slates does not improve on it there.
- Adding a feature to the tower means writing its gradient. That is a real tax on future work, and it
  is the right trade only while the model stays this size.
- Batch-size and learning-rate tuning is manual, with no scheduler and no early stopping. Epoch count
  is a config field chosen by looking at the loss curve.

## Alternatives considered

- **PyTorch on CPU.** Rejected on dependency weight, import cost and determinism, not on capability.
  If the model needed to grow past a two-layer MLP this decision should be revisited, and a new ADR
  should supersede this one.
- **JAX.** Same dependency objection, plus a JIT warm-up that would dominate a 14 s training run.
- **scikit-learn's `MLPRegressor` or similar.** No support for a two-tower structure with a shared
  output space and a sampled-softmax objective; it would have to be reimplemented anyway.
- **Matrix factorization (ALS) instead of towers.** Cheaper and well understood, but it has no way to
  use item side features, so cold items would get no representation at all — and the cold-start
  behaviour of the cascade is one of the things the repo measures.
