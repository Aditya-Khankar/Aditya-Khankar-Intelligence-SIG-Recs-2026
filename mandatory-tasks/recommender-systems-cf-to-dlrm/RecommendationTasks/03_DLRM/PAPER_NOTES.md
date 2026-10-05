# DLRM Paper Notes

**Paper:** Naumov et al., "Deep Learning Recommendation Model for Personalization and
Recommendation Systems", Facebook AI Research, 2019.
arXiv: 1906.00091

---

## The Problem

Modern recommendation systems need to handle two very different kinds of features
simultaneously:

- **Dense (continuous) features** — things like normalised counts, time-on-page, or
  engagement rates. These are already numeric and can flow directly into an MLP.
- **Sparse (categorical) features** — things like user ID, ad category, publisher ID.
  Each one is a one-hot vector over a potentially huge vocabulary (millions of items),
  so you cannot concatenate raw one-hots into an MLP without blowing up memory.

Earlier systems either (a) hashed the categoricals into a fixed-size dense vector
(lossy) or (b) used separate shallow models for each feature type. DLRM proposes a
unified architecture that handles both cleanly and scales to billions of parameters by
splitting the model across many machines.

---

## Architecture

### 1. Bottom MLP (dense branch)
The 13 numeric features are passed through a small MLP:

```
[B, 13]  →  FC(512) → ReLU  →  FC(256) → ReLU  →  FC(d)
```

The output has dimension `d` — **the same as the embedding dimension**. This is the key
design constraint: the dense representation must be the same size as each embedding row
so that it can participate in the dot-product interaction on equal footing.

### 2. Embedding tables (sparse branch)
Each categorical field `j` has its own `nn.Embedding(cardinality_j, d)` table.
A lookup replaces the sparse one-hot with a dense `d`-dimensional vector.
With 26 categorical fields and embedding dimension `d`, this produces 26 vectors,
each of shape `[B, d]`.

### 3. Interaction layer (the key novelty)
Stack the bottom MLP output and all 26 embeddings into a tensor of shape `[B, 27, d]`.
Call this `X`. Compute `Z = X @ X^T → [B, 27, 27]`.

Each entry `Z[b, i, j] = dot(X[b, i], X[b, j])` captures the interaction between
feature `i` and feature `j`. Only the **strictly lower triangle** is kept (351 values
for 27 features), because the upper triangle is redundant (dot product is symmetric)
and the diagonal is just the self-dot-product which carries less cross-feature information.

**Why dot products instead of concatenation?**
- Concatenating all 27 vectors gives `27 * d` inputs to the top MLP. The MLP then
  has to learn which pairs of features interact — requiring at least two layers to model
  second-order terms.
- The dot-product interaction computes all pairwise second-order interactions
  *explicitly and cheaply* (a single batched matrix multiply), and passes only 351
  numbers to the top MLP instead of `27d`. The top MLP therefore starts with structure
  already in the input rather than having to rediscover it.

### 4. Top MLP (classification head)
Concatenate:
- The dense vector `[B, d]` from the bottom MLP (carried forward as a residual).
- The 351 interaction scalars.

→ `[B, d + 351]` → FC(512) → ReLU → FC(256) → ReLU → FC(1) → logit.

---

## Parallelism Notes

At production scale (billions of embeddings, hundreds of millions of users), the
embedding tables are too large to fit on one GPU:

- **Model parallelism for embeddings:** Each embedding table (or shard of a large one)
  lives on a different device. Each device looks up its own table and communicates its
  output vectors via all-to-all collectives.
- **Data parallelism for MLPs:** The bottom and top MLPs are small enough to be
  replicated on every device. Each device processes a different mini-batch slice through
  the same MLP weights, then aggregates gradients with all-reduce.

This hybrid strategy is motivated by the asymmetry: embeddings are memory-bound
(huge, rarely updated), while MLPs are compute-bound (small, updated every step).

---

## What the Paper Reports

On the Criteo Kaggle dataset (45M rows, same format as this task's data):
- DLRM achieves comparable AUC to DCN and DeepFM at a fraction of the inference cost
  once the embeddings are sharded across accelerators.
- The paper emphasises that AUC differences of 0.001 can correspond to large revenue
  differences in production — motivating careful ablations.
- Model parallelism reduces end-to-end latency by ~2× versus pure data parallelism
  when embedding tables are large.

---

## Key Differences from Task 02 Models

| Aspect | MLP (Task 02) | DLRM (Task 03) |
|--------|--------------|----------------|
| Feature interaction | Implicit (MLP has to learn it) | Explicit pairwise dot products |
| Input to top MLP | All embeddings concatenated (27d) | Interactions only (351) + dense (d) |
| Bottom MLP | None | Projects dense features to embedding dim |
| Scale strategy | Data parallel | Model parallel (embeddings) + data parallel (MLP) |
| Inductive bias | None | Second-order feature cross |
