# Final Comparison: Collaborative Filtering to DLRM

## 1. Neural CTR & DLRM Benchmark Comparison (Tasks 02 vs 03)

*Identical 90/10 stratified split on the Criteo-style CTR dataset.
Same preprocessing (fit on train), same seed (42), same batch size (2048),
same early stopping rule (patience=2, val log loss), same metrics function.*

*Val metrics reported (test used once for best model per task).*

| Model | ROC-AUC | PR-AUC | Log Loss | F1 | ECE | Parameters | Sec/Epoch | Peak Mem (MB) | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|
| **Logistic Regression** | 0.4729 | 0.0394 | 0.7839 | 0.0568 | 0.2283 | 5,724 | 0.6s | 444 | 10 |
| MLP-A (d=16, 1x128) | 0.7032 | 0.0866 | 0.1331 | 0.1745 | 0.0002 | 146,529 | 0.8s | 462 | 7 |
| MLP-B (d=16, 2x256) | 0.6984 | 0.0791 | 0.1343 | 0.1234 | 0.0075 | 267,489 | 1.2s | 472 | 3 |
| MLP-C (d=16, 3x512) | 0.7033 | 0.0840 | 0.1343 | 0.1210 | 0.0051 | 837,345 | 2.4s | 521 | 3 |
| **DCN (bonus)** | 0.7247 | 0.0888 | 0.1322 | 0.1424 | 0.0075 | 270,492 | 1.4s | 511 | 4 |
| **MLP-B-d8 (best Task 02)** | **0.7279** | **0.1064** | **0.1315** | 0.1534 | 0.0068 | 168,561 | 1.2s | 491 | 7 |
| DLRM no-interaction | 0.6931 | 0.0861 | 0.1350 | 0.1165 | 0.0033 | 587,249 | 1.1s | 505 | 4 |
| DLRM (d=8) | 0.6739 | 0.0632 | 0.1395 | 0.1153 | 0.0057 | 502,137 | 1.3s | 500 | 5 |
| DLRM (d=16) | 0.6774 | 0.0708 | 0.1431 | 0.1508 | 0.0108 | 553,969 | 1.2s | 498 | 5 |
| DLRM (d=32) | 0.6812 | 0.0753 | 0.1441 | 0.1127 | 0.0151 | 657,633 | 1.4s | 535 | 4 |

**Key observations:**

- The best overall model is **MLP-B-d8** (Task 02): ROC-AUC 0.7279, log loss 0.1315.
  A smaller embedding dim (d=8) reduced overfitting relative to d=16 and d=32.
- **DCN** (the bonus Task 02 model) is second at 0.7247 AUC, validating that explicit
  cross-feature interactions help — but DCN learns the interaction weights, while DLRM
  uses fixed dot products.
- **DLRM (no-interaction)** wins within the DLRM family on log loss (0.1350), but is below
  MLP-B-d8 (0.1315). The no-interaction DLRM is essentially a larger MLP with a mandatory
  bottom-MLP projection stage.
- **Full DLRM** (d=16) performs below its ablation. The dot-product interaction adds an
  inductive bias that helps at scale (45M+ rows per the paper) but does not emerge clearly
  on ~36K training rows.
- **Memory** is similar across models on CPU (RSS ~460-535 MB). At production scale,
  embedding tables grow proportionally to sum(cardinalities) * d, dominating model memory.

---

## 2. Conceptual Evolution: kNN -> MF -> MLP -> DLRM

> **Note:** Task 01 uses MovieLens-100K (explicit 1-5 ratings, 943 users, 1,682 items),
> while Tasks 02 & 03 use a Criteo-style CTR dataset (implicit binary click labels,
> 13 numeric features, 26 categorical features). Numbers are **not comparable**.
> The comparison below is about **model expressiveness and design philosophy**.

| Aspect | kNN (Task 01) | MF (Task 01) | MLP (Task 02) | DLRM (Task 03) |
|---|---|---|---|---|
| **What it represents** | Local neighbourhood similarity | Global latent factors | Non-linear feature interactions | Explicit second-order feature crosses |
| **Input** | User-item co-occurrence matrix | User ID, item ID | Dense numerics + categorical IDs | Dense numerics + categorical IDs |
| **Side features** | No | No | Yes (dense + cat) | Yes (dense + cat) |
| **Training** | None (similarity at query time) | SGD/Adam on ratings | Adam on BCE | Adam on BCE |
| **Prediction cost** | O(n_neighbors * n_items) query-time | O(d) dot product | O(depth * width) MLP forward | O(d * 27^2) interaction + MLP |
| **Cold start (new user)** | Cannot predict (no ratings) | Cannot predict (no embedding) | Can predict (dense features help) | Can predict (dense features help) |
| **Cold start (new item)** | Cannot predict | Cannot predict | Can predict (item cat features) | Can predict (item cat features) |
| **Interpretability** | High: "because users like you liked X" | Medium: latent dimensions | Low: hidden neurons | Low: dot-product scores |
| **Scale** | O(n_users * n_items) memory | O((n_users + n_items) * d) | O(sum_cardinalities * d + params) | O(sum_cardinalities * d + params) |
| **Key weakness** | Cold start, no context, slow query | Cold start, no side features | No explicit interactions | Same data need as MLP, more complexity |

### When each model is appropriate

- **kNN:** Small dataset with explicit ratings, need for explainable recommendations,
  no training budget, or as a fast baseline. "Because users like you also liked X" is
  inherently interpretable.

- **MF:** Larger collaborative filtering dataset, users and items are stable (IDs don't
  change), no side features available or needed, fast inference from precomputed embeddings.
  MF generalizes better than kNN on sparse data because it shares statistical strength
  across similar users/items via the shared latent space.

- **MLP (Neural CTR):** When side features exist (context, ad features, user features),
  when items change rapidly (new ads appear every day — MF can't embed them), when you
  need a click probability rather than a rating, when item/user cold start is a real
  problem. The shared embedding + MLP architecture handles new categorical values via OOV.

- **DLRM:** Same setting as the neural MLP, but when you believe second-order feature
  interactions are important and you have enough data (millions+ of examples) to train
  the interaction layer meaningfully. Also when you need to operate at Facebook-like
  scale where embedding tables must be sharded across many GPUs (DLRM's model-parallel
  design for embeddings + data-parallel MLPs is the key systems contribution).

---

## 3. What Was Gained at Each Step

Moving from kNN to MF eliminated the expensive query-time nearest-neighbor search and gained
better generalization on sparse data by sharing latent factors across all users and items.
Moving from MF to the neural MLP added the ability to incorporate rich side features —
dense numeric context and many categorical fields — enabling predictions for new items and
new users, which pure collaborative filtering cannot do. Moving from a vanilla MLP to DCN
added explicit second-order feature crosses at low cost (a cross network computes O(d) per
layer rather than O(d^2)), and on the Criteo-style data DCN was the best Task 02 model
after the MLP-B-d8 variant. Moving from the MLP to DLRM restructured the same inputs into
a bottom MLP (to match dense features to embedding dimension) and a dot-product interaction
layer (to explicitly compute all pairwise second-order terms), offering a principled
inductive bias for learning which feature pairs matter. On this small dataset the DLRM
gain was not realized, but the architectural insight — that dense features need to be
projected to the same space as embeddings before interaction, and that interactions should
be explicit rather than discovered by depth — is validated at production scale in the paper.
