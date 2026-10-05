# Task 03: DLRM Report

## 1. Goal and How to Rerun

**Goal:** Implement the Deep Learning Recommendation Model (DLRM) from scratch following
Naumov et al. (2019), train it on the shared Criteo-style CTR dataset used in Task 02,
run ablations, and compare against the Task 02 neural baselines.

**How to rerun:**
```bash
pip install -r RecommendationTasks/requirements.txt
# Dataset must be at: Mandatory_Tasks/Recommender_Systems/dataset_2_3/
cd RecommendationTasks/03_DLRM
python run_dlrm.py
# or: jupyter nbconvert --to notebook --execute dlrm.ipynb
```

### Environment & Hardware
- **Hardware**: AMD Ryzen 7 8000-series laptop, 16 GB RAM, no discrete GPU (CPU-only execution)
- **Environment**: Python 3.14.7, PyTorch 2.14.0+cpu, NumPy 2.2.6, Pandas 2.2.3, Scikit-Learn 1.6.1, SciPy 1.15.2, Matplotlib 3.10.1
- **Random Seed**: 42 (covers Python `random`, `numpy`, `torch` via `common/utils.set_seed`)
- **Split**: Same `split.npz` and preprocessor as Task 02 (90/10 stratified split, seed 42)

---

## 2. Data and Preprocessing

Reuses the **exact shared pipeline from Task 02** (`common/ctr_data.py`):

- **Split:** 90/10 stratified on the training file, seed 42. Indices saved in `common/split.npz`.
  Task 03 loads the same file so train/val sets are identical to Task 02.
- **Numeric (13 features):** fill missing with train median → clip negatives to 0 → log1p → standardize (train mean/std).
- **Categorical (26 features):** fill missing with `__missing__` token → count on train only → values with count < 10 map to OOV index 0; others get IDs 1..n. Unseen values in val/test → OOV.
- **Test set:** transformed with the train-fitted preprocessor, evaluated **once** with the threshold chosen on val.

The class imbalance (positive rate ≈ 3.35%) makes accuracy at 0.5 misleading.
We report ROC-AUC, PR-AUC, log loss, and F1 at a val-chosen threshold.

---

## 3. Model Architecture (`dlrm.py`)

### DLRM (full, from scratch)

```
x_num [B, 13]
  └─> Bottom MLP: 13 -> 512 -> 256 -> 16           [B, 16]

x_cat [B, 26]
  └─> 26 Embedding tables (cardinality_j, 16)       [B, 16] each

Stack dense + 26 embeddings:                         [B, 27, 16]
  Z = X @ X^T                                        [B, 27, 27]
  Lower triangle (i > j): 27*26/2 = 351 values       [B, 351]

Top MLP: (16 + 351=367) -> 512 -> 256 -> 1          [B]
```

**Initialization:**
- Embeddings: uniform(-1/sqrt(d), 1/sqrt(d)). Small initial range keeps dot products near zero before training, preventing gradient saturation.
- MLP weights: Kaiming uniform (PyTorch default for `nn.Linear`), appropriate for ReLU activations.

**Why dot products instead of plain concat?**
Concatenating all 27 vectors gives a 27d-dimensional input to the top MLP, forcing the MLP to discover pairwise interactions through depth. The dot-product interaction computes all 351 second-order terms explicitly in O(27²d) time, then passes only 351 scalars to the top MLP — which already has structured cross-feature information rather than having to rediscover it.

**Training conventions (same as Task 02):**
- BCE-with-logits loss, Adam, lr=1e-3, batch=2048
- Weight decay 1e-5 on dense (Linear) layers only — no decay on embeddings (would shrink rows that haven't been updated in a batch)
- Early stopping on val log loss, patience=2, max 10 epochs
- F1-optimal threshold chosen on val, applied to test once

---

## 4. Results Table

### All DLRM variants (val metrics)

| Model | ROC-AUC | PR-AUC | Log Loss | Accuracy | F1 | Precision | Recall | ECE | Threshold | Params | Sec/Epoch | Best Epoch |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **DLRM (no-interaction)** | 0.6931 | 0.0861 | **0.1350** | 0.968 | 0.1165 | 0.0678 | 0.4141 | 0.0033 | 0.05 | 587,249 | 1.1s | 4 |
| DLRM (d=8) | 0.6739 | 0.0632 | 0.1395 | 0.968 | 0.1153 | 0.0749 | 0.2500 | 0.0057 | 0.08 | 502,137 | 1.3s | 5 |
| DLRM (d=16) | 0.6774 | 0.0708 | 0.1431 | 0.968 | 0.1508 | 0.1053 | 0.2656 | 0.0108 | 0.09 | 553,969 | 1.2s | 5 |
| DLRM (d=32) | 0.6812 | 0.0753 | 0.1441 | 0.968 | 0.1127 | 0.0691 | 0.3047 | 0.0151 | 0.07 | 657,633 | 1.4s | 4 |

### Best model (DLRM no-interaction) on test set

| Metric | Value |
|---|---|
| ROC-AUC | 0.6793 |
| PR-AUC | 0.0742 |
| Log Loss | 0.1411 |
| Accuracy @ 0.5 | 0.9665 |
| F1 (threshold=0.05) | 0.1186 |
| Precision | 0.0701 |
| Recall | 0.3851 |
| ECE | 0.0060 |
| Positive rate baseline | 3.35% |

---

## 5. Plots

| Figure | Description |
|---|---|
| `figures/dlrm_loss_curves.png` | Train vs val BCE loss per epoch for all 4 variants |
| `figures/dlrm_embed_dim_ablation.png` | ROC-AUC and log loss vs d in {8, 16, 32} |
| `figures/dlrm_interaction_ablation.png` | DLRM (dot-product) vs DLRM (no-interaction) |
| `figures/dlrm_reliability_diagram.png` | Calibration plot for best model on val |

---

## 6. Analysis

### Ablation 1: No-Interaction (Concat) Baseline

The no-interaction variant **wins on val log loss (0.1350 vs 0.1431)**, performing better
than the full DLRM with explicit dot-product interactions.

**Why this can happen on small data:**
- The no-interaction top MLP receives 27d = 432 inputs, giving it more raw capacity than
  the 367-input full DLRM. With only ~36,000 training examples (the data is very small),
  the MLP's extra capacity from seeing raw embeddings is more useful than structured
  dot-product interactions.
- The dot-product interaction is a fixed, non-learned operation. Its value emerges when the
  embeddings are trained to represent latent feature spaces that naturally interact via inner
  products — which requires more data and more training epochs to emerge.
- The paper (Naumov et al.) validates on 45M+ examples. On 36K examples, any architecture
  gains are hard to distinguish from noise.

### Ablation 2: Embedding Dimension (d in {8, 16, 32})

Larger d consistently gives better ROC-AUC (d=32 > d=16 > d=8) but worse log loss, because
larger embeddings memorize training examples faster, causing the val loss to rise sooner.
d=8 is the most regularized and generalizes best on log loss, but leaves capacity on the table
for discrimination (AUC). d=16 is a reasonable middle ground.

### Overfitting Pattern

All DLRM variants stop early at epoch 4–5. The val loss starts rising after the best epoch
in each case, consistent with the pattern observed in Task 02: high-cardinality embedding
tables can memorize training examples after just a few epochs on this subsample.

### DLRM vs MF (Task 01 conceptual comparison)
MF factorizes a user-item co-occurrence matrix: `r_hat = mu + b_u + b_i + p_u . q_i`.
It is limited to known user-item pairs and learns nothing from side features.
DLRM processes dense numeric context (e.g., time-of-day, position) and 26 categorical
fields (e.g., ad category, publisher), predicting the click probability for any impression,
including new user-item pairs that never appeared in training.

### DLRM vs vanilla MLP (Task 02 comparison)
The best Task 02 model (MLP-B-d8) achieved ROC-AUC 0.7279. DLRM (no-interaction) reached
0.6931 — somewhat below. This is not surprising: the no-interaction DLRM is essentially a
larger MLP-B with a dedicated bottom MLP, which adds parameters but also adds regularization
pressure through the bottom MLP transform. Full DLRM (d=16) reached 0.6774. The performance
difference is within the expected range for this data size.

The likely causes of DLRM underperforming the simpler MLP here:
1. Very small training set (~36K rows) — DLRM's bottom MLP adds an extra stage before the
   interaction that is harder to train on limited data.
2. The dot-product interaction benefits are most visible when embedding tables are large and
   well-trained, which requires more data.
3. No feature engineering or hyperparameter tuning specific to DLRM (bottom MLP depth, top
   MLP depth, embedding dim) was done — the defaults from the plan were used.

### Memory
Peak RSS on CPU: ~500 MB for all variants. At production scale (cardinalities in the millions,
d=128), the embedding tables alone would be gigabytes — requiring model-parallel sharding
across multiple accelerators as described in the paper and `PAPER_NOTES.md`.

---

## 7. Conclusion and Lessons Learned

1. **DLRM from scratch is implementable** in ~200 lines. The key architectural insight —
   project dense features to embedding dim so they can participate in dot-product interactions
   on equal footing — is elegant and non-obvious.
2. **The dot-product interaction did not help on this small dataset.** On 45M+ examples the
   paper shows clear gains; on ~36K examples the MLP's implicit interaction discovery is
   sufficient, and the structured interaction layer doesn't have enough data to manifest its
   inductive bias.
3. **Early stopping and no embedding weight decay are critical.** Without them, the embedding
   tables memorize training rows within 2–3 epochs.
4. **The architectural chain kNN -> MF -> MLP -> DLRM makes sense conceptually.** Each step
   adds input generality (IDs only -> IDs with biases -> dense features -> dense + sparse
   features with explicit interactions) at the cost of more hyperparameters and compute.
5. **On this data, simpler wins.** The best overall model (MLP-B-d8, Task 02) outperforms all
   DLRM variants. With more data and longer training, DLRM's structured interactions would
   likely emerge as an advantage.

---

## 8. Limitations, Subsampling & What Was Not Done

- **Subsampled Dataset**: Evaluated on the same stratified ~36K row Criteo subsample as Task 02. The full 45M dataset was not processed due to hardware CPU bounds.
- **Model Parallelism**: Distributed model-parallel embedding table sharding (as described in the original DLRM paper) was not implemented because all models fit easily on local single-host CPU RAM (~500 MB peak RSS).
- **Interaction Options**: Explored standard dot-product and no-interaction (concat) variants; bilinear/matrix-vector interaction kernel extensions were not tested.

