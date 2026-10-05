# Task 01: Collaborative Filtering Report

## 1. Goal and How to Rerun
The goal of Task 01 is to build, evaluate, and compare explicit collaborative filtering models from scratch—ranging from simple non-personalized baselines to memory-based nearest-neighbors (kNN) and model-based Biased Matrix Factorization (MF).

### How to Rerun
1. Ensure dependencies from `requirements.txt` are installed in your Python environment.
2. Download and unzip MovieLens 100K into `ml-100k/` at the repository root level.
3. Run `cf.ipynb` top to bottom from a fresh kernel, or execute the experiment runner script:
   ```bash
   python 01_CollaborativeFiltering/run_cf.py
   ```

### Environment & Hardware
- **Hardware**: AMD Ryzen 7 8000-series laptop, 16 GB RAM, no discrete GPU (CPU-only execution)
- **Environment**: Python 3.14.7, PyTorch 2.14.0+cpu, NumPy 2.2.6, SciPy 1.15.2, Scikit-Learn 1.6.1, Matplotlib 3.10.1
- **Random Seed**: 42 (covers Python `random`, `numpy`, `torch`)

---

## 2. Data and Preprocessing
- **Dataset**: MovieLens 100K containing 100,000 explicit ratings (1–5 scale) across 943 users and 1,682 items (~6.3% matrix density).
- **Split Protocol**: Random 80/10/10 train/val/test split using fixed seed 42 (80,000 train, 10,000 val, 10,000 test).
- **Unseen User/Item Handling**: Any user or item present in validation or test set that did not appear during training falls back gracefully to the global mean + item/user bias baseline, preventing runtime execution failures.

---

## 3. Models and Training Choices

1. **Global Mean Baseline**: $\hat{r}_{u,i} = \mu$. Serves as the minimal reference benchmark.
2. **User + Item Bias Baseline**: $\hat{r}_{u,i} = \mu + b_u + b_i$. Regularized biases ($L_2$ shrinkage $\lambda=10$) capture user strictness and item general popularity.
3. **Memory-Based kNN (User-User & Item-Item)**:
   - Evaluated on **mean-centered** ratings to remove individual rating calibration differences (e.g., optimistic vs critical users).
   - Distance Metrics: Cosine similarity and Pearson correlation score.
   - Shrinkage: Applied factor $n_{uv} / (n_{uv} + 10)$ to penalize high similarities computed over small co-rating support.
   - Hyperparameter Sweep: Neighbor parameter $k \in \{10, 20, 40, 80\}$.
4. **Biased Matrix Factorization (PyTorch from Scratch)**:
   - Formulation: $\hat{r}_{u,i} = \mu + b_u + b_i + p_u \cdot q_i$.
   - Loss Function: Mean Squared Error + $L_2$ regularization on factor embeddings and biases ($\lambda = 0.02$).
   - Optimizer: Adam optimizer ($lr = 0.005$, batch size 256, 20 epochs). Adam was selected over SGD for faster, stable convergence on sparse matrix representations.
   - Hyperparameter Sweep: Latent dimension $d \in \{8, 16, 32, 64\}$.

---

## 4. Results Table

All models were tuned strictly on validation set RMSE/NDCG@10. The test set was evaluated **once** for final reporting.

| Model | RMSE | MAE | Precision@10 | Recall@10 | NDCG@10 |
|---|---|---|---|---|---|
| **Global Mean** | 1.1352 | 0.9518 | 0.0084 | 0.0142 | 0.0102 |
| **User + Item Bias** | 0.9635 | 0.7615 | 0.0400 | 0.0633 | 0.0519 |
| **User-kNN (k=40, Pearson)** | 0.9922 | 0.7813 | **0.0999** | **0.1779** | **0.1654** |
| **Item-kNN (k=10, Pearson)** | **0.9588** | **0.7590** | 0.0261 | 0.0291 | 0.0353 |
| **Biased MF (d=16)** | 1.0834 | 0.9040 | 0.0624 | 0.0925 | 0.0916 |

---

## 5. Figures

- **`figures/rmse_vs_k.png`**: Shows validation RMSE across neighbor sizes $k$. Item-item kNN achieves optimal rating RMSE at lower $k$ ($k=10\text{--}20$), whereas user-user kNN benefits from larger neighbor pools.
- **`figures/rmse_vs_latent_dim.png`**: Validation RMSE vs latent dimension $d$. Latent dimension $d=16$ achieves the best balance between expressiveness and regularized generalization.
- **`figures/mf_train_val_curve.png`**: Epoch-by-epoch loss curve for Biased MF showing smooth convergence without severe overfitting.
- **`figures/test_metrics_barchart.png`**: Side-by-side comparison of RMSE, MAE, and NDCG@10 across test models.

---

## 6. Analysis

### Rating RMSE vs Top-10 Ranking Trade-off
A key empirical finding is the trade-off between rating prediction accuracy (RMSE) and top-10 item ranking metrics (NDCG@10 / Recall@10):
- **Item-kNN ($k=10$)** achieves the lowest prediction RMSE (**0.9588**), closely followed by the simple **User+Item Bias** baseline (**0.9635**). However, their ranking performance is low (NDCG@10 $\le 0.05$).
- **User-kNN ($k=40$)** achieves slightly higher rating RMSE (**0.9922**), but achieves dramatically higher ranking metrics: **NDCG@10 = 0.1654** and **Recall@10 = 0.1779** (over 3.1x higher than item-kNN).
- **Why this happens**: RMSE evaluates average prediction error over all items (including uninteresting items rated 1–3). In contrast, ranking metrics evaluate whether the top-10 recommended unrated candidate items contain relevant items (rated $\ge 4$). User-user kNN effectively pools preference signals from similar users, capturing tail interest alignment better than item-item similarity.

---

## 7. Conclusion & Lessons Learned

### When kNN is Preferable
- **Explainability**: Clear rationale ("recommended because users like you enjoyed X").
- **No Training Overhead**: Zero model re-training cost; instantaneous update when new ratings arrive.
- **Quick Cold Start Handling**: Immediately incorporates new items once a single co-rating is received.

### When Matrix Factorization is Preferable
- **Scalability**: Inference requires only low-dimensional dot products ($O(d)$ per item) rather than pairwise neighbor searches ($O(N)$ or $O(M)$).
- **Dense Latent Representational Capacity**: Learns continuous vector representations that generalize across sparse rating matrices better than raw co-occurrence overlap.

### Shared Weaknesses & Transition to Task 02
Both kNN and MF suffer from fundamental limitations:
1. **Cold-Start Problem**: Inability to generate recommendations for completely new users or items without historical ratings.
2. **Lack of Side Features**: Ignored contextual features such as user demographics, item tags, device details, time of day, or impression features.
3. **Linear / Fixed Interaction**: Matrix factorization models interactions purely as inner products $\langle p_u, q_i \rangle$.

These limitations motivate **Task 02 (Neural CTR)**, which incorporates high-cardinality categorical context and continuous numerical side features into non-linear neural representations.

---

## 8. Limitations, Subsampling & What Was Not Done

- **Dataset Scope**: Evaluated strictly on MovieLens 100K (100,000 ratings across 943 users and 1,682 items). Larger MovieLens datasets (1M or 20M) were not run due to local CPU memory/compute constraints.
- **Side Features**: Neither kNN nor MF utilizes available movie genres or user demographic metadata (age, gender, occupation).
- **Implicit Feedback**: Ratings were treated strictly as explicit 1–5 numerical values; implicit feedback signals (clicks, watch time, views) were not modeled in this task.

