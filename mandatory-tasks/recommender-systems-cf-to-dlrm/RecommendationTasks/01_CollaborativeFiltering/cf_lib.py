"""Collaborative Filtering algorithms from scratch: Baselines, User-User & Item-Item kNN, Biased Matrix Factorization, and Ranking Metrics."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


def load_movielens_100k(data_path: str) -> pd.DataFrame:
    """Loads MovieLens 100K u.data into a DataFrame with columns [user_id, item_id, rating, timestamp]."""
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
    )
    df["user_id"] = df["user_id"] - 1
    df["item_id"] = df["item_id"] - 1
    return df


def split_data(
    df: pd.DataFrame, seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits ratings 80/10/10 into train/val/test using fixed random seed."""
    np.random.seed(seed)
    n = len(df)
    shuffled_indices = np.random.permutation(n)
    
    n_train = int(0.80 * n)
    n_val = int(0.10 * n)
    
    train_idx = shuffled_indices[:n_train]
    val_idx = shuffled_indices[n_train : n_train + n_val]
    test_idx = shuffled_indices[n_train + n_val :]
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    return df_train, df_val, df_test


class BaselineModel:
    """Global Mean and User+Item Bias baseline predictor."""

    def __init__(self, mode: str = "bias"):
        self.mode = mode
        self.mu: float = 0.0
        self.b_u: Dict[int, float] = {}
        self.b_i: Dict[int, float] = {}

    def fit(self, df_train: pd.DataFrame, reg: float = 10.0) -> None:
        """Fits global mean and regularized user/item biases."""
        self.mu = float(df_train["rating"].mean())
        if self.mode == "global":
            return

        user_grp = df_train.groupby("user_id")["rating"]
        for u, s in user_grp:
            self.b_u[u] = float((s.sum() - len(s) * self.mu) / (len(s) + reg))

        df_train_copy = df_train.copy()
        df_train_copy["u_bias"] = df_train_copy["user_id"].map(lambda u: self.b_u.get(u, 0.0))
        item_grp = df_train_copy.groupby("item_id")
        for i, grp in item_grp:
            res = grp["rating"] - self.mu - grp["u_bias"]
            self.b_i[i] = float(res.sum() / (len(grp) + reg))

    def predict(self, user_id: int, item_id: int) -> float:
        """Predicts rating for a single (user, item) pair."""
        if self.mode == "global":
            return self.mu
        bu = self.b_u.get(user_id, 0.0)
        bi = self.b_i.get(item_id, 0.0)
        return float(np.clip(self.mu + bu + bi, 1.0, 5.0))

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Predicts ratings for a batch of (user, item) pairs."""
        if self.mode == "global":
            return np.full(len(users), self.mu, dtype=np.float32)
        
        preds = np.zeros(len(users), dtype=np.float32)
        for idx in range(len(users)):
            bu = self.b_u.get(users[idx], 0.0)
            bi = self.b_i.get(items[idx], 0.0)
            preds[idx] = self.mu + bu + bi
        return np.clip(preds, 1.0, 5.0)

    def predict_full_matrix(self, n_users: int, n_items: int) -> np.ndarray:
        """Predicts entire rating matrix [N_users, N_items]."""
        R = np.full((n_users, n_items), self.mu, dtype=np.float32)
        if self.mode == "global":
            return R

        bu_arr = np.array([self.b_u.get(u, 0.0) for u in range(n_users)], dtype=np.float32)
        bi_arr = np.array([self.b_i.get(i, 0.0) for i in range(n_items)], dtype=np.float32)
        
        R += bu_arr[:, None] + bi_arr[None, :]
        return np.clip(R, 1.0, 5.0)


class KNNRecommender:
    """User-User and Item-Item kNN on mean-centered ratings with Pearson/Cosine & shrinkage."""

    def __init__(
        self,
        mode: str = "item",
        similarity: str = "pearson",
        k: int = 20,
        shrinkage: float = 10.0,
    ):
        self.mode = mode
        self.similarity = similarity
        self.k = k
        self.shrinkage = shrinkage
        self.n_users = 0
        self.n_items = 0
        self.R: np.ndarray = np.array([])
        self.R_centered: np.ndarray = np.array([])
        self.means: np.ndarray = np.array([])
        self.sim_matrix: np.ndarray = np.array([])
        self.global_mean: float = 0.0
        self.user_biases: np.ndarray = np.array([])
        self.item_biases: np.ndarray = np.array([])

    def fit(self, df_train: pd.DataFrame, n_users: int, n_items: int) -> None:
        """Fits similarity matrix and baseline means from training ratings."""
        self.n_users = n_users
        self.n_items = n_items
        self.global_mean = float(df_train["rating"].mean())

        self.R = np.zeros((n_users, n_items), dtype=np.float32)
        mask = np.zeros((n_users, n_items), dtype=bool)

        for _, row in df_train.iterrows():
            u, i, r = int(row["user_id"]), int(row["item_id"]), float(row["rating"])
            self.R[u, i] = r
            mask[u, i] = True

        if self.mode == "user":
            counts = mask.sum(axis=1)
            sums = self.R.sum(axis=1)
            self.means = np.where(counts > 0, sums / np.maximum(1, counts), self.global_mean)
            self.R_centered = np.where(mask, self.R - self.means[:, None], 0.0)
            target_matrix = self.R_centered
        else:
            counts = mask.sum(axis=0)
            sums = self.R.sum(axis=0)
            self.means = np.where(counts > 0, sums / np.maximum(1, counts), self.global_mean)
            self.R_centered = np.where(mask, self.R - self.means[None, :], 0.0)
            target_matrix = self.R_centered.T

        u_counts = mask.sum(axis=1)
        self.user_biases = np.where(u_counts > 0, (self.R.sum(axis=1) - u_counts * self.global_mean) / (u_counts + 10.0), 0.0)
        i_counts = mask.sum(axis=0)
        self.item_biases = np.where(i_counts > 0, (self.R.sum(axis=0) - i_counts * self.global_mean) / (i_counts + 10.0), 0.0)

        # --- Similarity computation ---
        # Pearson: cosine on mean-centered vectors (already in target_matrix).
        # Cosine:  cosine on raw (non-centered) rating vectors.
        if self.similarity == "cosine":
            # Build raw (non-centered) version of the same target matrix
            if self.mode == "user":
                sim_matrix_input = np.where(mask, self.R, 0.0)
            else:
                sim_matrix_input = np.where(mask, self.R, 0.0).T
        else:
            # "pearson" — use mean-centered vectors
            sim_matrix_input = target_matrix

        binary_mask = (sim_matrix_input != 0).astype(np.float32)
        co_counts = binary_mask @ binary_mask.T

        norms = np.sqrt(np.sum(sim_matrix_input ** 2, axis=1, keepdims=True))
        denom = norms @ norms.T
        denom = np.where(denom == 0, 1e-9, denom)
        dot = sim_matrix_input @ sim_matrix_input.T
        raw_sim = dot / denom

        shrinkage_factor = co_counts / (co_counts + self.shrinkage)
        self.sim_matrix = raw_sim * shrinkage_factor
        np.fill_diagonal(self.sim_matrix, 0.0)

    def predict_full_matrix(self) -> np.ndarray:
        """Predicts entire rating matrix [N_users, N_items] using top-k neighbors."""
        R_pred = np.zeros((self.n_users, self.n_items), dtype=np.float32)
        base = self.global_mean + self.user_biases[:, None] + self.item_biases[None, :]

        if self.mode == "user":
            # For each user u, get top-k user neighbors
            top_k_idx = np.argsort(-self.sim_matrix, axis=1)[:, :self.k]
            top_k_sims = np.take_along_axis(self.sim_matrix, top_k_idx, axis=1)
            # Replace negative sims with 0
            top_k_sims = np.maximum(0.0, top_k_sims)

            for u in range(self.n_users):
                sims = top_k_sims[u]
                sum_sims = np.sum(np.abs(sims))
                if sum_sims > 0:
                    nbr_idx = top_k_idx[u]
                    weighted_ratings = sims @ self.R_centered[nbr_idx, :]
                    R_pred[u] = self.means[u] + weighted_ratings / sum_sims
                else:
                    R_pred[u] = base[u]
        else:
            # Item-item mode
            top_k_idx = np.argsort(-self.sim_matrix, axis=1)[:, :self.k]
            top_k_sims = np.take_along_axis(self.sim_matrix, top_k_idx, axis=1)
            top_k_sims = np.maximum(0.0, top_k_sims)

            for i in range(self.n_items):
                sims = top_k_sims[i]
                sum_sims = np.sum(np.abs(sims))
                if sum_sims > 0:
                    nbr_idx = top_k_idx[i]
                    weighted_ratings = sims @ self.R_centered.T[nbr_idx, :]
                    R_pred[:, i] = self.means[i] + weighted_ratings / sum_sims
                else:
                    R_pred[:, i] = base[:, i]

        return np.clip(R_pred, 1.0, 5.0)

    def predict(self, u: int, i: int) -> float:
        """Predicts single rating."""
        if u >= self.n_users or i >= self.n_items:
            fallback = self.global_mean + (self.user_biases[u] if u < self.n_users else 0.0) + (self.item_biases[i] if i < self.n_items else 0.0)
            return float(np.clip(fallback, 1.0, 5.0))
        full = self.predict_full_matrix()
        return float(full[u, i])

    def predict_batch(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        """Predicts ratings for batch."""
        full_R = self.predict_full_matrix()
        preds = np.zeros(len(users), dtype=np.float32)
        for idx in range(len(users)):
            u, i = users[idx], items[idx]
            if u < self.n_users and i < self.n_items:
                preds[idx] = full_R[u, i]
            else:
                fallback = self.global_mean + (self.user_biases[u] if u < self.n_users else 0.0) + (self.item_biases[i] if i < self.n_items else 0.0)
                preds[idx] = np.clip(fallback, 1.0, 5.0)
        return preds


class BiasedMF(nn.Module):
    """Biased Matrix Factorization model: r_hat = mu + b_u + b_i + p_u . q_i."""

    def __init__(self, n_users: int, n_items: int, dim: int = 16, global_mean: float = 3.5):
        super().__init__()
        self.global_mean = nn.Parameter(torch.tensor(global_mean, dtype=torch.float32), requires_grad=False)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.user_embeddings = nn.Embedding(n_users, dim)
        self.item_embeddings = nn.Embedding(n_items, dim)

        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(self, u: torch.Tensor, i: torch.Tensor) -> torch.Tensor:
        """Forward pass outputting predicted ratings [B]."""
        mu = self.global_mean
        bu = self.user_bias(u).squeeze(-1)
        bi = self.item_bias(i).squeeze(-1)
        pu = self.user_embeddings(u)
        qi = self.item_embeddings(i)
        dot = (pu * qi).sum(dim=-1)
        return mu + bu + bi + dot

    def predict_full_matrix(self, n_users: int, n_items: int) -> np.ndarray:
        """Predicts entire rating matrix [N_users, N_items]."""
        self.eval()
        with torch.no_grad():
            u_all = torch.arange(n_users, dtype=torch.long)
            i_all = torch.arange(n_items, dtype=torch.long)
            
            bu = self.user_bias(u_all).numpy()  # [N_users, 1]
            bi = self.item_bias(i_all).squeeze(-1).numpy()  # [N_items]
            pu = self.user_embeddings(u_all).numpy()  # [N_users, d]
            qi = self.item_embeddings(i_all).numpy()  # [N_items, d]

            mu = self.global_mean.item()
            pred = mu + bu + bi[None, :] + (pu @ qi.T)
            return np.clip(pred, 1.0, 5.0)


def train_biased_mf(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    n_users: int,
    n_items: int,
    dim: int = 16,
    lr: float = 0.005,
    weight_decay: float = 0.02,
    epochs: int = 20,
    batch_size: int = 256,
    seed: int = 42,
) -> Tuple[BiasedMF, List[float], List[float]]:
    """Trains Biased MF model from scratch with Adam and tracks train/val RMSE history."""
    torch.manual_seed(seed)
    global_mean = float(df_train["rating"].mean())
    model = BiasedMF(n_users, n_items, dim=dim, global_mean=global_mean)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    u_train = torch.tensor(df_train["user_id"].values, dtype=torch.long)
    i_train = torch.tensor(df_train["item_id"].values, dtype=torch.long)
    r_train = torch.tensor(df_train["rating"].values, dtype=torch.float32)

    u_val = torch.tensor(df_val["user_id"].values, dtype=torch.long)
    i_val = torch.tensor(df_val["item_id"].values, dtype=torch.long)
    r_val = torch.tensor(df_val["rating"].values, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(u_train, i_train, r_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    train_rmse_hist = []
    val_rmse_hist = []

    for epoch in range(epochs):
        model.train()
        for batch_u, batch_i, batch_r in loader:
            optimizer.zero_grad()
            preds = model(batch_u, batch_i)
            loss = criterion(preds, batch_r)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            train_preds = model(u_train, i_train).clamp(1.0, 5.0)
            val_preds = model(u_val, i_val).clamp(1.0, 5.0)

            train_rmse = float(torch.sqrt(criterion(train_preds, r_train)).item())
            val_rmse = float(torch.sqrt(criterion(val_preds, r_val)).item())

            train_rmse_hist.append(train_rmse)
            val_rmse_hist.append(val_rmse)

    return model, train_rmse_hist, val_rmse_hist


def evaluate_rating_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Computes rating prediction RMSE and MAE."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return {"rmse": rmse, "mae": mae}


def evaluate_ranking_metrics(
    model,
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    n_users: int,
    n_items: int,
    top_k: int = 10,
    relevant_threshold: float = 4.0,
) -> Dict[str, float]:
    """Vectorized ranking metrics calculation for Precision@10, Recall@10, and NDCG@10."""
    train_user_items = df_train.groupby("user_id")["item_id"].apply(set).to_dict()
    df_rel = df_eval[df_eval["rating"] >= relevant_threshold]
    eval_user_rel = df_rel.groupby("user_id")["item_id"].apply(set).to_dict()

    if hasattr(model, "predict_full_matrix"):
        if isinstance(model, BiasedMF):
            full_R = model.predict_full_matrix(n_users, n_items)
        elif isinstance(model, BaselineModel):
            full_R = model.predict_full_matrix(n_users, n_items)
        else:
            full_R = model.predict_full_matrix()
    else:
        full_R = np.zeros((n_users, n_items), dtype=np.float32)

    precisions = []
    recalls = []
    ndcgs = []

    all_items = np.arange(n_items)

    for u, rel_items in eval_user_rel.items():
        if len(rel_items) == 0:
            continue

        train_rated = train_user_items.get(u, set())
        candidate_items = np.array([i for i in all_items if i not in train_rated])
        if len(candidate_items) == 0:
            continue

        scores = full_R[u, candidate_items]

        if len(scores) > top_k:
            top_k_idx = np.argpartition(scores, -top_k)[-top_k:]
            sorted_top_idx = top_k_idx[np.argsort(-scores[top_k_idx])]
        else:
            sorted_top_idx = np.argsort(-scores)

        rec_items = candidate_items[sorted_top_idx]

        hits = [1 if item in rel_items else 0 for item in rec_items]
        n_hits = sum(hits)

        prec = n_hits / top_k
        rec = n_hits / len(rel_items)

        dcg = sum(h / np.log2(idx + 2) for idx, h in enumerate(hits))
        idcg = sum(1.0 / np.log2(idx + 2) for idx in range(min(top_k, len(rel_items))))
        ndcg = dcg / idcg if idcg > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        ndcgs.append(ndcg)

    return {
        f"precision@{top_k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{top_k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{top_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
    }
