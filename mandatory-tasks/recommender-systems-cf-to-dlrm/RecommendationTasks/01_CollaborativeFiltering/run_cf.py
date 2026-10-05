"""Re-run all CF experiments and regenerate figures with the corrected similarity fix."""
import sys
import pathlib

# Make sure both the task folder and the parent (RecommendationTasks) are on path
TASK_DIR = pathlib.Path(__file__).resolve().parent          # 01_CollaborativeFiltering/
ROOT     = TASK_DIR.parent                                   # RecommendationTasks/
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # headless
import matplotlib.pyplot as plt
import torch

from common.utils import set_seed
from cf_lib import (
    load_movielens_100k, split_data, BaselineModel,
    KNNRecommender, train_biased_mf,
    evaluate_rating_metrics, evaluate_ranking_metrics, BiasedMF,
)

def run_all_cf():
    set_seed(42)
    data_path = pathlib.Path("d:/Wec/Mandatory_Tasks/Recommender_Systems/ml-100k/u.data")
    fig_dir   = TASK_DIR / "figures"
    res_dir   = TASK_DIR / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MovieLens 100K...")
    df = load_movielens_100k(str(data_path))
    df_train, df_val, df_test = split_data(df, seed=42)
    n_users = df["user_id"].max() + 1
    n_items = df["item_id"].max() + 1
    print(f"Users: {n_users}, Items: {n_items}, Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    results = {}

    # 1. Global Mean
    print("\n--- 1. Global Mean Baseline ---")
    bm_global = BaselineModel(mode="global")
    bm_global.fit(df_train)
    vp = bm_global.predict_batch(df_val["user_id"].values, df_val["item_id"].values)
    vr = evaluate_rating_metrics(df_val["rating"].values, vp)
    vk = evaluate_ranking_metrics(bm_global, df_train, df_val, n_users, n_items)
    print("Val:", {**vr, **vk})
    results["Global_Mean"] = {"val": {**vr, **vk}}

    # 2. User+Item Bias
    print("\n--- 2. User+Item Bias Baseline ---")
    bm_bias = BaselineModel(mode="bias")
    bm_bias.fit(df_train, reg=10.0)
    vp = bm_bias.predict_batch(df_val["user_id"].values, df_val["item_id"].values)
    vr = evaluate_rating_metrics(df_val["rating"].values, vp)
    vk = evaluate_ranking_metrics(bm_bias, df_train, df_val, n_users, n_items)
    print("Val:", {**vr, **vk})
    results["User_Item_Bias"] = {"val": {**vr, **vk}}

    # 3. kNN sweeps
    print("\n--- 3. kNN sweeps ---")
    k_values    = [10, 20, 40, 80]
    knn_configs = [("item","pearson"),("item","cosine"),("user","pearson"),("user","cosine")]
    knn_results = {}
    best_knn_cfg, best_knn_rmse = None, float("inf")

    for mode, sim in knn_configs:
        key = f"{mode}_{sim}"
        knn_results[key] = {}
        for k in k_values:
            print(f"  Fitting kNN mode={mode}, sim={sim}, k={k}...")
            mdl = KNNRecommender(mode=mode, similarity=sim, k=k, shrinkage=10.0)
            mdl.fit(df_train, n_users, n_items)
            vp  = mdl.predict_batch(df_val["user_id"].values, df_val["item_id"].values)
            vr  = evaluate_rating_metrics(df_val["rating"].values, vp)
            vk  = evaluate_ranking_metrics(mdl, df_train, df_val, n_users, n_items)
            knn_results[key][k] = {**vr, **vk}
            print(f"    Val RMSE={vr['rmse']:.4f}  NDCG@10={vk['ndcg@10']:.4f}")
            if vr["rmse"] < best_knn_rmse:
                best_knn_rmse = vr["rmse"]
                best_knn_cfg  = (mode, sim, k)

    print(f"\nBest kNN: {best_knn_cfg}  RMSE={best_knn_rmse:.4f}")

    # Plot RMSE vs k (4 distinct lines now)
    style_map = {
        "item_pearson": ("o-", "steelblue"),
        "item_cosine":  ("s--","darkorange"),
        "user_pearson": ("^-", "forestgreen"),
        "user_cosine":  ("D--","crimson"),
    }
    label_map = {
        "item_pearson": "Item-Item (Pearson)",
        "item_cosine":  "Item-Item (Cosine)",
        "user_pearson": "User-User (Pearson)",
        "user_cosine":  "User-User (Cosine)",
    }
    plt.figure(figsize=(8, 5))
    for mode, sim in knn_configs:
        key   = f"{mode}_{sim}"
        rmses = [knn_results[key][k]["rmse"] for k in k_values]
        fmt, color = style_map[key]
        plt.plot(k_values, rmses, fmt, color=color, label=label_map[key], linewidth=2, markersize=7)
    plt.xlabel("k (Number of Neighbors)")
    plt.ylabel("Validation RMSE")
    plt.title("kNN Performance: Validation RMSE vs k")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "rmse_vs_k.png", dpi=150)
    plt.close()
    print("Saved rmse_vs_k.png")

    # 4. Biased MF sweeps
    print("\n--- 4. Biased MF sweeps ---")
    dims = [8, 16, 32, 64]
    mf_results  = {}
    best_mf_dim, best_mf_rmse, best_mf_hist, best_mf_model = None, float("inf"), None, None

    for d in dims:
        print(f"  Training Biased MF dim={d}...")
        mdl, tr_hist, va_hist = train_biased_mf(
            df_train, df_val, n_users, n_items,
            dim=d, lr=0.005, weight_decay=0.02, epochs=20, batch_size=256, seed=42,
        )
        vp = mdl(
            torch.tensor(df_val["user_id"].values, dtype=torch.long),
            torch.tensor(df_val["item_id"].values, dtype=torch.long),
        ).clamp(1.0, 5.0).detach().numpy()
        vr = evaluate_rating_metrics(df_val["rating"].values, vp)
        vk = evaluate_ranking_metrics(mdl, df_train, df_val, n_users, n_items)
        mf_results[d] = {**vr, **vk}
        print(f"    Val RMSE={vr['rmse']:.4f}  NDCG@10={vk['ndcg@10']:.4f}")
        if vr["rmse"] < best_mf_rmse:
            best_mf_rmse, best_mf_dim, best_mf_hist, best_mf_model = vr["rmse"], d, (tr_hist, va_hist), mdl

    print(f"\nBest MF dim={best_mf_dim}  RMSE={best_mf_rmse:.4f}")

    # Plot RMSE vs latent dim
    plt.figure(figsize=(7, 5))
    plt.plot(dims, [mf_results[d]["rmse"] for d in dims], "s-", color="darkgreen", linewidth=2, markersize=8)
    plt.xlabel("Latent Dimension (d)")
    plt.ylabel("Validation RMSE")
    plt.title("Biased MF: Validation RMSE vs Latent Dimension")
    plt.xticks(dims)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "rmse_vs_latent_dim.png", dpi=150)
    plt.close()
    print("Saved rmse_vs_latent_dim.png")

    # Plot MF train/val curve
    plt.figure(figsize=(7, 5))
    e_range = range(1, len(best_mf_hist[0]) + 1)
    plt.plot(e_range, best_mf_hist[0], "o-", label="Train RMSE", color="steelblue")
    plt.plot(e_range, best_mf_hist[1], "s-", label="Val RMSE",   color="crimson")
    plt.xlabel("Epoch"); plt.ylabel("RMSE")
    plt.title(f"Biased MF (d={best_mf_dim}) Train & Val Curve")
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "mf_train_val_curve.png", dpi=150)
    plt.close()
    print("Saved mf_train_val_curve.png")

    # 5. Test evaluation (touch test once)
    print("\n--- 5. Test evaluation ---")
    test_summary = {}

    def _test(mdl, name):
        if isinstance(mdl, (BaselineModel,)):
            tp = mdl.predict_batch(df_test["user_id"].values, df_test["item_id"].values)
        else:
            tp = mdl(
                torch.tensor(df_test["user_id"].values, dtype=torch.long),
                torch.tensor(df_test["item_id"].values, dtype=torch.long),
            ).clamp(1.0, 5.0).detach().numpy()
        tr = evaluate_rating_metrics(df_test["rating"].values, tp)
        tk = evaluate_ranking_metrics(mdl, df_train, df_test, n_users, n_items)
        test_summary[name] = {**tr, **tk}
        print(f"  {name}: RMSE={tr['rmse']:.4f}  NDCG@10={tk['ndcg@10']:.4f}")

    _test(bm_global, "Global_Mean")
    _test(bm_bias,   "User_Item_Bias")

    # Best user kNN
    bu_k = max(knn_results["user_pearson"], key=lambda k: knn_results["user_pearson"][k]["ndcg@10"])
    ku = KNNRecommender(mode="user", similarity="pearson", k=bu_k, shrinkage=10.0)
    ku.fit(df_train, n_users, n_items)
    tp = ku.predict_batch(df_test["user_id"].values, df_test["item_id"].values)
    tr = evaluate_rating_metrics(df_test["rating"].values, tp)
    tk = evaluate_ranking_metrics(ku, df_train, df_test, n_users, n_items)
    test_summary[f"User_kNN(k={bu_k})"] = {**tr, **tk}
    print(f"  User_kNN(k={bu_k}): RMSE={tr['rmse']:.4f}  NDCG@10={tk['ndcg@10']:.4f}")

    # Best item kNN
    bi_k = max(knn_results["item_pearson"], key=lambda k: knn_results["item_pearson"][k]["ndcg@10"])
    ki = KNNRecommender(mode="item", similarity="pearson", k=bi_k, shrinkage=10.0)
    ki.fit(df_train, n_users, n_items)
    tp = ki.predict_batch(df_test["user_id"].values, df_test["item_id"].values)
    tr = evaluate_rating_metrics(df_test["rating"].values, tp)
    tk = evaluate_ranking_metrics(ki, df_train, df_test, n_users, n_items)
    test_summary[f"Item_kNN(k={bi_k})"] = {**tr, **tk}
    print(f"  Item_kNN(k={bi_k}): RMSE={tr['rmse']:.4f}  NDCG@10={tk['ndcg@10']:.4f}")

    _test(best_mf_model, f"Biased_MF(d={best_mf_dim})")

    # Save JSON
    with open(res_dir / "cf_test_metrics.json", "w") as f:
        json.dump(test_summary, f, indent=2)

    # Bar chart
    models_list = list(test_summary.keys())
    x = np.arange(len(models_list)); w = 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - w, [test_summary[m]["rmse"]   for m in models_list], w, label="RMSE",    color="crimson",     alpha=0.85)
    ax.bar(x,     [test_summary[m]["mae"]    for m in models_list], w, label="MAE",     color="steelblue",   alpha=0.85)
    ax.bar(x + w, [test_summary[m]["ndcg@10"] for m in models_list], w, label="NDCG@10",color="forestgreen", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(models_list, rotation=20, ha="right")
    ax.set_ylabel("Metric Value"); ax.set_title("Task 01: Test Metrics Across All CF Models")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(fig_dir / "test_metrics_barchart.png", dpi=150)
    plt.close()
    print("Saved test_metrics_barchart.png")
    print("\nAll done.")

if __name__ == "__main__":
    run_all_cf()
