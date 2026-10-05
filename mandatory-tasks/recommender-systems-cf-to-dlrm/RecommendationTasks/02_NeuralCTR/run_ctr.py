"""Headless runner for Task 02 Neural CTR experiments."""
import sys
import pathlib

TASK_DIR = pathlib.Path(__file__).resolve().parent   # 02_NeuralCTR/
ROOT     = TASK_DIR.parent                            # RecommendationTasks/
sys.path.insert(0, str(TASK_DIR))
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pandas as pd

from common.utils import set_seed, get_device, count_params
from common.ctr_data import load_ctr_data
from common.metrics import compute_metrics, find_best_f1_threshold, plot_reliability_diagram
from models import LogisticRegression, MLPA, MLPB, MLPC, DCN, BaseMLP
from train import train_model, eval_on_test, evaluate, save_results

FIGURES_DIR = TASK_DIR / "figures"
RESULTS_DIR = TASK_DIR / "results"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

DATA_DIR   = pathlib.Path("d:/Wec/Mandatory_Tasks/Recommender_Systems/dataset_2_3")
SPLIT_PATH = ROOT / "common" / "split.npz"

set_seed(42)
DEVICE = get_device()
print(f"Device: {DEVICE}")

print("Loading CTR data...")
train_loader, val_loader, test_loader, NUM_NUMERIC, CAT_CARDS = load_ctr_data(
    data_dir=DATA_DIR,
    split_save_path=SPLIT_PATH,
    batch_size=2048,
    min_freq=10,
    seed=42,
)
print(f"Numeric features: {NUM_NUMERIC}, Categorical fields: {len(CAT_CARDS)}")
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")


def run_experiment(name, model, tag):
    print(f"\n{'='*55}")
    print(f"  Training: {name}  ({count_params(model):,} params)")
    print(f"{'='*55}")
    set_seed(42)
    info = train_model(
        model, train_loader, val_loader,
        lr=1e-3, weight_decay=1e-5,
        max_epochs=10, patience=2,
        device=DEVICE,
    )
    result = {
        "model": name,
        "n_params": info["n_params"],
        "best_val_loss": info["best_val_loss"],
        "best_epoch": info["best_epoch"],
        "history": info["history"],
        "peak_memory_mb": info["peak_memory_mb"],
        "best_threshold": info["best_threshold"],
    }
    save_results(result, RESULTS_DIR / f"{tag}.json")
    return model, info


all_results = {}

all_results["LR"]       = run_experiment("LR",              LogisticRegression(NUM_NUMERIC, CAT_CARDS),         "lr_baseline")
all_results["MLP-A"]    = run_experiment("MLP-A",           MLPA(NUM_NUMERIC, CAT_CARDS, embed_dim=16),         "mlp_a")
all_results["MLP-B"]    = run_experiment("MLP-B",           MLPB(NUM_NUMERIC, CAT_CARDS, embed_dim=16),         "mlp_b")
all_results["MLP-C"]    = run_experiment("MLP-C",           MLPC(NUM_NUMERIC, CAT_CARDS, embed_dim=16),         "mlp_c")
all_results["MLP-B-d8"] = run_experiment("MLP-B (dim=8)",   MLPB(NUM_NUMERIC, CAT_CARDS, embed_dim=8),          "mlp_b_d8")
all_results["MLP-B-d32"]= run_experiment("MLP-B (dim=32)",  MLPB(NUM_NUMERIC, CAT_CARDS, embed_dim=32),         "mlp_b_d32")
all_results["DCN"]      = run_experiment("DCN",             DCN(NUM_NUMERIC, CAT_CARDS, embed_dim=16,
                                                                 mlp_hidden=[256,256], mlp_dropout=0.2,
                                                                 cross_layers=3),                                "dcn")

# ---------- Loss curves ----------
print("\nGenerating loss curves...")
fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=False)
axes = axes.flatten()
for ax, (name, (_, info)) in zip(axes, all_results.items()):
    hist   = info["history"]
    epochs = range(1, len(hist["train_loss"]) + 1)
    ax.plot(epochs, hist["train_loss"], "o-",  label="Train", color="steelblue")
    ax.plot(epochs, hist["val_loss"],   "s--", label="Val",   color="crimson")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=7); ax.grid(True, linestyle="--", alpha=0.4)
for ax in axes[len(all_results):]:
    ax.set_visible(False)
plt.suptitle("Train vs Val Loss — All Neural CTR Models", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved loss_curves.png")

# ---------- Val metrics table ----------
print("\nBuilding val metrics table...")
criterion = nn.BCEWithLogitsLoss()
rows = []
for name, (model, info) in all_results.items():
    _, y_true_val, y_prob_val = evaluate(model, val_loader, criterion, DEVICE)
    thresh = info["best_threshold"] or find_best_f1_threshold(y_true_val, y_prob_val)
    m = compute_metrics(y_true_val, y_prob_val, threshold=thresh)
    sec_per_epoch = (
        sum(info["history"]["epoch_seconds"]) / len(info["history"]["epoch_seconds"])
        if info["history"]["epoch_seconds"] else 0
    )
    rows.append({
        "Model":        name,
        "ROC-AUC":      round(m["roc_auc"], 4),
        "PR-AUC":       round(m["pr_auc"], 4),
        "Log Loss":     round(m["log_loss"], 4),
        "Accuracy":     round(m["accuracy_0.5"], 4),
        "F1":           round(m["f1"], 4),
        "Precision":    round(m["precision"], 4),
        "Recall":       round(m["recall"], 4),
        "ECE":          round(m["ece"], 4),
        "Threshold":    round(thresh, 3),
        "Params":       info["n_params"],
        "Peak Mem(MB)": info["peak_memory_mb"],
        "Sec/Epoch":    round(sec_per_epoch, 1),
        "Best Epoch":   info["best_epoch"],
    })

df = pd.DataFrame(rows).sort_values("Log Loss")
print(df.to_string(index=False))
df.to_csv(RESULTS_DIR / "summary_table.csv", index=False)
print("Saved summary_table.csv")

# ---------- Best model on test (ONCE) ----------
best_name = df.iloc[0]["Model"]
best_model, best_info = all_results[best_name]
best_threshold = best_info["best_threshold"] or 0.5
print(f"\nBest model: {best_name}  val_loss={best_info['best_val_loss']:.4f}  threshold={best_threshold:.3f}")

test_metrics = eval_on_test(best_model, test_loader, threshold=best_threshold, device=DEVICE)
print("Test metrics:", test_metrics)
save_results({"model": best_name, "threshold": best_threshold, **test_metrics},
             RESULTS_DIR / "best_model_test.json")

# ---------- Reliability diagram ----------
print("Generating reliability diagram...")
_, y_true_val, y_prob_val = evaluate(best_model, val_loader, criterion, DEVICE)
plot_reliability_diagram(y_true_val, y_prob_val,
                         model_name=best_name,
                         save_path=FIGURES_DIR / "reliability_diagram.png")
print("Saved reliability_diagram.png")

# ---------- ROC & PR curves for top 3 ----------
print("Generating ROC/PR curves...")
from sklearn.metrics import roc_curve, precision_recall_curve

top3 = df["Model"].head(3).tolist()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for name in top3:
    model, info = all_results[name]
    _, y_tv, y_pv = evaluate(model, val_loader, criterion, DEVICE)
    fpr, tpr, _   = roc_curve(y_tv, y_pv)
    prec, rec, _  = precision_recall_curve(y_tv, y_pv)
    auc_v = compute_metrics(y_tv, y_pv)["roc_auc"]
    ap_v  = compute_metrics(y_tv, y_pv)["pr_auc"]
    ax1.plot(fpr, tpr, label=f"{name} (AUC={auc_v:.4f})")
    ax2.plot(rec, prec, label=f"{name} (AP={ap_v:.4f})")

ax1.plot([0,1],[0,1],"k--"); ax1.set_xlabel("FPR"); ax1.set_ylabel("TPR")
ax1.set_title("ROC Curves — Top 3 (Val)"); ax1.legend(); ax1.grid(True, linestyle="--", alpha=0.4)
ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
ax2.set_title("PR Curves — Top 3 (Val)"); ax2.legend(); ax2.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "roc_pr_curves.png", dpi=150)
plt.close()
print("Saved roc_pr_curves.png")

# ---------- Overfitting gap table ----------
gap_rows = []
for name, (_, info) in all_results.items():
    be = info["best_epoch"] - 1
    gap_rows.append({
        "Model":       name,
        "Train Loss":  info["history"]["train_loss"][be],
        "Val Loss":    info["history"]["val_loss"][be],
        "Gap":         round(info["history"]["val_loss"][be] - info["history"]["train_loss"][be], 6),
        "Epochs run":  len(info["history"]["train_loss"]),
    })
df_gap = pd.DataFrame(gap_rows).sort_values("Gap", ascending=False)
print("\nOverfitting gap:\n", df_gap.to_string(index=False))
df_gap.to_csv(RESULTS_DIR / "overfitting_gap.csv", index=False)
print("Saved overfitting_gap.csv")

print("\n=== Task 02 complete. All figures and results saved. ===")
