"""
Standalone training script for Task 03: DLRM from scratch.
Run from the 03_DLRM/ directory:
    python run_dlrm.py

Reuses the same split.npz, preprocessing, training loop, and metrics
as Task 02 (neural_ctr) for a fair comparison.
"""

import sys
import pathlib
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import platform

# Path setup
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "02_NeuralCTR"))

from common.utils import set_seed, get_device, count_params
from common.ctr_data import load_ctr_data
from common.metrics import compute_metrics, find_best_f1_threshold, plot_reliability_diagram
from train import train_model, evaluate, eval_on_test, save_results
from dlrm import DLRM, _make_mlp

DATA_DIR    = pathlib.Path(__file__).resolve().parents[2] / "dataset_2_3"
SPLIT_PATH  = ROOT / "common" / "split.npz"
RESULTS_DIR = HERE / "results"
FIGURES_DIR = HERE / "figures"
RESULTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DEVICE = get_device()

print("=" * 55)
print("  Python  : " + platform.python_version())
print("  PyTorch : " + torch.__version__)
gpu_info = (" (" + torch.cuda.get_device_name(0) + ")") if torch.cuda.is_available() else ""
print("  Device  : " + str(DEVICE) + gpu_info)
print("  Seed    : 42")
print("=" * 55)

# Load data
print("\n[1/6] Loading data ...")
train_loader, val_loader, test_loader, NUM_NUMERIC, CAT_CARDS = load_ctr_data(
    data_dir=DATA_DIR,
    split_save_path=SPLIT_PATH,
    batch_size=2048,
    min_freq=10,
    seed=42,
)
print("  Numeric features  : " + str(NUM_NUMERIC))
print("  Categorical fields: " + str(len(CAT_CARDS)))
print("  Train batches: " + str(len(train_loader)) + " | Val: " + str(len(val_loader)) + " | Test: " + str(len(test_loader)))

# Shape sanity check
print("\n[2/6] Shape sanity check ...")
set_seed(42)
_m = DLRM(NUM_NUMERIC, CAT_CARDS, embed_dim=16)
_xn = torch.randn(4, NUM_NUMERIC)
_xc = torch.zeros(4, len(CAT_CARDS), dtype=torch.long)
_lg = _m(_xn, _xc)
assert _lg.shape == (4,), "Shape mismatch: " + str(_lg.shape)
print("  n_interactions : " + str(_m.n_interactions) + "  (27*26/2)")
print("  top input dim  : " + str(16 + _m.n_interactions))
print("  Shape test PASSED")
del _m, _xn, _xc, _lg

# Experiment helper
all_results = {}

def run_experiment(name, model, tag):
    print("\n" + "="*60)
    print("  Training: " + name + "  (" + str(count_params(model)) + " params)")
    print("="*60)
    set_seed(42)
    run_info = train_model(
        model, train_loader, val_loader,
        lr=1e-3, weight_decay=1e-5,
        max_epochs=10, patience=2,
        device=DEVICE,
    )
    result = {
        "model": name,
        "n_params": run_info["n_params"],
        "best_val_loss": run_info["best_val_loss"],
        "best_epoch": run_info["best_epoch"],
        "history": run_info["history"],
        "peak_memory_mb": run_info["peak_memory_mb"],
        "best_threshold": run_info["best_threshold"],
    }
    save_results(result, RESULTS_DIR / (tag + ".json"))
    all_results[name] = (model, run_info)
    return model, run_info

# Train DLRM variants
print("\n[3/6] Training DLRM variants ...")

dlrm_d16 = DLRM(NUM_NUMERIC, CAT_CARDS, embed_dim=16,
                 bottom_mlp_sizes=[512, 256], top_mlp_sizes=[512, 256])
run_experiment("DLRM (d=16)", dlrm_d16, "dlrm_d16")


class DLRMNoInteraction(nn.Module):
    """DLRM with interaction replaced by plain concat [dense | emb_0 | ... | emb_25]."""
    def __init__(self, num_numeric, cat_cardinalities, embed_dim=16,
                 bottom_mlp_sizes=None, top_mlp_sizes=None):
        super().__init__()
        if bottom_mlp_sizes is None:
            bottom_mlp_sizes = [512, 256]
        if top_mlp_sizes is None:
            top_mlp_sizes = [512, 256]
        self.embed_dim = embed_dim
        self.n_cat = len(cat_cardinalities)
        bottom_sizes = [num_numeric] + bottom_mlp_sizes + [embed_dim]
        self.bottom_mlp = _make_mlp(bottom_sizes)
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, embed_dim) for card in cat_cardinalities
        ])
        bound = 1.0 / (embed_dim ** 0.5)
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -bound, bound)
        top_input_dim = (self.n_cat + 1) * embed_dim
        self.top_mlp = _make_mlp([top_input_dim] + top_mlp_sizes + [1])

    def forward(self, x_num, x_cat):
        dense_out = self.bottom_mlp(x_num)
        emb_outs = [emb(x_cat[:, j]) for j, emb in enumerate(self.embeddings)]
        concat = torch.cat([dense_out] + emb_outs, dim=1)
        return self.top_mlp(concat).squeeze(1)


dlrm_no_interact = DLRMNoInteraction(NUM_NUMERIC, CAT_CARDS, embed_dim=16)
run_experiment("DLRM (no-interaction)", dlrm_no_interact, "dlrm_no_interact")

dlrm_d8 = DLRM(NUM_NUMERIC, CAT_CARDS, embed_dim=8,
                bottom_mlp_sizes=[512, 256], top_mlp_sizes=[512, 256])
run_experiment("DLRM (d=8)", dlrm_d8, "dlrm_d8")

dlrm_d32 = DLRM(NUM_NUMERIC, CAT_CARDS, embed_dim=32,
                 bottom_mlp_sizes=[512, 256], top_mlp_sizes=[512, 256])
run_experiment("DLRM (d=32)", dlrm_d32, "dlrm_d32")

# Best model test evaluation
print("\n[4/6] Evaluating best model on test set (once) ...")
best_name = min(all_results, key=lambda n: all_results[n][1]["best_val_loss"])
best_model, best_info = all_results[best_name]
best_threshold = best_info["best_threshold"]

print("  Best: " + best_name)
print("  val_loss=" + str(best_info["best_val_loss"]) + "  epoch=" + str(best_info["best_epoch"]))
print("  F1-threshold (val): " + str(round(best_threshold, 3)))

test_metrics = eval_on_test(best_model, test_loader, threshold=best_threshold, device=DEVICE)
print("\n  Test metrics:")
for k, v in test_metrics.items():
    print("    " + str(k) + ": " + str(v))
save_results(
    {"model": best_name, "threshold": best_threshold, **test_metrics},
    RESULTS_DIR / "dlrm_best_test.json"
)

# Summary table
print("\n[5/6] Building summary table ...")
criterion = nn.BCEWithLogitsLoss()
rows = []
for name, (model, info) in all_results.items():
    _, y_t, y_p = evaluate(model, val_loader, criterion, DEVICE)
    thresh = info["best_threshold"] or find_best_f1_threshold(y_t, y_p)
    m = compute_metrics(y_t, y_p, threshold=thresh)
    n_epochs = len(info["history"]["epoch_seconds"])
    avg_sec = sum(info["history"]["epoch_seconds"]) / n_epochs if n_epochs else 0
    rows.append({
        "Model": name,
        "ROC-AUC": round(m["roc_auc"], 4),
        "PR-AUC": round(m["pr_auc"], 4),
        "Log Loss": round(m["log_loss"], 4),
        "Accuracy": round(m["accuracy_0.5"], 4),
        "F1": round(m["f1"], 4),
        "Precision": round(m["precision"], 4),
        "Recall": round(m["recall"], 4),
        "ECE": round(m["ece"], 4),
        "Threshold": round(thresh, 3),
        "Params": info["n_params"],
        "Peak Mem(MB)": info["peak_memory_mb"],
        "Sec/Epoch": round(avg_sec, 1),
        "Best Epoch": info["best_epoch"],
    })
df = pd.DataFrame(rows).sort_values("Log Loss")
print(df.to_string(index=False))
df.to_csv(RESULTS_DIR / "dlrm_summary_table.csv", index=False)
print("  Saved -> results/dlrm_summary_table.csv")

# Figures
print("\n[6/6] Generating figures ...")

# Loss curves
fig, axes = plt.subplots(2, 2, figsize=(14, 9))
axes = axes.flatten()
for ax, (name, (model, info)) in zip(axes, all_results.items()):
    hist = info["history"]
    eps = range(1, len(hist["train_loss"]) + 1)
    ax.plot(eps, hist["train_loss"], "o-", label="Train loss", color="steelblue")
    ax.plot(eps, hist["val_loss"], "s--", label="Val loss", color="crimson")
    ax.set_title(name, fontsize=10)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("BCE Loss")
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)
for ax in axes[len(all_results):]:
    ax.set_visible(False)
plt.suptitle("DLRM - Train vs Val Loss Curves", fontsize=13)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "dlrm_loss_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> figures/dlrm_loss_curves.png")

# Embedding dim ablation
dim_tags   = ["DLRM (d=8)", "DLRM (d=16)", "DLRM (d=32)"]
dims       = [8, 16, 32]
dim_aucs   = []
dim_losses = []
for tag in dim_tags:
    if tag in all_results:
        model, _ = all_results[tag]
        _, y_t, y_p = evaluate(model, val_loader, criterion, DEVICE)
        m = compute_metrics(y_t, y_p)
        dim_aucs.append(m["roc_auc"])
        dim_losses.append(m["log_loss"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.bar([str(d) for d in dims], dim_aucs, color=["#4e79a7", "#f28e2b", "#59a14f"])
ax1.set_xlabel("d")
ax1.set_ylabel("ROC-AUC (val)")
ax1.set_title("DLRM: ROC-AUC vs Embedding Dim")
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(dim_aucs):
    ax1.text(i, v + 0.001, str(round(v, 4)), ha="center", fontsize=9)

ax2.bar([str(d) for d in dims], dim_losses, color=["#4e79a7", "#f28e2b", "#59a14f"])
ax2.set_xlabel("d")
ax2.set_ylabel("Log Loss (val)")
ax2.set_title("DLRM: Log Loss vs Embedding Dim")
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(dim_losses):
    ax2.text(i, v + 0.0005, str(round(v, 4)), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "dlrm_embed_dim_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> figures/dlrm_embed_dim_ablation.png")

# Interaction ablation
ab_tags   = ["DLRM (d=16)", "DLRM (no-interaction)"]
ab_labels = ["DLRM\n(dot-product)", "DLRM\n(no-interaction)"]
ab_aucs   = []
ab_losses = []
for tag in ab_tags:
    if tag in all_results:
        model, _ = all_results[tag]
        _, y_t, y_p = evaluate(model, val_loader, criterion, DEVICE)
        m = compute_metrics(y_t, y_p)
        ab_aucs.append(m["roc_auc"])
        ab_losses.append(m["log_loss"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4))
colors = ["#4e79a7", "#e15759"]
ax1.bar(ab_labels, ab_aucs, color=colors)
ax1.set_ylabel("ROC-AUC (val)")
ax1.set_title("Interaction Ablation: ROC-AUC")
ax1.grid(True, axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(ab_aucs):
    ax1.text(i, v + 0.001, str(round(v, 4)), ha="center", fontsize=9)

ax2.bar(ab_labels, ab_losses, color=colors)
ax2.set_ylabel("Log Loss (val)")
ax2.set_title("Interaction Ablation: Log Loss")
ax2.grid(True, axis="y", linestyle="--", alpha=0.4)
for i, v in enumerate(ab_losses):
    ax2.text(i, v + 0.0005, str(round(v, 4)), ha="center", fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / "dlrm_interaction_ablation.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved -> figures/dlrm_interaction_ablation.png")

# Reliability diagram
_, y_t_val, y_p_val = evaluate(best_model, val_loader, criterion, DEVICE)
plot_reliability_diagram(y_t_val, y_p_val, model_name=best_name,
                         save_path=FIGURES_DIR / "dlrm_reliability_diagram.png")
print("  Saved -> figures/dlrm_reliability_diagram.png")

print("\nAll DLRM experiments complete.")
print("Results in 03_DLRM/results/ and 03_DLRM/figures/")
