"""Training loop for Neural CTR models with early stopping and metrics logging."""

import sys
import pathlib
import json
import time
from typing import Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Allow imports from common/ regardless of working directory
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from common.utils import get_device, count_params, get_peak_memory, Timer
from common.metrics import compute_metrics, find_best_f1_threshold


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Runs one training epoch and returns mean train loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluates model on a loader. Returns (mean loss, y_true, y_prob)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_probs = []
    all_labels = []

    for x_num, x_cat, y in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        y = y.to(device)

        logits = model(x_num, x_cat)
        loss = criterion(logits, y)
        total_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.cpu().numpy())

    y_prob = np.concatenate(all_probs)
    y_true = np.concatenate(all_labels)
    return total_loss / n_batches, y_true, y_prob


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 10,
    patience: int = 2,
    device: Optional[torch.device] = None,
    val_f1_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Full training loop with early stopping on val log loss.

    Weight decay is applied to dense (Linear) layers only, not embeddings.
    This prevents the optimizer from shrinking low-frequency embedding rows
    that haven't been updated in a mini-batch toward zero, which would bias them.

    Returns a history dict with per-epoch metrics and the best threshold for F1.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    # Separate dense params from embedding params for selective weight decay
    dense_params, embed_params = [], []
    for name, param in model.named_parameters():
        if "embedding" in name.lower():
            embed_params.append(param)
        else:
            dense_params.append(param)

    optimizer = torch.optim.Adam(
        [
            {"params": dense_params, "weight_decay": weight_decay},
            {"params": embed_params, "weight_decay": 0.0},
        ],
        lr=lr,
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = -1
    best_threshold = val_f1_threshold  # Will be computed from F1 sweep after loop
    _best_y_true: np.ndarray = np.array([])
    _best_y_prob: np.ndarray = np.array([])

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_auc": [],
        "epoch_seconds": [],
    }

    # Reset peak memory counter
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(1, max_epochs + 1):
        t_start = time.perf_counter()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, y_true_val, y_prob_val = evaluate(model, val_loader, criterion, device)

        elapsed = time.perf_counter() - t_start

        # Compute val AUC
        val_metrics = compute_metrics(y_true_val, y_prob_val)
        val_auc = val_metrics["roc_auc"]

        history["train_loss"].append(round(train_loss, 6))
        history["val_loss"].append(round(val_loss, 6))
        history["val_auc"].append(round(val_auc, 6))
        history["epoch_seconds"].append(round(elapsed, 2))

        print(
            f"  Epoch {epoch:2d} | train_loss={train_loss:.4f}  val_loss={val_loss:.4f}"
            f"  val_AUC={val_auc:.4f}  {elapsed:.1f}s"
        )

        # Early stopping check + track val probs at best epoch for threshold sweep
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            _best_y_true = y_true_val
            _best_y_prob = y_prob_val
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (best was epoch {best_epoch}).")
                break

    # Compute F1-optimal threshold on val at best epoch (never touch test here)
    if best_threshold is None and len(_best_y_prob) > 0:
        best_threshold = find_best_f1_threshold(_best_y_true, _best_y_prob)
        print(f"  F1-optimal threshold (val): {best_threshold:.3f}")

    peak_mem_mb = get_peak_memory()
    n_params = count_params(model)

    return {
        "history": history,
        "best_val_loss": round(best_val_loss, 6),
        "best_epoch": best_epoch,
        "n_params": n_params,
        "peak_memory_mb": round(peak_mem_mb, 2),
        "best_threshold": best_threshold,
    }


def eval_on_test(
    model: nn.Module,
    test_loader: DataLoader,
    threshold: float,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """
    Evaluates the model on the test set using the threshold chosen on val.
    Call this ONCE per model, only for the final report.
    """
    if device is None:
        device = get_device()

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()

    test_loss, y_true, y_prob = evaluate(model, test_loader, criterion, device)
    metrics = compute_metrics(y_true, y_prob, threshold=threshold)
    metrics["test_loss"] = round(test_loss, 6)
    return metrics


def save_results(results: Dict[str, Any], path: pathlib.Path) -> None:
    """Saves a results dict as a JSON file, creating parent directories if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved results -> {path}")
