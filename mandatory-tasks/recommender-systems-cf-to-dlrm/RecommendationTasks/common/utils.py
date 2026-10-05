"""Utility functions for seeding, device selection, timing, and memory tracking."""

import os
import random
import time
import pathlib
import psutil
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """Sets random seeds across Python, NumPy, PyTorch, and CUDA for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Returns CUDA device if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_params(model: torch.nn.Module) -> int:
    """Counts total trainable parameters in a PyTorch module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Timer:
    """Simple context manager to measure execution time in seconds."""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


def get_peak_memory() -> float:
    """Returns peak CUDA memory in MB if CUDA is available, else process RSS in MB."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)
