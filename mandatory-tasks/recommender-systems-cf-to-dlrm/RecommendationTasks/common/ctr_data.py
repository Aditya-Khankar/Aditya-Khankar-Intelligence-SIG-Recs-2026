"""Shared CTR data pipeline: stratified split, fit-on-train preprocessing, and PyTorch DataLoaders."""

from typing import Tuple, List, Dict, Optional
import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class CTRDataset(Dataset):
    """PyTorch Dataset yielding numeric features, categorical feature IDs, and binary labels."""

    def __init__(self, x_num: np.ndarray, x_cat: np.ndarray, y: np.ndarray):
        self.x_num = torch.tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.tensor(x_cat, dtype=torch.int64)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


class CTRPreprocessor:
    """Preprocesses numeric and categorical features strictly fitted on the training split."""

    def __init__(self, min_freq: int = 10):
        self.min_freq = min_freq
        self.num_medians: Optional[np.ndarray] = None
        self.num_means: Optional[np.ndarray] = None
        self.num_stds: Optional[np.ndarray] = None
        self.cat_vocabs: List[Dict[str, int]] = []
        self.cat_cardinalities: List[int] = []

    def fit(self, df_num_train: pd.DataFrame, df_cat_train: pd.DataFrame) -> None:
        """Fits numeric medians/means/stds and categorical vocabularies on train split only."""
        # 1. Fit numerics
        num_arr = df_num_train.to_numpy(dtype=np.float64)
        self.num_medians = np.nanmedian(num_arr, axis=0)

        # Impute median, clip at 0, apply log1p
        num_imputed = np.where(np.isnan(num_arr), self.num_medians, num_arr)
        num_clipped = np.maximum(0.0, num_imputed)
        num_log = np.log1p(num_clipped)

        self.num_means = np.mean(num_log, axis=0)
        self.num_stds = np.std(num_log, axis=0)
        # Avoid division by zero
        self.num_stds = np.where(self.num_stds == 0, 1.0, self.num_stds)

        # 2. Fit categoricals
        self.cat_vocabs = []
        self.cat_cardinalities = []

        for col in df_cat_train.columns:
            s = df_cat_train[col].fillna("__missing__").astype(str)
            val_counts = s.value_counts()

            vocab = {}
            next_id = 1
            for val, count in val_counts.items():
                if count >= self.min_freq:
                    vocab[val] = next_id
                    next_id += 1

            self.cat_vocabs.append(vocab)
            # 0 is reserved for OOV/rare categories, so cardinality = next_id
            self.cat_cardinalities.append(next_id)

    def transform_numeric(self, df_num: pd.DataFrame) -> np.ndarray:
        """Transforms numeric features using train-fitted parameters."""
        num_arr = df_num.to_numpy(dtype=np.float64)
        num_imputed = np.where(np.isnan(num_arr), self.num_medians, num_arr)
        num_clipped = np.maximum(0.0, num_imputed)
        num_log = np.log1p(num_clipped)
        return (num_log - self.num_means) / self.num_stds

    def transform_categorical(self, df_cat: pd.DataFrame) -> np.ndarray:
        """Transforms categorical features to category IDs using train-fitted vocabs."""
        n_rows, n_cols = df_cat.shape
        cat_ids = np.zeros((n_rows, n_cols), dtype=np.int64)

        for j, col in enumerate(df_cat.columns):
            vocab = self.cat_vocabs[j]
            col_vals = df_cat[col].fillna("__missing__").astype(str).to_numpy()
            mapped = [vocab.get(val, 0) for val in col_vals]
            cat_ids[:, j] = mapped

        return cat_ids


def load_ctr_data(
    data_dir: pathlib.Path,
    split_save_path: pathlib.Path,
    batch_size: int = 2048,
    min_freq: int = 10,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, List[int]]:
    """Loads, splits, preprocesses CTR data and returns PyTorch DataLoaders and dataset metadata."""
    train_csv = data_dir / "train.csv"
    test_csv = data_dir / "test.csv"

    df_train_full = pd.read_csv(train_csv)
    df_test_full = pd.read_csv(test_csv)

    num_cols = [f"integer_feature_{i}" for i in range(1, 14)]
    cat_cols = [f"categorical_feature_{i}" for i in range(1, 27)]
    label_col = "label"

    # Get or create train/val split
    if split_save_path.exists():
        split_data = np.load(split_save_path)
        train_idx = split_data["train_idx"]
        val_idx = split_data["val_idx"]
    else:
        labels = df_train_full[label_col].to_numpy()
        indices = np.arange(len(df_train_full))
        train_idx, val_idx = train_test_split(
            indices, test_size=0.10, stratify=labels, random_state=seed
        )
        split_save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(split_save_path, train_idx=train_idx, val_idx=val_idx)

    # Slice train / val splits
    df_train_split = df_train_full.iloc[train_idx].reset_index(drop=True)
    df_val_split = df_train_full.iloc[val_idx].reset_index(drop=True)

    # Fit preprocessor on train split ONLY
    preprocessor = CTRPreprocessor(min_freq=min_freq)
    preprocessor.fit(df_train_split[num_cols], df_train_split[cat_cols])

    # Transform numeric and categorical features
    x_num_train = preprocessor.transform_numeric(df_train_split[num_cols])
    x_cat_train = preprocessor.transform_categorical(df_train_split[cat_cols])
    y_train = df_train_split[label_col].to_numpy()

    x_num_val = preprocessor.transform_numeric(df_val_split[num_cols])
    x_cat_val = preprocessor.transform_categorical(df_val_split[cat_cols])
    y_val = df_val_split[label_col].to_numpy()

    x_num_test = preprocessor.transform_numeric(df_test_full[num_cols])
    x_cat_test = preprocessor.transform_categorical(df_test_full[cat_cols])
    y_test = df_test_full[label_col].to_numpy()

    # Create PyTorch datasets and loaders
    train_dataset = CTRDataset(x_num_train, x_cat_train, y_train)
    val_dataset = CTRDataset(x_num_val, x_cat_val, y_val)
    test_dataset = CTRDataset(x_num_test, x_cat_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        len(num_cols),
        preprocessor.cat_cardinalities,
    )
