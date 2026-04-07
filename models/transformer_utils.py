from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class TransformerTrainingConfig:
    model_name: str
    max_length: int
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    max_samples: int | None
    seed: int


def load_transformer_data(train_path: str, val_path: str, max_samples: int | None, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path, low_memory=False)
    val_df = pd.read_csv(val_path, low_memory=False)

    for df in (train_df, val_df):
        df["text"] = df["text"].fillna("")
        df["subject"] = df.get("subject", "").fillna("")
        df["body"] = df.get("body", "").fillna("")
        if "text" not in df or df["text"].isna().all():
            df["text"] = (df["subject"].astype(str) + "\n" + df["body"].astype(str)).str.strip()

    if max_samples is not None and max_samples > 0:
        if len(train_df) > max_samples:
            train_df = train_df.sample(n=max_samples, random_state=seed).reset_index(drop=True)

    return train_df, val_df


def build_label_map(labels: pd.Series) -> Tuple[Dict[str, int], Dict[int, str]]:
    unique = sorted(labels.dropna().unique().tolist())
    label_to_id = {label: idx for idx, label in enumerate(unique)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def save_label_map(path: str, label_to_id: Dict[str, int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(label_to_id, f, indent=2)


def compute_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())

