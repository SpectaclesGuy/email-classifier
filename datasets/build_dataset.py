from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from app.config import INTERIM_DIR, PROCESSED_DIR, RAW_DIR, RANDOM_STATE, TEST_SIZE
from app.utils import has_reply_prefix, merge_subject_body
from datasets.labeling_rules import derive_label_with_explanations

try:
    from tqdm import tqdm

    tqdm.pandas()
except Exception:  # pragma: no cover
    tqdm = None


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _find_first_csv(folder: Path) -> Optional[Path]:
    if not folder.exists():
        return None
    for path in folder.rglob("*.csv"):
        return path
    return None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    column_map: Dict[str, str] = {}
    for col in df.columns:
        lowered = col.strip().lower()
        if lowered in {"subject", "subj"}:
            column_map[col] = "subject"
        elif lowered in {"body", "message", "email", "text", "content"}:
            column_map[col] = "body"
        elif lowered in {"from", "sender", "from_email"}:
            column_map[col] = "sender"
        elif lowered in {"date", "timestamp", "time"}:
            column_map[col] = "timestamp"
        elif lowered in {"label", "spam", "class", "category"}:
            column_map[col] = "original_label"
    df = df.rename(columns=column_map)
    return df


def _ensure_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    for col in ["subject", "body", "sender", "timestamp", "original_label"]:
        if col not in df.columns:
            df[col] = None
    df["source_dataset"] = source
    return df


def load_enron() -> pd.DataFrame:
    folder = RAW_DIR / "enron"
    csv_path = _find_first_csv(folder)
    if not csv_path:
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)
    df = _ensure_columns(df, "enron")
    return df


def load_bc3() -> pd.DataFrame:
    folder = RAW_DIR / "bc3"
    csv_path = _find_first_csv(folder)
    if not csv_path:
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)
    df = _ensure_columns(df, "bc3")
    return df


def _load_trec_from_labels(folder: Path) -> pd.DataFrame:
    labels_path = folder / "labels"
    data_dir = folder / "data"
    if not labels_path.exists() or not data_dir.exists():
        return pd.DataFrame()

    records: List[Dict[str, Optional[str]]] = []
    lines = labels_path.read_text(encoding="latin-1", errors="ignore").splitlines()
    for line in _progress(lines, desc="TREC labels", unit="msg"):
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        label, rel_path = parts[0], parts[1]
        msg_path = (folder / rel_path).resolve()
        if not msg_path.exists():
            msg_path = data_dir / Path(rel_path).name
        if not msg_path.exists():
            continue
        try:
            content = msg_path.read_text(encoding="latin-1", errors="ignore")
        except OSError:
            continue
        records.append(
            {
                "subject": None,
                "body": content,
                "sender": None,
                "timestamp": None,
                "original_label": label,
                "source_dataset": "trec07",
            }
        )
    return pd.DataFrame(records)


def _load_trec_from_csv(folder: Path) -> pd.DataFrame:
    csv_path = _find_first_csv(folder)
    if not csv_path:
        return pd.DataFrame()
    df = pd.read_csv(csv_path, low_memory=False)
    df = _normalize_columns(df)
    df = _ensure_columns(df, "trec07")
    return df


def load_trec07() -> pd.DataFrame:
    folder = RAW_DIR / "trec07"
    df = _load_trec_from_labels(folder)
    if df.empty:
        df = _load_trec_from_csv(folder)
    return df


def build_unified_dataset() -> pd.DataFrame:
    datasets = [load_enron(), load_bc3(), load_trec07()]
    datasets = [df for df in datasets if not df.empty]
    if not datasets:
        raise FileNotFoundError(
            "No datasets found. Place data under data/raw/enron, data/raw/bc3, and/or data/raw/trec07."
        )

    df = pd.concat(datasets, ignore_index=True)
    df = df.drop_duplicates(subset=["subject", "body"], keep="first")
    df["subject"] = df["subject"].fillna("")
    df["body"] = df["body"].fillna("")
    df["has_reply_prefix"] = df["subject"].apply(has_reply_prefix)
    df["text"] = df.apply(lambda row: merge_subject_body(row["subject"], row["body"]), axis=1)

    if tqdm is not None:
        labels = df.progress_apply(
            lambda row: derive_label_with_explanations(
                row["subject"], row["body"], row.get("original_label"), row.get("has_reply_prefix")
            ),
            axis=1,
        )
    else:
        labels = df.apply(
            lambda row: derive_label_with_explanations(
                row["subject"], row["body"], row.get("original_label"), row.get("has_reply_prefix")
            ),
            axis=1,
        )

    df[["derived_label", "label_explanation", "label_source"]] = pd.DataFrame(labels.tolist(), index=df.index)
    return df


def main() -> None:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = build_unified_dataset()
    df.to_csv(PROCESSED_DIR / "email_dataset.csv", index=False)

    train_df, val_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df["derived_label"]
    )

    train_df.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df.to_csv(PROCESSED_DIR / "val.csv", index=False)

    stats = {
        "total": int(len(df)),
        "train": int(len(train_df)),
        "val": int(len(val_df)),
        "labels": df["derived_label"].value_counts().to_dict(),
        "sources": df["source_dataset"].value_counts().to_dict(),
    }
    (PROCESSED_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("Unified dataset saved to", PROCESSED_DIR / "email_dataset.csv")


if __name__ == "__main__":
    main()

