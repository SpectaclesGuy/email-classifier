from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))


import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from app.config import ARTIFACTS_DIR, MODEL_PATH, MODEL_REGISTRY_PATH, PROCESSED_DIR


def _load_registry() -> Dict[str, Dict[str, object]]:
    if MODEL_REGISTRY_PATH.exists():
        return json.loads(MODEL_REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"models": {"default": {"path": str(MODEL_PATH), "metrics": {}}}, "best_model": "default"}


def _evaluate_model(model_path: Path, X_val: pd.DataFrame, y_val: pd.Series, name: str) -> Dict[str, float]:
    model = joblib.load(model_path)
    preds = model.predict(X_val)

    acc = accuracy_score(y_val, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="macro", zero_division=0
    )

    report = classification_report(y_val, preds, zero_division=0)
    cm = confusion_matrix(y_val, preds, labels=sorted(y_val.unique().tolist()))

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / f"metrics_{name}.json").write_text(
        json.dumps(
            {
                "accuracy": acc,
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (ARTIFACTS_DIR / f"classification_report_{name}.txt").write_text(report, encoding="utf-8")

    cm_df = pd.DataFrame(cm, index=sorted(y_val.unique().tolist()), columns=sorted(y_val.unique().tolist()))
    cm_df.to_csv(ARTIFACTS_DIR / f"confusion_matrix_{name}.csv")

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
    }


def main() -> None:
    val_path = PROCESSED_DIR / "val.csv"
    if not val_path.exists():
        raise FileNotFoundError("Validation data not found. Run datasets/build_dataset.py first.")

    df = pd.read_csv(val_path)
    X_val = df[["text", "subject", "body", "sender", "has_reply_prefix"]]
    y_val = df["derived_label"]

    registry = _load_registry()
    models = registry.get("models", {})

    summary_rows: List[Dict[str, object]] = []
    for name, info in models.items():
        model_path = Path(info["path"])
        if not model_path.exists():
            continue
        metrics = _evaluate_model(model_path, X_val, y_val, name)
        summary_rows.append({"model": name, **metrics})

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(by="macro_f1", ascending=False)
        summary_df.to_csv(ARTIFACTS_DIR / "model_comparison.csv", index=False)
        print(summary_df)
    else:
        raise FileNotFoundError("No model artifacts found. Run models/train.py first.")


if __name__ == "__main__":
    main()


