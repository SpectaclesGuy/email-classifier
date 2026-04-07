from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from app.config import (
    ARTIFACTS_DIR,
    TRANSFORMER_BASE_MODEL,
    TRANSFORMER_CHECKPOINT_PATH,
    TRANSFORMER_MAX_LENGTH,
    TRANSFORMER_MODEL_DIR,
)
from app.utils import merge_subject_body
from models.scoring import score_email

try:  # pragma: no cover
    from huggingface_hub.utils import enable_progress_bars

    enable_progress_bars()
except Exception:
    pass


class TransformerPredictor:
    def __init__(self, model_dir: Path, checkpoint_path: Path | None = None) -> None:
        self.model_dir = model_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_source = self._resolve_model_source(model_dir)
        label_map = self._load_label_map(model_dir)
        ckpt_num_labels = self._infer_num_labels(checkpoint_path)

        if label_map:
            id2label = {int(v): k for k, v in label_map.items()}
            label2id = {k: int(v) for k, v in label_map.items()}
            num_labels = len(label2id)
            config = AutoConfig.from_pretrained(
                model_source, id2label=id2label, label2id=label2id, num_labels=num_labels
            )
        elif ckpt_num_labels:
            config = AutoConfig.from_pretrained(model_source, num_labels=ckpt_num_labels)
        else:
            config = AutoConfig.from_pretrained(model_source)

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_source, config=config)
        self.model.eval()
        self.model.to(self.device)

        if checkpoint_path and checkpoint_path.exists():
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
            if state:
                self.model.load_state_dict(state)
                self.model.to(self.device)

        self.id_to_label = self.model.config.id2label

    @staticmethod
    def _resolve_model_source(model_dir: Path) -> str:
        if model_dir.exists() and (model_dir / "config.json").exists():
            return str(model_dir)
        return TRANSFORMER_BASE_MODEL

    @staticmethod
    def _load_label_map(model_dir: Path) -> Dict[str, int]:
        path = model_dir / "label_map.json"
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _infer_num_labels(checkpoint_path: Path | None) -> int | None:
        if not checkpoint_path or not checkpoint_path.exists():
            return None
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state = ckpt.get("model_state") if isinstance(ckpt, dict) else None
        if not state:
            return None
        for key in ("classifier.weight", "pre_classifier.weight"):
            if key in state:
                return int(state[key].shape[0])
        return None

    def predict(self, subject: str, body: str, sender: str | None = None) -> Dict[str, Any]:
        text = merge_subject_body(subject, body)
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=TRANSFORMER_MAX_LENGTH,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            outputs = self.model(**enc)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            conf, pred = torch.max(probs, dim=-1)

        pred_id = int(pred.item())
        pred_label = self.id_to_label.get(pred_id, str(pred_id))
        confidence = float(conf.item())

        score_data = score_email(
            subject=subject,
            body=body,
            sender=sender,
            has_reply_prefix=None,
            predicted_label=pred_label,
            confidence=confidence,
        )

        return {
            "predicted_category": pred_label,
            "confidence_score": round(confidence, 4),
            "priority_score": score_data["priority_score"],
            "priority_band": score_data["priority_band"],
            "explanation": score_data["reasons"],
            "extracted_signals": score_data["signals"],
        }


def load_default_predictor() -> TransformerPredictor:
    model_dir = TRANSFORMER_MODEL_DIR
    ckpt_path = TRANSFORMER_CHECKPOINT_PATH if TRANSFORMER_CHECKPOINT_PATH.exists() else None
    return TransformerPredictor(model_dir, checkpoint_path=ckpt_path)


def main() -> None:
    predictor = load_default_predictor()
    sample = {
        "subject": "Urgent: client issue needs resolution today",
        "body": "Please resolve this before EOD. The client is waiting.",
        "sender": "manager@company.com",
    }
    result = predictor.predict(**sample)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

