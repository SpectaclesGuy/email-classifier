from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from app.config import ARTIFACTS_DIR, PROCESSED_DIR, RANDOM_STATE
from app.logging_config import setup_logging
from models.transformer_utils import (
    TransformerTrainingConfig,
    build_label_map,
    compute_accuracy,
    load_transformer_data,
    save_label_map,
)

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

try:  # pragma: no cover
    import matplotlib.pyplot as plt

    _PLOTS_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None
    _PLOTS_AVAILABLE = False


class EmailDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int) -> None:
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in encoded.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def train_epoch(model, loader, optimizer, scheduler, device) -> float:
    model.train()
    losses = []
    for batch in _progress(loader, desc="Train", unit="batch"):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def eval_epoch(model, loader, device) -> Dict[str, float]:
    model.eval()
    all_preds = []
    all_labels = []
    losses = []
    with torch.no_grad():
        for batch in _progress(loader, desc="Eval", unit="batch"):
            labels = batch["labels"].to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits.detach().cpu().numpy()
            preds = logits.argmax(axis=1)
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            losses.append(loss.item())

    acc = compute_accuracy(np.array(all_preds), np.array(all_labels))
    return {"loss": float(np.mean(losses)) if losses else 0.0, "accuracy": acc}


def parse_args() -> TransformerTrainingConfig:
    parser = argparse.ArgumentParser(description="Train transformer model for email classification")
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-samples", type=int, default=100000)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return TransformerTrainingConfig(
        model_name=args.model,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_samples=args.max_samples,
        seed=RANDOM_STATE,
    )


def _save_checkpoint(out_dir: Path, epoch: int, model, optimizer, scheduler, metrics: List[Dict[str, float]]) -> None:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "metrics": metrics,
    }
    torch.save(ckpt, ckpt_dir / f"epoch_{epoch}.pt")
    torch.save(ckpt, ckpt_dir / "latest.pt")


def _load_checkpoint(out_dir: Path):
    ckpt_path = out_dir / "checkpoints" / "latest.pt"
    if not ckpt_path.exists():
        return None
    return torch.load(ckpt_path, map_location="cpu")


def _save_plots(out_dir: Path, metrics: List[Dict[str, float]]) -> None:
    if not _PLOTS_AVAILABLE:
        return
    epochs = [m["epoch"] for m in metrics]
    train_loss = [m["train_loss"] for m in metrics]
    val_loss = [m.get("loss", 0.0) for m in metrics]
    val_acc = [m.get("accuracy", 0.0) for m in metrics]

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, val_acc, label="val_accuracy")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "accuracy_curve.png")
    plt.close()


def main() -> None:
    setup_logging()
    logger = logging.getLogger(__name__)
    cfg = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Transformer device: %s", device)

    train_df, val_df = load_transformer_data(
        str(PROCESSED_DIR / "train.csv"),
        str(PROCESSED_DIR / "val.csv"),
        cfg.max_samples,
        cfg.seed,
    )

    logger.info("Train size: %s", len(train_df))
    logger.info("Val size: %s", len(val_df))

    label_to_id, id_to_label = build_label_map(train_df["derived_label"])
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    train_labels = train_df["derived_label"].map(label_to_id).tolist()
    val_labels = val_df["derived_label"].map(label_to_id).tolist()

    train_ds = EmailDataset(train_df["text"].tolist(), train_labels, tokenizer, cfg.max_length)
    val_ds = EmailDataset(val_df["text"].tolist(), val_labels, tokenizer, cfg.max_length)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, total_steps // 10), num_training_steps=total_steps
    )

    out_dir = ARTIFACTS_DIR / "distilbert"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: List[Dict[str, float]] = []
    start_epoch = 0
    if "--resume" in sys.argv:
        ckpt = _load_checkpoint(out_dir)
        if ckpt:
            model.load_state_dict(ckpt["model_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            scheduler.load_state_dict(ckpt["scheduler_state"])
            metrics = ckpt.get("metrics", [])
            start_epoch = ckpt["epoch"]
            logger.info("Resuming from epoch %s", start_epoch + 1)

    for epoch in range(start_epoch, cfg.epochs):
        logger.info("Epoch %s/%s", epoch + 1, cfg.epochs)
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_metrics = eval_epoch(model, val_loader, device)
        logger.info("Train loss: %.4f", train_loss)
        logger.info("Val metrics: %s", val_metrics)
        metrics.append({"epoch": epoch + 1, "train_loss": train_loss, **val_metrics})

        _save_checkpoint(out_dir, epoch + 1, model, optimizer, scheduler, metrics)
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        _save_plots(out_dir, metrics)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    save_label_map(str(out_dir / "label_map.json"), label_to_id)

    logger.info("Saved transformer model to %s", out_dir)


if __name__ == "__main__":
    main()

