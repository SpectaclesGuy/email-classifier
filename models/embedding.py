from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from app.config import EMBED_BATCH_SIZE, EMBED_MODEL_NAME

try:  # pragma: no cover
    import torch
    from sentence_transformers import SentenceTransformer

    _EMBED_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None
    SentenceTransformer = None
    _EMBED_AVAILABLE = False


@dataclass
class EmbeddingConfig:
    model_name: str = EMBED_MODEL_NAME
    batch_size: int = EMBED_BATCH_SIZE
    device: Optional[str] = None
    normalize: bool = True
    show_progress_bar: bool = True


class EmbeddingTransformer(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        model_name: str = EMBED_MODEL_NAME,
        batch_size: int = EMBED_BATCH_SIZE,
        device: Optional[str] = None,
        normalize: bool = True,
        show_progress_bar: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.show_progress_bar = show_progress_bar
        self._model = None

    def _resolve_device(self) -> str:
        if self.device:
            return self.device
        if torch is not None and torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _get_model(self):
        if self._model is None:
            if not _EMBED_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is not installed. Install it to use embedding models."
                )
            device = self._resolve_device()
            self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        model = self._get_model()
        texts = ["" if x is None else str(x) for x in X]
        embeddings = model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
        )
        return np.asarray(embeddings, dtype=float)


def is_embedding_available() -> bool:
    return _EMBED_AVAILABLE

