"""Qwen3-Embedding-0.6B feature extractor via sentence-transformers."""

import logging
from typing import List

import numpy as np
import torch

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class Qwen3EmbExtractor(HuggingFaceExtractor):
    """Extracts embeddings from Qwen/Qwen3-Embedding-0.6B via sentence-transformers."""

    cache_name: str = "qwen3_emb"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.batch_size = batch_size

        torch.manual_seed(2)
        from sentence_transformers import SentenceTransformer

        self.st_model = SentenceTransformer(
            self.model_name,
            device=str(self.device),
            model_kwargs={"torch_dtype": torch.bfloat16},
        )

    def _compute_embeddings(self, queries, batch_size: int) -> List[np.ndarray]:
        vecs = self.st_model.encode(
            queries.tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return list(vecs)
