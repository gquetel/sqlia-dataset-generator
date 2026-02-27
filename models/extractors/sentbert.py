"""SentenceBERT (sentence-transformers) feature extractor."""

import logging
from typing import List

import numpy as np
import torch

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class SentBERTExtractor(HuggingFaceExtractor):
    """Extracts 768-dim embeddings from all-mpnet-base-v2 via sentence-transformers."""

    cache_name: str = "sentbert_mpnet"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.batch_size = batch_size

        torch.manual_seed(2)
        from sentence_transformers import SentenceTransformer

        self.st_model = SentenceTransformer(self.model_name, device=str(self.device))

    def _compute_embeddings(self, queries, batch_size: int) -> List[np.ndarray]:
        vecs = self.st_model.encode(
            queries.tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return list(vecs)
