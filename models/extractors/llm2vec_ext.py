"""llm2vec-based feature extractor (Mistral-7B-Instruct-v2)."""

import logging
from typing import List

import numpy as np
import torch

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class LLM2VecExtractor(HuggingFaceExtractor):
    """Extracts 4096-dim embeddings using McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp."""

    cache_name: str = "llm2vec_mistral"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        model_name: str = "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.batch_size = batch_size

        torch.manual_seed(2)
        from llm2vec import LLM2Vec

        self.l2v = LLM2Vec.from_pretrained(
            self.model_name,
            device_map=str(self.device),
            torch_dtype=torch.bfloat16,
        )

    def _compute_embeddings(self, queries, batch_size: int) -> List[np.ndarray]:
        vecs = self.l2v.encode(
            queries.tolist(),
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return list(vecs)
