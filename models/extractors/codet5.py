"""CodeT5+ embedding feature extractor."""

import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class CodeT5Extractor(HuggingFaceExtractor):
    """Extracts 256-dim embeddings from Salesforce/codet5p-110m-embedding."""

    cache_name: str = "codet5p_emb"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        model_name: str = "Salesforce/codet5p-110m-embedding",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.batch_size = batch_size

        torch.manual_seed(2)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.model.to(self.device)
        self.model.eval()

    def _compute_embeddings(self, queries, batch_size: int) -> List[np.ndarray]:
        embeddings = []
        with torch.no_grad():
            for i in range(0, len(queries), batch_size):
                batch = queries[i : i + batch_size].tolist()
                inputs = self.tokenizer(
                    batch,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512,
                )
                input_ids = inputs["input_ids"].to(self.device)
                # Model returns (batch_size, 256) L2-normalized embeddings directly.
                batch_embeddings = self.model(input_ids)
                embeddings.extend(batch_embeddings.cpu().numpy())
        return embeddings
