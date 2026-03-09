"""ModernBERT-base embedding-based feature extractor."""

import logging
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class ModernBERTExtractor(HuggingFaceExtractor):
    """Extracts 768-dim embeddings from the answerdotai/ModernBERT-base model.

    Uses the CLS token (first token of last_hidden_state) as the sentence embedding.
    """

    cache_name: str = "modernbert"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        bert_model: str = "answerdotai/ModernBERT-base",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.bert_model = bert_model
        self.batch_size = batch_size

        torch.manual_seed(2)
        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model)
        self.rb_model = AutoModel.from_pretrained(self.bert_model)
        self.rb_model.to(self.device)
        self.rb_model.eval()

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
                    max_length=1024,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.rb_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.extend(cls_embeddings.cpu().numpy())
        return embeddings
