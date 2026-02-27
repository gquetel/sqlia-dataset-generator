"""Flan-T5 Small encoder-based feature extractor."""

import logging
from typing import List

import numpy as np
import torch
from transformers import T5EncoderModel, T5TokenizerFast

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class FlanT5Extractor(HuggingFaceExtractor):
    """Extracts 512-dim embeddings from google/flan-t5-small via mean pooling."""

    cache_name: str = "flan_t5_small"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        model_name: str = "google/flan-t5-small",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.model_name = model_name
        self.batch_size = batch_size

        torch.manual_seed(2)
        self.tokenizer = T5TokenizerFast.from_pretrained(self.model_name)
        self.t5_model = T5EncoderModel.from_pretrained(self.model_name)
        self.t5_model.to(self.device)
        self.t5_model.eval()

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
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.t5_model(**inputs)
                # Mean pool over token dimension, excluding padding tokens.
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                embeddings.extend(pooled.cpu().numpy())
        return embeddings
