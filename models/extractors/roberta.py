"""RoBERTa-base embedding-based feature extractor."""

import logging
from typing import List

import numpy as np
import torch
import transformers
from transformers import RobertaTokenizerFast

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class RobertaExtractor(HuggingFaceExtractor):
    """Extracts 768-dim embeddings from the roberta-base model."""

    cache_name: str = "roberta_base"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        bert_model: str = "roberta-base",
        batch_size: int = 16,
    ):
        self.device = device
        self.embeddings_path = embeddings_path
        self.bert_model = bert_model
        self.batch_size = batch_size

        torch.manual_seed(2)
        self.tokenizer = RobertaTokenizerFast.from_pretrained(self.bert_model)
        self.rb_model = transformers.RobertaModel.from_pretrained(self.bert_model)
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
                    max_length=512,
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.rb_model(**inputs)
                embeddings.extend(outputs.pooler_output.cpu().numpy())
        return embeddings
