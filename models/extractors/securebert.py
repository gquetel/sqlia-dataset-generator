"""SecureBERT embedding-based feature extractors."""

import logging
from typing import List

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer, RobertaTokenizerFast

from extractors.hf_base import HuggingFaceExtractor

logger = logging.getLogger(__name__)


class SecureBERTExtractor(HuggingFaceExtractor):
    """Extracts 768-dim embeddings from the ehsanaghaei/SecureBERT model."""

    # Backward-compatible: no prefix so existing cache files still match.
    cache_name: str = ""

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        bert_model: str = "ehsanaghaei/SecureBERT",
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


class SecureBERT2Extractor(HuggingFaceExtractor):
    """Extracts 768-dim embeddings from the cisco-ai/SecureBERT2.0-base model.

    SecureBERT 2.0 is based on ModernBERT, which lacks a pooler head.
    We use the CLS token (first token of last_hidden_state) as the sentence embedding.
    """

    cache_name: str = "securebert2"

    def __init__(
        self,
        device: torch.device,
        embeddings_path: str,
        bert_model: str = "cisco-ai/SecureBERT2.0-base",
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
