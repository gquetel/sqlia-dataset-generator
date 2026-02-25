"""RoBERTa-base embedding-based feature extractor."""

import logging
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import RobertaTokenizerFast

from base import BaseExtractor
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)


class RobertaExtractor(BaseExtractor):
    """Extracts 768-dim embeddings from the roberta-base model.

    Caches embeddings to disk to avoid recomputation.
    """

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

    def _cache_path(self, df: pd.DataFrame) -> str:
        return os.path.join(
            self.embeddings_path,
            f"embeddings-roberta_base-{hash_df(df)}.pkl",
        )

    def _load_or_compute_embeddings(
        self, df: pd.DataFrame, batch_size: int
    ) -> List[np.ndarray]:
        cache_path = self._cache_path(df)
        cached = load_cache(cache_path)
        if cached is not None:
            logger.info("Loaded cached RoBERTa embeddings from %s", cache_path)
            return cached

        queries = df["full_query"].values
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
                outputs = self.rb_model(**inputs, output_hidden_states=True)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.extend(batch_embeddings)

        save_cache(cache_path, embeddings)
        return embeddings

    def extract_features(self, df: pd.DataFrame) -> list[np.ndarray]:
        return self._load_or_compute_embeddings(df, self.batch_size)

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[list[np.ndarray], np.ndarray]:
        embeddings = self._load_or_compute_embeddings(df, self.batch_size)
        labels = df["label"].to_numpy()
        return embeddings, labels
