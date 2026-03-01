"""Shared base class for HuggingFace-based feature extractors."""

import logging
import os
from abc import abstractmethod
from typing import List

import numpy as np
import pandas as pd

from base import BaseExtractor
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)


class HuggingFaceExtractor(BaseExtractor):
    """Abstract base for HuggingFace model extractors with disk caching.

    Subclasses must implement _compute_embeddings() and set self.cache_name.
    """

    cache_name: str = ""  # Override in subclasses

    def _cache_path(self, df: pd.DataFrame) -> str:
        prefix = f"embeddings-{self.cache_name}-" if self.cache_name else "embeddings-"
        return os.path.join(self.embeddings_path, f"{prefix}{hash_df(df)}.pkl")

    @abstractmethod
    def _compute_embeddings(self, queries, batch_size: int) -> List[np.ndarray]:
        """Compute embeddings for a list of query strings."""

    def _load_or_compute_embeddings(
        self, df: pd.DataFrame, batch_size: int
    ) -> List[np.ndarray]:
        if self.embeddings_path is None:
            queries = df["full_query"].values
            return self._compute_embeddings(queries, batch_size)

        cache_path = self._cache_path(df)
        cached = load_cache(cache_path)
        if cached is not None:
            logger.info(
                "Loaded cached %s embeddings from %s", self.cache_name, cache_path
            )
            return cached

        queries = df["full_query"].values
        embeddings = self._compute_embeddings(queries, batch_size)
        save_cache(cache_path, embeddings)
        return embeddings

    def extract_features(self, df: pd.DataFrame) -> List[np.ndarray]:
        return self._load_or_compute_embeddings(df, self.batch_size)

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[List[np.ndarray], np.ndarray]:
        embeddings = self._load_or_compute_embeddings(df, self.batch_size)
        labels = df["label"].to_numpy()
        return embeddings, labels
