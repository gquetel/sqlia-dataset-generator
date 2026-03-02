"""Kakisim enriched-view (E) + Word2Vec + max/mean pooling feature extractor."""

import hashlib
import logging
import os

import numpy as np
import pandas as pd

from base import BaseExtractor
from cache_utils import hash_df, load_cache, save_cache
from extractors.kakisim import _sql_to_views

logger = logging.getLogger(__name__)


class KakisimW2VVectorizer:
    """Word2Vec over Kakisim E-view tokens, pooled to a fixed-size vector."""

    def __init__(
        self,
        vector_size: int = 256,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
        pooling: str = "max",
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.pooling = pooling
        self._w2v = None

    def _to_sequences(self, queries) -> list[list[str]]:
        # Index 2 of _sql_to_views is the E-view (interleaved tags + values)
        return [_sql_to_views(str(q))[2].split() for q in queries]

    def fit_transform(self, queries) -> np.ndarray:
        from gensim.models import Word2Vec

        sequences = self._to_sequences(queries)
        self._w2v = Word2Vec(
            sentences=sequences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        return self._pool(sequences)

    def transform(self, queries) -> np.ndarray:
        return self._pool(self._to_sequences(queries))

    def _pool(self, sequences) -> np.ndarray:
        out = np.zeros((len(sequences), self.vector_size), dtype=np.float32)
        wv = self._w2v.wv
        for i, tokens in enumerate(sequences):
            vecs = [wv[t] for t in tokens if t in wv]
            if vecs:
                if self.pooling == "max":
                    out[i] = np.max(vecs, axis=0)
                else:
                    out[i] = np.mean(vecs, axis=0)
        return out

    @property
    def views_tag(self) -> str:
        return f"kakisim_e_w2v_{self.pooling}"

    def _vocab_tag(self) -> str:
        vocab = sorted(self._w2v.wv.key_to_index.keys())
        return hashlib.md5(str(vocab).encode()).hexdigest()[:8]

    def get_feature_names_out(self) -> np.ndarray:
        return np.array([f"kakisim_w2v_{i}" for i in range(self.vector_size)])


class KakisimW2VExtractor(BaseExtractor):
    """Kakisim E-view Word2Vec pooling extractor with internal caching."""

    def __init__(self, vector_size: int = 256, pooling: str = "max"):
        self.vectorizer = KakisimW2VVectorizer(vector_size=vector_size, pooling=pooling)
        self._fitted = False
        self.cache_dir = None

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            if self.cache_dir:
                key = f"{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
                cached = load_cache(os.path.join(self.cache_dir, key))
                if cached is not None:
                    self.vectorizer = cached["vectorizer"]
                    self._fitted = True
                    return cached["features"]

            result = self.vectorizer.fit_transform(df["full_query"])
            self._fitted = True

            if self.cache_dir:
                save_cache(
                    os.path.join(self.cache_dir, key),
                    {"vectorizer": self.vectorizer, "features": result},
                )
            return result
        return self.vectorizer.transform(df["full_query"])

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        labels = df["label"].to_numpy()
        cache_path = None
        if self.cache_dir:
            vocab_tag = self.vectorizer._vocab_tag()
            cache_path = os.path.join(
                self.cache_dir,
                f"{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached), labels
        pp_queries = self.vectorizer.transform(df["full_query"])
        if cache_path:
            save_cache(cache_path, pp_queries)
        return pd.DataFrame(pp_queries), labels

    def get_feature_names_out(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()
