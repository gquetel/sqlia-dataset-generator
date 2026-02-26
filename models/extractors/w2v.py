"""Plain whitespace Word2Vec + mean pooling feature extractor."""

import hashlib
import logging
import os

import numpy as np
import pandas as pd

from base import BaseExtractor
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)


class Word2VecMeanPoolVectorizer:
    """Word2Vec over whitespace-tokenized SQL queries, mean-pooled to fixed size."""

    def __init__(
        self,
        vector_size: int = 256,
        window: int = 5,
        min_count: int = 2,
        workers: int = 4,
    ):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self._w2v = None

    def _to_sequences(self, queries) -> list[list[str]]:
        return [str(q).split() for q in queries]

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
        return self._max_pool(sequences)

    def transform(self, queries) -> np.ndarray:
        return self._max_pool(self._to_sequences(queries))

    # def _mean_pool(self, sequences) -> np.ndarray:
    #     out = np.zeros((len(sequences), self.vector_size), dtype=np.float32)
    #     wv = self._w2v.wv
    #     for i, tokens in enumerate(sequences):
    #         vecs = [wv[t] for t in tokens if t in wv]
    #         if vecs:
    #             out[i] = np.mean(vecs, axis=0)
    #     return out

    def transform(self, queries) -> np.ndarray:
        return self._max_pool(self._to_sequences(queries))

    def _max_pool(self, sequences) -> np.ndarray:
        out = np.zeros((len(sequences), self.vector_size), dtype=np.float32)
        wv = self._w2v.wv
        for i, tokens in enumerate(sequences):
            vecs = [wv[t] for t in tokens if t in wv]
            if vecs:
                out[i] = np.max(vecs, axis=0)
        return out

    @property
    def views_tag(self) -> str:
        return "whitespace_w2v_mean"

    def _vocab_tag(self) -> str:
        vocab = sorted(self._w2v.wv.key_to_index.keys())
        return hashlib.md5(str(vocab).encode()).hexdigest()[:8]

    def get_feature_names_out(self) -> np.ndarray:
        return np.array([f"w2v_{i}" for i in range(self.vector_size)])


class W2VMeanPoolExtractor(BaseExtractor):
    """Word2Vec mean-pool extractor with internal caching."""

    def __init__(self, vector_size: int = 256):
        self.vectorizer = Word2VecMeanPoolVectorizer(vector_size=vector_size)
        self._fitted = False
        self.cache_dir = None

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            if self.cache_dir:
                key = f"w2v_mean-{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
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
                f"w2v_mean-{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
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
