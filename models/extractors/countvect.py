"""CountVectorizer-based feature extractor."""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer

from base import BaseExtractor


class CountVectExtractor(BaseExtractor):
    """Wraps sklearn CountVectorizer. Must be fit before transform."""

    def __init__(self, max_features: int | None = None):
        self.vectorizer = CountVectorizer(max_features=max_features)
        self._fitted = False

    def extract_features(self, df: pd.DataFrame) -> csr_matrix:
        """Fit-transform (first call) or transform (subsequent calls)."""
        if not self._fitted:
            result = self.vectorizer.fit_transform(df["full_query"])
            self._fitted = True
            return result
        return self.vectorizer.transform(df["full_query"])

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        labels = df["label"].to_numpy()
        pp_queries = self.vectorizer.transform(df["full_query"])
        return pd.DataFrame(pp_queries.toarray()), labels

    def get_feature_names_out(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()
