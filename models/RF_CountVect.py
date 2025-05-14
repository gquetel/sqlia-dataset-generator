import logging
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


logger = logging.getLogger(__name__)


class CustomRF_CountVectorizer:
    def __init__(
        self,
        GENERIC,
        max_depth: int | None = None,
        max_features: int | None = None,
    ):
        self.GENERIC = GENERIC
        self.max_depth = max_depth
        self.vectorizer = CountVectorizer(max_features=max_features)

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        f_matrix, labels = self.preprocess_for_preds(df)
        preds = self.clf.predict(f_matrix)
        return labels, preds

    def preprocess_for_train(
        self, df: pd.DataFrame
    ) -> tuple[csr_matrix, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])

        # And fit transform CountVectorizer.
        pp_queries = self.vectorizer.fit_transform(df_pped["full_query"])
        return pp_queries, labels

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[csr_matrix, np.ndarray]:
        
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        pp_queries = self.vectorizer.transform(df_pped["full_query"])
        
        return pp_queries, labels

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        f_matrix, train_labels = self.preprocess_for_train(df)

        rf = RandomForestClassifier(
            random_state=self.GENERIC.RANDOM_SEED, max_depth=self.max_depth, n_jobs=-1
        )
        rf.fit(f_matrix, train_labels)
        self.clf = rf
    