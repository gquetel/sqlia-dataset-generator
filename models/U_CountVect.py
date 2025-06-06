import logging
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class OCSVM_CV:
    def __init__(
        self,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        max_features: int | None = None,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.vectorizer = CountVectorizer(max_features=max_features)
        self.GENERIC = GENERIC
        self.max_iter = max_iter

        self.clf = None
        self.model_name = None
        self.feature_names = None

    def preprocess_for_train(self, df: pd.DataFrame) -> tuple[csr_matrix, np.ndarray]:
        df_pped = df.copy()
        # Fit Vectorizer and transform queries at the same time.
        pp_queries = self.vectorizer.fit_transform(df_pped["full_query"])
        return pp_queries

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        f_matrix = self.preprocess_for_train(df)
        model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            max_iter=self.max_iter,
        )
        self.feature_names = self.vectorizer.get_feature_names_out()
        model.fit(f_matrix)

        self.clf = model

    def preprocess_for_preds(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return preprocessed queries.

        WARNING: A New DataFrame with both features and original columns is returned if
        drop_og_columns is set to true. This means a new index is generated.

        Args:
            df (pd.DataFrame): _description_
            drop_og_columns (bool, optional): _description_. Defaults to True.

        Returns:
            tuple[pd.DataFrame, np.ndarray]: _description_
        """
        labels = df["label"]
        pp_queries = self.vectorizer.transform(df["full_query"])

        if drop_og_columns:
            return pp_queries, labels

        # Else, we need to keep track of initial columns. We artificially create a
        # new dataframe. The index is resetted for cases where passed df does not
        # possess a 0-based index.

        pp_queries_df = pd.DataFrame(pp_queries.toarray())
        df_copy = df.copy().reset_index(drop=True)
        df_pped = pd.concat([df_copy, pp_queries_df], axis=1)
        return df_pped, labels


class LOF_CV:
    def __init__(
        self, GENERIC, n_jobs: int = -1, vectorizer_max_features: int | None = None
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.vectorizer = CountVectorizer(max_features=vectorizer_max_features)
        self.n_jobs = n_jobs

        self.GENERIC = GENERIC
        self.clf = None
        self.model_name = None
        self.feature_names = None

    def preprocess_for_train(self, df: pd.DataFrame) -> tuple[csr_matrix, np.ndarray]:
        df_pped = df.copy()
        # Fit Vectorizer and transform queries at the same time.
        pp_queries = self.vectorizer.fit_transform(df_pped["full_query"])
        return pp_queries

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        f_matrix = self.preprocess_for_train(df)

        model = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)

        self.feature_names = self.vectorizer.get_feature_names_out()
        model.fit(f_matrix)
        self.clf = model

    def preprocess_for_preds(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return preprocessed queries.

        WARNING: A New DataFrame with both features and original columns is returned if
        drop_og_columns is set to true. This means a new index is generated.

        Args:
            df (pd.DataFrame): _description_
            drop_og_columns (bool, optional): _description_. Defaults to True.

        Returns:
            tuple[pd.DataFrame, np.ndarray]: _description_
        """
        labels = df["label"]
        pp_queries = self.vectorizer.transform(df["full_query"])

        if drop_og_columns:
            return pp_queries, labels

        # Else, we need to keep track of initial columns. We artificially create a
        # new dataframe. The index is resetted for cases where passed df does not
        # possess a 0-based index.
        pp_queries_df = pd.DataFrame(pp_queries.toarray())
        df_copy = df.copy().reset_index(drop=True)
        df_pped = pd.concat([df_copy, pp_queries_df], axis=1)
        return df_pped, labels
