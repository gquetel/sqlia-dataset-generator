import logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.neighbors import LocalOutlierFactor

from sklearn.svm import OneClassSVM
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from constants import MyAutoEncoder, MyAutoEncoderRelu
import torch.nn as nn
import torch.nn.functional as F
import torch

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
        use_scaler: bool = True,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

        self.vectorizer = CountVectorizer(max_features=max_features)
        self.GENERIC = GENERIC
        self.max_iter = max_iter

        self._scaler = StandardScaler(with_mean=False)

        self.clf = None
        self.model_name = None
        self.feature_columns = None
        self.use_scaler = use_scaler

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

        self.feature_columns = self.vectorizer.get_feature_names_out()

        if self.use_scaler:
            f_matrix = self._scaler.fit_transform(f_matrix)
        
        model.fit(f_matrix)
        self.clf = model

    # @profile
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
        # TODO: remove batching here, is done in caller.
        labels = df["label"]
        pp_queries = self.vectorizer.transform(df["full_query"])

        if drop_og_columns:
            return pp_queries, labels

        # if not drop: we need to keep track of initial columns. We artificially create a
        # new dataframe. The index is resetted for cases where passed df does not
        # possess a 0-based index.
        df_copy = df.copy().reset_index(drop=True)
        batch_size = 5000

        batch_dfs = []

        # CountVectorizer is too big to perform a toarray directly, we process
        # samples by batches then:
        for start_idx in range(0, len(df_copy), batch_size):
            # Select end idx
            end_idx = min(start_idx + batch_size, len(df_copy))

            # Fetch corresponding samples, transform to df (transform if needed)
            batch_queries = pp_queries[start_idx:end_idx].toarray()
            batch_queries_df = pd.DataFrame(batch_queries)

            # And then retrieve original columns
            batch_original = df_copy.iloc[start_idx:end_idx].copy()
            batch_combined = pd.concat([batch_original.reset_index(drop=True), batch_queries_df], axis=1)
            batch_dfs.append(batch_combined)

        # Then merge all.
        df_pped = pd.concat(batch_dfs, ignore_index=True)
        return df_pped, labels


class LOF_CV:
    def __init__(
        self,
        GENERIC,
        n_jobs: int = -1,
        vectorizer_max_features: int | None = None,
        use_scaler: bool = False,
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.vectorizer = CountVectorizer(max_features=vectorizer_max_features)
        self.n_jobs = n_jobs

        self._scaler = StandardScaler(with_mean=False)

        self.GENERIC = GENERIC
        self.clf = None
        self.model_name = None
        self.feature_columns = None
        self.use_scaler = use_scaler

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

        self.feature_columns = self.vectorizer.get_feature_names_out()
        if self.use_scaler:
            f_matrix = self._scaler.fit_transform(f_matrix)
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
        # TODO: remove batching here, is done in caller.
        labels = df["label"]
        pp_queries = self.vectorizer.transform(df["full_query"])

        if drop_og_columns:
            return pp_queries, labels

        # if not drop: we need to keep track of initial columns. We artificially create a
        # new dataframe. The index is resetted for cases where passed df does not
        # possess a 0-based index.
        df_copy = df.copy().reset_index(drop=True)
        batch_size = 50000

        batch_dfs = []

        # CountVectorizer is too big to perform a toarray directly, we process
        # samples by batches then:
        for start_idx in range(0, len(df), batch_size):
            # Select end idx
            end_idx = min(start_idx + batch_size, len(df))

            # Fetch corresponding samples, transform to df (transform if needed)
            batch_queries = pp_queries[start_idx:end_idx].toarray()
            batch_queries_df = pd.DataFrame(batch_queries)

            # And then retrieve original columns
            batch_original = df_copy.iloc[start_idx:end_idx].copy()
            batch_combined = pd.concat([batch_original.reset_index(drop=True), batch_queries_df], axis=1)
            batch_dfs.append(batch_combined)

        # Then merge all.
        df_pped = pd.concat(batch_dfs, ignore_index=True)
        return df_pped, labels


class AutoEncoder_CV:
    def __init__(
        self,
        GENERIC,
        device, 
        vectorizer_max_features: int | None = None,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.clf = None
        self.GENERIC = GENERIC
        self.model_name = None


        self.vectorizer = CountVectorizer(max_features=vectorizer_max_features)
        self.use_scaler = use_scaler
        self.device = device

        # Preprocess with counter, MaxAbsScaler should return values between [0, 1]
        self._scaler = MaxAbsScaler()
        self._scaler_min = None
        self._scaler_max = None

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.feature_columns = None

    def preprocess_for_preds(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Return preprocessed queries.

        WARNING: A New DataFrame with both features and original columns is returned if
        drop_og_columns is set to true. This means a new index is generated.
        
        WARNING 2: This function performs scaling, no need to do it again afterwards.
        
        Args:
            df (pd.DataFrame): _description_
            drop_og_columns (bool, optional): _description_. Defaults to True.

        Returns:
            tuple[pd.DataFrame, np.ndarray]: _description_
        """
        # TODO: remove batching here, is done in caller.

        labels = df["label"]
        pp_queries = self.vectorizer.transform(df["full_query"])

        if drop_og_columns:
            return pp_queries, labels

        # if not drop: we need to keep track of initial columns. We artificially create a
        # new dataframe. The index is resetted for cases where passed df does not
        # possess a 0-based index.
        df_copy = df.copy().reset_index(drop=True)
        batch_size = 50000

        batch_dfs = []

        # CountVectorizer is too big to perform a toarray directly, we process
        # samples by batches then:
        for start_idx in range(0, len(df), batch_size):
            # Select end idx
            end_idx = min(start_idx + batch_size, len(df))

            # Fetch corresponding samples, transform to df (transform if needed)
            batch_queries = pp_queries[start_idx:end_idx].toarray()

            if self.use_scaler:
                batch_queries = self._scaler.transform(batch_queries)
                batch_queries = np.clip(
                    batch_queries, self._scaler_min, self._scaler_max
                )

            batch_queries_df = pd.DataFrame(batch_queries)
            # And then retrieve original columns
            batch_original = df_copy.iloc[start_idx:end_idx].copy()
            batch_combined = pd.concat([batch_original.reset_index(drop=True), batch_queries_df], axis=1)
            batch_dfs.append(batch_combined)

        # Then merge all.
        df_pped = pd.concat(batch_dfs, ignore_index=True)
        return df_pped, labels

    def preprocess_for_train(self, df: pd.DataFrame) -> tuple[csr_matrix, np.ndarray]:
        df_pped = df.copy()
        pp_queries = self.vectorizer.fit_transform(df_pped["full_query"])
        self._scaler_min = pp_queries.min(axis=None)
        assert(self._scaler_min >= 0)
        self._scaler_max = pp_queries.max(axis=None)
        return pp_queries

    def _dataframe_to_tensor_batched(self, df, batch_size=4096):
        """
        Used during testing.

        Args:
            df (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 4096.

        Returns:
            _type_: _description_
        """
        n_samples = len(df)
        tensors = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            df_batch = df.iloc[i:batch_end]
            batch_dense = df_batch.values
            tensors.append(torch.FloatTensor(batch_dense))

        return torch.cat(tensors, dim=0).to(self.device)

    def _sparse_to_tensor_batched(self, sparse_matrix, batch_size=4096):
        """
        Used during training.
        Args:
            sparse_matrix (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 4096.

        Returns:
            _type_: _description_
        """
        n_samples = sparse_matrix.shape[0]
        tensors = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_dense = sparse_matrix[i:batch_end].toarray()
            tensors.append(torch.FloatTensor(batch_dense))
        return torch.cat(tensors, dim=0)

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        # Init variables for training + model
        self.model_name = model_name
        f_matrix = self.preprocess_for_train(df)
        self.feature_columns = self.vectorizer.get_feature_names_out()
        input_dim = len(self.feature_columns)

        # If use scaler => Sigmoid in forward (MyAutoEncoder)
        # If not => Relu (MyAutoEncoderRelu)
        if self.use_scaler:
            self.clf = MyAutoEncoder(
                input_dim=input_dim,
            )
        else:
            self.clf = MyAutoEncoderRelu(input_dim=input_dim)
        self.clf = self.clf.to(self.device)

        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)

        if self.use_scaler:
            scaled_data = self._scaler.fit_transform(f_matrix)
            train_data = self._sparse_to_tensor_batched(scaled_data, batch_size=10000)
        else:
            train_data = self._sparse_to_tensor_batched(f_matrix)

        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size]
                batch = batch.to(self.device)
                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(train_data):.6f}"
            )
