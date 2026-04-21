"""Base classes for feature extractors and anomaly detection model wrappers."""

import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import OneClassSVM

from autoencoder import AutoEncoder
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    """Feature extraction interface.

    Subclasses implement extract_features() which turns a raw DataFrame
    (with at least a 'full_query' column) into a feature matrix.
    """

    @abstractmethod
    def extract_features(self, df: pd.DataFrame) -> np.ndarray | pd.DataFrame:
        """Turn raw df into feature matrix (no labels)."""

    def prepare_for_training(self, df: pd.DataFrame):
        """Hook called before extract_features during training.

        Override in subclasses that need to learn state from training data
        (e.g. LoginovExtractor learning valid_schars).
        """
        pass

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
        labels = np.array(df["label"])
        X = self.extract_features(df)
        return X, labels


class BaseOCSVM:
    """One-Class SVM wrapping any extractor."""

    def __init__(
        self,
        extractor: BaseExtractor,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = False,
    ):
        self.extractor = extractor
        self.GENERIC = GENERIC
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.use_scaler = use_scaler
        self._scaler = None  # Created lazily based on data type
        self.clf = None
        self.model_name = None
        self.feature_columns = None
        self.threshold = None

    def _ensure_scaler(self, X):
        """Create scaler on first use, with_mean=False for sparse data."""
        if self._scaler is None:
            if sparse.issparse(X):
                self._scaler = StandardScaler(with_mean=False)
            else:
                self._scaler = StandardScaler()

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
        return self.extractor.preprocess_for_preds(df)

    def train_model(self, df: pd.DataFrame, project_paths=None, model_name: str = None):
        self.model_name = model_name
        self.extractor.prepare_for_training(df)
        X = self.extractor.extract_features(df)

        if hasattr(X, "columns"):
            self.feature_columns = X.columns.tolist()
        elif hasattr(self.extractor, "get_feature_names_out"):
            self.feature_columns = self.extractor.get_feature_names_out()

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.use_scaler:
            self._ensure_scaler(X)
            X = self._scaler.fit_transform(X)

        self.clf = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            max_iter=self.max_iter,
        )
        self.clf.fit(X)

    def save_model(self, save_path: str, threshold: float = None):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model_path = save_path.with_suffix(".pth")
        meta_path = save_path.parent / f"{save_path.stem}_meta.pkl"

        joblib.dump(self.clf, model_path)

        metadata = {
            "use_scaler": self.use_scaler,
            "scaler": self._scaler if self.use_scaler else None,
            "feature_columns": self.feature_columns,
            "threshold": threshold,
        }
        if hasattr(self.extractor, "valid_schars"):
            metadata["valid_schars"] = self.extractor.valid_schars
        if hasattr(self.extractor, "vectorizer"):
            metadata["vectorizer"] = self.extractor.vectorizer

        pd.to_pickle(metadata, meta_path, compression="zstd")
        logger.info(f"Saved OCSVM model to {model_path}")

    def load_model(self, load_path: str):
        load_path = Path(load_path)
        if load_path.suffix != ".pth":
            load_path = load_path.with_suffix(".pth")
        meta_path = load_path.parent / f"{load_path.stem}_meta.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"Model weights not found: {load_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        self.clf = joblib.load(load_path)

        try:
            metadata = pd.read_pickle(meta_path, compression="zstd")
        except Exception:
            metadata = pd.read_pickle(meta_path, compression=None)

        self.use_scaler = metadata.get("use_scaler", self.use_scaler)
        self.feature_columns = metadata.get("feature_columns", None)
        self.threshold = metadata.get("threshold", None)
        if self.use_scaler and metadata.get("scaler") is not None:
            self._scaler = metadata["scaler"]
        if "valid_schars" in metadata:
            self.extractor.valid_schars = metadata["valid_schars"]
        if "vectorizer" in metadata:
            self.extractor.vectorizer = metadata["vectorizer"]


class BaseLOF:
    """Local Outlier Factor wrapping any extractor."""

    def __init__(
        self,
        extractor: BaseExtractor,
        GENERIC,
        n_jobs: int = -1,
        use_scaler: bool = False,
    ):
        self.extractor = extractor
        self.GENERIC = GENERIC
        self.n_jobs = n_jobs
        self.use_scaler = use_scaler
        self._scaler = None
        self.clf = None
        self.model_name = None

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
        return self.extractor.preprocess_for_preds(df)

    def train_model(self, df: pd.DataFrame, project_paths=None, model_name: str = None):
        self.model_name = model_name
        self.extractor.prepare_for_training(df)
        X = self.extractor.extract_features(df)

        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.use_scaler:
            if sparse.issparse(X):
                self._scaler = StandardScaler(with_mean=False)
            else:
                self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        self.clf = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)
        self.clf.fit(X)


class BaseAutoEncoderModel:
    """AutoEncoder model wrapping any extractor.

    Handles training loop, scaling, save/load. Subclasses can override
    ``_prepare_train_data`` for special sparse→dense conversion.
    """

    def __init__(
        self,
        extractor: BaseExtractor,
        GENERIC,
        device: torch.device,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
        output_activation: str = "sigmoid",
    ):
        self.extractor = extractor
        self.GENERIC = GENERIC
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_scaler = use_scaler
        self.output_activation = output_activation

        self._scaler = MaxAbsScaler()
        self.clf = None
        self.model_name = None
        self.feature_columns = None
        self.threshold = None

    def preprocess_for_preds(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame | np.ndarray, np.ndarray]:
        return self.extractor.preprocess_for_preds(df)

    def X_to_tensor(self, X) -> torch.Tensor:
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)
        if self.use_scaler:
            X = self._scaler.transform(X)
        return torch.FloatTensor(X).to(self.device)

    def _to_dense_tensor(self, data) -> torch.Tensor:
        """Convert feature data (dense or sparse) into a float tensor on self.device."""
        if sparse.issparse(data):
            # Batch-convert sparse -> dense to avoid OOM
            n = data.shape[0]
            parts = []
            for i in range(0, n, 10000):
                parts.append(torch.FloatTensor(data[i : min(i + 10000, n)].toarray()))
            return torch.cat(parts, dim=0).to(self.device)
        if isinstance(data, pd.DataFrame):
            data = data.values
        if isinstance(data, list):
            data = np.array(data)
        return torch.FloatTensor(data).to(self.device)

    def train_model(self, df: pd.DataFrame, project_paths=None, model_name: str = None):
        self.model_name = model_name
        self.extractor.prepare_for_training(df)
        X = self.extractor.extract_features(df)

        if hasattr(X, "columns"):
            self.feature_columns = X.columns.tolist()
        elif hasattr(self.extractor, "get_feature_names_out"):
            self.feature_columns = self.extractor.get_feature_names_out()

        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(X, list):
            X = np.array(X)

        # Determine input dimension
        input_dim = X.shape[1]

        # Apply scaler if needed
        if self.use_scaler:
            X = self._scaler.fit_transform(X)

        train_data = self._to_dense_tensor(X)

        self.clf = AutoEncoder(
            input_dim=input_dim,
            output_activation=self.output_activation,
        )
        self.clf.to(self.device)

        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)

        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size].to(self.device)
                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss / len(train_data):.6f}"
            )

    def save_model(self, save_path: str, threshold: float = None):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_path = save_path.with_suffix(".pth")
        meta_path = save_path.parent / f"{save_path.stem}_meta.pkl"

        torch.save(self.clf.state_dict(), model_path)

        metadata = self._build_metadata(threshold)
        pd.to_pickle(metadata, meta_path, compression="zstd")

        logger.info(f"Saved model to {model_path}")
        if threshold is not None:
            logger.info(f"Saved threshold: {threshold}")

    def _build_metadata(self, threshold: float = None) -> dict:
        """Assemble the metadata dict for save_model. Override to add extra keys."""
        meta = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "input_dim": self.clf.input_dim,
            "use_scaler": self.use_scaler,
            "output_activation": self.output_activation,
            "scaler": self._scaler if self.use_scaler else None,
            "feature_columns": self.feature_columns,
            "threshold": threshold,
        }
        # Save extractor-specific state
        if hasattr(self.extractor, "valid_schars"):
            meta["valid_schars"] = self.extractor.valid_schars
        if hasattr(self.extractor, "vectorizer"):
            meta["vectorizer"] = self.extractor.vectorizer
            if hasattr(self.extractor.vectorizer, "_views"):
                meta["views"] = list(self.extractor.vectorizer._views)
        return meta

    def load_model(self, load_path: str):
        load_path = Path(load_path)
        if load_path.suffix != ".pth":
            load_path = load_path.with_suffix(".pth")

        meta_path = load_path.parent / f"{load_path.stem}_meta.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"Model weights not found: {load_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        try:
            metadata = pd.read_pickle(meta_path, compression="zstd")
        except Exception:
            metadata = pd.read_pickle(meta_path, compression=None)

        self.learning_rate = metadata.get("learning_rate", self.learning_rate)
        self.epochs = metadata.get("epochs", self.epochs)
        self.batch_size = metadata.get("batch_size", self.batch_size)
        self.use_scaler = metadata.get("use_scaler", self.use_scaler)
        self.feature_columns = metadata.get("feature_columns", None)
        self.threshold = metadata.get("threshold", None)
        input_dim = metadata["input_dim"]

        if self.use_scaler and metadata.get("scaler") is not None:
            self._scaler = metadata["scaler"]

        # Determine activation: saved metadata takes priority, then instance default
        activation = metadata.get("output_activation", self.output_activation)

        self._restore_extra_metadata(metadata)

        self.clf = AutoEncoder(input_dim=input_dim, output_activation=activation)
        self.clf.to(self.device)

        state_dict = torch.load(load_path, map_location=self.device)
        self.clf.load_state_dict(state_dict)
        self.clf.eval()

        logger.info(f"Loaded model from {load_path}")
        if self.threshold is not None:
            logger.info(f"Loaded threshold: {self.threshold}")

    def _restore_extra_metadata(self, metadata: dict):
        """Restore extractor-specific state from metadata."""
        if "valid_schars" in metadata and hasattr(self.extractor, "valid_schars"):
            self.extractor.valid_schars = metadata["valid_schars"]
        if "vectorizer" in metadata and hasattr(self.extractor, "vectorizer"):
            self.extractor.vectorizer = metadata["vectorizer"]
            self.extractor._fitted = True
            self.feature_columns = self.extractor.vectorizer.get_feature_names_out()
        if "w2v_model" in metadata and hasattr(self.extractor, "_w2v"):
            self.extractor._w2v = metadata["w2v_model"]
