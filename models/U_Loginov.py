import logging
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler, StandardScaler
from sklearn.svm import OneClassSVM

from constants import MyAutoEncoderRelu, mysql_functions, mysql_keywords

logger = logging.getLogger(__name__)

SQL_KEYWORDS = {kw.upper() for kw in mysql_keywords | mysql_functions}

_SCHAR_RE = re.compile(r"[^a-zA-Z0-9\s]+")
_S2_SEP_RE = re.compile(r"[\s+]+")


def extract_s_chars(query: str) -> list[str]:
    """Return all special-character sequences found in *query* (in order)."""
    return _SCHAR_RE.findall(query)


def _classify_tokens(tokens: list[str]) -> tuple[int, int, int, int]:
    """Count (keyword, alphabetical, numerical, mixed) tokens.

    Categories are mutually exclusive and applied in that order:
    - keyword  : token.upper() in SQL_KEYWORDS
    - alpha    : token.isalpha() and not keyword
    - numeric  : token.isnumeric()
    - mixed    : contains both alpha and digit characters
    """
    n_kw = n_alpha = n_num = n_mix = 0
    for tok in tokens:
        if not tok:
            continue
        upper = tok.upper()
        if upper in SQL_KEYWORDS:
            n_kw += 1
        elif tok.isalpha():
            n_alpha += 1
        elif tok.isnumeric():
            n_num += 1
        elif any(c.isalpha() for c in tok) and any(c.isdigit() for c in tok):
            n_mix += 1
    return n_kw, n_alpha, n_num, n_mix


def extract_loginov_features(query: str, valid_schars: frozenset) -> dict:
    """Extract the 9 Loginov et al. numerical features from a single query.

    Args:
        query: Raw SQL query string.
        valid_schars: Frozenset of s-char sequences seen in normal training data.

    Returns:
        Dict with keys ``f1`` … ``f9``.
    """
    # Feature 1: count s-chars NOT in the valid set.
    schars = extract_s_chars(query)
    f1 = sum(1 for sc in schars if sc not in valid_schars)

    # Build S1: replace every s-char sequence with a single space.
    s1 = _SCHAR_RE.sub(" ", query)
    s1_tokens = [t for t in s1.split() if t]
    f2, f3, f4, f5 = _classify_tokens(s1_tokens)

    # Build S2: remove all s-chars EXCEPT '+', then tokenise on whitespace and '+'.
    s2 = re.sub(r"[^a-zA-Z0-9\s+]+", "", query)
    s2_tokens = [t for t in _S2_SEP_RE.split(s2) if t]
    f6, f7, f8, f9 = _classify_tokens(s2_tokens)

    return {
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "f4": f4,
        "f5": f5,
        "f6": f6,
        "f7": f7,
        "f8": f8,
        "f9": f9,
    }


def _learn_valid_schars_from_df(df: pd.DataFrame) -> frozenset:
    """Collect all unique s-char sequences from all queries in *df*."""
    seen: set[str] = set()
    for query in df["full_query"]:
        seen.update(extract_s_chars(str(query)))
    return frozenset(seen)


def _features_from_df(df: pd.DataFrame, valid_schars: frozenset) -> pd.DataFrame:
    rows = [extract_loginov_features(str(q), valid_schars) for q in df["full_query"]]
    return pd.DataFrame(rows, index=df.index)


class OCSVM_Loginov:
    def __init__(
        self,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = False,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter
        self.use_scaler = use_scaler
        self.GENERIC = GENERIC
        self.model_name = None
        self.clf = None
        self.valid_schars: frozenset = frozenset()
        self._scaler = StandardScaler()

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        features = _features_from_df(df_pped, self.valid_schars)
        return features, labels

    def preprocess_for_train(self, df: pd.DataFrame) -> pd.DataFrame:
        features, _ = self.preprocess_for_preds(df)
        return features

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        self.valid_schars = _learn_valid_schars_from_df(df)
        logger.info(f"Loginov valid_schars size: {len(self.valid_schars)}")

        df_pped = self.preprocess_for_train(df)
        self.feature_columns = df_pped.columns.tolist()
        X = df_pped.values

        if self.use_scaler:
            X = self._scaler.fit_transform(X)

        self.clf = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            max_iter=self.max_iter,
        )
        self.clf.fit(X)


class AutoEncoder_Loginov:
    def __init__(
        self,
        GENERIC,
        device,
        learning_rate: float = 0.005,
        epochs: int = 100,
        batch_size: int = 8192,
        use_scaler: bool = False,
    ):
        self.GENERIC = GENERIC
        self.device = device
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_scaler = use_scaler
        self.model_name = None
        self.clf = None
        self.valid_schars: frozenset = frozenset()
        self.feature_columns = None
        self.threshold = None
        self._scaler = MaxAbsScaler()

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        features = _features_from_df(df_pped, self.valid_schars)
        return features, labels

    def preprocess_for_train(self, df: pd.DataFrame) -> pd.DataFrame:
        features, _ = self.preprocess_for_preds(df)
        return features

    def X_to_tensor(self, X: pd.DataFrame) -> torch.Tensor:
        arr = X.values
        if self.use_scaler:
            arr = self._scaler.transform(arr)
        return torch.FloatTensor(arr).to(self.device)

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        self.valid_schars = _learn_valid_schars_from_df(df)
        logger.info(f"Loginov valid_schars size: {len(self.valid_schars)}")

        df_pped = self.preprocess_for_train(df)
        self.feature_columns = df_pped.columns.tolist()
        input_dim = len(self.feature_columns)  # always 9

        arr = df_pped.values.astype(np.float32)

        if self.use_scaler:
            arr = self._scaler.fit_transform(arr).astype(np.float32)

        train_data = torch.FloatTensor(arr)
        self.clf = MyAutoEncoderRelu(input_dim=input_dim).to(self.device)

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
        """Save model weights and metadata (including valid_schars).

        Args:
            save_path: Path without extension. Creates {save_path}.pth and
                       {save_path}_meta.pkl.
            threshold: Decision threshold computed on validation set.
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        model_path = save_path.with_suffix(".pth")
        meta_path = save_path.parent / f"{save_path.stem}_meta.pkl"

        torch.save(self.clf.state_dict(), model_path)

        metadata = {
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "input_dim": len(self.feature_columns),
            "use_scaler": self.use_scaler,
            "feature_columns": self.feature_columns,
            "scaler": self._scaler if self.use_scaler else None,
            "threshold": threshold,
            "valid_schars": self.valid_schars,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved AutoEncoder_Loginov model to {model_path}")
        if threshold is not None:
            logger.info(f"Saved threshold: {threshold}")

    def load_model(self, load_path: str):
        """Load model from saved checkpoint.

        Args:
            load_path: Path to the .pth file (extension optional).
        """
        load_path = Path(load_path)
        if load_path.suffix != ".pth":
            load_path = load_path.with_suffix(".pth")

        meta_path = load_path.parent / f"{load_path.stem}_meta.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"Model weights not found: {load_path}")
        if not meta_path.exists():
            raise FileNotFoundError(f"Model metadata not found: {meta_path}")

        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)

        self.learning_rate = metadata.get("learning_rate", self.learning_rate)
        self.epochs = metadata.get("epochs", self.epochs)
        self.batch_size = metadata.get("batch_size", self.batch_size)
        self.use_scaler = metadata.get("use_scaler", self.use_scaler)
        self.feature_columns = metadata.get("feature_columns", None)
        self.threshold = metadata.get("threshold", None)
        self.valid_schars = metadata.get("valid_schars", frozenset())
        input_dim = metadata["input_dim"]

        if self.use_scaler and metadata.get("scaler") is not None:
            self._scaler = metadata["scaler"]

        self.clf = MyAutoEncoderRelu(input_dim=input_dim).to(self.device)
        state_dict = torch.load(load_path, map_location=self.device)
        self.clf.load_state_dict(state_dict)
        self.clf.eval()

        logger.info(f"Loaded AutoEncoder_Loginov model from {load_path}")
        if self.threshold is not None:
            logger.info(f"Loaded threshold: {self.threshold}")


def preprocess_loginov(
    model: OCSVM_Loginov | AutoEncoder_Loginov,
    df: pd.DataFrame,
    use_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Transform dataframe into feature array and labels, ready for scoring.

    Args:
        model: Fitted OCSVM_Loginov or AutoEncoder_Loginov.
        df: Input dataframe.
        use_scaler: Whether to apply the model's fitted scaler.

    Returns:
        Tuple of (X: ndarray, labels: ndarray).
    """
    X, labels = model.preprocess_for_preds(df=df)
    X = X.to_numpy()

    if use_scaler:
        X = model._scaler.transform(X)

    return X, labels
