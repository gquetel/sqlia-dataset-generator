import hashlib
import logging
import os
import pandas as pd
from scipy import sparse
from scipy.sparse import hstack, csr_matrix
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.svm import OneClassSVM
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import sqlparse
from sqlparse import tokens as T
from constants import MyAutoEncoder, MyAutoEncoderRelu
import torch.nn as nn
import torch
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)


# If an uppercased token matches a key, we set the value as semantic tag. As show
# in Table 1.

_KEYWORD_OVERRIDES: dict[str, str] = {
    "GRANT": "DCL",
    "REVOKE": "DCL",
    "WITH": "CTE",
    "WHERE": "Where",
    "ASC": "Order",
    "DESC": "Order",
    "DATE": "Builtin",
    "TIME": "Builtin",
    "TIMESTAMP": "Builtin",
    "VARCHAR": "Builtin",
    "CHAR": "Builtin",
    "INT": "Builtin",
    "INTEGER": "Builtin",
    "UNSIGNED": "Builtin",
    "BINARY": "Builtin",
    "SLEEP": "Func",
    "BENCHMARK": "Func",
    "SUBSTRING": "Func",
    "SUBSTR": "Func",
    "MID": "Func",
    "LENGTH": "Func",
    "LEN": "Func",
    "COUNT": "Func",
    "SUM": "Func",
    "AVG": "Func",
    "MAX": "Func",
    "MIN": "Func",
    "UPPER": "Func",
    "LOWER": "Func",
    "CONCAT": "Func",
    "GROUP_CONCAT": "Func",
    "COALESCE": "Func",
    "IFNULL": "Func",
    "IF": "Func",
    "CAST": "Func",
    "CONVERT": "Func",
    "CHAR_LENGTH": "Func",
    "LOAD_FILE": "Func",
    "VERSION": "Func",
    "DATABASE": "Func",
    "USER": "Func",
    "CURRENT_USER": "Func",
    "SCHEMA": "Func",
    "HEX": "Func",
    "UNHEX": "Func",
    "ASCII": "Func",
    "ORD": "Func",
    "FLOOR": "Func",
    "CEILING": "Func",
    "RAND": "Func",
    "EXTRACTVALUE": "Func",
    "UPDATEXML": "Func",
}

# Drop token value in T view for these tags (3.1.1): "As it is seen, integer numbers
# are deleted from SQL queries during the tokenization process."
_NOISY_FOR_T: set[str] = {"Int"}

# Keep tag but drop value in E view for these tags (3.1.3): "Considering that the
# semantic equivalents of some expressions integrated into the enriched version of
#  SQL inputs are sufficient, some expressions such as numbers, punctuation, and
# parentheses are deleted from the enriched versions."
_NOISY_FOR_E: set[str] = {"Int", "Punct", "Par"}


def _get_tag(ttype, token_val: str, next_tok=None) -> str:
    """Resolve a flattened sqlparse token to a Semantic tag."""
    if token_val == "(" or token_val == ")":
        return "Par"

    if ttype is T.Keyword.DDL:
        return "DLL"
    if ttype is T.Keyword.DML:
        return "DML"

    if ttype in T.Keyword:
        upper = token_val.upper()
        if upper in _KEYWORD_OVERRIDES:
            return _KEYWORD_OVERRIDES[upper]
        return "Keyw"

    if ttype is T.Number.Integer or ttype is T.Number.Float:
        return "Int"
    if ttype is T.Number.Hexadecimal:
        return "Hexadecimal"
    if ttype is T.Literal.String.Single:
        return "Quot"
    if ttype is T.Punctuation:
        return "Punct"
    if ttype is T.Wildcard:
        return "Wildcard"
    if ttype is T.Comparison:
        return "Comparison"
    if ttype in T.Operator:
        return "Oper"
    if ttype is T.Name:
        # Function call: Name immediately followed by opening parenthesis
        if next_tok is not None and next_tok.value == "(":
            return "Func"
        return "Identifi"
    if ttype in T.Comment:
        return "Escap"
    if ttype is T.Error:
        if token_val == "'":
            return "Escap"
        return "Error"
    return "Error"


def _tokenize_and_tag(query: str) -> list[tuple[str, str]]:
    """Flatten sqlparse parse tree; return [(token_value, kakisim_tag), ...]."""
    statements = sqlparse.parse(query)
    if not statements:
        return []
    flat = [tok for tok in statements[0].flatten() if not tok.is_whitespace]
    result = []
    for i, tok in enumerate(flat):
        next_tok = flat[i + 1] if i + 1 < len(flat) else None
        tag = _get_tag(tok.ttype, tok.value, next_tok)
        result.append((tok.value, tag))
    return result


def _sql_to_views(query: str) -> tuple[str, str, str]:
    """Return (T_str, C_str, E_str) space-joined token sequences.

    T: token values with numeric literals dropped
    C: tag sequence only
    E: interleaved [tag, value] pairs; noisy tags retain tag but drop value
    """
    tagged = _tokenize_and_tag(query)

    t_parts: list[str] = []
    c_parts: list[str] = []
    e_parts: list[str] = []

    for val, tag in tagged:
        c_parts.append(tag)

        if tag not in _NOISY_FOR_T:
            t_parts.append(val)

        e_parts.append(tag)
        if tag not in _NOISY_FOR_E:
            e_parts.append(val)

    return " ".join(t_parts), " ".join(c_parts), " ".join(e_parts)


class KakisimVectorizer:
    """Three-view count vectorizer to be fed to OCVM / Autoencoder

    Produces three semantic views of each SQL query (T, C, E), counts token
    frequencies per view, then hstacks the sparse matrices.
    """

    def __init__(self):
        self._cv_t = CountVectorizer()
        self._cv_c = CountVectorizer()
        self._cv_e = CountVectorizer()

    def _to_view_strings(self, queries) -> tuple[list[str], list[str], list[str]]:
        t_strs, c_strs, e_strs = [], [], []
        for q in queries:
            t, c, e = _sql_to_views(str(q))
            t_strs.append(t)
            c_strs.append(c)
            e_strs.append(e)
        return t_strs, c_strs, e_strs

    def fit_transform(self, queries) -> csr_matrix:
        t_strs, c_strs, e_strs = self._to_view_strings(queries)
        mat_t = self._cv_t.fit_transform(t_strs)
        mat_c = self._cv_c.fit_transform(c_strs)
        mat_e = self._cv_e.fit_transform(e_strs)
        return hstack([mat_t, mat_c, mat_e], format="csr")

    def transform(self, queries) -> csr_matrix:
        t_strs, c_strs, e_strs = self._to_view_strings(queries)
        mat_t = self._cv_t.transform(t_strs)
        mat_c = self._cv_c.transform(c_strs)
        mat_e = self._cv_e.transform(e_strs)
        return hstack([mat_t, mat_c, mat_e], format="csr")

    def get_feature_names_out(self) -> np.ndarray:
        return np.concatenate(
            [
                self._cv_t.get_feature_names_out(),
                self._cv_c.get_feature_names_out(),
                self._cv_e.get_feature_names_out(),
            ]
        )

    def _vocab_tag(self) -> str:
        vocab = str(sorted(self._cv_t.vocabulary_.items()))
        return hashlib.md5(vocab.encode()).hexdigest()[:8]


class OCSVM_Kakisim:
    def __init__(
        self,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = True,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter

        self.vectorizer = KakisimVectorizer()
        self.GENERIC = GENERIC

        self._scaler = StandardScaler(with_mean=False)

        self.clf = None
        self.model_name = None
        self.cache_dir = None
        self.feature_columns = None
        self.use_scaler = use_scaler

    def preprocess_for_train(
        self, df: pd.DataFrame, cache_dir: str | None = None
    ) -> csr_matrix:
        if cache_dir:
            key = f"kakisim-vectorized-train-{hash_df(df)}.pkl"
            cached = load_cache(os.path.join(cache_dir, key))
            if cached is not None:
                self.vectorizer = cached["vectorizer"]
                return cached["features"]

        pp_queries = self.vectorizer.fit_transform(df["full_query"])

        if cache_dir:
            save_cache(
                os.path.join(cache_dir, key),
                {
                    "vectorizer": self.vectorizer,
                    "features": pp_queries,
                },
            )
        return pp_queries

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        f_matrix = self.preprocess_for_train(df, cache_dir=cache_dir)
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

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        labels = df["label"].to_numpy()
        cache_path = None
        if self.cache_dir:
            vocab_tag = self.vectorizer._vocab_tag()
            cache_path = os.path.join(
                self.cache_dir,
                f"kakisim-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached.toarray()), labels
        pp_queries = self.vectorizer.transform(df["full_query"])
        if cache_path:
            save_cache(cache_path, pp_queries)
        return pd.DataFrame(pp_queries.toarray()), labels


class AutoEncoder_Kakisim:
    def __init__(
        self,
        GENERIC,
        device,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.clf = None
        self.GENERIC = GENERIC
        self.model_name = None
        self.cache_dir = None

        self.vectorizer = KakisimVectorizer()
        self.use_scaler = use_scaler
        self.device = device

        self._scaler = MaxAbsScaler()
        self._scaler_min = None
        self._scaler_max = None

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.feature_columns = None

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        labels = df["label"].to_numpy()
        cache_path = None
        if self.cache_dir:
            vocab_tag = self.vectorizer._vocab_tag()
            cache_path = os.path.join(
                self.cache_dir,
                f"kakisim-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached.toarray()), labels
        pp_queries = self.vectorizer.transform(df["full_query"])
        if cache_path:
            save_cache(cache_path, pp_queries)
        return pd.DataFrame(pp_queries.toarray()), labels

    def preprocess_for_train(
        self, df: pd.DataFrame, cache_dir: str | None = None
    ) -> csr_matrix:
        if cache_dir:
            key = f"kakisim-vectorized-train-{hash_df(df)}.pkl"
            cached = load_cache(os.path.join(cache_dir, key))
            if cached is not None:
                self.vectorizer = cached["vectorizer"]
                pp_queries = cached["features"]
                self._scaler_min = pp_queries.min(axis=None)
                self._scaler_max = pp_queries.max(axis=None)
                return pp_queries

        pp_queries = self.vectorizer.fit_transform(df["full_query"])
        self._scaler_min = pp_queries.min(axis=None)
        assert self._scaler_min >= 0
        self._scaler_max = pp_queries.max(axis=None)

        if cache_dir:
            save_cache(
                os.path.join(cache_dir, key),
                {
                    "vectorizer": self.vectorizer,
                    "features": pp_queries,
                },
            )
        return pp_queries

    def X_to_tensor(self, X) -> torch.Tensor:
        X = X.values
        if self.use_scaler:
            X = self._scaler.transform(X)
        X_tensors = torch.FloatTensor(X).to(self.device)
        return X_tensors

    def _sparse_to_tensor_batched(self, sparse_matrix, batch_size=4096):
        n_samples = sparse_matrix.shape[0]
        tensors = []
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            batch_dense = sparse_matrix[i:batch_end].toarray()
            tensors.append(torch.FloatTensor(batch_dense))
        return torch.cat(tensors, dim=0).to(self.device)

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
        cache_dir: str | None = None,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        f_matrix = self.preprocess_for_train(df, cache_dir=cache_dir)
        self.feature_columns = self.vectorizer.get_feature_names_out()
        input_dim = len(self.feature_columns)

        if self.use_scaler:
            self.clf = MyAutoEncoder(input_dim=input_dim)
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


def preprocessing_kakisim(
    model: OCSVM_Kakisim | AutoEncoder_Kakisim,
    df: pd.DataFrame,
    use_scaler: bool = False,
):
    X, labels = model.preprocess_for_preds(df=df)
    X = X.to_numpy()

    if use_scaler:
        X_scaled = sparse.csr_matrix(X)
        X_scaled = model._scaler.transform(X)
        return X_scaled, labels

    return X, labels
