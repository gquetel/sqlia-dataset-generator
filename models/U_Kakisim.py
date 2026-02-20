import hashlib
import logging
import os
import pickle
from multiprocessing import Pool
from pathlib import Path
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

    The `views` argument selects which views to include (default: all three).
    """

    def __init__(self, views: list[str] | None = None, min_df: int = 1):
        self._views = set(views) if views is not None else {"T", "C", "E"}
        if "T" in self._views:
            self._cv_t = CountVectorizer(min_df=min_df)
        if "C" in self._views:
            self._cv_c = CountVectorizer(min_df=min_df)
        if "E" in self._views:
            self._cv_e = CountVectorizer(min_df=min_df)

    def _to_view_strings(self, queries) -> tuple[list[str], list[str], list[str]]:
        str_queries = [str(q) for q in queries]
        if len(str_queries) > 1000:
            with Pool() as pool:
                results = pool.map(_sql_to_views, str_queries, chunksize=500)
        else:
            results = [_sql_to_views(q) for q in str_queries]
        t_strs = [r[0] for r in results]
        c_strs = [r[1] for r in results]
        e_strs = [r[2] for r in results]
        return t_strs, c_strs, e_strs

    def fit_transform(self, queries) -> csr_matrix:
        t_strs, c_strs, e_strs = self._to_view_strings(queries)
        mats = []
        if "T" in self._views:
            mats.append(self._cv_t.fit_transform(t_strs))
        if "C" in self._views:
            mats.append(self._cv_c.fit_transform(c_strs))
        if "E" in self._views:
            mats.append(self._cv_e.fit_transform(e_strs))
        return hstack(mats, format="csr")

    def transform(self, queries) -> csr_matrix:
        t_strs, c_strs, e_strs = self._to_view_strings(queries)
        mats = []
        if "T" in self._views:
            mats.append(self._cv_t.transform(t_strs))
        if "C" in self._views:
            mats.append(self._cv_c.transform(c_strs))
        if "E" in self._views:
            mats.append(self._cv_e.transform(e_strs))
        return hstack(mats, format="csr")

    def get_feature_names_out(self) -> np.ndarray:
        parts = []
        if "T" in self._views:
            parts.append(self._cv_t.get_feature_names_out())
        if "C" in self._views:
            parts.append(self._cv_c.get_feature_names_out())
        if "E" in self._views:
            parts.append(self._cv_e.get_feature_names_out())
        return np.concatenate(parts)

    @property
    def views_tag(self) -> str:
        """Short string identifying the active views, e.g. 'E' or 'TCE'."""
        return "".join(sorted(self._views))

    def _vocab_tag(self) -> str:
        views_str = self.views_tag
        if "T" in self._views:
            vocab = str(sorted(self._cv_t.vocabulary_.items()))
        elif "C" in self._views:
            vocab = str(sorted(self._cv_c.vocabulary_.items()))
        else:
            vocab = str(sorted(self._cv_e.vocabulary_.items()))
        return hashlib.md5((views_str + vocab).encode()).hexdigest()[:8]


class OCSVM_Kakisim:
    def __init__(
        self,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = True,
        views: list[str] | None = None,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter

        self.vectorizer = KakisimVectorizer(views=views)
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
            key = f"kakisim_{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
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
                f"kakisim_{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
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
        views: list[str] | None = None,
        min_df: int = 1,
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.clf = None
        self.GENERIC = GENERIC
        self.model_name = None
        self.cache_dir = None

        self.vectorizer = KakisimVectorizer(views=views, min_df=min_df)
        self.use_scaler = use_scaler
        self.device = device
        self.threshold = None

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
                f"kakisim_{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
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
            key = f"kakisim_{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
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
            f_matrix = self._scaler.fit_transform(f_matrix)

        n_samples = f_matrix.shape[0]
        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, n_samples, self.batch_size):
                batch = torch.FloatTensor(
                    f_matrix[i : i + self.batch_size].toarray()
                ).to(self.device)
                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/n_samples:.6f}"
            )

    def save_model(self, save_path: str, threshold: float = None):
        """Save model weights, vectorizer, threshold, and metadata.

        Args:
            save_path: Path to save the model (without extension).
                       Creates {save_path}.pth for weights and {save_path}_meta.pkl for metadata.
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
            "vectorizer": self.vectorizer,
            "threshold": threshold,
            "views": list(self.vectorizer._views),
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved AutoEncoder_Kakisim model to {model_path}")
        if threshold is not None:
            logger.info(f"Saved threshold: {threshold}")

    def load_model(self, load_path: str):
        """Load model from saved checkpoint.

        Args:
            load_path: Path to the model file (with or without .pth extension).
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
        self.threshold = metadata.get("threshold", None)
        input_dim = metadata["input_dim"]

        # Restore vectorizer (holds trained CountVectorizer vocabularies)
        self.vectorizer = metadata["vectorizer"]
        self.feature_columns = self.vectorizer.get_feature_names_out()

        if self.use_scaler:
            self.clf = MyAutoEncoder(input_dim=input_dim)
        else:
            self.clf = MyAutoEncoderRelu(input_dim=input_dim)

        self.clf.to(self.device)
        state_dict = torch.load(load_path, map_location=self.device)
        self.clf.load_state_dict(state_dict)
        self.clf.eval()

        logger.info(f"Loaded AutoEncoder_Kakisim model from {load_path}")


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


class Word2VecKakisimVectorizer:
    """Word2Vec over concatenated T+C+E token sequences, max-pooled to fixed size."""

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
        self._w2v = None  # gensim Word2Vec, set during fit_transform

    def _to_sequences(self, queries) -> list[list[str]]:
        """Return one token list per query (T+C+E concatenated)."""
        str_queries = [str(q) for q in queries]
        if len(str_queries) > 1000:
            with Pool() as pool:
                results = pool.map(_sql_to_views, str_queries, chunksize=500)
        else:
            results = [_sql_to_views(q) for q in str_queries]
        return [t.split() + c.split() + e.split() for t, c, e in results]

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
        return "TCE_w2v"

    def _vocab_tag(self) -> str:
        vocab = sorted(self._w2v.wv.key_to_index.keys())
        return hashlib.md5(str(vocab).encode()).hexdigest()[:8]

    def get_feature_names_out(self) -> np.ndarray:
        return np.array([f"w2v_{i}" for i in range(self.vector_size)])


class OCSVM_Kakisim_W2V:
    def __init__(
        self,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = True,
        vector_size: int = 256,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.max_iter = max_iter

        self.vectorizer = Word2VecKakisimVectorizer(vector_size=vector_size)
        self.GENERIC = GENERIC

        self._scaler = StandardScaler(with_mean=False)

        self.clf = None
        self.model_name = None
        self.cache_dir = None
        self.feature_columns = None
        self.use_scaler = use_scaler

    def preprocess_for_train(
        self, df: pd.DataFrame, cache_dir: str | None = None
    ) -> np.ndarray:
        if cache_dir:
            key = f"kakisim_{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
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
                f"kakisim_{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached), labels
        pp_queries = self.vectorizer.transform(df["full_query"])
        if cache_path:
            save_cache(cache_path, pp_queries)
        return pd.DataFrame(pp_queries), labels


class AutoEncoder_Kakisim_W2V:
    def __init__(
        self,
        GENERIC,
        device,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
        vector_size: int = 256,
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.clf = None
        self.GENERIC = GENERIC
        self.model_name = None
        self.cache_dir = None

        self.vectorizer = Word2VecKakisimVectorizer(vector_size=vector_size)
        self.use_scaler = use_scaler
        self.device = device
        self.threshold = None

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
                f"kakisim_{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached), labels
        pp_queries = self.vectorizer.transform(df["full_query"])
        if cache_path:
            save_cache(cache_path, pp_queries)
        return pd.DataFrame(pp_queries), labels

    def preprocess_for_train(
        self, df: pd.DataFrame, cache_dir: str | None = None
    ) -> np.ndarray:
        if cache_dir:
            key = f"kakisim_{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
            cached = load_cache(os.path.join(cache_dir, key))
            if cached is not None:
                self.vectorizer = cached["vectorizer"]
                pp_queries = cached["features"]
                self._scaler_min = pp_queries.min(axis=None)
                self._scaler_max = pp_queries.max(axis=None)
                return pp_queries

        pp_queries = self.vectorizer.fit_transform(df["full_query"])
        self._scaler_min = pp_queries.min(axis=None)
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

        self.clf = MyAutoEncoderRelu(input_dim=input_dim)
        self.clf = self.clf.to(self.device)

        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)

        n_samples = f_matrix.shape[0]
        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, n_samples, self.batch_size):
                batch = torch.FloatTensor(f_matrix[i : i + self.batch_size]).to(
                    self.device
                )
                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/n_samples:.6f}"
            )

    def save_model(self, save_path: str, threshold: float = None):
        """Save model weights, vectorizer, threshold, and metadata."""
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
            "vectorizer": self.vectorizer,
            "threshold": threshold,
            "vector_size": self.vectorizer.vector_size,
        }
        with open(meta_path, "wb") as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved AutoEncoder_Kakisim_W2V model to {model_path}")
        if threshold is not None:
            logger.info(f"Saved threshold: {threshold}")

    def load_model(self, load_path: str):
        """Load model from saved checkpoint."""
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
        self.threshold = metadata.get("threshold", None)
        input_dim = metadata["input_dim"]

        # Restore vectorizer (holds trained Word2Vec model)
        self.vectorizer = metadata["vectorizer"]
        self.feature_columns = self.vectorizer.get_feature_names_out()

        self.clf = MyAutoEncoderRelu(input_dim=input_dim)
        self.clf.to(self.device)
        state_dict = torch.load(load_path, map_location=self.device)
        self.clf.load_state_dict(state_dict)
        self.clf.eval()

        logger.info(f"Loaded AutoEncoder_Kakisim_W2V model from {load_path}")


def preprocessing_kakisim_w2v(
    model: OCSVM_Kakisim_W2V | AutoEncoder_Kakisim_W2V,
    df: pd.DataFrame,
    use_scaler: bool = False,
):
    X, labels = model.preprocess_for_preds(df=df)
    return X.to_numpy(), labels
