"""Kakisim semantic tokenization feature extractors (CountVectorizer and Word2Vec variants)."""

import hashlib
import logging
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
import sqlparse
from sqlparse import tokens as T

from base import BaseExtractor
from cache_utils import hash_df, load_cache, save_cache

logger = logging.getLogger(__name__)

# ---- Tokenization & semantic tagging (unchanged from U_Kakisim.py) ----

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

_NOISY_FOR_T: set[str] = {"Int"}
_NOISY_FOR_E: set[str] = {"Int", "Punct", "Par"}


def _get_tag(ttype, token_val: str, next_tok=None) -> str:
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
    if ttype in T.Name:
        if ttype is T.Name.Builtin:
            return "Builtin"
        if next_tok is not None and next_tok.value == "(":
            return "Func"
        upper = token_val.upper()
        if upper in _KEYWORD_OVERRIDES:
            return _KEYWORD_OVERRIDES[upper]
        return "Identifi"
    if ttype in T.Comment:
        return "Escap"
    if ttype is T.Error:
        if token_val == "'":
            return "Escap"
        return "Error"
    return "Unknown"


def _walk_tree(node, result: list[tuple[str, str]]):
    if isinstance(node, sqlparse.sql.IdentifierList):
        result.append((node.value, "Identifierlist"))
        return
    if node.ttype is not None:
        if node.is_whitespace:
            return
        tag = _get_tag(node.ttype, node.value, next_tok=None)
        result.append((node.value, tag))
    else:
        for child in node.tokens:
            _walk_tree(child, result)


def _tokenize_and_tag(query: str) -> list[tuple[str, str]]:
    statements = sqlparse.parse(query)
    if not statements:
        return []
    result: list[tuple[str, str]] = []
    _walk_tree(statements[0], result)
    fixed: list[tuple[str, str]] = []
    for i, (val, tag) in enumerate(result):
        if tag == "Identifi" and i + 1 < len(result) and result[i + 1][0] == "(":
            fixed.append((val, "Func"))
        else:
            fixed.append((val, tag))
    return fixed


def _sql_to_views(query: str) -> tuple[str, str, str]:
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


# ---- Vectorizers ----


class KakisimVectorizer:
    """Three-view count vectorizer producing sparse feature matrices."""

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
        return [r[0] for r in results], [r[1] for r in results], [r[2] for r in results]

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
        self._w2v = None

    def _to_sequences(self, queries) -> list[list[str]]:
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


# ---- Extractors ----


class KakisimExtractor(BaseExtractor):
    """Kakisim CountVectorizer-based extractor with internal caching."""

    def __init__(self, views: list[str] | None = None, min_df: int = 1):
        self.vectorizer = KakisimVectorizer(views=views, min_df=min_df)
        self._fitted = False
        self.cache_dir = None

    def extract_features(self, df: pd.DataFrame) -> csr_matrix:
        if not self._fitted:
            if self.cache_dir:
                key = f"kakisim_{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
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
                f"kakisim_{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
            )
            cached = load_cache(cache_path)
            if cached is not None:
                return pd.DataFrame(cached.toarray()), labels
        pp_queries = self.vectorizer.transform(df["full_query"])
        if cache_path:
            save_cache(cache_path, pp_queries)
        return pd.DataFrame(pp_queries.toarray()), labels

    def get_feature_names_out(self) -> np.ndarray:
        return self.vectorizer.get_feature_names_out()


class KakisimW2VExtractor(BaseExtractor):
    """Kakisim Word2Vec-based extractor with internal caching."""

    def __init__(self, vector_size: int = 256):
        self.vectorizer = Word2VecKakisimVectorizer(vector_size=vector_size)
        self._fitted = False
        self.cache_dir = None

    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            if self.cache_dir:
                key = f"kakisim_{self.vectorizer.views_tag}-vectorized-train-{hash_df(df)}.pkl"
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
                f"kakisim_{self.vectorizer.views_tag}-vectorized-{hash_df(df)}-{vocab_tag}.pkl",
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
