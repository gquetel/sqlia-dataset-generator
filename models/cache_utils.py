"""Lightweight file-based caching utilities for feature matrices."""

import hashlib
import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def hash_df(df: pd.DataFrame) -> str:
    """Return a SHA-256 hex digest of a DataFrame's content."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def load_cache(path: str):
    """Return the unpickled object at *path*, or None if the file is missing."""
    if os.path.isfile(path):
        logger.info("Cache hit: %s", path)
        return pd.read_pickle(path, compression="zstd")
    return None


def save_cache(path: str, obj) -> None:
    """Pickle *obj* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.to_pickle(obj, path, compression="zstd")
    logger.debug("Cache saved: %s", path)
