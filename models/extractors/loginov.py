"""Loginov et al. feature extractor — 9 numerical features from SQL queries."""

import logging
import re

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

from base import BaseExtractor
from constants import mysql_functions, mysql_keywords

SQL_KEYWORDS = {kw.upper() for kw in mysql_keywords | mysql_functions}

_SCHAR_RE = re.compile(r"[^a-zA-Z0-9\s]+")
_S2_SEP_RE = re.compile(r"[\s+]+")


def extract_s_chars(query: str) -> list[str]:
    return _SCHAR_RE.findall(query)


def _classify_tokens(tokens: list[str]) -> tuple[int, int, int, int]:
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
    schars = extract_s_chars(query)
    f1 = sum(1 for sc in schars if sc not in valid_schars)

    s1 = _SCHAR_RE.sub(" ", query)
    s1_tokens = [t for t in s1.split() if t]
    f2, f3, f4, f5 = _classify_tokens(s1_tokens)

    s2 = re.sub(r"[^a-zA-Z0-9\s+]+", "", query)
    s2_tokens = [t for t in _S2_SEP_RE.split(s2) if t]
    f6, f7, f8, f9 = _classify_tokens(s2_tokens)

    return {
        "n_anomalous_schars": f1,
        "s1_n_keywords": f2,
        "s1_n_alpha": f3,
        "s1_n_numeric": f4,
        "s1_n_mixed": f5,
        "s2_n_keywords": f6,
        "s2_n_alpha": f7,
        "s2_n_numeric": f8,
        "s2_n_mixed": f9,
    }


def _learn_valid_schars_from_df(df: pd.DataFrame) -> frozenset:
    seen: set[str] = set()
    for query in df["full_query"]:
        seen.update(extract_s_chars(str(query)))
    return frozenset(seen)


def _features_from_df(df: pd.DataFrame, valid_schars: frozenset) -> pd.DataFrame:
    rows = [extract_loginov_features(str(q), valid_schars) for q in df["full_query"]]
    return pd.DataFrame(rows, index=df.index)


class LoginovExtractor(BaseExtractor):
    """Loginov feature extractor.

    Requires learning valid_schars from training data before prediction.
    """

    def __init__(self):
        self.valid_schars: frozenset = frozenset()

    def prepare_for_training(self, df: pd.DataFrame):
        self.valid_schars = _learn_valid_schars_from_df(df)
        logger.info(f"Loginov valid_schars size: {len(self.valid_schars)}")

    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return _features_from_df(df, self.valid_schars)
