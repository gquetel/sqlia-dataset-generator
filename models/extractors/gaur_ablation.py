"""GAUR ablation extractor — Li + a specific subset of GAUR features.

Feature subsets are determined by gaur_features:
  - Explicit list: uses those column names directly (e.g. GAUR_LEX, GAUR_SYNT)
  - None: uses all GAUR columns not in GAUR_LEX and not in GAUR_SYNT (semantic)
"""

import numpy as np
import pandas as pd
from gaur_sqld.utils.constants import GAUR_LEX, GAUR_SYNT

from extractors.gaur import GaurExtractor
from extractors.li import LiExtractor

_GAUR_NON_SEM = set(GAUR_LEX) | set(GAUR_SYNT)


class GaurAblationExtractor(GaurExtractor):
    """Extracts Li features concatenated with a subset of GAUR features.

    gaur_features: explicit list of GAUR column names to keep, or None for semantic
                   (all GAUR columns that are not lexical and not syntactic).
    """

    def __init__(
        self,
        gaur_features: list[str] | None = None,
        mode: str = "expert",
        cache_dir: str | None = None,
    ):
        super().__init__(use_hybrid=False, mode=mode, cache_dir=cache_dir)
        self._gaur_features = gaur_features
        self._li = LiExtractor()

    def preprocess_for_preds(self, df: pd.DataFrame):
        labels = np.array(df["label"])
        X = self.extract_features(df)
        return X, labels

    def extract_features(self, df: pd.DataFrame):
        df = self._ensure_traces(df)
        X_gaur, _ = self._preprocessor.preprocess_for_preds(df)
        X_gaur = X_gaur.select_dtypes(include="number")

        if self._gaur_features == "SEMANTIC_TAGS":
            # Keep all GAUR columns that are not lexical and not syntactic.
            cols = [c for c in X_gaur.columns if c not in _GAUR_NON_SEM]
        else:
            cols = [c for c in self._gaur_features if c in X_gaur.columns]

        X_gaur_subset = X_gaur[cols].to_numpy(dtype=float)
        X_li = self._li.extract_features(df).to_numpy(dtype=float)
        return np.hstack([X_li, X_gaur_subset])
