"""GAUR feature extractor — hybrid mode (22 semantic + 23 Li-style features)."""

import pandas as pd

from base import BaseExtractor


class GaurExtractor(BaseExtractor):
    """Wraps GAUR's  preprocessing pipeline.

    Traces are collected lazily at extraction time via get_traces_from_df.
    The extractor is fully stateless — no vocabulary or weights to persist.
    """

    def __init__(
        self,
        use_hybrid: bool = True,
        cache_dir: str | None = None,
        mode: str = "expert",
    ):
        from gaur_sqld.models.Gaur import OCSVM_Gaur

        self.use_hybrid = use_hybrid
        self.cache_dir = cache_dir
        self.mode = mode
        # We use OCSVM_GAUR but this is the same preprocessing for another decision model.
        self._preprocessor = OCSVM_Gaur(use_hybrid=use_hybrid, mode=mode)

    def prepare_for_training(self, df: pd.DataFrame):
        pass  # Stateless no vocabulary to learn.

    def extract_features(self, df: pd.DataFrame):
        df = self._ensure_traces(df)
        X, _ = self._preprocessor.preprocess_for_preds(df)
        return X.select_dtypes(include="number")

    def preprocess_for_preds(self, df: pd.DataFrame):
        """Override base to avoid collecting traces twice.

        Returns a 3-tuple (X, labels, valid_index) where valid_index contains
        the pandas index values of rows that survived trace collection. Some rows
        may be dropped when gaur_sqld cannot build a semantic tree for a query.
        """
        df = self._ensure_traces(df)
        X, labels = self._preprocessor.preprocess_for_preds(df)
        return X.select_dtypes(include="number").to_numpy(dtype=float), labels, df.index

    def _ensure_traces(self, df: pd.DataFrame) -> pd.DataFrame:
        if "semantic_tree" not in df.columns:
            import gaur_sqld.config as gaur_config
            from gaur_sqld.utils.traces_collector import get_traces_from_df
            from typing import get_args

            valid_trace_types = get_args(gaur_config.ExistingTraces)
            trace_type = self.mode if self.mode in valid_trace_types else "expert"
            gaur_config.update_location_mysqlfiles(trace_type)

            df_traces = get_traces_from_df(
                df, use_cache=self.cache_dir is not None, disable_tqdm=True
            )
            df = pd.concat([df_traces, df], axis=1, join="inner")
        return df
