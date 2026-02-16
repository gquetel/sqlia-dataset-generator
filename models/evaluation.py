"""Shared evaluation logic for computing metrics from pre-computed scores."""

import logging

import numpy as np
import pandas as pd

from explain import (
    get_metrics_treshold,
    get_balanced_accuracy_per_attack,
    get_recall_per_attack,
    get_recall_per_statement_type,
)

logger = logging.getLogger(__name__)


def get_threshold_for_max_rate(s_val, max_rate=0.001):
    """Compute threshold given a max allowed FPR.

    Args:
        s_val: Validation scores (normal samples).
        max_rate (float, optional): Maximum false positive rate. Defaults to 0.001.

    Returns:
        float: Threshold value.
    """
    s_val = np.array(s_val)
    percentile = (1 - max_rate) * 100
    return np.percentile(s_val, percentile)


def compute_all_metrics(
    df_test: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model_name: str,
) -> tuple[dict, np.ndarray]:
    """Compute all evaluation metrics from pre-computed scores.

    Args:
        df_test: Test DataFrame (must contain 'attack_technique', 'statement_type', 'label' columns).
        labels: Ground truth labels.
        scores: Anomaly scores.
        threshold: Decision threshold.
        model_name: Identifier for logging and results.

    Returns:
        (metrics_dict, preds)
    """
    d_res, preds = get_metrics_treshold(
        labels=labels,
        scores=scores,
        model_name=model_name,
        threshold=threshold,
    )

    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"].values,
            "statement_type": df_test["statement_type"].values,
            "label": labels,
            "preds": preds,
        }
    )
    recall_per_attack = get_recall_per_attack(df=_df, model_name=model_name)
    d_res.update(recall_per_attack)
    d_res.update(get_recall_per_statement_type(df=_df, model_name=model_name))
    d_res.update(
        get_balanced_accuracy_per_attack(
            df=_df, model_name=model_name, recall_per_attack=recall_per_attack
        )
    )

    return d_res, preds
