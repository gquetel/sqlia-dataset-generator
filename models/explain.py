from pathlib import Path
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import logging
import math

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)

def get_ci(score, n):
    """ 
    From: https://machinelearningmastery.com/confidence-intervals-for-machine-learning/
    
    Args:
        score (_type_): _description_
        n (_type_): _description_

    Returns:
        _type_: _description_
    """
    z = 1.96  # 95%
    interval = z * math.sqrt((score * (1 - score)) / n)
    return interval


def get_metrics_treshold(
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model_name: str = "",
):

    preds = (scores >= threshold).astype(int)

    accuracy = f"{accuracy_score(labels, preds)* 100:.2f}%"
    f1 = f"{f1_score(labels, preds)* 100:.2f}%"
    precision = f"{precision_score(labels, preds) * 100:.2f}%"
    recall = f"{recall_score(labels, preds) * 100:.2f}%"

    p, r, _ = precision_recall_curve(labels, scores, pos_label=1)
    auprc = auc(r, p)

    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)

    C = confusion_matrix(labels, preds, labels=[0, 1])
    TN, FP, _, _ = C.ravel()
    FPR = FP / (FP + TN)
    achieved_fpr = f"{FPR* 100:.5f}%"

    auroc_ci = get_ci(auroc, len(scores))
    auprc_ci = get_ci(auprc, len(scores))

    logger.info(f"Metrics for {model_name}.")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"False Positive Rate: {achieved_fpr}")

    logger.info(f"ROC-AUC: {auroc:.4f}, CI {auroc_ci:.4f}")
    logger.info(f"AUPRC: {auprc:.4f}, CI {auprc_ci:.4f}")

    return (
        {
            "model": model_name,
            "fone": f1,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "fpr": achieved_fpr,
            "auprc": f"{auprc:.4f}",
            "rocauc": f"{auroc:.4f}",
            "auroc_ci": f"{auroc_ci:.4f}",
            "auprc_ci": f"{auprc_ci:.4f}",
        },
        preds,
    )

def get_recall_per_attack(df: pd.DataFrame, model_name: str, suffix: str = ""):
    """Display Recall score per technique from a dataframe with preds."""
    techniques = df.loc[df["label"] == 1, "attack_technique"].unique().tolist()
    logger.info(f"Computing recall for model: {model_name}{suffix}")

    d_res = {}

    for i, technique in enumerate(techniques):
        mask = df["attack_technique"] == technique
        preds = df.loc[mask, "preds"]
        labels = df.loc[mask, "label"]
        srecall = f"{recall_score(labels, preds, average='binary')* 100:.2f}%"
        logger.info(f"Recall for technique {technique}: {srecall}")
        d_res[f"recall{technique}"] = srecall

    return d_res

def plot_pr_curves_plt_from_scores(
    labels, l_scores: list, l_model_names: list, project_paths, suffix: str = ""
):
    fig, ax = plt.subplots(figsize=(12, 10))
    folder_name = f"{project_paths.output_path}pr_curves/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    for scores, model_name in zip(l_scores, l_model_names):
        assert isinstance(scores, np.ndarray)
        precision, recall, _ = precision_recall_curve(labels, scores, pos_label=1)
        auprc = auc(recall, precision)

        # Plot the curve
        ax.plot(recall, precision, label=f"{model_name} ({auprc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
            filepath, index=False
        )

    # y = prevalence
    x = [0, 1]
    y = [sum(labels) / len(labels)] * len(x)
    ax.plot(x, y, "k--", alpha=0.6, label=f"Random Classifier = {y[0]:.4f}")

    # Customize plot
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AUPRC Comparison")
    ax.legend()

    ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    plt.savefig(f"{folder_name}auprc_curves{suffix}.png")


def plot_roc_curves_plt_from_scores(
    labels: list, l_scores: list, l_model_names: list, project_paths, suffix: str = ""
):
    fig, ax = plt.subplots(figsize=(12, 10))
    folder_name = f"{project_paths.output_path}roc_curves/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    for scores, model_name in zip(l_scores, l_model_names):
        # I lost myself with all of the type conversion for the different pipelines
        # Let's just impose a numpy array:
        assert isinstance(scores, np.ndarray)

        fpr, tpr, thresholds = roc_curve(labels, scores)
        auroc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{model_name} ({auroc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(filepath, index=False)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Classifier")
    ax.legend()
    ax.set_title("ROC Curves Comparison")
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{folder_name}roc_curves{suffix}.png")