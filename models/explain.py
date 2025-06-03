
from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import logging

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve, auc
)

logger = logging.getLogger(__name__)

def print_and_save_metrics(
    all_targets: list,
    all_predictions: list,
    all_probas: np.ndarray,
    average: Literal["binary", "macro", "micro"] = "binary",
    model: str = "",
    silent: bool = False,
):
    """Print the metrics for the given targets and predictions
    Metrics are: Accuracy, F1 Score, Precision, Recall

    Args:
        all_targets (list): List of targets
        all_predictions (list): List of predictions
    """
    accuracy = f"{accuracy_score(all_targets, all_predictions)* 100:.2f}%"
    f1 = f"{f1_score(all_targets, all_predictions, average=average)* 100:.2f}%"
    sprecision = (
        f"{precision_score(all_targets, all_predictions, average=average)* 100:.2f}%"
    )
    srecall = f"{recall_score(all_targets, all_predictions, average=average)* 100:.2f}%"

    C = confusion_matrix(all_targets, all_predictions)
    TN, FP, _, _ = C.ravel()
    FPR = FP / (FP + TN)
    sfpr = f"{FPR* 100:.2f}%"

    precision, recall, _ = precision_recall_curve(all_targets, all_probas[:, 1])
    auc_pr = auc(recall, precision)

    fpr, tpr, _ = roc_curve(all_targets, all_probas[:, 1])
    auc_roc = auc(fpr, tpr)

    # plot_pr_curves_plt(all_targets, [all_probas], [model], suffix=f"_{model}")
    # plot_roc_curves_plt(all_targets, [all_probas], [model], suffix=f"_{model}")
    if not silent:
        logger.info(f"Metrics for {model}, using {average} average.")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"Precision: {sprecision}")
        logger.info(f"Recall: {srecall}")
        logger.info(f"False Positive Rate: {sfpr}")
        logger.info(f"AUPRC : {auc_pr:.4f}")
        logger.info(f"ROC-AUC: {auc_roc:.4f}")

    return (
        {
            "model": model,
            "fone": f1,
            "accuracy": accuracy,
            "precision": sprecision,
            "recall": srecall,
            "fpr": sfpr,
            "auprc": f"{auc_pr:.4f}",
            "rocauc": f"{auc_roc:.4f}",
        }
    )


def plot_confusion_matrices_by_technique(
    df_test: pd.DataFrame, model_name: str, project_paths, suffix: str = ""
):
    """Compute confusion matrices from dataframe with preds.

    Args:
        model (_type_): _description_
        df_test (pd.DataFrame): Dataframe with columns 'label', 'attack_technique'
            and 'preds'
        model_name (str): _description_
    """
    techniques = (
        df_test.loc[df_test["label"] == 1, "attack_technique"].unique().tolist()
    )
    # techniques holds all attack techniques in test set.
    # We iterate over them to compute confusion matrices
    fig, axes = plt.subplots(
        nrows=1, ncols=len(techniques) + 1, figsize=(5 * len(techniques), 10)
    )

    for i, technique in enumerate(techniques):
        # Create boolean mask for the subset we want
        mask = (df_test["label"] == 0) | (df_test["attack_technique"] == technique)

        labels = df_test.loc[mask, "label"]
        preds = df_test.loc[mask, "preds"]

        axes[i].set_title(f"{technique} technique")
        ConfusionMatrixDisplay.from_predictions(
            y_true=labels,
            y_pred=preds,
            ax=axes[i],
            colorbar=False,
        )

    # Plot full confusion matrix
    i = len(techniques)
    labels = df_test["label"]
    preds = df_test["preds"]

    axes[i].set_title(f"Total")

    ConfusionMatrixDisplay.from_predictions(
        y_true=labels,
        y_pred=preds,
        ax=axes[i],
        colorbar=True,
    )
    folder_path = f"{project_paths.output_path}confmatrices/"
    Path(folder_path).mkdir(exist_ok=True, parents=True)
    fp_fig = f"{folder_path}confusion_matrix_{model_name}{suffix}.png"
    plt.savefig(fp_fig)


def plot_pr_curves_plt(labels, l_preds: list, l_model_names: list, project_paths, suffix: str = ""):
    fig, ax = plt.subplots(figsize=(8, 6))
    folder_name = f"{project_paths.output_path}pr_curves/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    for preds, model_name in zip(l_preds, l_model_names):
        # I lost myself with all of the type conversion for the different pipelines
        # Let's just impose a numpy array:
        assert isinstance(preds, np.ndarray)

        # Process predictions to get probabilities
        preds = preds[:, 1]

        precision, recall, _ = precision_recall_curve(labels, preds, pos_label=1)
        auprc = auc(recall, precision)

        # Plot the curve
        ax.plot(recall, precision, label=f"{model_name} (AUC = {auprc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
            filepath, index=False
        )

    # y = prevalence
    x = [0, 1]
    y = [sum(labels) / len(labels)] * len(x)
    ax.plot(x, y, label=f"Prevalence = {y[0]:.4f}")

    # Customize plot
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AUPRC Comparison")
    ax.legend()

    ax.grid(True, alpha=0.3)

    # plt.tight_layout()
    plt.savefig(f"{folder_name}auprc_curves{suffix}.png")


def plot_roc_curves_plt(
    labels: list, l_preds: list, l_model_names: list, project_paths, suffix: str = ""
):
    fig, ax = plt.subplots(figsize=(8, 6))
    folder_name = f"{project_paths.output_path}roc_curves/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)

    for preds, model_name in zip(l_preds, l_model_names):
        # I lost myself with all of the type conversion for the different pipelines
        # Let's just impose a numpy array:
        assert isinstance(preds, np.ndarray)

        preds = preds[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, preds)
        auroc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auroc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(filepath, index=False)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Classifier")
    ax.legend()
    ax.set_title("ROC Curves Comparison")
    ax.grid(True, alpha=0.3)

    plt.savefig(f"{folder_name}roc_curves{suffix}.png")
