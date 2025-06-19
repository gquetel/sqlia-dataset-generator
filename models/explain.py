from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
from sklearn.tree import plot_tree
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
    roc_curve,
    auc,
)

logger = logging.getLogger(__name__)


def print_and_save_metrics_from_treshold(
    labels: np.ndarray,
    scores: np.ndarray,
    project_paths,
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
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auroc = auc(fpr, tpr)
    

    C = confusion_matrix(labels, preds, labels=[0, 1])
    TN, FP, _, _ = C.ravel()
    FPR = FP / (FP + TN)
    achieved_fpr = f"{FPR* 100:.5f}%"
    
    logger.info(f"Metrics for {model_name}.")
    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"False Positive Rate: {achieved_fpr}")
    logger.info(f"AUPRC : {auprc:.4f}")
    logger.info(f"ROC-AUC: {auroc:.4f}")
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
        },
        preds,
    )


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

    # Addind labels information prevent from only returning a single value
    # causing the not enough values to unpack error.
    C = confusion_matrix(all_targets, all_predictions, labels=[0, 1])
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

    return {
        "model": model,
        "fone": f1,
        "accuracy": accuracy,
        "precision": sprecision,
        "recall": srecall,
        "fpr": sfpr,
        "auprc": f"{auc_pr:.4f}",
        "rocauc": f"{auc_roc:.4f}",
    }


def get_recall_per_attack(df: pd.DataFrame, model_name: str, suffix: str = ""):
    """Display Recall score per technique from a dataframe with preds."""
    techniques = df.loc[df["label"] == 1, "attack_technique"].unique().tolist()
    logger.info(f"Computing recall for model: {model_name}{suffix}")

    d_res = {}

    for i, technique in enumerate(techniques):
        mask = df["attack_technique"] == technique
        preds = df.loc[mask, "preds"]
        labels = df.loc[mask, "label"]
        srecall = f"{recall_score(labels, preds, average="binary")* 100:.2f}%"
        logger.info(f"Recall for technique {technique}: {srecall}")
        d_res[f"recall{technique}"] = srecall

    return d_res


def plot_tree_clf(model, project_paths, max_depth: int | None = None):
    n_leaves = model.clf.get_n_leaves()
    w = np.sqrt(n_leaves * 10)
    h = np.sqrt(n_leaves * 10)
    plt.figure(figsize=(w, h))

    plt.title(f"Decision tree for {model}")
    folder_path = f"{project_paths.output_path}tree_plots/"
    Path(folder_path).mkdir(exist_ok=True, parents=True)
    _full_path = f"{folder_path}{model.model_name}"
    plot_tree(
        model.clf,
        proportion=False,
        feature_names=model.feature_names,
        fontsize=5,
        max_depth=max_depth,
    )
    plt.savefig(_full_path, dpi=200)
    logger.info(f"Saved decision tree plot at {_full_path}. ")
    plt.clf()


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


def plot_pca(X: np.ndarray, y: np.ndarray, project_paths, model_name: str):
    from sklearn.decomposition import PCA

    assert isinstance(X, np.ndarray)

    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)
    print(
        "explained variance ratio (first two components): %s"
        % str(pca.explained_variance_ratio_)
    )

    plt.figure(figsize=(8, 6))

    classes = np.unique(y)
    colors = ["darkorange", "turquoise"]

    for i, class_label in enumerate(classes):
        mask = y == class_label
        plt.scatter(
            X_r[mask, 0],
            X_r[mask, 1],
            c=colors[i],
            alpha=0.3,
            s=50,
            label=f"Class {class_label}",
        )

    plt.xlabel(
        f"First Principal Component ({pca.explained_variance_ratio_[0]:.2%} variance)"
    )
    plt.ylabel(
        f"Second Principal Component ({pca.explained_variance_ratio_[1]:.2%} variance)"
    )
    plt.title("PCA of Binary Classification Data")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    folder_name = f"{project_paths.output_path}pca_analysis/"
    Path(folder_name).mkdir(exist_ok=True, parents=True)
    plt.savefig(f"{folder_name}pca_{model_name}.png")


def plot_pr_curves_plt(
    labels, l_preds: list, l_model_names: list, project_paths, suffix: str = ""
):
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
        # TODO: Enlever "AUC =" , Manual Features -> Li,et figure en plus grand.
        ax.plot(recall, precision, label=f"{model_name} (AUC = {auprc:.4f})")

        # Also let's save the results:
        filepath = folder_name + f"{model_name}{suffix}.csv"
        pd.DataFrame({"precision": precision, "recall": recall}).to_csv(
            filepath, index=False
        )

    # y = prevalence
    x = [0, 1]
    y = [sum(labels) / len(labels)] * len(x)
    # TODO: Pointillés et en noir
    ax.plot(x, y, label=f"Random = {y[0]:.4f}")

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
