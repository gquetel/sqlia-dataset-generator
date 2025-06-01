"""Definition of ML models configuration."""

from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import RocCurveDisplay, roc_curve, auc
import numpy as np
import random
import pandas as pd
import sys
import logging
import plotly.express as px

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
)
from sklearn.tree import plot_tree

from RF_Li import CustomRF_Li, CustomDT_Li
from RF_CountVect import CustomRF_CountVectorizer


# ------------ Custom Data Structures  ------------
class DotDict(dict):
    def __getattr__(self, attr):
        try:
            return self[attr]
        except KeyError:
            raise AttributeError(f"Attribute {attr} not found")

    def __setattr__(self, attr, value):
        self[attr] = value


class ProjectPaths:
    def __init__(
        self,
        base_path: str,
    ):
        self.base_path = base_path

    @property
    def dataset_path(self) -> str:
        return f"{self.base_path}/../dataset.csv"

    @property
    def output_path(self) -> str:
        path = f"{self.base_path}/output/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path

    @property
    def logs_path(self) -> str:
        path = f"{self.base_path}../logs/"
        Path(path).mkdir(exist_ok=True, parents=True)
        return path


# ------------ Global variables  ------------

GENERIC = DotDict(
    {
        "RANDOM_SEED": 7,
        "BASE_PATH": os.path.join(os.path.dirname(__file__), ""),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)

# Bootstrap a custom object path.
project_paths = ProjectPaths(GENERIC.BASE_PATH)
# project_paths = ProjectPaths("/home/gquetel/repos/sqlia-dataset/models")
logger = logging.getLogger(__name__)


def init_logging():
    lf = TimedRotatingFileHandler(
        project_paths.logs_path + "/training.log",
        when="midnight",
    )
    lf.setLevel(logging.INFO)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(logging.INFO)

    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=logging.INFO, handlers=[lf, lstdo])


def print_and_ret_metrics(
    all_targets: list,
    all_predictions: list,
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
    precision = (
        f"{precision_score(all_targets, all_predictions, average=average)* 100:.2f}%"
    )
    recall = f"{recall_score(all_targets, all_predictions, average=average)* 100:.2f}%"

    C = confusion_matrix(all_targets, all_predictions)
    TN, FP, _, _ = C.ravel()
    FPR = FP / (FP + TN)

    fpr = f"{FPR* 100:.2f}%"

    if not silent:
        logger.info(f"Metrics for {model}, using {average} average.")
        logger.info(f"Accuracy: {accuracy}")
        logger.info(f"F1 Score: {f1}")
        logger.info(f"Precision: {precision}")
        logger.info(f"Recall: {recall}")
        logger.info(f"False Positive Rate: {fpr}")

    return accuracy, f1, precision, recall, fpr


def plot_confusion_matrices_by_technique(
    df_test: pd.DataFrame, model_name: str, suffix: str = ""
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
        nrows=1, ncols=len(techniques), figsize=(5 * len(techniques), 10)
    )
    if len(techniques) == 1:
        axes = [axes]  # allows subscription
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
            colorbar=(i == len(techniques) - 1),
        )
    fp_fig = f"{project_paths.output_path}confusion_matrix_{model_name}{suffix}.png"
    plt.savefig(fp_fig)


def plot_pr_curves_plt(labels, l_preds: list, l_model_names: list):
    fig, ax = plt.subplots(figsize=(8, 6))

    for preds, model_name in zip(l_preds, l_model_names):
        # Process predictions to get probabilities
        if isinstance(preds, list):
            preds = np.array(preds)[:, 1]
        elif isinstance(preds, pd.Series):
            preds = preds.apply(lambda x: x[1])
        else:
            preds = preds[:, 1]
        
        precision, recall, _ = precision_recall_curve(labels, preds)
        auc_pr = auc(recall, precision)

        # Plot the curve
        ax.plot(recall, precision, label=f"{model_name} (AUC = {auc_pr:.3f})")

    # Customize plot
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("AUPRC Comparison")
    ax.legend()

    ax.grid(True, alpha=0.3)  
    
    # plt.tight_layout()
    plt.savefig(project_paths.output_path + "auprc_curves.png")


def plot_roc_curves_plt(labels, l_preds: list, l_model_names: list):
    fig, ax = plt.subplots(figsize=(8, 6))

    for preds, model_name in zip(l_preds, l_model_names):

        if isinstance(preds, list):
            preds = np.array(preds)[:, 1]
        elif isinstance(preds, pd.Series):
            preds = preds.apply(lambda x: x[1])
        else:
            preds = preds[:, 1]
        fpr, tpr, thresholds = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)

        display = RocCurveDisplay(
            fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=f"{model_name}"
        )
        display.plot(ax=ax)  # Plot on the same axis

    # Add reference line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.6, label="Random Classifier")
    ax.legend()
    ax.set_title("ROC Curves Comparison")
    ax.grid(True, alpha=0.3)

    plt.savefig(project_paths.output_path + "roc_curves.png")


def plot_roc_curve_plotly(model, df_test: pd.DataFrame, model_name: str):
    labels, ppreds = model.predict_proba(df_test)
    ppreds = ppreds[:, 1]
    fpr, tpr, thresholds = roc_curve(labels, ppreds)

    fig = px.area(
        x=fpr,
        y=tpr,
        title=f"ROC Curve for model {model_name} (AUC={auc(fpr, tpr):.4f})",
        labels=dict(x="False Positive Rate", y="True Positive Rate"),
        width=1600,
        height=1920,
    )

    # Random line
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    fig.update_yaxes(scaleanchor="x", scaleratio=1)
    fig.update_xaxes(constrain="domain")
    fp_fig = f"{project_paths.output_path}roc_score_{model_name}.png"
    fig.write_image(fp_fig)


def compute_metrics(model, df_test: pd.DataFrame, model_name: str):
    # 0 => pped + original columns
    df_pped, labels = model.preprocess_for_preds(df=df_test, drop_og_columns=False)
    # 1 => Probas (ppeds only)
    # Warning: having this variable and df_pped is VERY memory consuming
    # for CountVectorizer.
    df_pped_wout_og_cols = df_pped.drop(df_test.columns.to_list(), axis=1)
    probas = model.clf.predict_proba(df_pped_wout_og_cols.to_numpy())

    # 2 => Preds
    preds = np.argmax(probas, axis=1)
    df_pped["probas"] = probas.tolist()
    df_pped["preds"] = preds
    # 3 => print_and_ret_metrics all
    _, _, _, _, _ = print_and_ret_metrics(
        df_pped["label"],
        df_pped["preds"],
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )

    # 4 =>  print_and_ret_metrics challenging only
    df_chall = df_pped[df_pped["template_split"] == "challenging"]
    logger.info("Metrics for challenging set only:")
    _, _, _, _, _ = print_and_ret_metrics(
        df_chall["label"],
        df_chall["preds"],
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )

    # 5 =>  print_and_ret_metrics original only
    df_og = df_pped[df_pped["template_split"] == "original"]
    logger.info("Metrics for original set only:")
    _, _, _, _, _ = print_and_ret_metrics(
        df_og["label"].to_list(),
        df_og["preds"].to_list(),
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )

    # 6 => Confusion matrix all
    # This function needs the dataframe with the preds and the original
    # columns information (attack_technique).
    plot_confusion_matrices_by_technique(df_test=df_pped, model_name=model_name)

    # 7 => Confusion matrix challenging
    plot_confusion_matrices_by_technique(
        df_test=df_chall, model_name=model_name, suffix="_challenge"
    )

    # For AUC plot
    return df_pped["label"], df_pped["probas"]


def train_rf_li(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Li-LSyn_RF"
    model = CustomRF_Li(GENERIC=GENERIC, max_depth=None)
    model.train_model(df=df_train, model_name=model_name)
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_rf_cv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "CountVectorizer_RF"
    model = CustomRF_CountVectorizer(GENERIC=GENERIC, max_depth=None, max_features=None)
    model.train_model(df=df_train, model_name=model_name)
    df_test = df_test.copy().reset_index(drop=True)
    # 0 => pped
    f_matrix, labels = model.preprocess_for_preds(df=df_test, drop_og_columns=True)

    # 1 => Probas (ppeds only)
    probas = model.clf.predict_proba(f_matrix)

    # 2 => Preds
    preds = np.argmax(probas, axis=1)

    # 3 => print_and_ret_metrics all
    _, _, _, _, _ = print_and_ret_metrics(
        labels,
        preds,
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )

    # 4 =>  print_and_ret_metrics challenging only
    # To get challenging only, we need to fetch their indices:
    # And then retrieve rows from labels and preds to be given
    # to print_and_ret_metrics
    ids_chall = df_test[df_test["template_split"] == "challenging"].index.tolist()

    logger.info("Metrics for challenging set only:")
    _, _, _, _, _ = print_and_ret_metrics(
        labels[ids_chall],
        preds[ids_chall],
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )

    # 5 =>  print_and_ret_metrics original only
    ids_og = df_test[df_test["template_split"] == "original"].index.tolist()
    logger.info("Metrics for original set only:")
    _, _, _, _, _ = print_and_ret_metrics(
        labels[ids_og],
        preds[ids_og],
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )

    # 6 => Confusion matrix all
    # This function needs the dataframe with the preds and the original
    # columns information (attack_technique). I want to keep the same function
    # For both models, let's hack by creating the dataframe the function requires:
    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"],
            "label": labels,
            "preds": preds,
            "probas" : list(probas)
        }
    )
    plot_confusion_matrices_by_technique(df_test=_df, model_name=model_name)

    _df_chall = _df[df_test["template_split"] == "challenging"]
    # 7 => Confusion matrix challenging
    plot_confusion_matrices_by_technique(
        df_test=_df_chall, model_name=model_name, suffix="_challenge"
    )

    # For AUC plot
    return labels, probas


def train_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    logger.info(
        f"Training - number of attacks {len(df_train[df_train['label'] == 1])}"
        f" and number of normals {len(df_train[df_train['label'] == 0])}"
    )
    logger.info(
        f"Testing - number of attacks {len(df_test[df_test['label'] == 1])}"
        f" and number of normals {len(df_test[df_test['label'] == 0])}"
    )

    _, ppreds_li = train_rf_li(df_train, df_test)
    labels, ppreds_cv = train_rf_cv(df_train=df_train, df_test=df_test)

    plot_roc_curves_plt(
        labels=labels,
        l_preds=[ppreds_li, ppreds_cv],
        l_model_names=["Manual Features and RF", "CountVectorizer and RF"],
    )

    plot_pr_curves_plt(
        labels=labels,
        l_preds=[ppreds_li, ppreds_cv],
        l_model_names=["Manual Features and RF", "CountVectorizer and RF"],
    )


if __name__ == "__main__":
    init_logging()
    np.random.seed(GENERIC.RANDOM_SEED)
    random.seed(GENERIC.RANDOM_SEED)

    df = pd.read_csv(
        project_paths.dataset_path,
        # "/home/gquetel/repos/sqlia-dataset/dataset.csv",
        # dtype is specified to prevent a DtypeWarning
        dtype={
            "full_query": str,
            "label": int,
            "statement_type": str,
            "query_template_id": str,
            "user_inputs": str,
            "attack_id": str,
            "attack_technique": str,
            "attack_desc": str,
            "split": str,
            "attack_status": str,
            "attack_stage": str,
            "tamper_method": str,
            "template_split": str,
        },
    )

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    # from sklearn.model_selection import train_test_split
    # df_train, df_test = train_test_split(df,test_size=0.2)
    train_models(df_train, df_test)
