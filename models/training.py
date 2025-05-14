"""Definition of ML models configuration."""

from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path
from typing import Literal
from matplotlib import pyplot as plt
import numpy as np
import random
import pandas as pd
import sys
import logging

from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from RF_Li import CustomRF_Li
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
        path = f"{self.base_path}/logs/"
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
    precision = f"{precision_score(all_targets, all_predictions, average=average)* 100:.2f}%"
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


def test_model_separating_techniques(
    model, df_test: pd.DataFrame, model_name: str
):
    techniques = (
        df_test.loc[df_test["label"] == 1, "attack_technique"]
        .unique()
        .tolist()
    )
    # techniques holds all attack techniques in test set.
    # We iterate over them to compute confusion matrices

    fig, axes = plt.subplots(nrows=1, ncols=len(techniques), figsize=(5*len(techniques), 10))
    for i, technique in enumerate(techniques):
        subset_df_test = pd.concat(
            [
                df_test[df_test["label"] == 0],
                df_test[df_test["attack_technique"] == technique],
            ]
        )
        labels, preds = model.predict(subset_df_test)
        axes[i].set_title(f"{technique} technique")
        ConfusionMatrixDisplay.from_predictions(
            y_true=labels,
            y_pred=preds,
            ax=axes[i],
            colorbar=(i == len(techniques) - 1),
        )
    fp_fig = f"{project_paths.output_path}confusion_matrix_{model_name}.png"
    plt.savefig(fp_fig)


def train_rf_li(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Li-LSyn_RF"
    model = CustomRF_Li(GENERIC=GENERIC, max_depth=None)
    model.train_model(df=df_train, model_name=model_name)
    labels, preds = model.predict(df_test)

    accuracy, f1, precision, recall, fpr = print_and_ret_metrics(
        labels.tolist(),
        preds.tolist(),
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )
    test_model_separating_techniques(
        model=model, df_test=df_test, model_name=model_name
    )


def train_rf_cv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "CountVectorizer_RF"
    model = CustomRF_CountVectorizer(
        GENERIC=GENERIC, max_depth=None, max_features=None
    )
    model.train_model(df=df_train, model_name=model_name)
    labels, preds = model.predict(df_test)
    accuracy, f1, precision, recall, fpr = print_and_ret_metrics(
        labels.tolist(),
        preds.tolist(),
        average=GENERIC.METRICS_AVERAGE_METHOD,
        model=model_name,
    )
    test_model_separating_techniques(
        model=model, df_test=df_test, model_name=model_name
    )


def train_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    logger.info(
        f"Training - number of attacks {len(df_train[df_train['label'] == 1])}"
        f" and number of normals {len(df_train[df_train['label'] == 0])}"
    )
    logger.info(
        f"Testing - number of attacks {len(df_test[df_test['label'] == 1])}"
        f" and number of normals {len(df_test[df_test['label'] == 0])}"
    )
    train_rf_li(df_train, df_test)
    train_rf_cv(df_train=df_train, df_test=df_test)


if __name__ == "__main__":
    init_logging()
    np.random.seed(GENERIC.RANDOM_SEED)
    random.seed(GENERIC.RANDOM_SEED)

    df = pd.read_csv(
        project_paths.dataset_path,
        # dtype is specified to prevent a DtypeWarning
        dtype={
            "full_query": str,
            "label": int,
            "statement_type": str,
            "query_template_id": str,
            "attack_payload": str,
            "attack_id": str,
            "attack_technique": str,
            "attack_desc": str,
            "split": str,
        },
    )
    # df = df.sample(5000)  # This is when testing functions, remove later

    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    print(df_test["query_template_id"].unique().tolist())
    train_models(df_train, df_test)
