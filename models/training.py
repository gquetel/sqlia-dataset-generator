"""Definition of ML models configuration."""

import os

from U_Sentence_BERT import OCSVM_SecureBERT


# We force device on which training happens.
# device = torch.device("cuda:0" if USE_CUDA else "cpu") is not taken
# into account apparently...
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import argparse
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import numpy as np
import random
import pandas as pd
import sys
import logging
import torch

from U_Li import LOF_Li, OCSVM_Li
from U_CountVect import  LOF_CV, OCSVM_CV

from explain import (
    get_recall_per_attack,
    plot_confusion_matrices_by_technique,
    plot_pca,
    plot_pr_curves_plt_from_scores,
    plot_roc_curves_plt_from_scores,
    plot_tree_clf,
    print_and_save_metrics,
)


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

    @property
    def models_path(self) -> str:
        path = f"{self.base_path}/cache/"
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
training_results = []


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


def init_device() -> torch.device:
    """Initialize the device to use for experiments

    Returns:
        torch.device: device to use
    """
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    if USE_CUDA:
        logger.info("Using device: %s for experiments.", torch.cuda.get_device_name())
        torch.cuda.set_per_process_memory_fraction(0.99, 0)
    else:
        logger.critical("Using CPU for experiments.")
    return device


def init_args() -> argparse.Namespace:
    """Argsparse initializing function.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Enable the training and testing of GPU models.",
    )

    args = parser.parse_args()
    return args


# ------------- MODELS TRAINING -------------


def compute_metrics(
    model: OCSVM_Li | OCSVM_CV | LOF_CV | LOF_Li, df_test: pd.DataFrame, model_name: str
):
    # 0 => pped + original columns
    df_pped, labels = model.preprocess_for_preds(df=df_test, drop_og_columns=False)

    # 1 => Probas (ppeds only)
    df_pped_wout_og_cols = df_pped.drop(df_test.columns.to_list(), axis=1)
    dists = model.clf.decision_function(df_pped_wout_og_cols.to_numpy())
    # dists are a distance to the separating hyperplane.
    # Negative distance is an outlier (attack)
    # Positive distance is an inlier (normal)

    # 2 => Process dists so that positive class is > 0 as asked by
    # average_precision_score & roc_auc_score
    scores = -dists

    # For AUC plot
    return (
        labels,
        scores,  # For AUPRC computation and AUC-ROC
    )


def compute_metrics_sbert(
    model: OCSVM_SecureBERT, df_test: pd.DataFrame, model_name: str
):
    # 0 => pped + original columns
    df_pped = model.preprocess()
    labels = np.array(df_pped["labels"].tolist())

    # 1 => Probas (ppeds only)
    scores = model.get_scores(df_pped)


def train_ocsvm_cv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "CountVectorizer_OCSVM"
    model = OCSVM_CV(GENERIC=GENERIC, nu=0.05, kernel="rbf", gamma="scale", max_iter=-1)
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_ocsvm_li(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Li-LSyn_OCSVM"
    model = OCSVM_Li(GENERIC=GENERIC, nu=0.05, kernel="rbf", gamma="scale", max_iter=-1)
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_lof_cv(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "CountVectorizer_LOF"
    model = LOF_CV(
        GENERIC=GENERIC,
        n_jobs=-1,
        vectorizer_max_features=None,
    )
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_lof_li(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "Li-LSyn_LOF"
    model = LOF_Li(
        GENERIC=GENERIC,
        n_jobs=-1,

    )
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_cpu_models(df_train: pd.DataFrame, df_test: pd.DataFrame):
    logger.info(
        f"Training - number of attacks {len(df_train[df_train['label'] == 1])}"
        f" and number of normals {len(df_train[df_train['label'] == 0])}"
    )
    logger.info(
        f"Testing - number of attacks {len(df_test[df_test['label'] == 1])}"
        f" and number of normals {len(df_test[df_test['label'] == 0])}"
    )

    # Train models and get their output.
    l_ocsvm_li, scores_ocsvm_li = train_ocsvm_li(df_train=df_train, df_test=df_test)
    l_ocsvm_cv, scores_ocsvm_cv = train_ocsvm_cv(df_train=df_train, df_test=df_test)
    l_lof_cv, scores_lof_cv = train_lof_cv(df_train=df_train, df_test=df_test)
    l_lof_li, scores_lof_li = train_lof_li(df_train=df_train, df_test=df_test)

    assert np.array_equal(l_ocsvm_li, l_ocsvm_cv)
    assert np.array_equal(l_ocsvm_li, l_lof_cv)
    assert np.array_equal(l_ocsvm_li, l_lof_li)

    # Plot AUCPRC & AUROC for original dataset
    plot_pr_curves_plt_from_scores(
        labels=l_ocsvm_li,
        l_scores=[scores_ocsvm_li, scores_ocsvm_cv, scores_lof_cv, scores_lof_li],
        l_model_names=[
            "Manual Features (Li) and OCSVM",
            "CountVectorizer and OCSVM",
            "CountVectorizer and LOF",
            "Manual Features (Li) and LOF",
        ],
        project_paths=project_paths,
    )

    plot_roc_curves_plt_from_scores(
        labels=l_ocsvm_li,
        l_scores=[scores_ocsvm_li, scores_ocsvm_cv, scores_lof_cv, scores_lof_li],
        l_model_names=[
            "Manual Features (Li) and OCSVM",
            "CountVectorizer and OCSVM",
            "CountVectorizer and LOF",
            "Manual Features (Li) and LOF",
        ],
        project_paths=project_paths,
    )

    # Finally, save results to csv.
    dfres = pd.DataFrame(training_results)
    dfres.to_csv("output/results.csv", index=False)


if __name__ == "__main__":
    init_logging()
    np.random.seed(GENERIC.RANDOM_SEED)
    random.seed(GENERIC.RANDOM_SEED)
    args = init_args()
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

    df_train = df_train[df_train["label"] == 0]

    if args.gpu:
        logger.critical(f"Not implemented yet.")
    else:
        train_cpu_models(df_train, df_test)
