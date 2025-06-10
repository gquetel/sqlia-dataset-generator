"""Definition of ML models configuration."""

import os

# We force device on which training happens.
# device = torch.device("cuda:0" if USE_CUDA else "cpu") is not taken
# into account apparently...
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"

import argparse
from logging.handlers import TimedRotatingFileHandler

import numpy as np
import random
import pandas as pd
import sys
import logging
import scipy
import torch

from U_Li import AutoEncoder_Li, LOF_Li, OCSVM_Li
from U_CountVect import LOF_CV, OCSVM_CV, AutoEncoder_CV
from U_Sentence_BERT import AutoEncoder_SecureBERT, LOF_SecureBERT, OCSVM_SecureBERT
from constants import DotDict, ProjectPaths

from explain import (
    get_recall_per_attack,
    plot_confusion_matrices_by_technique,
    plot_pca,
    plot_pr_curves_plt_from_scores,
    plot_roc_curves_plt_from_scores,
    plot_tree_clf,
    print_and_save_metrics,
)

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


def init_logging(args):
    lf = TimedRotatingFileHandler(
        project_paths.logs_path + "/training.log",
        when="midnight",
    )

    lg_lvl = logging.DEBUG if args.debug else logging.INFO
    lf.setLevel(lg_lvl)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(lg_lvl)

    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=lg_lvl, handlers=[lf, lstdo])


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
        "--debug",
        action="store_true",
        help="Prints more details on about model training",
    )
    args = parser.parse_args()
    return args


# ------------- MODELS TRAINING -------------


# @profile
def compute_metrics(
    model: OCSVM_Li | OCSVM_CV | LOF_CV | LOF_Li,
    df_test: pd.DataFrame,
    model_name: str,
    use_scaler: bool = False,
):
    """Process test set in batches of 20k samples to manage memory usage."""
    batch_size = 20000
    all_labels = []
    all_scores = []

    for start_idx in range(0, len(df_test), batch_size):
        end_idx = min(start_idx + batch_size, len(df_test))
        batch_df = df_test.iloc[start_idx:end_idx]

        # 0 => pped + original columns
        df_pped, labels = model.preprocess_for_preds(df=batch_df, drop_og_columns=False)

        # 1 => Probas (ppeds only)
        df_pped_wout_og_cols = df_pped.drop(batch_df.columns.to_list(), axis=1)

        # Some models use scaler some does not. Because we want to keep column information
        # For some tests, we perform the scaling here after the original columns have been
        # removed.
        if use_scaler:
            df_pped_wout_og_cols = model._scaler.transform(
                df_pped_wout_og_cols.to_numpy()
            )

        dists = model.clf.decision_function(df_pped_wout_og_cols)

        # dists are a distance to the separating hyperplane.
        # Negative distance is an outlier (attack)
        # Positive distance is an inlier (normal)

        # 2 => Process dists so that positive class is > 0 as asked by
        # average_precision_score & roc_auc_score
        scores = -dists

        all_labels.extend(labels)
        all_scores.extend(scores)
        logger.debug(
            f"Processed batch {start_idx//batch_size + 1}/{(len(df_test) + batch_size - 1)//batch_size}"
        )

    return (
        np.array(all_labels),
        np.array(all_scores),  # For AUPRC computation and AUC-ROC
    )


def compute_metrics_ae(
    model: AutoEncoder_CV,
    df_test: pd.DataFrame,
    model_name: str,
):
    """Process test set in batches of 20k samples to manage memory usage."""
    batch_size = 20000
    all_labels = []
    all_scores = []

    for start_idx in range(0, len(df_test), batch_size):
        end_idx = min(start_idx + batch_size, len(df_test))
        batch_df = df_test.iloc[start_idx:end_idx]

        # 0 => pped + original columns
        df_pped, labels = model.preprocess_for_preds(df=batch_df, drop_og_columns=False)

        # 1 => Probas (ppeds only)
        df_pped_wout_og_cols = df_pped.drop(batch_df.columns.to_list(), axis=1)

        tensors = model._dataframe_to_tensor_batched(
            df_pped_wout_og_cols, batch_size=4096
        )
        dists = model.clf.decision_function(tensors, is_tensor=True)

        # dists are a distance to the separating hyperplane.
        # Negative distance is an outlier (attack)
        # Positive distance is an inlier (normal)

        # 2 => Process dists so that positive class is > 0 as asked by
        # average_precision_score & roc_auc_score
        scores = -dists

        # Collect results from this batch
        all_labels.extend(labels)
        all_scores.extend(scores)
        logger.debug(
            f"Processed batch {start_idx//batch_size + 1}/{(len(df_test) + batch_size - 1)//batch_size}"
        )

    return (
        np.array(all_labels),
        np.array(all_scores),  # For AUPRC computation and AUC-ROC
    )


def compute_metrics_sbert(
    model: OCSVM_SecureBERT | LOF_SecureBERT | AutoEncoder_SecureBERT, df_test: pd.DataFrame, model_name: str
):
    """Process test set in batches of 20k samples to manage memory usage."""
    batch_size = 20000
    all_labels = []
    all_scores = []

    for start_idx in range(0, len(df_test), batch_size):
        end_idx = min(start_idx + batch_size, len(df_test))
        batch_df = df_test.iloc[start_idx:end_idx]

        # 0 => pped + original columns
        df_pped = model.preprocess(df=batch_df)
        labels = np.array(df_pped["label"].tolist())

        # 1 => Probas (ppeds only)
        labels_inf, dists = model.get_scores(df_pped)

        # dists are a distance to the separating hyperplane.
        # Negative distance is an outlier (attack)
        # Positive distance is an inlier (normal)
        scores = -dists

        all_labels.extend(labels)
        all_scores.extend(scores)

        logger.debug(
            f"Processed batch {start_idx//batch_size + 1}/{(len(df_test) + batch_size - 1)//batch_size}"
        )

    return (
        np.array(all_labels),
        np.array(all_scores),  # For AUPRC computation and AUC-ROC
    )


def train_ocsvm_cv(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler: bool = False
):
    model_name = "CountVectorizer and OCSVM"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = OCSVM_CV(
        GENERIC=GENERIC,
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        max_iter=1000,
        use_scaler=use_scaler,
    )
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_ocsvm_li(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler: bool = False
):
    model_name = "Li and OCSVM"
    if use_scaler:
        model_name += "-scaler"

    logger.info(f"Training model: {model_name}")
    model = OCSVM_Li(
        GENERIC=GENERIC,
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        max_iter=1000,
        use_scaler=use_scaler,
    )
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(
        model=model, df_test=df_test, model_name=model_name, use_scaler=use_scaler
    )


def train_ocsvm_sbert(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "SBERT and OCSVM"
    logger.info(f"Training model: {model_name}")
    model = OCSVM_SecureBERT(device=init_device(), max_iter=1000)
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics_sbert(model=model, df_test=df_test, model_name=model_name)


def train_lof_cv(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler: bool = False
):
    model_name = "CountVectorizer and LOF"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")

    model = LOF_CV(
        GENERIC=GENERIC, n_jobs=-1, vectorizer_max_features=None, use_scaler=use_scaler
    )
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(model=model, df_test=df_test, model_name=model_name)


def train_lof_li(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler: bool = False
):
    model_name = "Li and LOF"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = LOF_Li(GENERIC=GENERIC, n_jobs=-1, use_scaler=use_scaler)
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(
        model=model, df_test=df_test, model_name=model_name, use_scaler=use_scaler
    )


def train_lof_sbert(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "SBERT and LOF"
    logger.info(f"Training model: {model_name}")
    model = LOF_SecureBERT(device=init_device(), n_jobs=-1)
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_sbert(model, df_test=df_test, model_name=model_name)


# -- Autoencoders --
def train_ae_li(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler: bool = False
):
    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    model_name = "Li and AE"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_Li(
        GENERIC=GENERIC,
        learning_rate=0.005,
        epochs=100,
        batch_size=1024,
        use_scaler=use_scaler,
    )
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_ae(model, df_test=df_test, model_name=model_name)


def train_ae_cv(
    df_train: pd.DataFrame, df_test: pd.DataFrame, use_scaler: bool = False
):
    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    model_name = "CountVectorizer and AE"
    if use_scaler:
        model_name += "-scaler"

    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_CV(
        GENERIC=GENERIC,
        learning_rate=0.001,
        epochs=20,
        batch_size=1024,
        vectorizer_max_features=None,
        use_scaler=use_scaler,
    )
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_ae(model, df_test=df_test, model_name=model_name)


def train_ae_sbert(df_train: pd.DataFrame, df_test: pd.DataFrame):
    model_name = "SBERT and AE"
    # TODO: Verifier pooler_output format and adapt decode accordingly.
    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_SecureBERT(
        device=init_device(), learning_rate=0.001, epochs=100, batch_size=32
    )
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_sbert(model, df_test=df_test, model_name=model_name)


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
    models = {}

    labels, scores = train_ocsvm_li(df_train=df_train, df_test=df_test)
    models["Li and OCSVM"] = (labels, scores)
    labels, scores = train_ocsvm_li(df_train=df_train, df_test=df_test, use_scaler=True)
    models["Li and OCSVM - scaler"] = (labels, scores)

    labels, scores = train_ocsvm_cv(df_train=df_train, df_test=df_test)
    models["CountVectorizer and OCSVM"] = (labels, scores)
    labels, scores = train_ocsvm_cv(df_train=df_train, df_test=df_test, use_scaler=True)
    models["CountVectorizer and OCSVM - scaler"] = (labels, scores)

    labels, scores = train_lof_li(df_train=df_train, df_test=df_test)
    models["Li and LOF"] = (labels, scores)
    labels, scores = train_lof_li(df_train=df_train, df_test=df_test, use_scaler=True)
    models["Li and LOF - scaler"] = (labels, scores)

    labels, scores = train_lof_cv(df_train=df_train, df_test=df_test)
    models["CountVectorizer and LOF "] = (labels, scores)
    labels, scores = train_lof_cv(df_train=df_train, df_test=df_test, use_scaler=True)
    models["CountVectorizer and LOF - scaler"] = (labels, scores)

    labels, scores = train_ae_li(df_train=df_train, df_test=df_test)
    models["Li and AE"] = (labels, scores)
    labels, scores = train_ae_li(df_train=df_train, df_test=df_test, use_scaler=True)
    models["Li and AE - scaler"] = (labels, scores)

    labels, scores = train_ae_cv(df_train=df_train, df_test=df_test)
    models["CountVectorizer and AE (relu)"] = (labels, scores)
    labels, scores = train_ae_cv(df_train=df_train, df_test=df_test, use_scaler=True)
    models["CountVectorizer and AE - scaled (sigmoid)"] = (labels, scores)

    # labels, scores = train_ocsvm_sbert(df_train=df_train, df_test=df_test)
    # models["SBERT and OCSVM"] = (labels, scores)
    # labels, scores = train_lof_sbert(df_train=df_train, df_test=df_test)
    # models["SBERT and LOF"] = (labels, scores)
    # labels, scores = train_ae_sbert(df_train=df_train, df_test=df_test)
    # models["SBERT and AE"] = (labels, scores)

    labels_list = [labels for labels, _ in models.values()]
    scores_list = [scores for _, scores in models.values()]
    names_list = list(models.keys())

    ref_labels = labels_list[0]
    for labels in labels_list[1:]:
        # assert np.array_equal(ref_labels, labels)
        if not np.array_equal(ref_labels, labels):
            logger.critical(f"Label mismatch detected")

    plot_pr_curves_plt_from_scores(
        labels=ref_labels,
        l_scores=scores_list,
        l_model_names=names_list,
        project_paths=project_paths,
    )

    plot_roc_curves_plt_from_scores(
        labels=ref_labels,
        l_scores=scores_list,
        l_model_names=names_list,
        project_paths=project_paths,
    )

    # Finally, save results to csv.
    dfres = pd.DataFrame(training_results)
    dfres.to_csv("output/results.csv", index=False)


if __name__ == "__main__":
    np.random.seed(GENERIC.RANDOM_SEED)
    random.seed(GENERIC.RANDOM_SEED)
    args = init_args()
    init_logging(args)

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
    # df = df.sample(int(len(df) / 2))
    # df.to_csv("../dataset-small.csv", index=False)
    # exit()
    # df = df.sample(100)
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]

    df_train = df_train[df_train["label"] == 0]

    train_cpu_models(df_train, df_test)
