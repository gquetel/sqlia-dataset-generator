"""Definition of ML models configuration."""

import os

from sklearn.model_selection import train_test_split

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
from scipy import sparse
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
    print_and_save_metrics_from_treshold,
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

n_jobs = min(64, int(os.cpu_count() * 0.8))


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

    parser.add_argument(
        "--on-user-inputs",
        action="store_true",
        help="Train algorithm on user inputs rather than full query",
    )

    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Declare the training of GPU models, change result filename.",
    )
    args = parser.parse_args()
    return args


# ------------- MODELS TRAINING -------------


def preprocess_for_user_inputs_training(df: pd.DataFrame):
    """Preprocess DataFrame for model training on user inputs.
    Args:
        df (pd.DataFrame): _description_
    """
    # remove samples for which user_inputcolumn  is null.
    c = len(df)
    df.dropna(subset=["user_inputs"], inplace=True)
    # Then replace full_query by user_input content
    dropped_count = c - len(df)
    logger.info(f"Dropped {dropped_count} samples with no user_input")
    df["full_query"] = df["user_inputs"]


def get_threshold_for_max_rate(s_val, max_rate=0.001):
    """Compute threshold given a max allowed FPR.

    Args:
        s_val (_type_): _description_
        max_rate (float, optional): _description_. Defaults to 0.001.

    Returns:
        _type_: _description_
    """
    s_val = np.array(s_val)
    percentile = (1 - max_rate) * 100
    return np.percentile(s_val, percentile)


# @profile
def compute_metrics(
    model: OCSVM_Li | OCSVM_CV | LOF_CV | LOF_Li,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    use_scaler: bool = False,
):
    def _get_scores_in_batch(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Return scores and labels for given dataframes, they are computed in batches.
        Args:
            df (pd.DataFrame): _description_
        """
        batch_size = 10000  # 20k is too much for my laptop for CV
        all_labels = []
        all_scores = []

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # 0 => pped + original columns
            df_pped, labels = model.preprocess_for_preds(
                df=batch_df, drop_og_columns=False
            )

            # 1 => Probas (ppeds only)
            df_pped_wout_og_cols = df_pped.drop(batch_df.columns.to_list(), axis=1)

            # Some models use scaler some does not. Because we want to keep column information
            # For some tests, we perform the scaling here after the original columns have been
            # removed.
            df_pped_wout_og_cols = df_pped_wout_og_cols.to_numpy()

            if use_scaler:
                # TODO: better code to prevent back and forth conversions
                if isinstance(model, (LOF_CV, OCSVM_CV)):
                    # If CV -> Convert to sparse before transform
                    f_matrix = sparse.csr_matrix(df_pped_wout_og_cols)
                    f_matrix = model._scaler.transform(f_matrix)
                    dists = model.clf.decision_function(f_matrix)
                else:
                    # Else, directly transform from numpy.
                    df_pped_wout_og_cols = model._scaler.transform(df_pped_wout_og_cols)
                    dists = model.clf.decision_function(df_pped_wout_og_cols)
            else:
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
                f"Processed batch {start_idx//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}"
            )

        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)

        return all_labels, all_scores

    # We compute all scores for test dataset.
    l_test, s_test = _get_scores_in_batch(df=df_test)

    # We compute all scores for val dataset
    _, s_val = _get_scores_in_batch(df=df_val)

    # We infer a treshold given a maximum FPR
    threshold = get_threshold_for_max_rate(s_val=s_val, max_rate=0.001)
    num_above_threshold = np.sum(s_val > threshold)
    proportion = num_above_threshold / len(s_val)
    logger.info(
        f"Chosen threshold {threshold}, leads to {num_above_threshold} samples ({proportion:.1%}) above threshold"
    )

    # We compute metrics data from test scores, their labels and the treshold
    d_res, preds = print_and_save_metrics_from_treshold(
        labels=l_test,
        scores=s_test,
        model_name=model_name,
        threshold=threshold,
        project_paths=project_paths,
    )

    # 4 => Compute recall per technique given preds, and add them to d_res
    # We create an artificial dataframe with attack_techniques, labels and preds
    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"],
            "label": l_test,
            "preds": preds,
        }
    )
    d_res_recall = get_recall_per_attack(df=_df, model_name=model_name)
    # Add keys of recall dict to d_res and save it.
    d_res.update(d_res_recall)
    training_results.append(d_res)

    return l_test, s_test


def compute_metrics_ae(
    model: AutoEncoder_CV | AutoEncoder_Li,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    model_name: str,
):
    def _get_scores_in_batch(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Process test set in batches of 20k samples to manage memory usage."""
        batch_size = 4096
        all_labels = []
        all_scores = []

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

            # 0 => pped + original columns
            df_pped, labels = model.preprocess_for_preds(
                df=batch_df, drop_og_columns=False
            )

            # 1 => Probas (ppeds only)
            df_pped_wout_og_cols = df_pped.drop(batch_df.columns.to_list(), axis=1)

            tensors = model._dataframe_to_tensor_batched(
                df_pped_wout_og_cols, batch_size=4096
            )
            tensors = tensors.to(model.device)
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
                f"Processed batch {start_idx//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}"
            )

        # 3 => Print metrics
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        return all_labels, all_scores

    # We compute all scores for test dataset.
    l_test, s_test = _get_scores_in_batch(df=df_test)
    # We compute all scores for val dataset
    _, s_val = _get_scores_in_batch(df=df_val)
    # We infer a treshold given a maximum FPR
    threshold = get_threshold_for_max_rate(s_val=s_val, max_rate=0.001)
    num_above_threshold = np.sum(s_val > threshold)
    proportion = num_above_threshold / len(s_val)
    logger.info(
        f"Chosen threshold {threshold}, leads to {num_above_threshold} samples ({proportion:.1%}) above threshold"
    )

    # We compute metrics data from test scores, their labels and the treshold
    d_res, preds = print_and_save_metrics_from_treshold(
        labels=l_test,
        scores=s_test,
        model_name=model_name,
        threshold=threshold,
        project_paths=project_paths,
    )

    # 4 => Compute recall per technique given preds, and add them to d_res
    # We create an artificial dataframe with attack_techniques, labels and preds
    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"],
            "label": l_test,
            "preds": preds,
        }
    )
    d_res_recall = get_recall_per_attack(df=_df, model_name=model_name)
    # Add keys of recall dict to d_res and save it.
    d_res.update(d_res_recall)
    training_results.append(d_res)

    return l_test, s_test


def compute_metrics_sbert(
    model: OCSVM_SecureBERT | LOF_SecureBERT | AutoEncoder_SecureBERT,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
):
    def _get_scores_in_batch(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Process test set in batches of 20k samples to manage memory usage."""
        batch_size = 10000
        all_labels = []
        all_scores = []

        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]

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
                f"Processed batch {start_idx//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}"
            )

        # 3 => Print metrics
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        return all_labels, all_scores

    # We compute all scores for test dataset.
    l_test, s_test = _get_scores_in_batch(df=df_test)
    # We compute all scores for val dataset
    _, s_val = _get_scores_in_batch(df=df_val)
    # We infer a treshold given a maximum FPR
    threshold = get_threshold_for_max_rate(s_val=s_val, max_rate=0.001)
    num_above_threshold = np.sum(s_val > threshold)
    proportion = num_above_threshold / len(s_val)
    logger.info(
        f"Chosen threshold {threshold}, leads to {num_above_threshold} samples ({proportion:.1%}) above threshold"
    )

    # We compute metrics data from test scores, their labels and the treshold
    d_res, preds = print_and_save_metrics_from_treshold(
        labels=l_test,
        scores=s_test,
        model_name=model_name,
        threshold=threshold,
        project_paths=project_paths,
    )

    # 4 => Compute recall per technique given preds, and add them to d_res
    # We create an artificial dataframe with attack_techniques, labels and preds
    _df = pd.DataFrame(
        {
            "attack_technique": df_test["attack_technique"],
            "label": l_test,
            "preds": preds,
        }
    )
    d_res_recall = get_recall_per_attack(df=_df, model_name=model_name)
    # Add keys of recall dict to d_res and save it.
    d_res.update(d_res_recall)
    training_results.append(d_res)

    return l_test, s_test


def train_ocsvm_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
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
        max_iter=10000,
        use_scaler=use_scaler,
    )
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        use_scaler=use_scaler,
    )


def train_ocsvm_li(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
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
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        use_scaler=use_scaler,
    )


def train_ocsvm_sbert(
    df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame
):
    model_name = "SBERT and OCSVM"
    logger.info(f"Training model: {model_name}")
    model = OCSVM_SecureBERT(device=init_device(), max_iter=10000, batch_size=1024)
    model.train_model(df=df_train, model_name=model_name, project_paths=project_paths)
    return compute_metrics_sbert(
        model=model, df_val=df_val, df_test=df_test, model_name=model_name
    )


def train_lof_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    model_name = "CountVectorizer and LOF"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")

    model = LOF_CV(
        GENERIC=GENERIC,
        n_jobs=n_jobs,
        vectorizer_max_features=None,
        use_scaler=use_scaler,
    )
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        use_scaler=use_scaler,
    )


def train_lof_li(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    model_name = "Li and LOF"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = LOF_Li(GENERIC=GENERIC, n_jobs=n_jobs, use_scaler=use_scaler)
    model.train_model(
        df=df_train,
        model_name=model_name,
        project_paths=project_paths,
    )
    return compute_metrics(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        use_scaler=use_scaler,
    )


def train_lof_sbert(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
):
    model_name = "SBERT and LOF"
    logger.info(f"Training model: {model_name}")
    model = LOF_SecureBERT(device=init_device(), n_jobs=n_jobs, batch_size=1024)
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_sbert(
        model, df_test=df_test, df_val=df_val, model_name=model_name
    )


# -- Autoencoders --
def train_ae_li(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
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
        device=init_device(),
        learning_rate=0.005,
        epochs=100,
        batch_size=8192,
        use_scaler=use_scaler,
    )
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_ae(
        model, df_test=df_test, df_val=df_val, model_name=model_name
    )


def train_ae_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    model_name = "CountVectorizer and AE"
    if use_scaler:
        model_name += "-scaler"

    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_CV(
        device=init_device(),
        GENERIC=GENERIC,
        learning_rate=0.001,
        epochs=100,
        batch_size=4096,
        # Because a too big AE does not fit GPU Memory we limit the input_dim:
        # We need enough size for both the model and the features 
        vectorizer_max_features=20000,
        use_scaler=use_scaler,
    )
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_ae(
        model, df_test=df_test, df_val=df_val, model_name=model_name
    )


def train_ae_sbert(df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame):
    model_name = "SBERT and AE"
    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_SecureBERT(
        device=init_device(), learning_rate=0.001, epochs=100, batch_size=2048
    )
    model.train_model(df=df_train, project_paths=project_paths, model_name=model_name)
    return compute_metrics_sbert(
        model, df_test=df_test, df_val=df_val, model_name=model_name
    )


def train_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    args,
):
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

    # # We keep this one with scaling, it behaves way better.
    # labels, scores = train_ocsvm_li(
    #     df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=True
    # )
    # models["Li and OCSVM"] = (labels, scores)
    # labels, scores = train_lof_cv(df_train=df_train, df_test=df_test, df_val=df_val)
    # models["CountVectorizer and LOF "] = (labels, scores)

    # # We keep this one without scaler, it has the best results.
    # labels, scores = train_ocsvm_cv(df_train=df_train, df_test=df_test, df_val=df_val)
    # models["CountVectorizer and OCSVM"] = (labels, scores)

    # # We keep this one without scaler, it has the best results.
    # labels, scores = train_lof_li(df_train=df_train, df_test=df_test, df_val=df_val)
    # models["Li and LOF"] = (labels, scores)

    # AE is behaving way better with scaling
    # labels, scores = train_ae_li(
    #     df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=True
    # )
    # models["Li and AE"] = (labels, scores)

    # AE is behaving way better with scaling
    # labels, scores = train_ae_cv(
    #     df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=False
    # )
    # models["CountVectorizer and AE"] = (labels, scores)

    # labels, scores = train_ocsvm_sbert(df_train=df_train, df_test=df_test, df_val=df_val)
    # models["SBERT and OCSVM"] = (labels, scores)

    labels, scores = train_lof_sbert(df_train=df_train, df_test=df_test, df_val=df_val)
    models["SBERT and LOF"] = (labels, scores)

    labels, scores = train_ae_sbert(df_train=df_train, df_test=df_test, df_val=df_val)
    models["SBERT and AE"] = (labels, scores)

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
    filename = "output/results"

    if args.gpu:
        filename += "-gpu"
    if args.on_user_inputs:
        filename += "-on-user-inputs"

    filename += ".csv"
    dfres.to_csv(filename, index=False)


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

    if args.on_user_inputs:
        preprocess_for_user_inputs_training(df=df)

    # df = df.sample(int(len(df) / 10), random_state=GENERIC.RANDOM_SEED)
    _df_train = df[df["split"] == "train"]
    df_train, df_val = train_test_split(
        _df_train,
        test_size=0.1,
        random_state=GENERIC.RANDOM_SEED,
    )
    df_test = df[df["split"] == "test"]

    train_models(df_train, df_test, df_val, args)
