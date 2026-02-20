"""Definition of ML models configuration."""

import hashlib
import os
from typing import Any, Callable

from sklearn.model_selection import train_test_split

# We force device on which training happens.
# device = torch.device("cuda:0" if USE_CUDA else "cpu") is not taken
# into account apparently...
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"

import argparse
from logging.handlers import TimedRotatingFileHandler

import numpy as np
import random
import pandas as pd
import sys
import logging
import torch
from tqdm import tqdm


from U_Li import AutoEncoder_Li, LOF_Li, OCSVM_Li, preprocess_li
from U_CountVect import (
    LOF_CV,
    OCSVM_CV,
    AutoEncoder_CV,
    preprocessing_cv,
)
from U_Sentence_BERT import (
    AutoEncoder_SecureBERT,
    LOF_SecureBERT,
    OCSVM_SecureBERT,
    preprocessing_sbert,
)
from U_Kakisim import (
    AutoEncoder_Kakisim,
    OCSVM_Kakisim,
    preprocessing_kakisim,
)
from constants import DotDict, ProjectPaths

from cache_utils import hash_df, load_cache, save_cache
from evaluation import compute_all_metrics, get_threshold_for_max_rate
from explain import (
    plot_pr_curves_plt_from_scores,
    plot_roc_curves_plt_from_scores,
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
save_model_path = None  # Set via --save-model-path argument

n_jobs = min(64, int(os.cpu_count() * 0.8))
use_feature_cache = True  # Disable with --no-feature-cache


def init_logging(args):
    # the File handler has always debug.
    lf = TimedRotatingFileHandler(
        project_paths.logs_path + "/training.log",
        when="midnight",
    )
    lf.setLevel(logging.DEBUG)

    lg_lvl = logging.DEBUG if args.debug else logging.INFO
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(lg_lvl)
    lstdo.setFormatter(logging.Formatter(" %(message)s"))

    # Configure the root logger so submodule loggers also get captured
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(lf)
    root.addHandler(lstdo)


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
        logger.info("Using CPU for experiments.")
    return device


def init_args() -> argparse.Namespace:
    """Argsparse initializing function.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        dest="dataset",
        required=True,
        help="Filepath to the dataset.",
    )

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
        "--capture-insider",
        action="store_true",
        help="Treat insider attacks as observable (otherwise, they are treated as false negatives)",
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to train (e.g., --models ocsvm_li ae_cv). Use 'all' to run everything.",
    )

    parser.add_argument(
        "--subfolder",
        dest="subfolder",
        help="Save results in output subfolder. Used when computing on multiple nodes to prevent results overwrite.",
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce dataset size to test correct code execution",
    )

    parser.add_argument(
        "--save-model-path",
        type=str,
        dest="save_model_path",
        help="Path to save trained model (without extension). Works with ae_li, ae_sbert, ae_kakisim, and ae_kakisim_enriched.",
    )

    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        dest="no_feature_cache",
        help="Disable the feature matrix disk cache (enabled by default).",
    )

    args = parser.parse_args()
    return args


def set_global_seed():
    seed = GENERIC.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_state_tag(model) -> str:
    """Return a short hex string capturing the fitted vectorizer vocabulary.

    For models with no fitted vectorizer (Li, SBERT) the tag is constant, so
    cached test features are keyed purely on the df content. For model with a
    vocabulary (such as Kakisim), this tag varies. It is returned
    """
    v = getattr(model, "vectorizer", None)
    if v is None:
        return "nofeat"

    if hasattr(v, "_cv_t") and hasattr(v._cv_t, "vocabulary_"):
        vocab = str(sorted(v._cv_t.vocabulary_.items()))
        return hashlib.md5(vocab.encode()).hexdigest()[:8]

    if hasattr(v, "vocabulary_"):
        vocab = str(sorted(v.vocabulary_.items()))
        return hashlib.md5(vocab.encode()).hexdigest()[:8]

    return "nofeat"


def _cached_preprocess_preds(
    preprocess_fn,
    model,
    df: pd.DataFrame,
    cache_dir: str,
    split: str,
    use_scaler: bool = False,
):
    """Wrap a preprocessing function with file-based caching.

    Cache key = model class name + split tag + SHA-256 of the df content,
    consistent with SecureBERT's own embedding cache convention.
    Torch tensors are stored on CPU for portability and moved back to the
    model device on load.

    Note: SecureBERT models have their own embedding cache in embeddings/.
    The outer feature cache (features/) is compatible with it: on first run
    both caches are populated; on subsequent runs this cache is hit first,
    short-circuiting even SecureBERT's inner cache read.
    """
    # Kakisim models cache internally in preprocess_for_preds; skip outer wrapping.
    if getattr(model, "cache_dir", None) is not None:
        return preprocess_fn(model, df, use_scaler=use_scaler)

    df_hash = hash_df(df)
    state_tag = _model_state_tag(model)
    model_tag = type(model).__name__
    path = os.path.join(cache_dir, f"{model_tag}-{split}-{df_hash}-{state_tag}.pkl")

    cached = load_cache(path)
    if cached is not None:
        X, labels = cached
        if isinstance(X, torch.Tensor) and hasattr(model, "device"):
            X = X.to(model.device)
        return X, labels

    X, labels = preprocess_fn(model, df, use_scaler=use_scaler)

    # Always persist tensors on CPU so the cache is device-agnostic.
    save_obj = (X.cpu() if isinstance(X, torch.Tensor) else X, labels)
    save_cache(path, save_obj)
    return X, labels


def get_scores_with_cache(
    df: pd.DataFrame,
    model,
    preprocess_fn,
    score_fn,
    cache_dir: str,
    split: str,
    use_scaler: bool = False,
    batch_size: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess the full DataFrame (with caching), then score in batches."""
    X, labels = _cached_preprocess_preds(
        preprocess_fn=preprocess_fn,
        model=model,
        df=df,
        cache_dir=cache_dir,
        split=split,
        use_scaler=use_scaler,
    )
    labels = np.array(labels)

    if batch_size is None:
        return labels, np.array(score_fn(model, X))

    all_scores = []
    n = len(labels)
    for start in tqdm(range(0, n, batch_size), desc=f"Scoring {split}"):
        end = min(start + batch_size, n)
        all_scores.extend(score_fn(model, X[start:end]))
    return labels, np.array(all_scores)


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


# --------------- Generic Evaluation Functions ---------------
def decision_score_generic(model: OCSVM_Li | LOF_Li | OCSVM_CV | LOF_CV, X: np.ndarray):
    # dists are a distance to the separating hyperplane.
    # Negative distance is an outlier (attack)
    # Positive distance is an inlier (normal)
    dists = model.clf.decision_function(X)

    # Process dists so that positive class is > 0 as asked by
    # average_precision_score & roc_auc_score
    return -dists


def decision_score_ae(model: AutoEncoder_CV | AutoEncoder_Li, X: np.ndarray):
    # dists are a distance to the separating hyperplane.
    # Negative distance is an outlier (attack)
    # Positive distance is an inlier (normal)
    dists = model.clf.decision_function(X, is_tensor=True)

    # Process dists so that positive class is > 0 as asked by
    # average_precision_score & roc_auc_score
    return -dists


def preprocessing_generic_ae(
    model: AutoEncoder_CV | AutoEncoder_Li | AutoEncoder_SecureBERT,
    df: pd.DataFrame,
    use_scaler: bool = False,
) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocess queries from pandas DataFrame, returns tensors and associated labels.

    Args:
        model (AutoEncoder_CV | AutoEncoder_Li | AutoEncoder_SecureBERT ): _description_
        df (pd.DataFrame): _description_
        use_scaler (bool, optional): Ignored, kept for API compatibilty.. Defaults to False.

    Returns:
        tuple[torch.Tensor, np.ndarray]: _description_
    """
    X, labels = model.preprocess_for_preds(df=df)
    # The scaling is dealt with internally in `X_to_tensor`.
    # use_scaler is only kept to fit with how the function is called.
    X_tensors = model.X_to_tensor(X)
    return X_tensors, labels


def get_scores_generic(
    df: pd.DataFrame,
    model,
    preprocess_fn: Callable[[Any, pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    score_fn: Callable[[Any, np.ndarray], np.ndarray],
    batch_size: int | None = None,
    use_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Generic scoring loop."""
    all_labels, all_scores = [], []

    if batch_size:
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Scoring batches"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            X, labels = preprocess_fn(model, batch_df, use_scaler=use_scaler)
            scores = score_fn(model, X)

            all_labels.extend(labels)
            all_scores.extend(scores)

    else:
        X, all_labels = preprocess_fn(model, df, use_scaler=use_scaler)
        all_scores = score_fn(model, X)

    return np.array(all_labels), np.array(all_scores)


def compute_metrics_generic(
    model,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    preprocess_fn: Callable,
    get_decision_scores_fn: Callable,
    use_scaler: bool,
    insider_as_fn: bool = False,
    use_batches: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generic function to evaluate an anomaly detection model by computing prediction scores,
    selecting a decision threshold, and logging evaluation metrics.

    Args:
        model: Trained anomaly detection model.
        df_test (pd.DataFrame): Test dataset containing samples and true labels.
        df_val (pd.DataFrame): Validation dataset used to compute the decision threshold.
        model_name (str): Identifier used for logging and result tracking.
        preprocess_fn (Callable): Function that takes (model, batch_df) and returns
            a tuple (X, labels), where X can be passed to the model for scoring.
        get_decision_scores_fn (Callable): Function that takes (model, X) and returns
            anomaly scores (e.g., negative distances to decision boundary).
        use_scaler (bool): Whether scaling should be used for the features.
        insider_as_fn (bool): False by default. If set to True, the queries with the
            attack_technique "insider" will be considered as False Negative: the data
            collector is not able to collect information for this attack.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - Ground truth labels from the test set.
            - Computed anomaly scores for the test set.
    """
    # Get test and val scores
    batch = 4096 if use_batches else None

    if use_feature_cache:
        cache_dir = project_paths.features_cache_path
        l_test, s_test = get_scores_with_cache(
            df=df_test,
            model=model,
            preprocess_fn=preprocess_fn,
            score_fn=get_decision_scores_fn,
            cache_dir=cache_dir,
            split="test",
            use_scaler=use_scaler,
            batch_size=batch,
        )
        _, s_val = get_scores_with_cache(
            df=df_val,
            model=model,
            preprocess_fn=preprocess_fn,
            score_fn=get_decision_scores_fn,
            cache_dir=cache_dir,
            split="val",
            use_scaler=use_scaler,
            batch_size=4096,
        )
    else:
        l_test, s_test = get_scores_generic(
            df=df_test,
            batch_size=batch,
            model=model,
            preprocess_fn=preprocess_fn,
            score_fn=get_decision_scores_fn,
            use_scaler=use_scaler,
        )
        _, s_val = get_scores_generic(
            df=df_val,
            batch_size=4096,
            model=model,
            use_scaler=use_scaler,
            preprocess_fn=preprocess_fn,
            score_fn=get_decision_scores_fn,
        )

    # Threshold selection
    threshold = get_threshold_for_max_rate(s_val=s_val)
    num_above_threshold = np.sum(s_val > threshold)
    proportion = num_above_threshold / len(s_val)
    logger.info(
        f"Chosen threshold {threshold}, leads to {num_above_threshold} "
        f"samples ({proportion:.1%}) above threshold"
    )

    # Here, set preds where attack_technique = "insider" to min(scores) ->
    # It should be classifier as normal to be a false negative.
    if insider_as_fn:
        insider_mask = df_test["attack_technique"].eq("insider")
        if insider_mask.any():
            min_score = np.min(s_test)
            s_test[insider_mask.values] = min_score
            logger.info(
                f"Set {insider_mask.sum()} 'insider' samples to min score ({min_score}) "
                "to be treated as false negatives."
            )

    d_res, preds = compute_all_metrics(
        df_test=df_test,
        labels=l_test,
        scores=s_test,
        threshold=threshold,
        model_name=model_name,
    )
    training_results.append(d_res)

    return l_test, s_test, threshold


def train_ocsvm_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
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

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_cv,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_ocsvm_li(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
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

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocess_li,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_ocsvm_sbert(
    df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame
):
    set_global_seed()
    model_name = "SecureBERT and OCSVM"
    logger.info(f"Training model: {model_name}")
    model = OCSVM_SecureBERT(
        device=init_device(),
        project_paths=project_paths,
        max_iter=10000,
        batch_size=1024,
    )

    model.train_model(df=df_train, model_name=model_name)

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_sbert,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=False,  # We never use scaler for SecureBERT based models.
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_lof_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
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

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_cv,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_lof_li(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
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

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocess_li,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_lof_sbert(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
):
    set_global_seed()
    model_name = "SBERT and LOF"
    logger.info(f"Training model: {model_name}")
    model = LOF_SecureBERT(
        device=init_device(),
        project_paths=project_paths,
        n_jobs=n_jobs,
        batch_size=1024,
    )
    model.train_model(df=df_train, model_name=model_name)
    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_sbert,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=False,  # We never use scaler for SecureBERT based models.
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


# -- Autoencoders --
def train_ae_li(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    global save_model_path
    set_global_seed()
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

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_generic_ae,
        get_decision_scores_fn=decision_score_ae,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )

    if save_model_path:
        model.save_model(save_model_path, threshold=threshold)

    return labels, scores, threshold


def train_ae_cv(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
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

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_generic_ae,
        get_decision_scores_fn=decision_score_ae,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_ae_sbert(df_train: pd.DataFrame, df_test: pd.DataFrame, df_val: pd.DataFrame):
    global save_model_path
    set_global_seed()
    model_name = "SecureBERT and AE"
    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_SecureBERT(
        device=init_device(),
        project_paths=project_paths,
        learning_rate=0.001,
        epochs=100,
        batch_size=512,
    )

    model.train_model(df=df_train, model_name=model_name)

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_generic_ae,
        get_decision_scores_fn=decision_score_ae,
        use_scaler=False,
        insider_as_fn=False,
        use_batches=True,
    )

    if save_model_path:
        model.save_model(save_path=save_model_path, threshold=threshold)

    return labels, scores, threshold


def train_ocsvm_kakisim(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
    model_name = "Kakisim and OCSVM"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = OCSVM_Kakisim(
        GENERIC=GENERIC,
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        max_iter=10000,
        use_scaler=use_scaler,
    )
    cache_dir = project_paths.features_cache_path if use_feature_cache else None
    model.train_model(df=df_train, model_name=model_name, cache_dir=cache_dir)

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_kakisim,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_ae_kakisim(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    global save_model_path
    set_global_seed()
    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    model_name = "Kakisim and AE"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_Kakisim(
        GENERIC=GENERIC,
        device=init_device(),
        learning_rate=0.001,
        epochs=100,
        batch_size=4096,
        use_scaler=use_scaler,
    )
    cache_dir = project_paths.features_cache_path if use_feature_cache else None
    model.train_model(df=df_train, model_name=model_name, cache_dir=cache_dir)

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_generic_ae,
        get_decision_scores_fn=decision_score_ae,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )

    if save_model_path:
        model.save_model(save_model_path, threshold=threshold)

    return labels, scores, threshold


def train_ocsvm_kakisim_enriched(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    set_global_seed()
    model_name = "Kakisim-E and OCSVM"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = OCSVM_Kakisim(
        GENERIC=GENERIC,
        nu=0.05,
        kernel="rbf",
        gamma="scale",
        max_iter=10000,
        use_scaler=use_scaler,
        views=["E"],
    )
    cache_dir = project_paths.features_cache_path if use_feature_cache else None
    model.train_model(df=df_train, model_name=model_name, cache_dir=cache_dir)

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_kakisim,
        get_decision_scores_fn=decision_score_generic,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )
    return labels, scores, threshold


def train_ae_kakisim_enriched(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    use_scaler: bool = False,
):
    global save_model_path
    set_global_seed()
    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    model_name = "Kakisim-E and AE"
    if use_scaler:
        model_name += "-scaler"
    logger.info(f"Training model: {model_name}")
    model = AutoEncoder_Kakisim(
        GENERIC=GENERIC,
        device=init_device(),
        learning_rate=0.001,
        epochs=100,
        batch_size=4096,
        use_scaler=use_scaler,
        views=["E"],
    )
    cache_dir = project_paths.features_cache_path if use_feature_cache else None
    model.train_model(df=df_train, model_name=model_name, cache_dir=cache_dir)

    labels, scores, threshold = compute_metrics_generic(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocessing_generic_ae,
        get_decision_scores_fn=decision_score_ae,
        use_scaler=use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )

    if save_model_path:
        model.save_model(save_model_path, threshold=threshold)

    return labels, scores, threshold


def save_results(args):
    dfres = pd.DataFrame(training_results)
    resdir = project_paths.output_path
    filepath = f"{resdir}/results"

    if args.on_user_inputs:
        filepath += "-on-user-inputs"

    filepath += ".csv"
    dfres.to_csv(filepath, index=False)


def select_models(args):
    AUTHORIZED_GROUPS = {
        "li": ["ocsvm_li", "lof_li", "ae_li"],
        "cv": ["ocsvm_cv", "lof_cv", "ae_cv"],
        "sbert": ["ocsvm_sbert", "lof_sbert", "ae_sbert"],
        "kakisim": ["ocsvm_kakisim", "ae_kakisim"],
        "kakisim_enriched": ["ocsvm_kakisim_enriched", "ae_kakisim_enriched"],
    }

    MODEL_REGISTRY = {
        "ocsvm_li": lambda df_train, df_test, df_val: train_ocsvm_li(
            df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=True
        ),
        "lof_cv": lambda df_train, df_test, df_val: train_lof_cv(
            df_train=df_train, df_test=df_test, df_val=df_val
        ),
        "ocsvm_cv": lambda df_train, df_test, df_val: train_ocsvm_cv(
            df_train=df_train, df_test=df_test, df_val=df_val
        ),
        "lof_li": lambda df_train, df_test, df_val: train_lof_li(
            df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=True
        ),
        "ae_li": lambda df_train, df_test, df_val: train_ae_li(
            df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=True
        ),
        "ae_cv": lambda df_train, df_test, df_val: train_ae_cv(
            df_train=df_train, df_test=df_test, df_val=df_val, use_scaler=False
        ),
        "ocsvm_sbert": train_ocsvm_sbert,
        "lof_sbert": train_lof_sbert,
        "ae_sbert": train_ae_sbert,
        "ocsvm_kakisim": lambda df_train, df_test, df_val: train_ocsvm_kakisim(
            df_train=df_train, df_test=df_test, df_val=df_val
        ),
        "ae_kakisim": lambda df_train, df_test, df_val: train_ae_kakisim(
            df_train=df_train, df_test=df_test, df_val=df_val
        ),
        "ocsvm_kakisim_enriched": lambda df_train, df_test, df_val: train_ocsvm_kakisim_enriched(
            df_train=df_train, df_test=df_test, df_val=df_val
        ),
        "ae_kakisim_enriched": lambda df_train, df_test, df_val: train_ae_kakisim_enriched(
            df_train=df_train, df_test=df_test, df_val=df_val
        ),
    }

    if "all" in args.models:
        return MODEL_REGISTRY

    requested = []

    # List all requested models (by name or group name)
    for item in args.models:
        if item in AUTHORIZED_GROUPS:
            requested.extend(AUTHORIZED_GROUPS[item])
        else:
            requested.append(item)

    # Now check if names are valid.
    valid = {}
    for model_name in requested:
        if model_name in MODEL_REGISTRY:
            valid[model_name] = MODEL_REGISTRY[model_name]
        else:
            logger.warning(f"Unrecognized model {model_name}, skipping.")
    return valid


def train_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    selected_models: dict,
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
    models_output = {}

    for model_name, model_fn in selected_models.items():
        labels, scores, threshold = model_fn(df_train, df_test, df_val)
        models_output[model_name] = (labels, scores)
        save_results(args=args)

    # consistency checks, curve plotting, etc.
    labels_list = [l for l, _ in models_output.values()]
    scores_list = [s for _, s in models_output.values()]
    names_list = list(models_output.keys())

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


if __name__ == "__main__":
    set_global_seed()
    args = init_args()
    init_logging(args)

    selected_models = select_models(args)
    if len(selected_models) == 0:
        logger.critical("No valid model selected, exiting.")
        exit()

    df = pd.read_csv(
        args.dataset,
        dtype={
            "full_query": str,
            "label": int,
            "user_inputs": str,
            "attack_stage": str,
            "tamper_method": str,
            "attack_status": str,
            "statement_type": str,
            "query_template_id": str,
            "attack_id": str,
            "attack_technique": str,
            "split": str,
        },
    )
    logger.info(f"Training on model: {args.dataset}")
    if args.testing:
        df = df.sample(5000)

    if args.subfolder:
        project_paths.set_subfolder_output_path(args.subfolder)

    if args.save_model_path:
        save_model_path = args.save_model_path

    if args.no_feature_cache:
        use_feature_cache = False

    if args.on_user_inputs:
        preprocess_for_user_inputs_training(df=df)

    _df_train = df[df["split"] == "train"]
    df_train, df_val = train_test_split(
        _df_train,
        test_size=0.1,
        random_state=GENERIC.RANDOM_SEED,
    )
    df_test = df[df["split"] == "test"]

    train_models(df_train, df_test, df_val, selected_models, args)
