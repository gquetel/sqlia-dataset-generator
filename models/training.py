"""Training pipeline for SQL injection detection models.

Entry point: ``python training.py --dataset <path> --models <names>``

Uses the config-driven registry (registry.py) instead of per-model train_*()
functions.  The CLI, logging, data loading, and evaluation orchestration are
unchanged from the original.
"""

import hashlib
import os
from typing import Any, Callable

from sklearn.model_selection import train_test_split

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

from constants import DotDict, ProjectPaths
from cache_utils import hash_df, load_cache, save_cache
from evaluation import compute_all_metrics, get_threshold_for_max_rate
from explain import (
    plot_pr_curves_plt_from_scores,
    plot_roc_curves_plt_from_scores,
)
from registry import (
    MODEL_CONFIGS,
    build_model,
    decision_score_ae,
    decision_score_lodo,
    get_preprocess_fn,
    get_score_fn,
    preprocessing_lodo_ae,
    preprocessing_sklearn,
)

# ------------ Global variables  ------------

GENERIC = DotDict(
    {
        "RANDOM_SEED": 7,
        "BASE_PATH": os.path.join(os.path.dirname(__file__), ""),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)

project_paths = ProjectPaths(GENERIC.BASE_PATH)
logger = logging.getLogger(__name__)
training_results = []
save_model_path = None  # Set via --save-model-path argument
skip_eval = False  # Set via --skip-eval argument

n_jobs = min(
    64, int(os.cpu_count() * 0.8)
)  # We are nice and limit the number of jobs on the cluster.
use_feature_cache = True  # Disable with --no-feature-cache


def init_logging(args):
    lf = TimedRotatingFileHandler(
        project_paths.logs_path + "/training.log",
        when="midnight",
    )
    lf.setLevel(logging.DEBUG)

    lg_lvl = logging.DEBUG if args.debug else logging.INFO
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(lg_lvl)
    lstdo.setFormatter(logging.Formatter(" %(message)s"))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG)
    root.addHandler(lf)
    root.addHandler(lstdo)

    # gaur_sqld uses logging.getLogger(__name__) throughout, so all its messages
    # are under the gaur_sqld.* namespace. Isolate them with a dedicated handler
    # so they can be prefixed and suppressed independently from our own logs.
    gaur_logger = logging.getLogger("gaur_sqld")
    gaur_logger.propagate = False
    gaur_handler = logging.StreamHandler(sys.stdout)
    gaur_handler.setFormatter(logging.Formatter("[gaur_sqld] %(message)s"))
    gaur_handler.setLevel(logging.DEBUG if args.debug else logging.WARNING)
    gaur_logger.setLevel(logging.DEBUG if args.debug else logging.WARNING)
    gaur_logger.addHandler(gaur_handler)


def init_device() -> torch.device:
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    if USE_CUDA:
        logger.info("Using device: %s for experiments.", torch.cuda.get_device_name())
        torch.cuda.set_per_process_memory_fraction(0.99, 0)
    else:
        logger.info("Using CPU for experiments.")
    return device


def init_args() -> argparse.Namespace:
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
        "--n-samples",
        type=int,
        default=None,
        dest="n_samples",
        help="Limit dataset to N samples (deterministic, seed-fixed) before splitting. Overrides --testing's default of 5000. Use for reproducibility checks (e.g. --n-samples 20000).",
    )
    parser.add_argument(
        "--save-model-path",
        type=str,
        dest="save_model_path",
        help="Path to save trained model (without extension). Works with ae_li, ae_securebert, ae_kakisim, ae_kakisim_enriched, and ae_loginov.",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        dest="no_feature_cache",
        help="Disable the feature matrix disk cache (enabled by default).",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        dest="skip_eval",
        help="Skip test-set evaluation after training (only compute val threshold). "
        "Use when evaluate_model.py is run separately.",
    )
    parser.add_argument(
        "--with-shap",
        action="store_true",
        dest="with_shap",
        help="Run SHAP analysis after training (only valid for lodo/in_domain datasets).",
    )

    return parser.parse_args()


def set_global_seed():
    seed = GENERIC.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_state_tag(model) -> str:
    """Return a short hex string capturing the fitted vectorizer vocabulary."""
    extractor = getattr(model, "extractor", None)
    v = (
        getattr(extractor, "vectorizer", None)
        if extractor
        else getattr(model, "vectorizer", None)
    )
    if v is None:
        # Include extractor type in cache key to avoid collisions between different extractors
        if extractor:
            tag = type(extractor).__name__
            # For ablation extractors, include the gaur_features config so lex/synt/sem
            # variants don't collide on the same cache file.
            gaur_features = getattr(extractor, "_gaur_features", None)
            if gaur_features is not None:
                tag += "-" + hashlib.md5(str(gaur_features).encode()).hexdigest()[:6]
            return f"nofeat-{tag}"
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
    """Wrap a preprocessing function with file-based caching."""
    # Kakisim models cache the full feature matrix internally; skip outer wrapping.
    extractor = getattr(model, "extractor", None)
    from extractors.kakisim import KakisimExtractor

    if extractor and isinstance(extractor, KakisimExtractor):
        X, labels, valid_index = preprocess_fn(model, df, use_scaler=use_scaler)
        return X, labels, valid_index

    df_hash = hash_df(df)
    state_tag = _model_state_tag(model)
    model_tag = type(model).__name__
    path = os.path.join(cache_dir, f"{model_tag}-{split}-{df_hash}-{state_tag}.pkl")

    cached = load_cache(path)
    if cached is not None:
        X, labels, valid_index = cached
        if isinstance(X, torch.Tensor) and hasattr(model, "device"):
            X = X.to(model.device)
        return X, labels, valid_index

    X, labels, valid_index = preprocess_fn(model, df, use_scaler=use_scaler)

    save_obj = (X.cpu() if isinstance(X, torch.Tensor) else X, labels, valid_index)
    save_cache(path, save_obj)
    return X, labels, valid_index


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
    X, labels, _ = _cached_preprocess_preds(
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


def preprocess_for_user_inputs_training(df: pd.DataFrame):
    c = len(df)
    df.dropna(subset=["user_inputs"], inplace=True)
    dropped_count = c - len(df)
    logger.info(f"Dropped {dropped_count} samples with no user_input")
    df["full_query"] = df["user_inputs"]


def get_scores_lodo(
    df: pd.DataFrame,
    model,
    preprocess_fn: Callable[[Any, pd.DataFrame], tuple[np.ndarray, np.ndarray]],
    score_fn: Callable[[Any, np.ndarray], np.ndarray],
    batch_size: int | None = None,
    use_scaler: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared scoring loop.

    Returns (labels, scores, valid_index) where valid_index contains the pandas
    index values of rows that were actually scored. Some preprocessors (e.g. gaur)
    may silently drop rows; valid_index lets callers align df metadata with results.
    """
    all_labels, all_scores, all_valid_indices = [], [], []

    if batch_size:
        for start_idx in tqdm(range(0, len(df), batch_size), desc="Scoring batches"):
            end_idx = min(start_idx + batch_size, len(df))
            batch_df = df.iloc[start_idx:end_idx]
            # For good tqdm bars, we need the preprocess_function to be silent.
            result = preprocess_fn(model, batch_df, use_scaler=use_scaler)
            if len(result) == 3:
                X, labels, valid_index = result
            else:
                X, labels = result
                valid_index = batch_df.index
            scores = score_fn(model, X)
            all_labels.extend(labels)
            all_scores.extend(scores)
            all_valid_indices.extend(valid_index)
    else:
        result = preprocess_fn(model, df, use_scaler=use_scaler)
        if len(result) == 3:
            X, all_labels, valid_index = result
            all_valid_indices = list(valid_index)
        else:
            X, all_labels = result
            all_valid_indices = list(df.index)
        all_scores = score_fn(model, X)

    return np.array(all_labels), np.array(all_scores), np.array(all_valid_indices)


def compute_metrics_lodo(
    model,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    model_name: str,
    preprocess_fn: Callable,
    get_decision_scores_fn: Callable,
    use_scaler: bool,
    insider_as_fn: bool = False,
    use_batches: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    batch = 4096 if use_batches else None

    if use_feature_cache:
        cache_dir = project_paths.features_cache_path
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
        if not skip_eval:
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
    else:
        _, s_val, _ = get_scores_lodo(
            df=df_val,
            batch_size=batch,
            model=model,
            use_scaler=use_scaler,
            preprocess_fn=preprocess_fn,
            score_fn=get_decision_scores_fn,
        )
        if not skip_eval:
            l_test, s_test, _ = get_scores_lodo(
                df=df_test,
                batch_size=batch,
                model=model,
                preprocess_fn=preprocess_fn,
                score_fn=get_decision_scores_fn,
                use_scaler=use_scaler,
            )

    threshold = get_threshold_for_max_rate(s_val=s_val)
    num_above_threshold = np.sum(s_val > threshold)
    proportion = num_above_threshold / len(s_val)
    logger.info(
        f"Chosen threshold {threshold}, leads to {num_above_threshold} "
        f"samples ({proportion:.1%}) above threshold"
    )

    if skip_eval:
        return np.array([]), np.array([]), threshold

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


# --------------- Registry-driven training ---------------

AUTHORIZED_GROUPS = {
    "li": ["ocsvm_li", "lof_li", "ae_li"],
    "cv": ["ocsvm_cv", "ae_cv"],
    "securebert": ["ocsvm_securebert", "lof_securebert", "ae_securebert"],
    "securebert2": ["ocsvm_securebert2", "ae_securebert2"],
    "modernbert": ["ocsvm_modernbert", "ae_modernbert"],
    "kakisim_c": ["ocsvm_kakisim_c", "ae_kakisim_c"],
    "loginov": ["ocsvm_loginov", "ae_loginov"],
    "gaur": [
        "ocsvm_gaur",
        "ae_gaur",
        "ocsvm_gaur_chatgpt",
        "ae_gaur_chatgpt",
        "ae_gaur_mistral",
        "ae_li_gaur_chatgpt_sem",
        "ae_li_gaur_mistral_sem",
    ],
    "codebert": ["ae_codebert"],
    "flan_t5": ["ae_flan_t5"],
    "sentbert": ["ae_sentbert"],
    "llm2vec": ["ae_llm2vec"],
    "qwen3_emb": ["ocsvm_qwen3_emb", "ae_qwen3_emb"],
}


def select_models(args) -> list[str]:
    """Return list of model config names to train."""
    if "all" in args.models:
        return list(MODEL_CONFIGS.keys())

    requested = []
    for item in args.models:
        if item in AUTHORIZED_GROUPS:
            requested.extend(AUTHORIZED_GROUPS[item])
        else:
            requested.append(item)

    valid = []
    for name in requested:
        if name in MODEL_CONFIGS:
            valid.append(name)
        else:
            logger.warning(f"Unrecognized model {name}, skipping.")
    return valid


def _train_single_model(
    config_name: str,
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    testing: bool = False,
    with_shap: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Build, train, evaluate, and optionally save one model."""
    global save_model_path
    set_global_seed()

    config = MODEL_CONFIGS[config_name]
    model_name = config.display_name
    logger.info(f"Training model: {model_name}")

    device = init_device()
    cache_dir = project_paths.features_cache_path if use_feature_cache else None

    model = build_model(
        config_name=config_name,
        GENERIC=GENERIC,
        device=device,
        project_paths=project_paths,
        n_jobs=n_jobs,
        cache_dir=cache_dir,
        no_cache=not use_feature_cache,
    )

    model.train_model(
        df=df_train,
        project_paths=project_paths,
        model_name=model_name,
    )

    preprocess_fn = get_preprocess_fn(config_name)
    score_fn = get_score_fn(config_name)

    labels, scores, threshold = compute_metrics_lodo(
        model=model,
        df_test=df_test,
        df_val=df_val,
        model_name=model_name,
        preprocess_fn=preprocess_fn,
        get_decision_scores_fn=score_fn,
        use_scaler=config.use_scaler,
        insider_as_fn=False,
        use_batches=True,
    )

    if save_model_path and config.model_type in ("ae", "ocsvm"):
        model.save_model(save_model_path, threshold=threshold)

    if with_shap:
        from shap_analysis import SHAP_COMPATIBLE_MODELS, run_shap_analysis

        if config_name in SHAP_COMPATIBLE_MODELS:
            run_shap_analysis(
                model=model,
                config_name=config_name,
                df_train=df_train,
                df_test=df_test,
                output_dir=project_paths.output_path,
                testing=testing,
            )

    return labels, scores, threshold


def save_results(args):
    dfres = pd.DataFrame(training_results)
    resdir = project_paths.output_path
    filepath = f"{resdir}/results"
    if args.on_user_inputs:
        filepath += "-on-user-inputs"
    filepath += ".csv"
    dfres.to_csv(filepath, index=False)


def train_models(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_val: pd.DataFrame,
    selected_models: list[str],
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

    models_output = {}
    for config_name in selected_models:
        labels, scores, threshold = _train_single_model(
            config_name,
            df_train,
            df_test,
            df_val,
            testing=args.testing,
            with_shap=args.with_shap,
        )
        models_output[config_name] = (labels, scores)
        if not skip_eval:
            save_results(args=args)

    if skip_eval:
        return

    labels_list = [l for l, _ in models_output.values()]
    scores_list = [s for _, s in models_output.values()]
    names_list = list(models_output.keys())

    ref_labels = labels_list[0]
    for labels in labels_list[1:]:
        if not np.array_equal(ref_labels, labels):
            logger.critical("Label mismatch detected")

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
    n_samples = args.n_samples or (5000 if args.testing else None)
    if n_samples:
        df = df.sample(n_samples, random_state=GENERIC.RANDOM_SEED)

    if args.subfolder:
        project_paths.set_subfolder_output_path(args.subfolder)

    if args.save_model_path:
        save_model_path = args.save_model_path
        if args.testing:
            save_model_path = save_model_path + "_testing"

    if args.no_feature_cache:
        use_feature_cache = False

    if args.skip_eval:
        skip_eval = True

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
