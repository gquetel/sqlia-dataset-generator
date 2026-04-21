"""
Evaluate saved models on test datasets.

Loads a trained model (with saved threshold) and evaluates it on a test dataset,
computing ROC-AUC, PR-AUC, and other metrics.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import random
import torch

# Add models directory to path
SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict, ProjectPaths
from evaluation import compute_all_metrics, get_threshold_for_max_rate
from explain import (
    plot_pr_curves_plt_from_scores,
    plot_roc_curves_plt_from_scores,
)
from registry import (
    MODEL_CONFIGS,
    build_model,
    get_preprocess_fn,
    get_score_fn,
)
from training import get_scores_lodo

logger = logging.getLogger(__name__)

GENERIC = DotDict(
    {
        "RANDOM_SEED": 7,
        "BASE_PATH": str(REPO_ROOT / "models"),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)


def set_global_seed(seed: int = 7):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_device() -> torch.device:
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    if USE_CUDA:
        logger.info("Using device: %s", torch.cuda.get_device_name())
    else:
        logger.info("Using CPU")
    return device


def load_test_data(test_path: str, test_size: int = None) -> pd.DataFrame:
    """Load test data from CSV file."""
    logger.info(f"Loading test data from {test_path}")

    chunks = []
    for chunk in pd.read_csv(test_path, chunksize=500_000, low_memory=False):
        filtered = chunk[chunk["split"] == "test"]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    logger.info(f"Loaded {len(df)} test samples")

    if test_size and len(df) > test_size:
        df = df.sample(n=test_size, random_state=GENERIC.RANDOM_SEED)
        logger.info(f"Sampled down to {len(df)} samples")

    return df


def load_model(
    model_type: str,
    model_path: str,
    device: torch.device,
    embeddings_cache_dir: str = None,
    no_cache: bool = False,
):
    """Load a model from disk using the registry."""
    project_paths = ProjectPaths(GENERIC.BASE_PATH)
    if embeddings_cache_dir:
        Path(embeddings_cache_dir).mkdir(parents=True, exist_ok=True)
        project_paths._embeddings_override = embeddings_cache_dir

    model = build_model(
        config_name=model_type,
        GENERIC=GENERIC,
        device=device,
        project_paths=project_paths,
        cache_dir=project_paths.features_cache_path,
        no_cache=no_cache,
    )
    model.load_model(model_path)

    if model.threshold is None:
        raise ValueError(
            f"Model at {model_path} does not have a saved threshold. "
            "Please retrain the model to save the threshold."
        )

    return model


def evaluate_model(
    df_test: pd.DataFrame,
    labels: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    model_name: str,
) -> tuple[dict, np.ndarray]:
    logger.info(f"Using threshold: {threshold:.6f}")
    return compute_all_metrics(
        df_test=df_test,
        labels=labels,
        scores=scores,
        threshold=threshold,
        model_name=model_name,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved models on test datasets"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the saved model (.pth file)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        required=True,
        choices=[
            "ae_li",
            "ae_cv",
            "ocsvm_li",
            "ocsvm_loginov",
            "ocsvm_gaur",
            "ocsvm_securebert",
            "ocsvm_sentbert",
            "ocsvm_codebert",
            "ae_securebert",
            "ae_securebert2",
            "ae_modernbert",
            "ae_kakisim_c",
            "ae_loginov",
            "ae_roberta",
            "ae_gaur",
            "ae_gaur_chatgpt",
            "ae_gaur_mistral",
            "ae_li_gaur_chatgpt_sem",
            "ae_li_gaur_mistral_sem",
            "ae_li_gaur_lex",
            "ae_li_gaur_synt",
            "ae_li_gaur_sem",
            "ae_codebert",
            "ae_codet5",
            "ae_flan_t5",
            "ae_sentbert",
            "ae_llm2vec",
            "ae_qwen3_emb",
        ],
        help="Type of model",
    )
    parser.add_argument(
        "--test-dataset",
        type=str,
        default=None,
        help="Path to test dataset CSV file (single dataset)",
    )
    parser.add_argument(
        "--test-datasets",
        type=str,
        nargs="+",
        default=None,
        help="Multiple test datasets as path:label pairs "
        "(e.g. bcd-a.csv:A acd-b.csv:B)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Testing mode: limit test set to 500 samples",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        dest="n_samples",
        help="Limit test set to N samples (deterministic). Overrides --testing's default of 500.",
    )
    parser.add_argument(
        "--embeddings-cache",
        type=str,
        default=None,
        help="Directory to cache SecureBERT embeddings",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2,
        help="Random seed",
    )
    parser.add_argument(
        "--fixed-fpr",
        type=float,
        default=None,
        help="Override saved threshold by computing one from test set normal samples "
        "at the given FPR (e.g. 0.01 for 1%% FPR)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-feature-cache",
        action="store_true",
        dest="no_feature_cache",
        help="Disable HuggingFace embeddings disk cache.",
    )
    # Initialization code
    args = parser.parse_args()

    # Validate test dataset arguments
    if args.test_dataset is None and args.test_datasets is None:
        parser.error("one of --test-dataset or --test-datasets is required")
    if args.test_dataset is not None and args.test_datasets is not None:
        parser.error("--test-dataset and --test-datasets are mutually exclusive")

    # Build list of (path, label) pairs
    if args.test_datasets:
        test_items = []
        for item in args.test_datasets:
            if ":" not in item:
                parser.error(
                    f"--test-datasets entries must be path:label, got '{item}'"
                )
            path, label = item.rsplit(":", 1)
            test_items.append((path, label))
    else:
        test_items = [(args.test_dataset, None)]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(message)s",
    )
    # In debug mode, we prefix the gaur_sqld logger by [gaur_sqld].
    # Otherwise, we set the level to WARNING.
    gaur_logger = logging.getLogger("gaur_sqld")
    gaur_logger.propagate = False
    gaur_handler = logging.StreamHandler(sys.stdout)
    gaur_handler.setFormatter(logging.Formatter("[gaur_sqld] %(message)s"))
    gaur_handler.setLevel(log_level)
    gaur_logger.setLevel(log_level if args.debug else logging.WARNING)
    gaur_logger.addHandler(gaur_handler)
    set_global_seed(args.seed)

    device = init_device()
    model_name = Path(args.model_path).stem
    model = load_model(
        model_type=args.model_type,
        model_path=args.model_path,
        device=device,
        embeddings_cache_dir=args.embeddings_cache,
        no_cache=args.no_feature_cache,
    )

    for test_path, test_label in test_items:
        if test_label:
            cur_output_dir = Path(args.output_dir) / f"{model_name}_on_{test_label}"
            logger.info(f"=== Evaluating on {test_label} ===")
        else:
            cur_output_dir = Path(args.output_dir)
        cur_output_dir.mkdir(parents=True, exist_ok=True)

        # Load test data
        test_size = args.n_samples or (500 if args.testing else None)
        df_test = load_test_data(test_path, test_size)

        # Score
        config = MODEL_CONFIGS[args.model_type]
        _, partial_scores, valid_idx = get_scores_lodo(
            df=df_test,
            batch_size=4096,
            model=model,
            preprocess_fn=get_preprocess_fn(args.model_type),
            score_fn=get_score_fn(args.model_type),
            use_scaler=config.use_scaler,
        )
        # Rows dropped by the extractor (e.g. gaur unparseable queries) get score=0,
        # which falls below any threshold and is treated as predicted-normal.
        n_dropped = len(df_test) - len(valid_idx)
        if n_dropped > 0:
            logger.info(
                f"Extractor dropped {n_dropped} rows; assigning score=0 (predicted normal)"
            )
        scores = np.zeros(len(df_test))
        scores[df_test.index.get_indexer(valid_idx)] = partial_scores
        labels = df_test["label"].to_numpy()

        # Determine threshold
        if args.fixed_fpr is not None:
            logger.info(
                f"Computing threshold for fixed FPR={args.fixed_fpr} from test set normal samples"
            )
            normal_scores = scores[labels == 0]
            threshold = get_threshold_for_max_rate(
                normal_scores, max_rate=args.fixed_fpr
            )
            logger.info(f"Fixed-FPR threshold: {threshold:.6f}")
        else:
            threshold = model.threshold

        # Evaluate model and save results
        metrics, preds = evaluate_model(df_test, labels, scores, threshold, model_name)
        results_df = pd.DataFrame([metrics])
        results_path = cur_output_dir / "results.csv"
        results_df.to_csv(results_path, index=False)
        logger.info(f"Saved results to {results_path}")

        eval_paths = DotDict({"output_path": str(cur_output_dir) + "/"})
        plot_roc_curves_plt_from_scores(
            labels=labels,
            l_scores=[scores],
            l_model_names=[model_name],
            project_paths=eval_paths,
        )
        plot_pr_curves_plt_from_scores(
            labels=labels,
            l_scores=[scores],
            l_model_names=[model_name],
            project_paths=eval_paths,
        )


if __name__ == "__main__":
    main()
