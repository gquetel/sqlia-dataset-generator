"""
Threshold Re-Fitting Experiment (Linear Probe / Head-Only Adaptation)

For each generic model, freeze the encoder and re-fit the anomaly threshold
using k normal samples from the target domain. Sweep k over a log scale and
report AUROC vs. k to find how few samples are needed to match specialised
model performance.

Protocol:
  - Load pre-trained generic model (fails if not found)
  - Sample k normal samples from target domain train split
  - Score those samples with the frozen model → use as s_val
  - Recompute threshold via get_threshold_for_max_rate(s_val)
  - Evaluate on up to 50k test samples from target domain
  - Repeat n_runs times with different seeds for each k
  - Report mean ± std AUROC per k

Usage:
    python experiments/threshold_refitting.py \\
        --model-type ae_li \\
        --model-path output/checkpoints/ae_li_generic/ae_li_BCD.pth \\
        --target-dataset ~/datasets/100k-training/generic-OurAirports.csv \\
        --output-dir output/results/threshold_refitting/ae_li_BCD_on_A

    # Testing mode
    python experiments/threshold_refitting.py \\
        --model-type ae_li \\
        --model-path output/checkpoints/ae_li_generic/ae_li_BCD.pth \\
        --target-dataset ~/datasets/100k-training/generic-OurAirports.csv \\
        --output-dir /tmp/out/threshold_refitting \\
        --testing
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict, ProjectPaths
from evaluation import compute_all_metrics, get_threshold_for_max_rate
from registry import build_model, decision_score_ae, preprocessing_generic_ae
from training import get_scores_generic

logger = logging.getLogger(__name__)

GENERIC = DotDict(
    {
        "RANDOM_SEED": 2,
        "BASE_PATH": str(REPO_ROOT / "models"),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)

K_VALUES = [5, 10, 50, 100, 500, 1000, 10000]
TEST_SIZE = 50_000


def load_model(model_type: str, model_path: str, device):
    project_paths = ProjectPaths(GENERIC.BASE_PATH)
    model = build_model(
        config_name=model_type,
        GENERIC=GENERIC,
        device=device,
        project_paths=project_paths,
        cache_dir=project_paths.features_cache_path,
        no_cache=False,
    )
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    model.load_model(str(path))
    logger.info("Loaded model from %s", model_path)
    return model


def load_target_data(
    dataset_path: str, test_size: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (df_train_normal, df_test) from the target dataset CSV."""
    logger.info("Loading target dataset from %s", dataset_path)
    train_chunks, test_chunks = [], []
    for chunk in pd.read_csv(dataset_path, chunksize=500_000, low_memory=False):
        train_part = chunk[(chunk["split"] == "train") & (chunk["label"] == 0)]
        test_part = chunk[chunk["split"] == "test"]
        if len(train_part):
            train_chunks.append(train_part)
        if len(test_part):
            test_chunks.append(test_part)

    df_train_normal = pd.concat(train_chunks, ignore_index=True)
    df_test = pd.concat(test_chunks, ignore_index=True)

    if len(df_test) > test_size:
        df_test = df_test.sample(n=test_size, random_state=GENERIC.RANDOM_SEED)

    logger.info(
        "Train normal samples: %d | Test samples: %d (attacks: %d)",
        len(df_train_normal),
        len(df_test),
        (df_test["label"] == 1).sum(),
    )
    return df_train_normal, df_test


def refit_threshold(model, df_k: pd.DataFrame) -> float:
    """Score k normal samples with frozen model and compute new threshold."""
    _, s_val, _ = get_scores_generic(
        df=df_k,
        batch_size=4096,
        model=model,
        preprocess_fn=preprocessing_generic_ae,
        score_fn=decision_score_ae,
        use_scaler=False,
    )
    return get_threshold_for_max_rate(s_val=s_val)


def run_sweep(
    model,
    df_train_normal: pd.DataFrame,
    df_test: pd.DataFrame,
    k_values: list[int],
    n_runs: int,
) -> pd.DataFrame:
    """Sweep over k values, re-fit threshold, evaluate. Return results DataFrame."""
    # Score the test set once (model is frozen)
    _, partial_scores, valid_idx = get_scores_generic(
        df=df_test,
        batch_size=4096,
        model=model,
        preprocess_fn=preprocessing_generic_ae,
        score_fn=decision_score_ae,
        use_scaler=False,
    )
    n_dropped = len(df_test) - len(valid_idx)
    if n_dropped > 0:
        logger.warning(
            f"Extractor dropped {n_dropped} rows; assigning score=0 (predicted normal)"
        )
    scores = np.zeros(len(df_test))
    scores[df_test.index.get_indexer(valid_idx)] = partial_scores
    labels = df_test["label"].to_numpy()

    rows = []
    for k in k_values:
        if k > len(df_train_normal):
            logger.warning(
                "k=%d exceeds available train normal samples (%d), skipping",
                k,
                len(df_train_normal),
            )
            continue

        for run in range(n_runs):
            seed = GENERIC.RANDOM_SEED + run
            df_k = df_train_normal.sample(n=k, random_state=seed)
            threshold = refit_threshold(model, df_k)
            metrics, _ = compute_all_metrics(
                df_test=df_test,
                labels=labels,
                scores=scores,
                threshold=threshold,
                model_name=f"k{k}_run{run}",
            )
            rows.append(
                {
                    "k": k,
                    "run": run,
                    "seed": seed,
                    "rocauc": float(metrics["rocauc"]),
                    "auprc": float(metrics["auprc"]),
                }
            )
            logger.info(
                "k=%5d run=%d  AUROC=%s  threshold=%.6f",
                k,
                run,
                metrics["rocauc"],  # already a formatted string e.g. "0.5904"
                threshold,
            )

    return pd.DataFrame(rows)


def summarise(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean ± std over runs for each k."""
    numeric = df_runs.select_dtypes(include="number").drop(
        columns=["run", "seed"], errors="ignore"
    )
    numeric["k"] = df_runs["k"]
    agg = numeric.groupby("k").agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


def main():
    parser = argparse.ArgumentParser(description="Threshold re-fitting experiment")
    parser.add_argument(
        "--model-type",
        required=True,
        choices=[
            "ae_li",
            "ae_cv",
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
            "ae_unixcoder",
            "ae_flan_t5",
            "ae_sentbert",
            "ae_llm2vec",
            "ae_qwen3_emb",
        ],
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to .pth file of generic model"
    )
    parser.add_argument(
        "--target-dataset", required=True, help="Path to target domain CSV"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of random seeds per k"
    )
    parser.add_argument(
        "--testing", action="store_true", help="Use small k values and test subset"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s"
    )

    import random
    import torch

    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    model = load_model(args.model_type, args.model_path, device)

    test_size = 500 if args.testing else TEST_SIZE
    k_values = [5, 10, 50] if args.testing else K_VALUES

    df_train_normal, df_test = load_target_data(args.target_dataset, test_size)

    df_runs = run_sweep(model, df_train_normal, df_test, k_values, n_runs=args.n_runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    runs_path = output_dir / "runs.csv"
    df_runs.to_csv(runs_path, index=False)
    logger.info("Saved per-run results to %s", runs_path)

    df_summary = summarise(df_runs)
    summary_path = output_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    logger.info("Saved summary to %s", summary_path)

    # Print AUROC summary
    print("\nAUROC vs. k (mean ± std):")
    for _, row in df_summary.iterrows():
        print(
            f"  k={int(row['k']):>6}  {row['rocauc_mean']:.4f} ± {row['rocauc_std']:.4f}"
        )


if __name__ == "__main__":
    main()
