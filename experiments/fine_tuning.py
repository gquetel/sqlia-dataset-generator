"""
Fine-Tuning Experiment (Full AE Fine-Tune on Target Domain)

For each LODO model, fine-tune the full autoencoder on k normal samples from
the target domain, then recompute the threshold from those same samples.
Sweep k over a log scale and report balanced accuracy vs. k.

The feature extractor is not retrained (too expensive / no labels needed).
Only the AE weights are updated via continued training on target-domain normals.

Protocol:
  - Load pre-trained LODO model (fails if not found)
  - Pre-extract features for the test set once (extractor is frozen)
  - k=0 baseline: original model + original threshold, no fine-tuning
  - For each k:
    - Restore original AE weights
    - Sample k normal samples from target domain train split
    - Fine-tune AE on those k samples
    - Recompute threshold via get_threshold_for_max_rate on k samples
    - Score test set with fine-tuned AE on pre-extracted features
    - Evaluate
  - Repeat n_runs times with different seeds per k
  - Report mean +- std balanced accuracy per k
"""

import argparse
import copy
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict, ProjectPaths
from evaluation import compute_all_metrics, get_threshold_for_max_rate
from registry import build_model, decision_score_ae, preprocessing_lodo_ae

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


def finetune_ae(model, X_k: torch.Tensor, original_state: dict) -> float:
    """Restore original AE weights, fine-tune on X_k, return new threshold."""
    model.clf.load_state_dict(copy.deepcopy(original_state))
    model.clf.to(model.device)
    model.clf.train()

    criterion = nn.MSELoss().to(model.device)
    optimizer = torch.optim.Adam(model.clf.parameters(), lr=model.learning_rate * 0.1)

    for _ in range(model.epochs):
        for i in range(0, len(X_k), model.batch_size):
            batch = X_k[i : i + model.batch_size].to(model.device)
            optimizer.zero_grad()
            loss = criterion(model.clf(batch), batch)
            loss.backward()
            optimizer.step()

    model.clf.eval()
    s_val = -model.clf.decision_function(X_k, is_tensor=True)
    return get_threshold_for_max_rate(s_val=s_val)


def score_test(model, X_test: torch.Tensor, valid_idx, n_total: int) -> np.ndarray:
    """Run AE forward pass on pre-extracted test features, fill dropped rows with 0."""
    partial_scores = -model.clf.decision_function(X_test, is_tensor=True)
    scores = np.zeros(n_total)
    scores[valid_idx] = partial_scores
    return scores


def extract_metrics(
    metrics: dict, k: int, run: int, seed: int, threshold: float
) -> dict:
    return {
        "k": k,
        "run": run,
        "seed": seed,
        "threshold": threshold,
        "rocauc": float(metrics["rocauc"]),
    }


def run_sweep(
    model,
    df_train_normal: pd.DataFrame,
    df_test: pd.DataFrame,
    k_values: list[int],
    n_runs: int,
) -> pd.DataFrame:
    """Sweep over k values, fine-tune AE, evaluate. Return results DataFrame."""
    # Pre-extract test features once (extractor is frozen throughout)
    logger.info("Pre-extracting test features...")
    X_test_tensors, _, valid_index = preprocessing_lodo_ae(model, df_test)
    n_dropped = len(df_test) - len(X_test_tensors)
    if n_dropped > 0:
        logger.warning("Extractor dropped %d rows; assigning score=0", n_dropped)
    # Map valid pandas index positions to integer positions for np indexing
    valid_pos = np.where(df_test.index.isin(valid_index))[0]
    labels = df_test["label"].to_numpy()

    # Save original AE weights to restore before each run
    original_state = copy.deepcopy(model.clf.state_dict())

    rows = []

    # k=0: original model, no fine-tuning
    logger.info("k=    0 (original threshold=%.6f)", model.threshold)
    scores = score_test(model, X_test_tensors, valid_pos, len(df_test))
    metrics, _ = compute_all_metrics(
        df_test=df_test,
        labels=labels,
        scores=scores,
        threshold=model.threshold,
        model_name="k0_baseline",
    )
    rows.append(extract_metrics(metrics, k=0, run=0, seed=0, threshold=model.threshold))
    logger.info(
        "k=    0  AUROC=%.4f  ← baseline",
        rows[-1]["rocauc"],
    )

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

            # Extract features for k samples (extractor is frozen)
            X_k, _, _ = preprocessing_lodo_ae(model, df_k)

            # Fine-tune AE and get new threshold
            threshold = finetune_ae(model, X_k, original_state)

            # Score test with fine-tuned AE
            scores = score_test(model, X_test_tensors, valid_pos, len(df_test))

            metrics, _ = compute_all_metrics(
                df_test=df_test,
                labels=labels,
                scores=scores,
                threshold=threshold,
                model_name=f"k{k}_run{run}",
            )
            rows.append(
                extract_metrics(metrics, k=k, run=run, seed=seed, threshold=threshold)
            )
            logger.info(
                "k=%5d run=%d  AUROC=%.4f  threshold=%.6f",
                k,
                run,
                rows[-1]["rocauc"],
                threshold,
            )

    # Restore original weights before returning
    model.clf.load_state_dict(original_state)
    model.clf.eval()

    return pd.DataFrame(rows)


def summarise(df_runs: pd.DataFrame) -> pd.DataFrame:
    """Aggregate mean ± std over runs for each k (k=0 baseline kept as-is)."""
    numeric = df_runs.drop(columns=["run", "seed"], errors="ignore")
    agg = numeric.groupby("k").agg(["mean", "std"]).reset_index()
    agg.columns = ["_".join(c).strip("_") for c in agg.columns]
    return agg


def main():
    parser = argparse.ArgumentParser(
        description="AE fine-tuning experiment on target domain"
    )
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
            "ae_flan_t5",
            "ae_sentbert",
            "ae_llm2vec",
            "ae_qwen3_emb",
        ],
    )
    parser.add_argument(
        "--model-path", required=True, help="Path to .pth file of LODO model"
    )
    parser.add_argument(
        "--target-dataset", required=True, help="Path to target domain CSV"
    )
    parser.add_argument("--output-dir", required=True, help="Directory to save results")
    parser.add_argument(
        "--n-runs", type=int, default=5, help="Number of random seeds per k"
    )
    parser.add_argument(
        "--testing", action="store_true", help="Small k values and test subset"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO, format="%(message)s"
    )

    import random

    random.seed(GENERIC.RANDOM_SEED)
    np.random.seed(GENERIC.RANDOM_SEED)
    torch.manual_seed(GENERIC.RANDOM_SEED)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model_type, args.model_path, device)

    test_size = 500 if args.testing else TEST_SIZE
    k_values = [5, 10, 50] if args.testing else K_VALUES

    df_train_normal, df_test = load_target_data(args.target_dataset, test_size)
    df_runs = run_sweep(model, df_train_normal, df_test, k_values, n_runs=args.n_runs)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_runs.to_csv(output_dir / "runs.csv", index=False)
    logger.info("Saved per-run results to %s/runs.csv", output_dir)

    df_summary = summarise(df_runs)
    df_summary.to_csv(output_dir / "summary.csv", index=False)
    logger.info("Saved summary to %s/summary.csv", output_dir)

    print("\nAUROC vs. k (mean ± std over runs):")
    print(f"  {'k':>6}  {'AUROC':>12}")
    for _, row in df_summary.iterrows():
        k = int(row["k"])
        suffix = "  ← baseline (no fine-tune)" if k == 0 else ""
        print(f"  {k:>6}  {row['rocauc_mean']:.4f}±{row['rocauc_std']:.4f}{suffix}")


if __name__ == "__main__":
    main()
