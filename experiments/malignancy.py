"""Assess malignancy of distribution shifts for SQLIA detectors.

For each transfer-learning scenario (ABC→D), determines whether the detected
distribution shift hurts detector performance, following Rabanser et al. (2019) §3.4.

Pipeline:
  1. Load a pre-trained AE model (trained on source datasets, e.g. ABC).
  2. Train a domain classifier (LogisticRegression) on source train features
     (class 0) vs target train features (class 1).
  3. Score every target test sample by P(belongs to target domain).
  4. For each k in TOP_K_PCTS, take the top-k% most target-like test samples
     and evaluate the AE detector's ROC-AUC on that subset.
  5. delta = filtered_roc_auc - baseline_roc_auc; large negative values
     indicate a malignant shift.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict, ProjectPaths
from registry import (
    MODEL_CONFIGS,
    build_model,
    decision_score_ae,
    preprocessing_generic_ae,
)

logger = logging.getLogger(__name__)

CSV_DTYPES = {
    "full_query": str,
    "label": int,
    "statement_type": str,
    "query_template_id": str,
    "attack_payload": str,
    "attack_id": str,
    "attack_technique": str,
    "attack_desc": str,
    "split": str,
    "attack_status": str,
    "attack_stage": str,
}

SCENARIOS = {
    "ABC→D": (["A", "B", "C"], "D"),
    "ABD→C": (["A", "B", "D"], "C"),
    "ACD→B": (["A", "C", "D"], "B"),
    "BCD→A": (["B", "C", "D"], "A"),
}

# Percentages of top target-like samples to evaluate (paper §3.4)
TOP_K_PCTS = [0.01, 0.1, 1, 5, 10, 25]

GENERIC = DotDict(
    {
        "RANDOM_SEED": 2,
        "BASE_PATH": str(REPO_ROOT / "models"),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def load_ae_model(model_type: str, model_dir: str, train_datasets: list[str], device):
    datasets_str = "".join(train_datasets)
    model_path = Path(model_dir) / f"{model_type}_{datasets_str}.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    project_paths = ProjectPaths(GENERIC.BASE_PATH)
    model = build_model(
        config_name=model_type,
        GENERIC=GENERIC,
        device=device,
        project_paths=project_paths,
    )
    model.load_model(str(model_path))
    return model


def get_features(model, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Extract features in the AE's input space (scaled if applicable) as numpy.

    Uses the same preprocessing pipeline as AE scoring so the domain
    classifier sees the same representation the detector operates on.
    """
    X_tensor, labels = preprocessing_generic_ae(model, df)
    return X_tensor.cpu().numpy(), np.asarray(labels)


def train_domain_classifier(
    X_source: np.ndarray, X_target: np.ndarray
) -> LogisticRegression:
    X = np.concatenate([X_source, X_target], axis=0)
    y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
    clf = LogisticRegression(max_iter=1000, random_state=2)
    clf.fit(X, y)
    return clf


def evaluate_malignancy(
    model,
    domain_clf: LogisticRegression,
    df_target_test: pd.DataFrame,
) -> list[dict]:
    """Return per-k AUC rows; empty list if target test has only one class."""
    X_test, labels = get_features(model, df_target_test)

    if len(np.unique(labels)) < 2:
        logger.warning("Target test set has only one class — AUC undefined, skipping.")
        return []

    p_target = domain_clf.predict_proba(X_test)[:, 1]

    X_tensor = torch.FloatTensor(X_test).to(model.device)
    ae_scores = decision_score_ae(model, X_tensor)

    baseline_auc = roc_auc_score(labels, ae_scores)
    n_total = len(labels)

    rows = []
    for k in TOP_K_PCTS:
        n_take = max(1, int(n_total * k / 100))
        top_idx = np.argsort(p_target)[-n_take:]
        filtered_labels = labels[top_idx]
        filtered_scores = ae_scores[top_idx]

        n_attacks = int(filtered_labels.sum())
        if len(np.unique(filtered_labels)) < 2:
            filtered_auc = float("nan")
            delta = float("nan")
        else:
            filtered_auc = roc_auc_score(filtered_labels, filtered_scores)
            delta = filtered_auc - baseline_auc

        rows.append(
            {
                "k_pct": k,
                "n_filtered": n_take,
                "n_attacks_filtered": n_attacks,
                "baseline_roc_auc": baseline_auc,
                "filtered_roc_auc": filtered_auc,
                "delta_roc_auc": delta,
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Assess malignancy of distribution shifts for SQLIA detectors"
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Model type (e.g. ae_li)",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Directory containing trained .pth files (e.g. models/output/models/ae_li_generic/)",
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
        help="Dataset short name (e.g. A) and CSV path (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/experiments/malignancy",
        help="Output directory for CSV results (default: output/experiments/malignancy)",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Sample datasets down to 2000 rows for quick iteration",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    if args.testing:
        for name in datasets:
            datasets[name] = datasets[name].sample(
                n=min(2000, len(datasets[name])), random_state=2
            )
        print("[testing] Datasets sampled to 2000 rows each")

    dataset_names = {name for name, _ in args.dataset}
    active_scenarios = {
        s: (train_ds, test_ds)
        for s, (train_ds, test_ds) in SCENARIOS.items()
        if set(train_ds) | {test_ds} <= dataset_names
    }
    if not active_scenarios:
        print("No complete scenarios available (need all 4 datasets A/B/C/D).")
        return
    print(f"Active scenarios: {list(active_scenarios.keys())}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_rows = []

    for scenario, (train_ds, test_ds) in active_scenarios.items():
        datasets_str = "".join(train_ds)
        print(f"\n=== {scenario} ===")

        print(f"Loading model {args.model}_{datasets_str} ...")
        model = load_ae_model(args.model, args.model_dir, train_ds, device)

        # Source train features for domain classifier (normals only — train split)
        df_source_train = pd.concat(
            [datasets[n][datasets[n]["split"] == "train"] for n in train_ds],
            ignore_index=True,
        )
        print(f"Extracting source train features ({len(df_source_train)} rows) ...")
        X_source, _ = get_features(model, df_source_train)
        print(f"  Source: {X_source.shape}")

        # Target train features for domain classifier (D train, normals only)
        df_target_train = datasets[test_ds][datasets[test_ds]["split"] == "train"]
        print(f"Extracting target train features ({len(df_target_train)} rows) ...")
        X_target_train, _ = get_features(model, df_target_train)
        print(f"  Target train: {X_target_train.shape}")

        # Train domain classifier
        print("Training domain classifier ...")
        domain_clf = train_domain_classifier(X_source, X_target_train)

        # Evaluate malignancy on D test (all labels)
        df_target_test = datasets[test_ds][datasets[test_ds]["split"] == "test"]
        n_attacks = int((df_target_test["label"] == 1).sum())
        print(
            f"Evaluating on target test ({len(df_target_test)} samples, "
            f"{n_attacks} attacks) ..."
        )

        rows = evaluate_malignancy(model, domain_clf, df_target_test)
        for row in rows:
            row["model"] = args.model
            row["scenario"] = scenario
            all_rows.append(row)
            status = (
                "NaN"
                if np.isnan(row["filtered_roc_auc"])
                else f"{row['filtered_roc_auc']:.4f} (delta={row['delta_roc_auc']:+.4f})"
            )
            print(
                f"  k={row['k_pct']:5}%  n={row['n_filtered']:>6}"
                f"  attacks={row['n_attacks_filtered']:>5}"
                f"  baseline={row['baseline_roc_auc']:.4f}  filtered={status}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        print("No results to save.")
        return

    csv_path = output_dir / f"{args.model}_malignancy.csv"
    df_out = pd.DataFrame(all_rows)[
        [
            "model",
            "scenario",
            "k_pct",
            "n_filtered",
            "n_attacks_filtered",
            "baseline_roc_auc",
            "filtered_roc_auc",
            "delta_roc_auc",
        ]
    ]
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
