"""Assess malignancy of distribution shifts for SQLIA detectors.

For each transfer-learning scenario (ABC→D), determines whether the detected
distribution shift hurts detector performance, following Rabanser et al. (2019) §3.4.

Pipeline:
  1. Load a pre-trained AE model (trained on source datasets, e.g. ABC).
  2. Train a domain classifier (RandomForest) on source train features
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

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

DATASETS_DIR = Path("~/datasets/100k-training").expanduser()
DATASET_PATHS = {
    "A": DATASETS_DIR / "specialised-OurAirports.csv",
    "B": DATASETS_DIR / "specialised-sakila.csv",
    "C": DATASETS_DIR / "specialised-AdventureWorks.csv",
    "D": DATASETS_DIR / "specialised-OHR.csv",
}

# Percentages of top target-like samples to evaluate (paper §3.4)
TOP_K_PCTS = [0.01, 0.1, 1, 2, 5, 10]

# Cap samples per side (source / target) when training the domain classifier.
N_DOMAIN_CLF_SAMPLES = 15_000

# Chunk size for batched scoring of the target test set.
# Keeps GPU memory bounded regardless of test set size.
EVAL_CHUNK_SIZE = 50_000


GENERIC = DotDict(
    {
        "RANDOM_SEED": 2,
        "BASE_PATH": str(REPO_ROOT / "models"),
        "METRICS_AVERAGE_METHOD": "binary",
    }
)


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def load_ae_model(
    model_type: str,
    model_dir: str,
    train_datasets: list[str],
    device,
    no_cache: bool = False,
):
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
        no_cache=no_cache,
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
) -> RandomForestClassifier:
    X = np.concatenate([X_source, X_target], axis=0)
    y = np.concatenate([np.zeros(len(X_source)), np.ones(len(X_target))])
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2, stratify=y)
    clf = RandomForestClassifier(n_estimators=100, random_state=2, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    print(f"  Binary classifier — val accuracy: {accuracy_score(y_val, y_pred):.4f}")
    cm = confusion_matrix(y_val, y_pred)
    print(f"  Confusion matrix (rows=true, cols=pred) [source, target]:\n{cm}")
    return clf



def evaluate_malignancy(
    model,
    domain_clf: RandomForestClassifier,
    df_target_test: pd.DataFrame,
) -> list[dict]:
    """Return per-k AUC and FPR rows; empty list if target test has only one class.

    Processes the test set in chunks of EVAL_CHUNK_SIZE to keep GPU memory bounded.
    """
    all_labels, all_p_target, all_ae_scores = [], [], []
    for start in range(0, len(df_target_test), EVAL_CHUNK_SIZE):
        chunk = df_target_test.iloc[start : start + EVAL_CHUNK_SIZE]
        X_chunk, chunk_labels = get_features(model, chunk)
        all_labels.append(chunk_labels)
        all_p_target.append(domain_clf.predict_proba(X_chunk)[:, 1])
        X_tensor = torch.FloatTensor(X_chunk).to(model.device)
        all_ae_scores.append(decision_score_ae(model, X_tensor))

    labels = np.concatenate(all_labels)
    p_target = np.concatenate(all_p_target)
    ae_scores = np.concatenate(all_ae_scores)

    if len(np.unique(labels)) < 2:
        logger.warning("Target test set has only one class — AUC undefined, skipping.")
        return []
    threshold = model.threshold

    # Split by class so we rank normals and attacks independently
    mask_normal = labels == 0
    mask_attack = labels == 1
    p_normal = p_target[mask_normal]
    p_attack = p_target[mask_attack]
    scores_normal = ae_scores[mask_normal]
    scores_attack = ae_scores[mask_attack]

    rows = []
    for k in TOP_K_PCTS:
        n_take_normal = max(1, int(mask_normal.sum() * k / 100))
        n_take_attack = max(1, int(mask_attack.sum() * k / 100))

        top_normal_idx = np.argpartition(p_normal, -n_take_normal)[-n_take_normal:]
        top_attack_idx = np.argpartition(p_attack, -n_take_attack)[-n_take_attack:]

        topk_scores_normal = scores_normal[top_normal_idx]
        topk_scores = np.concatenate(
            [topk_scores_normal, scores_attack[top_attack_idx]]
        )
        topk_labels = np.concatenate([np.zeros(n_take_normal), np.ones(n_take_attack)])

        n_filtered = n_take_normal + n_take_attack
        topk_roc_auc = (
            roc_auc_score(topk_labels, topk_scores)
            if len(np.unique(topk_labels)) >= 2
            else float("nan")
        )
        topk_fpr = (topk_scores_normal >= threshold).sum() / n_take_normal

        rows.append(
            {
                "k_pct": k,
                "n_filtered": n_filtered,
                "n_attacks_filtered": n_take_attack,
                "topk_roc_auc": topk_roc_auc,
                "topk_fpr": topk_fpr,
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
        "--output-dir",
        default="output/experiments/malignancy",
        help="Output directory for CSV results (default: output/experiments/malignancy)",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Sample datasets down to 2000 rows for quick iteration",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable HuggingFace embeddings disk cache",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        metavar="TRAIN_DATASETS",
        help="Run only this scenario, identified by its train datasets string (e.g. ABC). Default: all.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in DATASET_PATHS.items():
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(str(path))

    if args.testing:
        for name in datasets:
            datasets[name] = datasets[name].sample(
                n=min(2000, len(datasets[name])), random_state=2
            )
        print("[testing] Datasets sampled to 2000 rows each")

    dataset_names = set(DATASET_PATHS.keys())
    active_scenarios = {
        s: (train_ds, test_ds)
        for s, (train_ds, test_ds) in SCENARIOS.items()
        if set(train_ds) | {test_ds} <= dataset_names
        and (args.scenario is None or "".join(train_ds) == args.scenario)
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
        model = load_ae_model(
            args.model, args.model_dir, train_ds, device, no_cache=args.no_cache
        )

        # Extract per-dataset train features (up to N_DOMAIN_CLF_SAMPLES each)
        all_ds_names = sorted(train_ds + [test_ds])
        ds_to_label = {name: i for i, name in enumerate(all_ds_names)}
        X_per_ds: dict[str, np.ndarray] = {}
        for ds_name in all_ds_names:
            df_ds = datasets[ds_name][datasets[ds_name]["split"] == "train"]
            if len(df_ds) > N_DOMAIN_CLF_SAMPLES:
                df_ds = df_ds.sample(n=N_DOMAIN_CLF_SAMPLES, random_state=2)
            print(f"Extracting features for {ds_name} ({len(df_ds)} rows) ...")
            X_ds, _ = get_features(model, df_ds)
            X_per_ds[ds_name] = X_ds
            print(f"  {ds_name} (class {ds_to_label[ds_name]}): {X_ds.shape}")

        X_source = np.concatenate([X_per_ds[n] for n in train_ds], axis=0)
        X_target_train = X_per_ds[test_ds]

        print("\nTraining domain classifier ...")
        domain_clf = train_domain_classifier(X_source, X_target_train)

        # Evaluate malignancy on full target test set (all labels)
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
            auc_str = (
                "NaN" if np.isnan(row["topk_roc_auc"]) else f"{row['topk_roc_auc']:.4f}"
            )
            print(
                f"  k={row['k_pct']:5}%  n={row['n_filtered']:>6}"
                f"  attacks={row['n_attacks_filtered']:>5}"
                f"  topk_roc_auc={auc_str}"
                f"  topk_fpr={row['topk_fpr']:.4f}"
            )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not all_rows:
        print("No results to save.")
        return

    scenario_suffix = f"_{args.scenario}" if args.scenario else ""
    csv_path = output_dir / f"{args.model}_malignancy{scenario_suffix}.csv"
    df_out = pd.DataFrame(all_rows)[
        [
            "model",
            "scenario",
            "k_pct",
            "n_filtered",
            "n_attacks_filtered",
            "topk_roc_auc",
            "topk_fpr",
        ]
    ]
    df_out.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")


if __name__ == "__main__":
    main()
