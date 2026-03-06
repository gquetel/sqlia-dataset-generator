"""Measure domain leakage in SQL injection feature extractors via Decision Tree classification.

Trains a Decision Tree to classify which dataset (A/B/C/D) a query originates from,
using only its extracted features. High accuracy indicates the extractor retains
domain-specific information (table/column names, vocabulary), which hurts transfer learning.

Expected outcome: structural extractors (Li, GAUR, Loginov) produce domain-agnostic
features → DT struggles. Semantic extractors (SecureBERT, Flan-T5, RoBERTa) encode
vocabulary → DT succeeds.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from scipy import sparse
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_graphviz

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict
from registry import MODEL_CONFIGS, _make_extractor

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


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, list):
        return np.stack(X)
    return np.asarray(X)


def build_extractor(extractor_name: str, device, embeddings_path: str, cache_dir: str):
    config = MODEL_CONFIGS[extractor_name]
    project_paths = DotDict({"embeddings_path": embeddings_path})
    return _make_extractor(
        config, device=device, project_paths=project_paths, cache_dir=cache_dir
    )


def fit_extractor(extractor, df_train: pd.DataFrame):
    """Prepare extractor on training data (same logic as feature_space_viz.py)."""
    extractor.prepare_for_training(df_train)
    if hasattr(extractor, "_fitted"):
        extractor.extract_features(df_train)


def sample_split(
    df: pd.DataFrame, split: str, n: int, normal_only: bool
) -> pd.DataFrame:
    """Sample up to n rows from the given split, optionally filtering to normal only."""
    subset = df[df["split"] == split]
    if normal_only:
        subset = subset[subset["label"] == 0]
    return subset.sample(n=min(n, len(subset)), random_state=42)


def main():
    parser = argparse.ArgumentParser(
        description="Measure domain leakage via Decision Tree dataset classification"
    )
    parser.add_argument(
        "--extractor",
        required=True,
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Feature extractor(s) to evaluate",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME_OR_PATH",
        required=True,
        help="Alternating NAME PATH pairs, e.g. A /path/a.csv B /path/b.csv",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=10000,
        help="Per-dataset samples from train split (default: 10000)",
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=1000,
        help="Per-dataset samples from test split (default: 1000)",
    )
    parser.add_argument(
        "--normal-only",
        action="store_true",
        help="Only use normal (label=0) queries",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiments/domain_leakage",
        help="Output directory for results CSV",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce sample counts to 50 per dataset for quick iteration",
    )
    args = parser.parse_args()

    tokens = args.dataset
    if len(tokens) % 2 != 0:
        parser.error("--dataset requires an even number of arguments (NAME PATH pairs)")
    args.dataset = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

    if args.testing:
        args.train_samples = 50
        args.test_samples = 50
        print(
            f"[testing] Reduced samples to {args.train_samples} train / {args.test_samples} test"
        )

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = str(output_dir / "embeddings")
    Path(embeddings_path).mkdir(parents=True, exist_ok=True)
    cache_dir = str(output_dir / "feature_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load datasets
    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    domain_names = [name for name, _ in args.dataset]

    # Sample train/test splits per domain and assign domain labels
    train_frames, test_frames = [], []
    for name, df in datasets.items():
        train_sample = sample_split(df, "train", args.train_samples, args.normal_only)
        test_sample = sample_split(df, "test", args.test_samples, args.normal_only)
        train_sample = train_sample.copy()
        test_sample = test_sample.copy()
        train_sample["_domain"] = name
        test_sample["_domain"] = name
        train_frames.append(train_sample)
        test_frames.append(test_sample)
        print(f"  {name}: {len(train_sample)} train, {len(test_sample)} test")

    df_train = pd.concat(train_frames, ignore_index=True)
    df_test = pd.concat(test_frames, ignore_index=True)

    y_train = df_train["_domain"].to_numpy()
    y_test = df_test["_domain"].to_numpy()

    # Combined train for fitting stateful extractors
    df_train_fit = df_train.drop(columns=["_domain"])
    df_test_feat = df_test.drop(columns=["_domain"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results = []
    for extractor_name in args.extractor:
        print(f"\n=== Extractor: {extractor_name} ===")
        extractor = build_extractor(
            extractor_name,
            device=device,
            embeddings_path=embeddings_path,
            cache_dir=cache_dir,
        )

        print("Fitting extractor on training data ...")
        fit_extractor(extractor, df_train_fit)

        print("Extracting training features ...")
        X_train = to_dense(extractor.extract_features(df_train_fit))
        print(f"  X_train shape: {X_train.shape}")

        print("Extracting test features ...")
        X_test = to_dense(extractor.extract_features(df_test_feat))
        print(f"  X_test shape: {X_test.shape}")

        print("Training Decision Tree ...")
        dt = DecisionTreeClassifier(random_state=2)
        dt.fit(X_train, y_train)

        y_pred = dt.predict(X_test)
        report = classification_report(
            y_test, y_pred, labels=domain_names, output_dict=True
        )

        accuracy = report["accuracy"]
        macro_f1 = report["macro avg"]["f1-score"]
        print(f"  Accuracy: {accuracy:.4f}  Macro-F1: {macro_f1:.4f}")

        cm = confusion_matrix(y_test, y_pred, labels=domain_names)
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=domain_names,
                y=domain_names,
                colorscale="Blues",
                text=cm,
                texttemplate="%{text}",
                showscale=True,
            )
        )
        fig.update_layout(
            title=f"Confusion Matrix — {extractor_name} (accuracy={accuracy:.3f})",
            xaxis_title="Predicted",
            yaxis_title="True",
            yaxis_autorange="reversed",
            width=500,
            height=450,
        )
        cm_path = output_dir / f"{extractor_name}_confusion_matrix.svg"
        fig.write_image(str(cm_path))
        print(f"  Saved {cm_path}")

        dot_path = output_dir / f"{extractor_name}_decision_tree.dot"
        export_graphviz(
            dt,
            out_file=str(dot_path),
            class_names=domain_names,
            filled=True,
            rounded=True,
            impurity=False,
        )
        print(f"  Saved {dot_path}")

        row = {"extractor": extractor_name, "accuracy": accuracy, "macro_f1": macro_f1}
        for domain in domain_names:
            metrics = report.get(domain, {})
            row[f"f1_{domain}"] = metrics.get("f1-score", float("nan"))
            row[f"precision_{domain}"] = metrics.get("precision", float("nan"))
            row[f"recall_{domain}"] = metrics.get("recall", float("nan"))
        results.append(row)

    output_path = output_dir / "domain_leakage_results.csv"
    pd.DataFrame(results).to_csv(output_path, index=False)
    print(f"\nSaved {output_path}")


if __name__ == "__main__":
    main()
