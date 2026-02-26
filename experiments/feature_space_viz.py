"""Visualize SQL query feature spaces across multiple datasets.

For a given extractor, projects queries from multiple datasets into 2D using PCA
and/or t-SNE. Points are colored by dataset and shaped by label (normal vs attack).
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict
from registry import MODEL_CONFIGS, _make_extractor

logger = logging.getLogger(__name__)

EXTRACTOR_CONFIG_MAP = {
    "li": "ae_li",
    "countvect": "ae_cv",
    "sbert": "ae_sbert",
    "roberta": "ae_roberta",
    "kakisim": "ae_kakisim_c",
    "kakisim_w2v": "ae_kakisim_w2v",
    "loginov": "ae_loginov",
    "bilstm_w2v": "ae_bilstm_w2v",
}

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

DATASET_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def sample_test_data(df: pd.DataFrame, name: str, n: int) -> pd.DataFrame:
    """Sample up to n normal and n attack rows from the test split."""
    test_df = df[df["split"] == "test"]
    normal = test_df[test_df["label"] == 0]
    attack = test_df[test_df["label"] == 1]
    sampled_normal = normal.sample(n=min(n, len(normal)), random_state=42)
    sampled_attack = attack.sample(n=min(n, len(attack)), random_state=42)
    result = pd.concat([sampled_normal, sampled_attack]).copy()
    result["dataset_name"] = name
    return result


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, list):
        return np.stack(X)
    return np.asarray(X)


def build_extractor(extractor_name: str, device, embeddings_path: str, cache_dir: str):
    config = MODEL_CONFIGS[EXTRACTOR_CONFIG_MAP[extractor_name]]
    project_paths = DotDict({"embeddings_path": embeddings_path})
    return _make_extractor(
        config, device=device, project_paths=project_paths, cache_dir=cache_dir
    )


def fit_extractor(extractor, df_train: pd.DataFrame):
    """Prepare extractor on training data.

    prepare_for_training handles state-learning (e.g. Loginov valid_schars).
    For vectorizer-based extractors (CountVect, Kakisim, BiLSTM-W2V), the first
    extract_features call triggers fit_transform internally.
    """
    extractor.prepare_for_training(df_train)
    if hasattr(extractor, "_fitted"):
        extractor.extract_features(df_train)  # triggers fit_transform, result discarded


def plot_reduction(
    coords: np.ndarray,
    dataset_names: np.ndarray,
    labels: np.ndarray,
    unique_datasets: list,
    extractor_name: str,
    reduction_name: str,
    output_dir: Path,
):
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, ds_name in enumerate(unique_datasets):
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        mask_ds = dataset_names == ds_name

        mask_normal = mask_ds & (labels == 0)
        if mask_normal.any():
            ax.scatter(
                coords[mask_normal, 0],
                coords[mask_normal, 1],
                c=color,
                marker="o",
                alpha=0.5,
                s=15,
                label=f"{ds_name} - normal",
            )

        mask_attack = mask_ds & (labels == 1)
        if mask_attack.any():
            ax.scatter(
                coords[mask_attack, 0],
                coords[mask_attack, 1],
                c=color,
                marker="x",
                alpha=0.7,
                s=20,
                label=f"{ds_name} - attack",
            )

    ax.set_title(f"{reduction_name} — {extractor_name} extractor (n={len(coords)})")
    ax.set_xlabel(f"{reduction_name} Component 1")
    ax.set_ylabel(f"{reduction_name} Component 2")
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=2, fontsize=8)
    fig.tight_layout()

    fname = output_dir / f"{reduction_name.lower()}_{extractor_name}.png"
    fig.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature space of SQL injection detection datasets"
    )
    parser.add_argument(
        "--extractor",
        required=True,
        choices=list(EXTRACTOR_CONFIG_MAP.keys()),
        help="Feature extractor to use",
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
        help="Dataset label and CSV path (repeatable)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of samples per label per dataset (default: 1000)",
    )
    parser.add_argument("--pca", action="store_true", help="Compute PCA projection")
    parser.add_argument("--tsne", action="store_true", help="Compute t-SNE projection")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/feature_space",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce sample count to 50 for quick iteration",
    )
    args = parser.parse_args()

    if not args.pca and not args.tsne:
        parser.error("At least one of --pca or --tsne must be specified")

    if args.testing:
        args.samples = 50
        print(f"[testing] Reduced samples to {args.samples}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = str(output_dir / "embeddings")
    Path(embeddings_path).mkdir(parents=True, exist_ok=True)
    cache_dir = str(output_dir / "feature_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    datasets = []
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        df = load_dataset(path)
        datasets.append((name, df))

    # Combined train split for fitting stateful extractors
    df_train_combined = pd.concat(
        [df[df["split"] == "train"] for _, df in datasets], ignore_index=True
    )
    print(f"Training set size: {len(df_train_combined)}")

    # Sample test data for visualization
    df_sampled_list = []
    for name, df in datasets:
        sampled = sample_test_data(df, name, args.samples)
        n_normal = (sampled["label"] == 0).sum()
        n_attack = (sampled["label"] == 1).sum()
        print(
            f"  {name}: {len(sampled)} samples ({n_normal} normal, {n_attack} attack)"
        )
        df_sampled_list.append(sampled)
    df_sampled = pd.concat(df_sampled_list, ignore_index=True)

    dataset_names = df_sampled["dataset_name"].to_numpy()
    labels = df_sampled["label"].to_numpy()
    unique_datasets = [name for name, _ in args.dataset]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    extractor = build_extractor(
        args.extractor,
        device=device,
        embeddings_path=embeddings_path,
        cache_dir=cache_dir,
    )

    print("Fitting extractor on training data ...")
    fit_extractor(extractor, df_train_combined)

    print("Extracting features from test samples ...")
    X_raw = extractor.extract_features(df_sampled)
    X = to_dense(X_raw)
    print(f"Feature matrix shape: {X.shape}")

    if args.pca:
        print("Computing PCA ...")
        coords_pca = PCA(n_components=2).fit_transform(X)
        plot_reduction(
            coords_pca,
            dataset_names,
            labels,
            unique_datasets,
            args.extractor,
            "pca",
            output_dir,
        )

    if args.tsne:
        print("Computing t-SNE ...")
        n = len(X)
        coords_tsne = TSNE(
            n_components=2,
            perplexity=min(50, n - 1),
            random_state=42,
            n_jobs=-1,
        ).fit_transform(X)
        plot_reduction(
            coords_tsne,
            dataset_names,
            labels,
            unique_datasets,
            args.extractor,
            "tsne",
            output_dir,
        )


if __name__ == "__main__":
    main()
