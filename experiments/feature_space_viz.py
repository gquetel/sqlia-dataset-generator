"""Visualize SQL query feature spaces across multiple datasets.

For a given extractor, projects queries from multiple datasets into 2D using PCA
and/or t-SNE. Points are colored by dataset and shaped by label (normal vs attack).
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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

TSNE_PERPLEXITIES = [5, 10, 20, 40, 50, 100]

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
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def sample_test_data(df: pd.DataFrame, name: str, n: int) -> pd.DataFrame:
    """Sample up to n normal and n attack rows from the test split."""
    test_df = df[df["split"] == "test"]
    normal = test_df[test_df["label"] == 0]
    attack = test_df[test_df["label"] == 1]
    sampled_normal = normal.sample(n=min(n, len(normal)), random_state=2)
    sampled_attack = attack.sample(n=min(n, len(attack)), random_state=2)
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
    config = MODEL_CONFIGS[extractor_name]
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


def axis_range(
    coords: np.ndarray, sigma: float | None
) -> tuple[list | None, list | None]:
    """Return [min, max] axis ranges clipped to mean ± sigma·std, or None for auto."""
    if sigma is None:
        return None, None
    x_mean, x_std = coords[:, 0].mean(), coords[:, 0].std()
    y_mean, y_std = coords[:, 1].mean(), coords[:, 1].std()
    return (
        [x_mean - sigma * x_std, x_mean + sigma * x_std],
        [y_mean - sigma * y_std, y_mean + sigma * y_std],
    )


def plot_reduction(
    coords: np.ndarray,
    dataset_names: np.ndarray,
    labels: np.ndarray,
    unique_datasets: list,
    extractor_name: str,
    reduction_name: str,
    output_dir: Path,
    sigma: float | None = None,
):
    fig = go.Figure()

    for i, ds_name in enumerate(unique_datasets):
        color = DATASET_COLORS[i % len(DATASET_COLORS)]
        mask_ds = dataset_names == ds_name

        mask_normal = mask_ds & (labels == 0)
        if mask_normal.any():
            fig.add_trace(
                go.Scatter(
                    x=coords[mask_normal, 0],
                    y=coords[mask_normal, 1],
                    mode="markers",
                    marker=dict(symbol="circle", color=color, size=5, opacity=0.5),
                    name=f"{ds_name} - normal",
                    legendgroup=ds_name,
                )
            )

        mask_attack = mask_ds & (labels == 1)
        if mask_attack.any():
            fig.add_trace(
                go.Scatter(
                    x=coords[mask_attack, 0],
                    y=coords[mask_attack, 1],
                    mode="markers",
                    marker=dict(symbol="x", color=color, size=6, opacity=0.7),
                    name=f"{ds_name} - attack",
                    legendgroup=ds_name,
                )
            )

    x_range, y_range = axis_range(coords, sigma)
    title = f"{reduction_name} — {extractor_name} extractor (n={len(coords)})"
    if sigma is not None:
        title += f" [clipped ±{sigma}σ]"
    fig.update_layout(
        title=title,
        xaxis_title=f"{reduction_name} Component 1",
        yaxis_title=f"{reduction_name} Component 2",
        xaxis_range=x_range,
        yaxis_range=y_range,
        legend=dict(itemsizing="constant"),
        width=900,
        height=700,
    )

    suffix = f"_clip{sigma}s" if sigma is not None else ""
    fname = output_dir / f"{reduction_name.lower()}_{extractor_name}{suffix}.svg"
    fig.write_image(str(fname))
    print(f"Saved {fname}")


def plot_tsne_grid(
    coords_per_perplexity: list[tuple[int, np.ndarray]],
    dataset_names: np.ndarray,
    labels: np.ndarray,
    unique_datasets: list,
    extractor_name: str,
    output_dir: Path,
    sigma: float | None = None,
):
    """Plot t-SNE results for multiple perplexities in a single subplot grid."""
    n_plots = len(coords_per_perplexity)
    ncols = 3
    nrows = (n_plots + ncols - 1) // ncols

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        subplot_titles=[f"perplexity={p}" for p, _ in coords_per_perplexity],
    )

    # Track which legend entries have been shown to avoid duplicates across subplots
    shown = set()

    for idx, (perplexity, coords) in enumerate(coords_per_perplexity):
        row, col = divmod(idx, ncols)
        row += 1
        col += 1

        for i, ds_name in enumerate(unique_datasets):
            color = DATASET_COLORS[i % len(DATASET_COLORS)]
            mask_ds = dataset_names == ds_name

            mask_normal = mask_ds & (labels == 0)
            if mask_normal.any():
                key_normal = f"{ds_name} - normal"
                fig.add_trace(
                    go.Scatter(
                        x=coords[mask_normal, 0],
                        y=coords[mask_normal, 1],
                        mode="markers",
                        marker=dict(symbol="circle", color=color, size=5, opacity=0.5),
                        name=key_normal,
                        legendgroup=key_normal,
                        showlegend=key_normal not in shown,
                    ),
                    row=row,
                    col=col,
                )
                shown.add(key_normal)

            mask_attack = mask_ds & (labels == 1)
            if mask_attack.any():
                key_attack = f"{ds_name} - attack"
                fig.add_trace(
                    go.Scatter(
                        x=coords[mask_attack, 0],
                        y=coords[mask_attack, 1],
                        mode="markers",
                        marker=dict(symbol="x", color=color, size=6, opacity=0.7),
                        name=key_attack,
                        legendgroup=key_attack,
                        showlegend=key_attack not in shown,
                    ),
                    row=row,
                    col=col,
                )
                shown.add(key_attack)

    title = f"t-SNE — {extractor_name} extractor (n={len(labels)})"
    if sigma is not None:
        title += f" [clipped ±{sigma}σ]"
    fig.update_layout(
        title=title,
        legend=dict(itemsizing="constant"),
        width=ncols * 900,
        height=nrows * 800,
    )

    # Apply per-subplot axis ranges after layout is set
    if sigma is not None:
        for idx, (_, coords) in enumerate(coords_per_perplexity):
            row, col = divmod(idx, ncols)
            x_range, y_range = axis_range(coords, sigma)
            axis_suffix = "" if idx == 0 else str(idx + 1)
            fig.update_layout(
                **{
                    f"xaxis{axis_suffix}_range": x_range,
                    f"yaxis{axis_suffix}_range": y_range,
                }
            )

    suffix = f"_clip{sigma}s" if sigma is not None else ""
    fname = output_dir / f"tsne_{extractor_name}{suffix}.svg"
    fig.write_image(str(fname))
    print(f"Saved {fname}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize feature space of SQL injection detection datasets"
    )
    parser.add_argument(
        "--extractor",
        required=True,
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Feature extractor(s) to use (repeatable)",
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
        default=100,
        help="Number of samples per label per dataset (default: 200)",
    )
    parser.add_argument("--pca", action="store_true", help="Compute PCA projection")
    parser.add_argument("--tsne", action="store_true", help="Compute t-SNE projection")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiments/feature_space_viz",
        help="Output directory for plots",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=None,
        metavar="N",
        help="Clip plot axes to mean ± N·σ to focus on the main cluster (e.g. 3)",
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

    for extractor_name in args.extractor:
        print(f"\n=== Extractor: {extractor_name} ===")
        extractor = build_extractor(
            extractor_name,
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
            for s in [None, args.sigma] if args.sigma is not None else [None]:
                plot_reduction(
                    coords_pca,
                    dataset_names,
                    labels,
                    unique_datasets,
                    extractor_name,
                    "pca",
                    output_dir,
                    sigma=s,
                )

        if args.tsne:
            n = len(X)
            valid_perplexities = [p for p in TSNE_PERPLEXITIES if p < n]
            coords_per_perplexity = []
            for perplexity in valid_perplexities:
                print(f"Computing t-SNE (perplexity={perplexity}) ...")
                coords_tsne = TSNE(
                    n_components=2,
                    perplexity=perplexity,
                    random_state=2,
                    n_jobs=-1,
                    max_iter=5000,  # Is typically enough to converge https://distill.pub/2016/misread-tsne/
                ).fit_transform(X)
                coords_per_perplexity.append((perplexity, coords_tsne))
            for s in [None, args.sigma] if args.sigma is not None else [None]:
                plot_tsne_grid(
                    coords_per_perplexity,
                    dataset_names,
                    labels,
                    unique_datasets,
                    extractor_name,
                    output_dir,
                    sigma=s,
                )


if __name__ == "__main__":
    main()
