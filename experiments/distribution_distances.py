"""Compute distribution distance metrics between train and test feature spaces.

For each extractor and each generic transfer-learning scenario (e.g. ABC→D),
computes distances between the training pool distribution and the held-out test
distribution in feature space.  Produces a heatmap per metric.

Metrics computed:
- KL divergence (estimated via KDE)
- Chi-squared divergence (histogram-based)
- Jensen-Shannon divergence (histogram-based, symmetric)
- L2 distance (between mean embeddings)
- Wasserstein distance (1D sliced approximation)
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from scipy import sparse
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sklearn.preprocessing import normalize

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

SCENARIOS = {
    "ABC→D": (["A", "B", "C"], "D"),
    "ABD→C": (["A", "B", "D"], "C"),
    "ACD→B": (["A", "C", "D"], "B"),
    "BCD→A": (["B", "C", "D"], "A"),
}

SCENARIO_ORDER = ["ABC→D", "ABD→C", "ACD→B", "BCD→A"]

METRICS = ["KL", "chi_squared", "jensen_shannon", "L2", "wasserstein"]


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
    extractor.prepare_for_training(df_train)
    if hasattr(extractor, "_fitted"):
        extractor.extract_features(df_train)


def sample_normal_test(df: pd.DataFrame, n: int) -> pd.DataFrame:
    normal = df[(df["split"] == "test") & (df["label"] == 0)]
    return normal.sample(n=min(n, len(normal)), random_state=2)


# --- Distance metrics ---

N_BINS = 50
N_SLICES = 100
EPS = 1e-10


def _histograms(X_train: np.ndarray, X_test: np.ndarray, n_bins: int = N_BINS):
    """Project to 1D via PCA-1 and compute aligned histograms."""
    combined = np.concatenate([X_train, X_test], axis=0)
    mean = combined.mean(axis=0)
    centered = combined - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_dir = Vt[0]

    p1 = X_train @ proj_dir
    p2 = X_test @ proj_dir

    lo = min(p1.min(), p2.min())
    hi = max(p1.max(), p2.max())
    bins = np.linspace(lo, hi, n_bins + 1)

    h1, _ = np.histogram(p1, bins=bins, density=True)
    h2, _ = np.histogram(p2, bins=bins, density=True)

    # Normalize to probability distributions
    h1 = h1 / (h1.sum() + EPS) + EPS
    h2 = h2 / (h2.sum() + EPS) + EPS

    return h1, h2


def kl_divergence(X_train: np.ndarray, X_test: np.ndarray) -> float:
    p, q = _histograms(X_train, X_test)
    return float(np.sum(p * np.log(p / q)))


def chi_squared_divergence(X_train: np.ndarray, X_test: np.ndarray) -> float:
    p, q = _histograms(X_train, X_test)
    return float(np.sum((p - q) ** 2 / q))


def jensen_shannon_divergence(X_train: np.ndarray, X_test: np.ndarray) -> float:
    p, q = _histograms(X_train, X_test)
    return float(jensenshannon(p, q) ** 2)  # scipy returns sqrt(JSD)


def l2_distance(X_train: np.ndarray, X_test: np.ndarray) -> float:
    mean_train = X_train.mean(axis=0)
    mean_test = X_test.mean(axis=0)
    return float(np.linalg.norm(mean_train - mean_test))


def sliced_wasserstein(
    X_train: np.ndarray, X_test: np.ndarray, n_slices: int = N_SLICES
) -> float:
    """Sliced Wasserstein distance: average 1D Wasserstein over random projections."""
    rng = np.random.default_rng(seed=2)
    d = X_train.shape[1]
    directions = rng.standard_normal((n_slices, d))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    distances = []
    for direction in directions:
        p1 = X_train @ direction
        p2 = X_test @ direction
        distances.append(wasserstein_distance(p1, p2))

    return float(np.mean(distances))


METRIC_FNS = {
    "KL": kl_divergence,
    "chi_squared": chi_squared_divergence,
    "jensen_shannon": jensen_shannon_divergence,
    "L2": l2_distance,
    "wasserstein": sliced_wasserstein,
}

# --- Plotting ---


def load_scenario_data(input_dir: Path) -> pd.DataFrame:
    csvs = sorted(input_dir.glob("*_distances_scenarios.csv"))
    if not csvs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)


def plot_heatmaps(df: pd.DataFrame, output_dir: Path):
    metrics = [m for m in METRICS if m in df.columns]
    n_metrics = len(metrics)

    fig = make_subplots(
        rows=1,
        cols=n_metrics,
        subplot_titles=[m.replace("_", " ").title() for m in metrics],
        horizontal_spacing=0.06,
    )

    for i, metric in enumerate(metrics, 1):
        pivot = df.pivot(index="scenario", columns="extractor", values=metric)
        pivot = pivot.reindex([s for s in SCENARIO_ORDER if s in pivot.index])
        pivot = pivot[sorted(pivot.columns)]
        pivot.columns = [c.removeprefix("ae_") for c in pivot.columns]

        text = [[f"{v:.4f}" for v in row] for row in pivot.values]
        fig.add_trace(
            go.Heatmap(
                z=pivot.values,
                x=list(pivot.columns),
                y=list(pivot.index),
                text=text,
                texttemplate="%{text}",
                colorscale="Reds",
                showscale=(i == n_metrics),
                colorbar=dict(title="Distance") if i == n_metrics else None,
            ),
            row=1,
            col=i,
        )

    fig.update_layout(
        title="Distribution Distances: Train Pool vs Held-Out Test — Generic Scenarios",
        width=400 * n_metrics,
        height=450,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = output_dir / f"distribution_distances.{ext}"
        fig.write_image(str(path))
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute distribution distance metrics across SQL injection datasets"
    )
    parser.add_argument(
        "--extractor",
        required=True,
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Feature extractor(s) to use",
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
        "--samples",
        type=int,
        default=1000,
        help="Normal samples per dataset from test split (default: 1000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiments/distribution_distances",
        help="Output directory for CSV results and heatmaps",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce sample count to 50 for quick iteration",
    )
    args = parser.parse_args()

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

    dataset_names = [name for name, _ in args.dataset]

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    df_train_combined = pd.concat(
        [df[df["split"] == "train"] for df in datasets.values()], ignore_index=True
    )
    print(f"Combined training set size: {len(df_train_combined)}")

    normal_samples: dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        sampled = sample_normal_test(df, args.samples)
        print(f"  {name}: {len(sampled)} normal test samples")
        normal_samples[name] = sampled

    available = set(dataset_names)
    active_scenarios = {
        scenario: (train_ds, test_ds)
        for scenario, (train_ds, test_ds) in SCENARIOS.items()
        if set(train_ds) | {test_ds} <= available
    }
    if active_scenarios:
        print(f"Active scenarios: {list(active_scenarios.keys())}")
    else:
        print("No complete scenarios available (need all 4 datasets A/B/C/D)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for extractor_name in args.extractor:
        print(f"\n=== Extractor: {extractor_name} ===")
        scenario_rows = []
        extractor = build_extractor(
            extractor_name,
            device=device,
            embeddings_path=embeddings_path,
            cache_dir=cache_dir,
        )

        print("Fitting extractor on combined training data ...")
        fit_extractor(extractor, df_train_combined)

        features: dict[str, np.ndarray] = {}
        for name, df in normal_samples.items():
            print(f"Extracting features for dataset {name} ...")
            X_raw = extractor.extract_features(df)
            X = to_dense(X_raw)
            features[name] = normalize(X, norm="l2")
            print(f"  {name}: feature matrix {features[name].shape}")

        for scenario, (train_ds, test_ds) in active_scenarios.items():
            X_train = np.concatenate([features[n] for n in train_ds], axis=0)
            X_test = features[test_ds]

            row = {
                "extractor": extractor_name,
                "scenario": scenario,
            }

            for metric_name, metric_fn in METRIC_FNS.items():
                val = metric_fn(X_train, X_test)
                row[metric_name] = val

            print(f"  {scenario}: " + "  ".join(f"{m}={row[m]:.4f}" for m in METRICS))
            scenario_rows.append(row)

        scenario_path = output_dir / f"{extractor_name}_distances_scenarios.csv"
        if scenario_rows:
            pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False)
            print(f"Saved {scenario_path}")

    df_scenarios = load_scenario_data(output_dir)
    if not df_scenarios.empty:
        print("\nPlotting heatmaps ...")
        plot_heatmaps(df_scenarios, output_dir)
    else:
        print("\nNo scenario data to plot.")


if __name__ == "__main__":
    main()
