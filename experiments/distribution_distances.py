"""Detect distribution shift between train and test feature spaces.

For each extractor and each generic transfer-learning scenario (e.g. ABC→D),
runs two complementary shift detectors from Rabanser et al. (2019):
  - MMD with permutation test (multivariate kernel two-sample test)
  - Domain classifier accuracy with binomial test (logistic regression probe)

Both produce a test statistic and a p-value under H0: "train and test distributions
are identical." Low p-values indicate detectable domain shift.
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
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
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

METRICS = ["MMD", "MMD_pval", "domain_acc", "domain_pval"]


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


# --- Shift detection (Rabanser et al., 2019) ---

N_PERMUTATIONS = 1000


def _kernel_matrix(X_all: np.ndarray) -> np.ndarray:
    """RBF kernel matrix with median bandwidth heuristic."""
    sq_dists = np.sum((X_all[:, None] - X_all[None, :]) ** 2, axis=-1)
    sigma_sq = float(np.median(sq_dists[sq_dists > 0]))
    return np.exp(-sq_dists / sigma_sq)


def _mmd_sq_from_kernel(K: np.ndarray, n: int, m: int) -> float:
    """Unbiased MMD² estimate from a precomputed kernel matrix."""
    K_XX = K[:n, :n].copy()
    K_YY = K[n:, n:].copy()
    K_XY = K[:n, n:]
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    return K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2 * K_XY.mean()


def compute_mmd(
    X_train: np.ndarray, X_test: np.ndarray, n_permutations: int = N_PERMUTATIONS
) -> dict:
    """MMD statistic and p-value via permutation test on the kernel matrix."""
    n, m = len(X_train), len(X_test)
    K = _kernel_matrix(np.concatenate([X_train, X_test], axis=0))
    observed = _mmd_sq_from_kernel(K, n, m)

    rng = np.random.default_rng(seed=2)
    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n + m)
        K_perm = K[np.ix_(perm[:n], perm[:n])]
        K_perm_YY = K[np.ix_(perm[n:], perm[n:])]
        K_perm_XY = K[np.ix_(perm[:n], perm[n:])]
        np.fill_diagonal(K_perm, 0)
        np.fill_diagonal(K_perm_YY, 0)
        mmd_sq_perm = (
            K_perm.sum() / (n * (n - 1))
            + K_perm_YY.sum() / (m * (m - 1))
            - 2 * K_perm_XY.mean()
        )
        if mmd_sq_perm >= observed:
            count += 1

    return {
        "MMD": float(np.sqrt(max(observed, 0))),
        "MMD_pval": (count + 1) / (n_permutations + 1),
    }


def compute_domain_clf(X_train: np.ndarray, X_test: np.ndarray) -> dict:
    """Domain classifier accuracy and binomial test p-value.

    Trains a logistic regression to distinguish source (train-pool) from target
    (held-out) embeddings on half the data, evaluates on the other half, then
    tests whether accuracy is significantly above 0.5.
    """
    n, m = len(X_train), len(X_test)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(m)])

    rng = np.random.default_rng(seed=2)
    idx = rng.permutation(n + m)
    half = (n + m) // 2
    train_idx, test_idx = idx[:half], idx[half:]

    clf = LogisticRegression(max_iter=1000, random_state=2)
    clf.fit(X[train_idx], y[train_idx])
    acc = float(clf.score(X[test_idx], y[test_idx]))

    n_test = len(test_idx)
    n_correct = round(acc * n_test)
    p_value = float(binomtest(n_correct, n_test, 0.5, alternative="greater").pvalue)

    return {"domain_acc": acc, "domain_pval": p_value}


METRIC_FNS = [compute_mmd, compute_domain_clf]

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

        max_val = pivot.values.max()
        pct = pivot.values / max_val * 100 if max_val > 0 else pivot.values
        text = [[f"{v:.1f}%" for v in row] for row in pct]
        fig.add_trace(
            go.Heatmap(
                z=pct,
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
        default=10000,
        help="Normal samples per dataset from test split (default: 10000)",
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

            for metric_fn in METRIC_FNS:
                row.update(metric_fn(X_train, X_test))

            print(
                f"  {scenario}: "
                f"MMD={row['MMD']:.4f} (p={row['MMD_pval']:.3f})  "
                f"domain_acc={row['domain_acc']:.3f} (p={row['domain_pval']:.3f})"
            )
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
