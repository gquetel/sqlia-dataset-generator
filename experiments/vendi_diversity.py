"""Compute Vendi Score diversity metrics across SQL injection detection datasets and plot results.

For each extractor, computes intra-domain VS per dataset, pooled VS across all
datasets, and per-scenario VS metrics (train pool, test, combined) for the 4
generic transfer-learning scenarios. After computing, produces a heatmap of
relative VS delta across extractors and scenarios.

The Vendi Score (Friedman & Dieng, 2022) is an entropy-based diversity measure
computed from a kernel similarity matrix. Higher VS = more diverse feature space.
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
from sklearn.metrics.pairwise import rbf_kernel
from vendi_score import vendi

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


def sample_normal_test(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Sample up to n normal rows from the test split."""
    normal = df[(df["split"] == "test") & (df["label"] == 0)]
    return normal.sample(n=min(n, len(normal)), random_state=2)


def vendi_score(X: np.ndarray) -> float:
    K = rbf_kernel(X)
    # Same similarity metric than the original paper.
    return float(vendi.score_K(K, q=1))


# --- Plotting ---


def _fmt(v: float) -> str:
    return f"{v:.1f}%"


def load_scenario_data(input_dir: Path) -> pd.DataFrame:
    csvs = sorted(input_dir.glob("*_scenarios.csv"))
    if not csvs:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in csvs], ignore_index=True)
    return df[df["scenario_type"] == "generic"]


def plot_heatmap(df: pd.DataFrame, output_dir: Path):
    df = df.copy()
    df["pct_delta_VS"] = df["delta_VS"] / df["VS_train"] * 100
    pivot = df.pivot(index="scenario", columns="extractor", values="pct_delta_VS")
    pivot = pivot.reindex([s for s in SCENARIO_ORDER if s in pivot.index])
    pivot = pivot[sorted(pivot.columns)]
    pivot.columns = [c.removeprefix("ae_") for c in pivot.columns]

    text = [[_fmt(v) for v in row] for row in pivot.values]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=list(pivot.columns),
            y=list(pivot.index),
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            colorbar=dict(title="Δ VS (%)"),
        )
    )
    fig.update_layout(
        title="Vendi Score Relative Delta (VS_pool − VS_train) / VS_train — Generic Scenarios",
        xaxis_title="Feature Extractor",
        yaxis_title="Training Scenario",
        width=900,
        height=400,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "png"):
        path = output_dir / f"vendi_score_delta.{ext}"
        fig.write_image(str(path))
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute Vendi Score diversity metrics across SQL injection datasets"
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
        default="output/experiments/vendi_score",
        help="Output directory for CSV results and heatmap",
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
        intra_rows = []
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
            features[name] = to_dense(X_raw)
            print(f"  {name}: feature matrix {features[name].shape}")

        for name in dataset_names:
            vs = vendi_score(features[name])
            intra_rows.append({"extractor": extractor_name, "dataset": name, "VS": vs})

        X_all = np.concatenate([features[n] for n in dataset_names], axis=0)
        vs_pool = vendi_score(X_all)
        pool_label = "".join(dataset_names)
        intra_rows.append(
            {"extractor": extractor_name, "dataset": pool_label, "VS": vs_pool}
        )

        for scenario, (train_ds, test_ds) in active_scenarios.items():
            X_train = np.concatenate([features[n] for n in train_ds], axis=0)
            X_test = features[test_ds]
            X_combined = np.concatenate([X_train, X_test], axis=0)

            vs_train = vendi_score(X_train)
            vs_test = vendi_score(X_test)
            vs_combined = vendi_score(X_combined)
            delta_vs = vs_combined - vs_train

            print(
                f"  {scenario}: VS_train={vs_train:.4f}  VS_test={vs_test:.4f}  ΔVS={delta_vs:.4f}"
            )
            scenario_rows.append(
                {
                    "extractor": extractor_name,
                    "scenario": scenario,
                    "scenario_type": "generic",
                    "VS_train": vs_train,
                    "VS_test": vs_test,
                    "VS_combined": vs_combined,
                    "delta_VS": delta_vs,
                }
            )

        intra_path = output_dir / f"{extractor_name}_vendi_scores_intra.csv"
        scenario_path = output_dir / f"{extractor_name}_vendi_scores_scenarios.csv"

        pd.DataFrame(intra_rows).to_csv(intra_path, index=False)
        if scenario_rows:
            pd.DataFrame(scenario_rows).to_csv(scenario_path, index=False)

        print(f"\nSaved {intra_path}")
        if scenario_rows:
            print(f"Saved {scenario_path}")

    # Plot heatmap from all scenario CSVs written above
    df_scenarios = load_scenario_data(output_dir)
    if not df_scenarios.empty:
        print("\nPlotting heatmap ...")
        plot_heatmap(df_scenarios, output_dir)
    else:
        print("\nNo generic scenario data to plot.")


if __name__ == "__main__":
    main()
