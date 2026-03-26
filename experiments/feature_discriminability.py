import argparse
import itertools
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from extractors.gaur import GaurExtractor
from extractors.li import LiExtractor

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

# Li features qall lexical (23 from extract_li_features + 9 from get_char_kinds_number)
_LI_FEATURES = [
    "len_query",
    "has_null",
    "has_comment",
    "has_query_keywords",
    "has_union",
    "has_database_keywords",
    "has_connection_keywords",
    "has_file_keywords",
    "has_exec",
    "has_string_functions",
    "c_comparison",
    "has_exist_keyword",
    "has_floor",
    "has_rand",
    "has_group",
    "has_order",
    "has_length",
    "has_ascii",
    "has_concat",
    "has_if",
    "has_count",
    "has_sleep",
    "has_tautology",
    "c_num",
    "c_upper",
    "c_space",
    "c_special",
    "c_arith",
    "c_square_brackets",
    "c_round_brackets",
    "has_multiline_comment",
    "c_curly_brackets",
]

# GAUR feature categories (from gaur_sqld/utils/constants.py)
_GAUR_LEX = ["avg_c_sqlkywds", "max_c_sqlkywds", "min_c_sqlkywds"]
_GAUR_SYNT = [
    "n_terminal",
    "n_nonterminal",
    "is_syntax_error",
    "depth",
    "n_parser_invoc",
]
_GAUR_SEM = [
    "DDL_ALTER",
    "DDL_CREATE",
    "DDL_DROP",
    "DML_DELETE_TRUNCATE",
    "DML_INSERT_REPLACE",
    "DML_MAINTENANCE",
    "DML_SELECT",
    "DML_UPDATE",
    "EXPRESSION_LOGIC",
    "PARTITIONING_STORAGE",
    "PRIVILEGES_SECURITY",
    "PROCEDURAL_LOGIC",
    "REPLICATION_MANAGEMENT",
    "SERVER_ADMIN",
    "SHOW_DESCRIBE_EXPLAIN",
    "STATEMENT_CONTROL",
    "STATEMENT_HELP",
    "STATEMENT_MANAGEMENT",
    "TRANSACTION_CONTROL",
    "WINDOW_ANALYTICS",
]

FEATURE_CATEGORIES: dict[str, str] = (
    {f: "lexical" for f in _LI_FEATURES}
    | {f: "lexical" for f in _GAUR_LEX}
    | {f: "syntactic" for f in _GAUR_SYNT}
    | {f: "semantic" for f in _GAUR_SEM}
)

CATEGORY_ORDER = ["lexical", "syntactic", "semantic"]
CATEGORY_ABBREV = {"lexical": "LEX", "syntactic": "SYN", "semantic": "SEM"}

N_SAMPLES = 50_000
TESTING_N_SAMPLES = 1_000


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def sample_normal_test(df: pd.DataFrame, n: int) -> pd.DataFrame:
    normal = df[(df["split"] == "test") & (df["label"] == 0)]
    return normal.sample(n=min(n, len(normal)), random_state=2)


def run_discriminability(feat_a: pd.DataFrame, feat_b: pd.DataFrame, col: str) -> float:
    """Train a DT on one feature column to distinguish two datasets; return test accuracy."""
    X_a = feat_a[[col]].fillna(0).to_numpy(dtype=float)
    X_b = feat_b[[col]].fillna(0).to_numpy(dtype=float)

    # Balance classes
    n = min(len(X_a), len(X_b))
    rng = np.random.default_rng(seed=2)
    X_a = X_a[rng.choice(len(X_a), n, replace=False)]
    X_b = X_b[rng.choice(len(X_b), n, replace=False)]

    X = np.concatenate([X_a, X_b])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    # Constant features carry no discriminative signal
    if np.std(X) == 0:
        return 0.5

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=2
    )
    clf = DecisionTreeClassifier(random_state=2)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def make_heatmap(results_df: pd.DataFrame, extractor_name: str, output_dir: Path):
    import plotly.graph_objects as go

    def sort_key(f):
        cat = FEATURE_CATEGORIES.get(f, "unknown")
        return (CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99, f)

    features_sorted = sorted(results_df["feature"].unique(), key=sort_key)
    pairs = sorted(results_df["pair"].unique())

    pivot = results_df.pivot(index="feature", columns="pair", values="accuracy")
    pivot = pivot.reindex(features_sorted)

    # Build interleaved rows: features + mean row after each category block
    row_labels = []
    row_data = []
    shapes = []
    prev_cat = None
    row_idx = 0
    cat_blocks: dict[str, list[int]] = {}  # cat -> list of feature row indices

    for f in features_sorted:
        cat = FEATURE_CATEGORIES.get(f, "unknown")
        if prev_cat is not None and cat != prev_cat:
            # Insert mean row for the previous category
            mean_vals = np.mean([row_data[i] for i in cat_blocks[prev_cat]], axis=0)
            row_labels.append(f"► mean {prev_cat}")
            row_data.append(mean_vals)
            row_idx += 1
            # Separator line after the mean row
            shapes.append(
                dict(
                    type="line",
                    x0=-0.5,
                    x1=len(pairs) - 0.5,
                    y0=row_idx - 0.5,
                    y1=row_idx - 0.5,
                    xref="x",
                    yref="y",
                    line=dict(color="white", width=3),
                )
            )
        cat_blocks.setdefault(cat, []).append(row_idx)
        row_labels.append(f"[{CATEGORY_ABBREV.get(cat, '?')}] {f}")
        row_data.append(pivot.loc[f, pairs].to_numpy(dtype=float))
        prev_cat = cat
        row_idx += 1

    # Insert mean row for the last category
    if prev_cat is not None:
        mean_vals = np.mean([row_data[i] for i in cat_blocks[prev_cat]], axis=0)
        row_labels.append(f"► mean {prev_cat}")
        row_data.append(mean_vals)

    z = np.array(row_data)

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=pairs,
            y=row_labels,
            colorscale="RdYlGn",
            zmid=0.5,
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Accuracy"),
            text=np.round(z, 2),
            texttemplate="%{text:.2f}",
            hovertemplate="Feature: %{y}<br>Pair: %{x}<br>Accuracy: %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=f"Per-feature discriminability — {extractor_name}",
        xaxis_title="Dataset pair",
        yaxis_title="Feature",
        shapes=shapes,
        height=max(400, len(row_labels) * 22 + 120),
        width=max(600, len(pairs) * 120 + 350),
        margin=dict(l=230, r=100, t=60, b=60),
        yaxis=dict(autorange="reversed"),
    )

    out_path = output_dir / f"discriminability_{extractor_name}.svg"
    fig.write_image(str(out_path))
    print(f"Saved heatmap: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-feature dataset discriminability via DecisionTree"
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
        help="Dataset short name and CSV path (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default="output/experiments/feature_discriminability",
        help="Output directory for CSV and HTML heatmaps",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=f"Subsample to {TESTING_N_SAMPLES} per dataset for quick iteration",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    n_samples = TESTING_N_SAMPLES if args.testing else N_SAMPLES
    if args.testing:
        print(f"[testing] using {n_samples} samples per dataset")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    samples: dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        s = sample_normal_test(df, n_samples)
        print(f"  {name}: {len(s)} normal test samples")
        samples[name] = s.reset_index(drop=True)

    pairs = list(itertools.combinations(sorted(datasets.keys()), 2))
    print(f"Dataset pairs: {[f'{a}-{b}' for a, b in pairs]}")

    extractors = [
        ("li", LiExtractor()),
        ("gaur", GaurExtractor(use_hybrid=False, mode="chatgpt")),
    ]

    all_results = []

    for extractor_name, extractor in extractors:
        print(f"\n=== Extractor: {extractor_name} ===")

        feats: dict[str, pd.DataFrame] = {}
        for name, df in samples.items():
            print(f"Extracting {extractor_name} features for {name} ...")
            feat = extractor.extract_features(df)
            feats[name] = feat if isinstance(feat, pd.DataFrame) else pd.DataFrame(feat)
            print(f"  {name}: {feats[name].shape}")

        feature_cols = feats[next(iter(feats))].columns.tolist()

        for a, b in pairs:
            pair_label = f"{a}-{b}"
            print(f"  Pair {pair_label} ...")
            for col in feature_cols:
                acc = run_discriminability(feats[a], feats[b], col)
                all_results.append(
                    {
                        "extractor": extractor_name,
                        "feature": col,
                        "pair": pair_label,
                        "accuracy": acc,
                        "category": FEATURE_CATEGORIES.get(col, "unknown"),
                    }
                )

    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "feature_discriminability.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    for extractor_name in results_df["extractor"].unique():
        subset = results_df[results_df["extractor"] == extractor_name]
        make_heatmap(subset, extractor_name, output_dir)


if __name__ == "__main__":
    main()
