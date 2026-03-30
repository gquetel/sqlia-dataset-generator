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


def sample_disc(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Random sample from test split. We don't need label balance."""
    test = df[df["split"] == "test"]
    return test.sample(n=min(n, len(test)), random_state=2).reset_index(drop=True)


def sample_label_balanced(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Balanced (50/50) sample from test split. This allows better interpretation."""
    test = df[df["split"] == "test"]
    n_each = min(n // 2, (test["label"] == 0).sum(), (test["label"] == 1).sum())
    normal = test[test["label"] == 0].sample(n=n_each, random_state=2)
    attack = test[test["label"] == 1].sample(n=n_each, random_state=2)
    return (
        pd.concat([normal, attack])
        .sample(frac=1, random_state=2)
        .reset_index(drop=True)
    )


def run_discriminability(
    feat_a_train: pd.DataFrame,
    feat_b_train: pd.DataFrame,
    feat_a_test: pd.DataFrame,
    feat_b_test: pd.DataFrame,
    col: str,
) -> float:
    """Train a DT (on train split) to distinguish two datasets; evaluate on test split."""
    rng = np.random.default_rng(seed=2)

    def _balanced_pair(fa, fb):
        Xa = fa[[col]].fillna(0).to_numpy(dtype=float)
        Xb = fb[[col]].fillna(0).to_numpy(dtype=float)
        n = min(len(Xa), len(Xb))
        Xa = Xa[rng.choice(len(Xa), n, replace=False)]
        Xb = Xb[rng.choice(len(Xb), n, replace=False)]
        return np.concatenate([Xa, Xb]), np.concatenate([np.zeros(n), np.ones(n)])

    X_train, y_train = _balanced_pair(feat_a_train, feat_b_train)
    X_test, y_test = _balanced_pair(feat_a_test, feat_b_test)

    if np.std(X_train) == 0:
        return 0.5

    clf = DecisionTreeClassifier(random_state=2)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def run_label_prediction(
    feat_a_train: pd.DataFrame,
    feat_b_train: pd.DataFrame,
    labels_a_train: np.ndarray,
    labels_b_train: np.ndarray,
    feat_a_test: pd.DataFrame,
    feat_b_test: pd.DataFrame,
    labels_a_test: np.ndarray,
    labels_b_test: np.ndarray,
    col: str,
) -> float:
    """Train a DT (on train split) to predict label (0/1); evaluate on test split."""
    X_train = np.concatenate(
        [
            feat_a_train[[col]].fillna(0).to_numpy(dtype=float),
            feat_b_train[[col]].fillna(0).to_numpy(dtype=float),
        ]
    )
    y_train = np.concatenate([labels_a_train, labels_b_train])

    X_test = np.concatenate(
        [
            feat_a_test[[col]].fillna(0).to_numpy(dtype=float),
            feat_b_test[[col]].fillna(0).to_numpy(dtype=float),
        ]
    )
    y_test = np.concatenate([labels_a_test, labels_b_test])

    if np.std(X_train) == 0:
        return 0.5

    clf = DecisionTreeClassifier(random_state=2)
    clf.fit(X_train, y_train)
    return float(clf.score(X_test, y_test))


def make_heatmap(results_df: pd.DataFrame, extractor_name: str, output_dir: Path):
    import plotly.colors as pc
    import plotly.graph_objects as go

    def sort_key(f):
        cat = FEATURE_CATEGORIES.get(f, "unknown")
        return (CATEGORY_ORDER.index(cat) if cat in CATEGORY_ORDER else 99, f)

    features_sorted = sorted(results_df["feature"].unique(), key=sort_key)
    pairs = sorted(results_df["pair"].unique())

    pivot_disc = results_df.pivot(
        index="feature", columns="pair", values="disc_acc"
    ).reindex(features_sorted)
    pivot_label = results_df.pivot(
        index="feature", columns="pair", values="label_acc"
    ).reindex(features_sorted)

    # Build interleaved rows: features + mean row after each category block
    row_labels = []
    row_disc = []
    row_label = []
    separator_before: list[int] = []  # row indices that get a separator line above them
    prev_cat = None
    row_idx = 0

    for f in features_sorted:
        cat = FEATURE_CATEGORIES.get(f, "unknown")
        if prev_cat is not None and cat != prev_cat:
            separator_before.append(row_idx)
        row_labels.append(f)
        row_disc.append(pivot_disc.loc[f, pairs].to_numpy(dtype=float))
        row_label.append(pivot_label.loc[f, pairs].to_numpy(dtype=float))
        prev_cat = cat
        row_idx += 1

    z_disc = np.array(row_disc)
    z_label = np.array(row_label)

    vmin = float(min(np.nanmin(z_disc), np.nanmin(z_label)))
    n_rows, n_cols = z_disc.shape

    def to_color(val: float) -> str:
        norm = max(0.0, min(1.0, (val - vmin) / (1.0 - vmin))) if 1.0 > vmin else 0.5
        return pc.sample_colorscale("RdYlGn", [norm])[0]

    shapes = []
    annotations = []

    # "domain" / "label" sub-headers above each pair column
    for j in range(n_cols):
        for xoff, label in [(-0.25, "Domain"), (0.25, "Label")]:
            annotations.append(
                dict(
                    x=j + xoff,
                    xref="x",
                    y=1.0,
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(size=11, color="#444"),
                    xanchor="center",
                    yanchor="bottom",
                )
            )

    for i in range(n_rows):
        for j in range(n_cols):
            # Left half — domain discriminability
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=j - 0.5,
                    x1=j,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    fillcolor=to_color(z_disc[i, j]),
                    line_width=0,
                    layer="below",
                )
            )
            # Right half — label predictive power
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=j,
                    x1=j + 0.5,
                    y0=i - 0.5,
                    y1=i + 0.5,
                    fillcolor=to_color(z_label[i, j]),
                    line_width=0,
                    layer="below",
                )
            )
            # Value labels: disc on left half, label on right half
            annotations.append(
                dict(
                    x=j - 0.25,
                    y=i,
                    xref="x",
                    yref="y",
                    text=f"{z_disc[i, j]:.2f}",
                    showarrow=False,
                    font=dict(size=8),
                    align="center",
                )
            )
            annotations.append(
                dict(
                    x=j + 0.25,
                    y=i,
                    xref="x",
                    yref="y",
                    text=f"{z_label[i, j]:.2f}",
                    showarrow=False,
                    font=dict(size=8),
                    align="center",
                )
            )

    # Dummy scatter trace for the single shared colorbar
    fig = go.Figure(
        data=[
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    colorscale="RdYlGn",
                    cmin=vmin,
                    cmax=1.0,
                    showscale=True,
                    size=0,
                    colorbar=dict(title="Accuracy", x=1.02, len=0.85, thickness=14),
                    color=[vmin],
                ),
                showlegend=False,
            ),
        ]
    )

    # Vertical separators between dataset pair columns
    for j in range(1, n_cols):
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=j - 0.5,
                x1=j - 0.5,
                y0=-0.5,
                y1=n_rows - 0.5,
                line=dict(color="white", width=2),
            )
        )

    # Category separator lines (horizontal, between category blocks)
    # for row in separator_before:
    #     shapes.append(
    #         dict(
    #             type="line",
    #             xref="x",
    #             yref="y",
    #             x0=-0.5,
    #             x1=n_cols - 0.5,
    #             y0=row - 0.5,
    #             y1=row - 0.5,
    #             line=dict(color="white", width=3),
    #         )
    #     )

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=pairs,
            title="Dataset pair",
            range=[-0.5, n_cols - 0.5],
            side="bottom",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=row_labels,
            title=f"Features from {extractor_name}",
            range=[n_rows - 0.5, -0.5],
            autorange=False,
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        height=max(400, n_rows * 24 + 160),
        width=max(600, n_cols * 140 + 450),
        margin=dict(l=230, r=160, t=40, b=60),
    )

    out_path = output_dir / f"discriminability_{extractor_name}.pdf"
    fig.write_image(str(out_path))
    print(f"Saved heatmap: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Per-feature dataset discriminability via DecisionTree"
    )
    parser.add_argument(
        "--from-csv",
        metavar="PATH",
        help="Skip computation and plot heatmaps from a previously saved results CSV",
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Dataset short name and CSV path (repeatable); required without --from-csv",
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

    if not args.from_csv and not args.dataset:
        parser.error("--dataset is required when --from-csv is not specified")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        print(f"Loading results from {args.from_csv} ...")
        results_df = pd.read_csv(args.from_csv)
        for extractor_name in results_df["extractor"].unique():
            subset = results_df[results_df["extractor"] == extractor_name]
            make_heatmap(subset, extractor_name, output_dir)
        return

    n_samples = TESTING_N_SAMPLES if args.testing else N_SAMPLES
    if args.testing:
        print(f"[testing] using {n_samples} samples per dataset")

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    disc_train_dfs: dict[str, pd.DataFrame] = {}
    disc_test_dfs: dict[str, pd.DataFrame] = {}
    label_train_dfs: dict[str, pd.DataFrame] = {}
    label_test_dfs: dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        disc = sample_disc(df, n_samples)
        disc_tr, disc_te = train_test_split(disc, test_size=0.5, random_state=2)
        disc_train_dfs[name] = disc_tr.reset_index(drop=True)
        disc_test_dfs[name] = disc_te.reset_index(drop=True)

        label = sample_label_balanced(df, n_samples)
        label_tr, label_te = train_test_split(
            label, test_size=0.5, random_state=2, stratify=label["label"]
        )
        label_train_dfs[name] = label_tr.reset_index(drop=True)
        label_test_dfs[name] = label_te.reset_index(drop=True)

        print(
            f" Dataset {name}: Domain train {len(disc_tr)}, disc test {len(disc_te)}."
            f" Label train{len(label_tr)}, label test{len(label_te)}."
        )

    pairs = list(itertools.combinations(sorted(datasets.keys()), 2))

    extractors = [
        ("Li et al.", LiExtractor()),
        ("GAUR (ChatGPT)", GaurExtractor(use_hybrid=False, mode="chatgpt")),
    ]

    all_results = []

    for extractor_name, extractor in extractors:
        print(f"\n=== Extractor: {extractor_name} ===")

        def _extract(df: pd.DataFrame) -> pd.DataFrame:
            feat = extractor.extract_features(df)
            return feat if isinstance(feat, pd.DataFrame) else pd.DataFrame(feat)

        disc_train_feats: dict[str, pd.DataFrame] = {}
        disc_test_feats: dict[str, pd.DataFrame] = {}
        label_train_feats: dict[str, pd.DataFrame] = {}
        label_train_labels: dict[str, np.ndarray] = {}
        label_test_feats: dict[str, pd.DataFrame] = {}
        label_test_labels: dict[str, np.ndarray] = {}
        for name in datasets:
            print(f"Extracting {extractor_name} features for {name} ...")
            disc_train_feats[name] = _extract(disc_train_dfs[name])
            disc_test_feats[name] = _extract(disc_test_dfs[name])
            label_train_feats[name] = _extract(label_train_dfs[name])
            label_train_labels[name] = label_train_dfs[name]["label"].to_numpy()
            label_test_feats[name] = _extract(label_test_dfs[name])
            label_test_labels[name] = label_test_dfs[name]["label"].to_numpy()
            print(
                f"  disc {disc_train_feats[name].shape}/{disc_test_feats[name].shape}"
                f" | label {label_train_feats[name].shape}/{label_test_feats[name].shape}"
            )

        feature_cols = disc_train_feats[next(iter(disc_train_feats))].columns.tolist()

        extractor_results = []
        for a, b in pairs:
            pair_label = f"{a}-{b}"
            print(f"  Pair {pair_label} ...")
            for col in feature_cols:
                disc_acc = run_discriminability(
                    disc_train_feats[a],
                    disc_train_feats[b],
                    disc_test_feats[a],
                    disc_test_feats[b],
                    col,
                )
                label_acc = run_label_prediction(
                    label_train_feats[a],
                    label_train_feats[b],
                    label_train_labels[a],
                    label_train_labels[b],
                    label_test_feats[a],
                    label_test_feats[b],
                    label_test_labels[a],
                    label_test_labels[b],
                    col,
                )
                extractor_results.append(
                    {
                        "extractor": extractor_name,
                        "feature": col,
                        "pair": pair_label,
                        "disc_acc": disc_acc,
                        "label_acc": label_acc,
                        "category": FEATURE_CATEGORIES.get(col, "unknown"),
                    }
                )

        # Features constant across all pairs return 0.5 for every pair
        ext_df = pd.DataFrame(extractor_results)
        constant_features = (
            ext_df.groupby("feature")["disc_acc"]
            .apply(lambda s: (s == 0.5).all())
            .pipe(lambda s: s[s].index.tolist())
        )
        if constant_features:
            print(
                f"  Constant features (excluded from output): {sorted(constant_features)}"
            )
        ext_df = ext_df[~ext_df["feature"].isin(constant_features)]
        all_results.extend(ext_df.to_dict("records"))

    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "feature_discriminability.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    for extractor_name in results_df["extractor"].unique():
        subset = results_df[results_df["extractor"] == extractor_name]
        make_heatmap(subset, extractor_name, output_dir)


if __name__ == "__main__":
    main()
