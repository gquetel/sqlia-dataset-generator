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
from extractors.loginov import LoginovExtractor

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

FEATURE_CATEGORIES: dict[str, str] = {
    "len_query": "syntactic",
    "has_null": "semantic",
    "has_comment": "semantic",
    "has_query_keywords": "semantic",
    "has_union": "semantic",
    "has_database_keywords": "semantic",
    "has_connection_keywords": "semantic",
    "has_file_keywords": "semantic",
    "has_exec": "semantic",
    "has_string_functions": "semantic",
    "c_comparison": "lexical",
    "has_exist_keyword": "semantic",
    "has_floor": "semantic",
    "has_rand": "semantic",
    "has_group": "semantic",
    "has_order": "semantic",
    "has_length": "semantic",
    "has_ascii": "semantic",
    "has_concat": "semantic",
    "has_if": "semantic",
    "has_count": "semantic",
    "has_sleep": "semantic",
    "has_tautology": "semantic",
    "c_num": "lexical",
    "c_upper": "lexical",
    "c_space": "lexical",
    "c_special": "lexical",
    "c_arith": "lexical",
    "c_square_brackets": "lexical",
    "c_round_brackets": "lexical",
    "has_multiline_comment": "lexical",
    "c_curly_brackets": "lexical",
    "avg_c_sqlkywds": "semantic",
    "max_c_sqlkywds": "semantic",
    "min_c_sqlkywds": "semantic",
    "n_terminal": "syntactic",
    "n_nonterminal": "syntactic",
    "is_syntax_error": "syntactic",
    "depth": "syntactic",
    "n_parser_invoc": "syntactic",
    "DDL_ALTER": "semantic",
    "DDL_CREATE": "semantic",
    "DDL_DROP": "semantic",
    "DML_DELETE_TRUNCATE": "semantic",
    "DML_INSERT_REPLACE": "semantic",
    "DML_MAINTENANCE": "semantic",
    "DML_SELECT": "semantic",
    "DML_UPDATE": "semantic",
    "EXPRESSION_LOGIC": "syntactic",
    "PARTITIONING_STORAGE": "semantic",
    "PRIVILEGES_SECURITY": "semantic",
    "PROCEDURAL_LOGIC": "semantic",
    "REPLICATION_MANAGEMENT": "semantic",
    "SERVER_ADMIN": "semantic",
    "SHOW_DESCRIBE_EXPLAIN": "semantic",
    "STATEMENT_CONTROL": "semantic",
    "STATEMENT_HELP": "semantic",
    "STATEMENT_MANAGEMENT": "semantic",
    "TRANSACTION_CONTROL": "semantic",
    "WINDOW_ANALYTICS": "semantic",
    "n_anomalous_schars": "lexical",
    "s1_n_keywords": "lexical",
    "s1_n_alpha": "lexical",
    "s1_n_numeric": "lexical",
    "s1_n_mixed": "lexical",
    "s2_n_keywords": "lexical",
    "s2_n_alpha": "lexical",
    "s2_n_numeric": "lexical",
    "s2_n_mixed": "lexical",
    # ── GAUR (expert) ────────────────────────────────────────────────────────
    # action tags
    "CREATE": "semantic",
    "DELETE": "semantic",
    "MODIFY": "semantic",
    "EXECUTE": "semantic",
    "READ": "semantic",
    # object tags
    "TABLESPACE": "semantic",
    "TABLE": "semantic",
    "INDEX": "semantic",
    "VIEW": "semantic",
    "USER": "semantic",
    "PROCEDURE": "semantic",
    "DATABASE": "semantic",
    "FUNCTION": "semantic",
    "INSTANCE": "semantic",
    "LOGFILE": "semantic",
    "SERVER": "semantic",
    "TRIGGER": "semantic",
}

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
        index="feature", columns="pair", values="domain_inv"
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
        for xoff, label in [(-0.25, "Inv."), (0.25, "Label")]:
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
            # Value labels: domain_inv on left half, label on right half
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
            title=f"Features — {extractor_name}",
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


def make_mean_scatter(results_df: pd.DataFrame, output_dir: Path):
    """2-D scatter: mean domain_inv (x) vs mean label_acc (y) per feature, colored by category."""
    import plotly.graph_objects as go

    means = (
        results_df.groupby("feature")[["domain_inv", "label_acc"]].mean().reset_index()
    )
    means["category"] = means["feature"].map(
        lambda f: FEATURE_CATEGORIES.get(f, "unknown")
    )

    category_colors = {
        "lexical": "#1f77b4",
        "syntactic": "#ff7f0e",
        "semantic": "#2ca02c",
        "unknown": "#999999",
    }

    fig = go.Figure()
    for cat in CATEGORY_ORDER + ["unknown"]:
        grp = means[means["category"] == cat]
        if grp.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=grp["domain_inv"],
                y=grp["label_acc"],
                mode="markers+text",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color=category_colors.get(cat, "#999999"),
                    line=dict(width=0.8, color="white"),
                ),
                text=grp["feature"],
                textposition="top center",
                textfont=dict(size=7),
                name=cat,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Category: {cat}<br>"
                    "Domain invariance: %{x:.3f}<br>"
                    "Label acc: %{y:.3f}<extra></extra>"
                ),
            )
        )

    # Quadrant cross: midpoint of each axis (domain_inv ∈ [0,1], label_acc ∈ [0.5,1])
    fig.add_vline(x=0.5, line=dict(color="gray", dash="dash", width=1))
    fig.add_hline(y=0.75, line=dict(color="gray", dash="dash", width=1))

    fig.update_layout(
        xaxis=dict(title="Mean domain invariance", range=[-0.05, 1.05]),
        yaxis=dict(title="Mean label prediction accuracy", range=[0.45, 1.02]),
        legend=dict(title="Feature type"),
        plot_bgcolor="white",
        width=900,
        height=750,
        margin=dict(l=60, r=40, t=40, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    out_path = output_dir / "discriminability_mean_scatter.pdf"
    fig.write_image(str(out_path))
    print(f"Saved scatter plot: {out_path}")


def derive_taxonomy_scores(
    results_df: pd.DataFrame, output_dir: Path, threshold: float = 0.75
) -> None:
    """Save a CSV with raw feature counts and normalized 0-10 scores per extractor.

    Only features whose mean label_acc (averaged across dataset pairs) exceeds
    *threshold* are counted. Counts per category are normalized so the largest
    category maps to 10 (rounded to the nearest integer).
    """
    means = (
        results_df.groupby(["extractor", "feature"])["label_acc"]
        .mean()
        .reset_index()
        .rename(columns={"label_acc": "mean_label_acc"})
    )
    means["category"] = means["feature"].map(
        lambda f: FEATURE_CATEGORIES.get(f, "unknown")
    )

    above = means[means["mean_label_acc"] > threshold]

    # Li features included in GAUR in practice; pre-compute Li's above-threshold features.
    li_above = above[above["extractor"] == "Li et al."][
        ["feature", "category"]
    ].drop_duplicates()

    rows = []
    for extractor_name in results_df["extractor"].unique():
        ext = above[above["extractor"] == extractor_name]
        if extractor_name.startswith("GAUR"):
            ext = pd.concat(
                [ext, li_above.assign(extractor=extractor_name)]
            ).drop_duplicates(subset="feature")
        counts_series = ext.groupby("category").size()
        cat_counts = {cat: int(counts_series.get(cat, 0)) for cat in CATEGORY_ORDER}
        max_count = max(cat_counts.values()) if any(cat_counts.values()) else 1

        row = {"extractor": extractor_name}
        for cat in CATEGORY_ORDER:
            row[f"raw_{cat}"] = cat_counts[cat]
            row[f"score_{cat}"] = (
                round(10 * cat_counts[cat] / max_count) if max_count > 0 else 0
            )
        rows.append(row)

    out_path = output_dir / "taxonomy_scores.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Saved taxonomy scores: {out_path}")


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
    parser.add_argument(
        "--derive-taxonomy",
        action="store_true",
        help="After computing/loading results, print derived 0-5 dimension scores",
    )
    parser.add_argument(
        "--taxonomy-threshold",
        type=float,
        default=0.75,
        metavar="T",
        help="Mean label_acc threshold for a feature to count (default: 0.75)",
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
        make_mean_scatter(results_df, output_dir)
        if args.derive_taxonomy:
            derive_taxonomy_scores(
                results_df, output_dir, threshold=args.taxonomy_threshold
            )
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
        ("GAUR (expert)", GaurExtractor(use_hybrid=False, mode="expert")),
        ("GAUR (ChatGPT)", GaurExtractor(use_hybrid=False, mode="chatgpt")),
        ("Loginov et al.", LoginovExtractor()),
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
                        "domain_inv": max(0.0, min(1.0, 2 * (1 - disc_acc))),
                        "label_acc": label_acc,
                        "category": FEATURE_CATEGORIES.get(col, "unknown"),
                    }
                )

        # Features constant across all pairs return 0.5 for every pair
        ext_df = pd.DataFrame(extractor_results)
        constant_features = (
            ext_df.groupby("feature")["domain_inv"]
            .apply(lambda s: (s == 1.0).all())
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
    make_mean_scatter(results_df, output_dir)
    if args.derive_taxonomy:
        derive_taxonomy_scores(
            results_df, output_dir, threshold=args.taxonomy_threshold
        )


if __name__ == "__main__":
    main()
