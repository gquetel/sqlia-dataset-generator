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

from constants import mysql_functions, mysql_keywords
from extractors.countvect import CountVectExtractor
from extractors.gaur import GaurExtractor
from extractors.li import LiExtractor
from extractors.loginov import LoginovExtractor
from extractors.kakisim import KakisimExtractor

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
    # Li
    "len_query": "syntactic",
    "has_null": "protocol-level",
    "has_comment": "protocol-level",
    "has_query_keywords": "protocol-level",
    "has_union": "protocol-level",
    "has_database_keywords": "protocol-level",
    "has_connection_keywords": "protocol-level",
    "has_file_keywords": "protocol-level",
    "has_exec": "protocol-level",
    "has_string_functions": "protocol-level",
    "c_comparison": "lexical",
    "has_exist_keyword": "protocol-level",
    "has_floor": "protocol-level",
    "has_rand": "protocol-level",
    "has_group": "protocol-level",
    "has_order": "protocol-level",
    "has_length": "protocol-level",
    "has_ascii": "protocol-level",
    "has_concat": "protocol-level",
    "has_if": "protocol-level",
    "has_count": "protocol-level",
    "has_sleep": "protocol-level",
    "has_tautology": "protocol-level",
    "c_num": "lexical",
    "c_upper": "lexical",
    "c_space": "lexical",
    "c_special": "lexical",
    "c_arith": "lexical",
    "c_square_brackets": "lexical",
    "c_round_brackets": "lexical",
    "has_multiline_comment": "lexical",
    "c_curly_brackets": "lexical",
    "avg_c_sqlkywds": "protocol-level",
    "max_c_sqlkywds": "protocol-level",
    "min_c_sqlkywds": "protocol-level",
    "n_terminal": "syntactic",
    "n_nonterminal": "syntactic",
    "is_syntax_error": "syntactic",
    "depth": "syntactic",
    "n_parser_invoc": "syntactic",
    # Gaur ChatGPT
    "DDL_ALTER": "protocol-level",
    "DDL_CREATE": "protocol-level",
    "DDL_DROP": "protocol-level",
    "DML_DELETE_TRUNCATE": "protocol-level",
    "DML_INSERT_REPLACE": "protocol-level",
    "DML_MAINTENANCE": "protocol-level",
    "DML_SELECT": "protocol-level",
    "DML_UPDATE": "protocol-level",
    "EXPRESSION_LOGIC": "syntactic",
    "PARTITIONING_STORAGE": "protocol-level",
    "PRIVILEGES_SECURITY": "protocol-level",
    "PROCEDURAL_LOGIC": "protocol-level",
    "REPLICATION_MANAGEMENT": "protocol-level",
    "SERVER_ADMIN": "protocol-level",
    "SHOW_DESCRIBE_EXPLAIN": "protocol-level",
    "STATEMENT_CONTROL": "protocol-level",
    "STATEMENT_HELP": "protocol-level",
    "STATEMENT_MANAGEMENT": "protocol-level",
    "TRANSACTION_CONTROL": "protocol-level",
    "WINDOW_ANALYTICS": "protocol-level",
    # Gaur Expert
    # action tags
    "CREATE": "protocol-level",
    "DELETE": "protocol-level",
    "MODIFY": "protocol-level",
    "EXECUTE": "protocol-level",
    "READ": "protocol-level",
    # object tags
    "TABLESPACE": "protocol-level",
    "TABLE": "protocol-level",
    "INDEX": "protocol-level",
    "VIEW": "protocol-level",
    "USER": "protocol-level",
    "PROCEDURE": "protocol-level",
    "DATABASE": "protocol-level",
    "FUNCTION": "protocol-level",
    "INSTANCE": "protocol-level",
    "LOGFILE": "protocol-level",
    "SERVER": "protocol-level",
    "TRIGGER": "protocol-level",
    # Mistral semantic tags
    "Data Definition": "protocol-level",
    "Data Import Export": "protocol-level",
    "Data Import/Export": "protocol-level",
    "Data Manipulation": "protocol-level",
    "Data Query": "protocol-level",
    "Database Management": "protocol-level",
    "Locking & Concurrency": "protocol-level",
    "Miscellaneous Operations": "protocol-level",
    "Replication & Clustering": "protocol-level",
    "Resource Management": "protocol-level",
    "Security & Privileges": "protocol-level",
    "Stored Procedures & Functions": "protocol-level",
    "System Information": "protocol-level",
    "System Maintenance": "protocol-level",
    "System Variables": "protocol-level",
    "Temporary Objects": "protocol-level",
    "Triggers & Events": "protocol-level",
    "User Management": "protocol-level",
    "Views": "protocol-level",
    "Statement Control": "protocol-level",
    "Transaction Control": "protocol-level",
    # Loginov
    "n_anomalous_schars": "lexical",
    "s1_n_keywords": "lexical",
    "s1_n_alpha": "lexical",
    "s1_n_numeric": "lexical",
    "s1_n_mixed": "lexical",
    "s2_n_keywords": "lexical",
    "s2_n_alpha": "lexical",
    "s2_n_numeric": "lexical",
    "s2_n_mixed": "lexical",
    # Kakisim (view C — semantic tags)
    "Par": "protocol-level",
    "DLL": "protocol-level",
    "DML": "protocol-level",
    "Keyw": "protocol-level",
    "Int": "protocol-level",
    "Hexadecimal": "protocol-level",
    "Quot": "protocol-level",
    "Punct": "protocol-level",
    "Wildcard": "protocol-level",
    "Comparison": "protocol-level",
    "Oper": "protocol-level",
    "Builtin": "protocol-level",
    "Func": "protocol-level",
    "Identifi": "protocol-level",
    "Escap": "protocol-level",
    "Error": "protocol-level",
    "Unknown": "protocol-level",
    "Identifierlist": "protocol-level",
}

EXTRACTOR_KEYS = {
    "li": "Li et al.",
    "gaur_expert": "GAUR (expert)",
    "gaur_chatgpt": "GAUR (ChatGPT)",
    "gaur_mistral": "GAUR (Mistral)",
    "loginov": "Loginov et al.",
    "kakisim": "Kakisim",
    "cv": "CountVect",
}


CATEGORY_ORDER = ["lexical", "syntactic", "protocol-level", "user-level"]
# LNCS figures: Times New Roman matches the paper body font.
# Alternatives: "STIX Two Text" (open-source Times clone), "serif" (system default).
PAPER_FONT = "Times New Roman"
PAPER_FONT_SIZE = 13  # base size; sub-labels and value annotations use smaller sizes

N_SAMPLES = 50_000
TESTING_N_SAMPLES = 1_000

DISC_THRESHOLD = 0.5
LABEL_THRESHOLD = 0.75


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

    feature_to_cat = (
        results_df.drop_duplicates("feature").set_index("feature")["category"].to_dict()
        if "category" in results_df.columns
        else {}
    )

    def sort_key(f):
        cat = feature_to_cat.get(f, "unknown")
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
        cat = feature_to_cat.get(f, "unknown")
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
        for xoff, label in [(-0.25, "Indisc."), (0.25, "Label")]:
            annotations.append(
                dict(
                    x=j + xoff,
                    xref="x",
                    y=1.0,
                    yref="paper",
                    text=label,
                    showarrow=False,
                    font=dict(size=PAPER_FONT_SIZE, color="#444", family=PAPER_FONT),
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
                    font=dict(size=10, family=PAPER_FONT),
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
                    font=dict(size=10, family=PAPER_FONT),
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
                    colorbar=dict(
                        title=dict(
                            text="Accuracy",
                            font=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
                        ),
                        tickfont=dict(size=PAPER_FONT_SIZE - 1, family=PAPER_FONT),
                        x=1.02,
                        len=0.85,
                        thickness=14,
                    ),
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

    fig.update_layout(
        shapes=shapes,
        annotations=annotations,
        font=dict(family=PAPER_FONT, size=PAPER_FONT_SIZE),
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(n_cols)),
            ticktext=pairs,
            title=dict(
                text="Dataset pair", font=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT)
            ),
            tickfont=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
            range=[-0.5, n_cols - 0.5],
            side="bottom",
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(n_rows)),
            ticktext=row_labels,
            title=dict(
                text=f"Features — {extractor_name}",
                font=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
            ),
            tickfont=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
            range=[n_rows - 0.5, -0.5],
            autorange=False,
            showgrid=False,
            zeroline=False,
        ),
        plot_bgcolor="white",
        height=max(400, n_rows * 28 + 160),
        width=max(600, n_cols * 160 + 450),
        margin=dict(l=230, r=160, t=40, b=60),
    )

    out_path = output_dir / f"discriminability_{extractor_name}.pdf"
    fig.write_image(str(out_path))
    print(f"Saved heatmap: {out_path}")


def compute_mean_scores(results_df: pd.DataFrame) -> pd.DataFrame:
    """Average domain_inv and label_acc per (extractor, feature) across dataset pairs."""
    agg_cols = [
        c for c in ["disc_acc", "domain_inv", "label_acc"] if c in results_df.columns
    ]
    means = results_df.groupby(["extractor", "feature"])[agg_cols].mean().reset_index()
    feature_to_cat = (
        results_df.drop_duplicates("feature").set_index("feature")["category"].to_dict()
    )
    means["category"] = means["feature"].map(lambda f: feature_to_cat.get(f, "unknown"))
    return means


def plot_mean_scatter(means_df: pd.DataFrame, output_dir: Path):
    """2-D scatter: mean domain_inv (x) vs mean label_acc (y) per feature, colored by category."""
    import plotly.graph_objects as go

    unknown = means_df[means_df["category"] == "unknown"]
    if not unknown.empty:
        features = sorted(unknown["feature"].unique())
        logger.warning(f"No category for features: {features}")

    category_colors = {
        "lexical": "#1f77b4",
        "syntactic": "#ff7f0e",
        "protocol-level": "#2ca02c",
        "user-level": "#9b6dce",
        "unknown": "#999999",
    }

    # TEMPORARY: label only the 3 lowest and 1 highest domain_inv features
    mean_disc = means_df.groupby("feature")["domain_inv"].mean()
    labeled_features = set(mean_disc.nsmallest(3).index) | {mean_disc.idxmax()}

    fig = go.Figure()
    for cat in CATEGORY_ORDER + ["unknown"]:
        grp = means_df[means_df["category"] == cat]
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
                text=grp["feature"].where(grp["feature"].isin(labeled_features), ""),
                textposition="top center",
                textfont=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
                name=cat,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Category: {cat}<br>"
                    "Domain indiscriminability: %{x:.3f}<br>"
                    "Label acc: %{y:.3f}<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=DISC_THRESHOLD, line=dict(color="gray", dash="dash", width=1))
    fig.add_hline(y=LABEL_THRESHOLD, line=dict(color="gray", dash="dash", width=1))

    fig.update_layout(
        font=dict(family=PAPER_FONT, size=PAPER_FONT_SIZE),
        xaxis=dict(
            title=dict(
                text="Mean domain indiscriminability",
                font=dict(size=1.5 * PAPER_FONT_SIZE, family=PAPER_FONT),
            ),
            tickfont=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
            range=[-0.05, 1.05],
        ),
        yaxis=dict(
            title=dict(
                text="Mean label prediction accuracy",
                font=dict(size=1.5 * PAPER_FONT_SIZE, family=PAPER_FONT),
            ),
            tickfont=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
            range=[0.45, 1.02],
        ),
        legend=dict(
            title=dict(
                text="Feature type",
                font=dict(size=1.5 * PAPER_FONT_SIZE, family=PAPER_FONT),
            ),
            font=dict(size=1.2 * PAPER_FONT_SIZE, family=PAPER_FONT),
            x=0.02,
            y=0.02,
            xanchor="left",
            yanchor="bottom",
            # bgcolor="rgba(255,255,255,0.8)",
            # bordercolor="lightgrey",
            # borderwidth=1,
        ),
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


def compute_quadrant_stats(means_df: pd.DataFrame, output_dir: Path):
    """Count features per category per scatter quadrant; save to CSV."""

    def _quadrant(row):
        x = "high-indisc" if row["domain_inv"] >= DISC_THRESHOLD else "low-indisc"
        y = "high-label" if row["label_acc"] >= LABEL_THRESHOLD else "low-label"
        return f"{x} / {y}"

    df = means_df.copy()
    df["quadrant"] = df.apply(_quadrant, axis=1)

    all_quadrants = [
        f"{x} / {y}"
        for x in ("low-indisc", "high-indisc")
        for y in ("low-label", "high-label")
    ]
    counts = df.groupby(["quadrant", "category"]).size().reset_index(name="count")
    pivot = (
        counts.pivot(index="quadrant", columns="category", values="count")
        .reindex(index=all_quadrants)
        .fillna(0)
        .astype(int)
    )
    ordered_cols = [c for c in CATEGORY_ORDER if c in pivot.columns]
    pivot = pivot.reindex(columns=ordered_cols).fillna(0).astype(int)
    pivot["total"] = pivot.sum(axis=1)
    pivot = pivot.reset_index()

    csv_path = output_dir / "feature_discriminability_quadrant_stats.csv"
    pivot.to_csv(csv_path, index=False)
    print(f"Saved quadrant stats: {csv_path}")


def _generate_plots(results_df: pd.DataFrame, output_dir: Path):
    """Shared pipeline: heatmaps → mean CSV → scatter plot."""
    for extractor_name in results_df["extractor"].unique():
        subset = results_df[results_df["extractor"] == extractor_name]
        if "constant" in subset.columns:
            subset = subset[~subset["constant"]]
        make_heatmap(subset, extractor_name, output_dir)

    means = compute_mean_scores(results_df)

    csv_path = output_dir / "feature_discriminability_mean.csv"
    means.to_csv(csv_path, index=False)
    print(f"Saved mean results: {csv_path}")

    plot_mean_scatter(means, output_dir)
    compute_quadrant_stats(means, output_dir)


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
        "--scatter-from-mean",
        metavar="PATH",
        help="Plot the mean scatter figure from a previously saved mean results CSV",
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
        default="output/results/feature_discriminability",
        help="Output directory for CSV and PDF plots",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help=f"Subsample to {TESTING_N_SAMPLES} per dataset for quick iteration",
    )
    parser.add_argument(
        "--fe",
        default="all",
        metavar="EXTRACTORS",
        help=(
            "Comma-separated list of extractors to run (default: all). "
            f"Valid keys: {', '.join(EXTRACTOR_KEYS)}"
        ),
    )
    args = parser.parse_args()

    if args.scatter_from_mean:
        print(f"Loading mean results from {args.scatter_from_mean} ...")
        means_df = pd.read_csv(args.scatter_from_mean)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_mean_scatter(means_df, output_dir)
        compute_quadrant_stats(means_df, output_dir)
        return

    if not args.from_csv and not args.dataset:
        parser.error("--dataset is required when --from-csv is not specified")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_csv:
        print(f"Loading results from {args.from_csv} ...")
        results_df = pd.read_csv(args.from_csv)
        _generate_plots(results_df, output_dir)
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

    all_extractors = [
        ("Li et al.", LiExtractor()),
        ("GAUR (expert)", GaurExtractor(use_hybrid=False, mode="expert")),
        ("GAUR (ChatGPT)", GaurExtractor(use_hybrid=False, mode="chatgpt")),
        ("GAUR (Mistral)", GaurExtractor(use_hybrid=False, mode="mistral")),
        ("Loginov et al.", LoginovExtractor()),
        ("Kakisim", KakisimExtractor(views=["C"])),
        ("CountVect", CountVectExtractor(max_features=10_000)),
    ]

    if args.fe == "all":
        extractors = all_extractors
    else:
        requested = [k.strip() for k in args.fe.split(",")]
        unknown = [k for k in requested if k not in EXTRACTOR_KEYS]
        if unknown:
            parser.error(
                f"Unknown extractor key(s): {unknown}. Valid: {list(EXTRACTOR_KEYS)}"
            )
        selected_names = {EXTRACTOR_KEYS[k] for k in requested}
        extractors = [
            (name, ext) for name, ext in all_extractors if name in selected_names
        ]

    all_results = []

    for extractor_name, extractor in extractors:
        print(f"\n=== Extractor: {extractor_name} ===")

        # CountVectorizer must be fit on combined training data before any transform
        if isinstance(extractor, CountVectExtractor):
            all_train = pd.concat(
                [disc_train_dfs[n] for n in datasets]
                + [label_train_dfs[n] for n in datasets]
            )
            extractor.vectorizer.fit(all_train["full_query"])
            extractor._fitted = True
            print(
                f"  CountVect vocabulary size: {len(extractor.vectorizer.vocabulary_)}"
            )

        def _extract(df: pd.DataFrame, _ext=extractor) -> pd.DataFrame:
            feat = _ext.extract_features(df)
            if hasattr(feat, "toarray"):
                return pd.DataFrame(
                    feat.toarray(), columns=_ext.get_feature_names_out()
                )
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
                category = FEATURE_CATEGORIES.get(col)
                if category is None:
                    if isinstance(extractor, CountVectExtractor):
                        category = (
                            "protocol-level"
                            if col.upper() in mysql_keywords
                            or col.upper() in mysql_functions
                            else "user-level"
                        )
                    else:
                        logger.warning(
                            f"No category for feature '{col}' "
                            f"(extractor: {extractor_name})"
                        )
                        category = "unknown"
                extractor_results.append(
                    {
                        "extractor": extractor_name,
                        "feature": col,
                        "pair": pair_label,
                        "disc_acc": disc_acc,
                        "domain_inv": max(0.0, min(1.0, 2 * (1 - disc_acc))),
                        "label_acc": label_acc,
                        "category": category,
                    }
                )

        # For CountVect, keep only features with mean label_acc > 0.75 across pairs,
        # plus 20 randomly sampled dimensions to represent the full vocabulary.
        if isinstance(extractor, CountVectExtractor):
            cv_df = pd.DataFrame(extractor_results)
            all_cv_features = cv_df["feature"].unique().tolist()
            n_before = len(all_cv_features)
            rng_cv = np.random.default_rng(seed=2)
            kept_features = set(
                rng_cv.choice(
                    all_cv_features, size=min(500, n_before), replace=False
                ).tolist()
            )
            extractor_results = [
                r for r in extractor_results if r["feature"] in kept_features
            ]
            print(
                f"  CountVect: plotting {len(kept_features)}/{n_before} randomly sampled features (seed=2)"
            )

        ext_df = pd.DataFrame(extractor_results)
        constant_features = (
            ext_df.groupby("feature")["domain_inv"]
            .apply(lambda s: (s == 1.0).all())
            .pipe(lambda s: s[s].index.tolist())
        )
        if constant_features:
            print(
                f"  Constant features (excluded from heatmap): {sorted(constant_features)}"
            )
        ext_df["constant"] = ext_df["feature"].isin(constant_features)
        all_results.extend(ext_df.to_dict("records"))

    results_df = pd.DataFrame(all_results)
    csv_path = output_dir / "feature_discriminability.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved results: {csv_path}")

    _generate_plots(results_df, output_dir)


if __name__ == "__main__":
    main()
