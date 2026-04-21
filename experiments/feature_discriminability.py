import argparse
import logging
import sys
from collections import defaultdict
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
from extractors.kakisim import KakisimExtractor
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
    # Exceptions from labelised:
    "columns": "protocol-level",  # https://dev.mysql.com/doc/refman/8.4/en/show-columns.html
    "information_schema": "protocol-level",  # https://dev.mysql.com/doc/refman/8.4/en/innodb-information-schema-tables.html
    "end": "protocol-level",  # https://dev.mysql.com/doc/refman/8.4/en/begin-end.html
    # Li
    "len_query": "lexical",
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
    "has_ascii": "lexical",
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
    "has_multiline_comment": "protocol-level",
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
    "CREATE": "protocol-level",
    "DELETE": "protocol-level",
    "MODIFY": "protocol-level",
    "EXECUTE": "protocol-level",
    "READ": "protocol-level",
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
    "s1_n_keywords": "syntactic",  # Need list of keywords (grammar)
    "s1_n_alpha": "lexical",
    "s1_n_numeric": "lexical",
    "s1_n_mixed": "lexical",
    "s2_n_keywords": "lexical",  # Need list of keywords (grammar)
    "s2_n_alpha": "lexical",
    "s2_n_numeric": "lexical",
    "s2_n_mixed": "lexical",
    # Kakisim (view C)
    "par": "lexical",
    "dll": "protocol-level",  # There is an a priori of the impact of part of parse tree on the system
    "dml": "protocol-level",  # There is an a priori of the impact of part of parse tree on the system
    "keyw": "syntactic",
    "int": "lexical",
    "cte": "protocol-level",  # Common table expression, syntactic group with a meaning, is protocol.
    "dcl": "protocol-level",  # There is an a priori of the impact of part of parse tree on the system
    "order": "protocol-level",  # Focus on specific keywords: because of the impact, protocol
    "where": "protocol-level",  # Focus on specific keywords: because of the impact, protocol
    "hexadecimal": "lexical",
    "quot": "lexical",
    "punct": "lexical",
    "wildcard": "protocol-level",  # Focus on specific keywords: because of the impact, protocol
    "comparison": "syntactic",  # need for AST knowledge: <= in identifier is different than as a keywords.
    "oper": "lexical",
    "builtin": "protocol-level",
    "func": "protocol-level",
    "identifi": "protocol-level",  # Focus on specific terminals: because of the impact, protocol
    "escap": "protocol-level",
    "error": "protocol-level",
    "unknown": "protocol-level",
    "identifierlist": "protocol-level",  # Focus on specific terminals: because of the impact, protocol
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

# information_schema, présent en très petites qté dans rq d'administration, mais est typiquement ciblé par sqlmap.
LABELS_TO_DISPLAY = [
    # "information_schema", overflows
    "c_space",
    "n_nonterminal",
    "n_terminal",
    "c_round_brackets",
    "dll",
    # "has_comment",
    "FUNCTION",
    "TABLE",
    # "has_connection_keywords",
    "has_database_keywords",
    "has_tautology",
    "has_count",
    # "columns",
    "Data Definition",
    # "avg_c_sqlkywds",
    "DML_SELECT",
    "keyw",
    "employees",
    # "airport",
    "businessentityid",
    # "name",
    "employee_id",
    "london",
]

FEATURES_TO_HIDE: dict[str, set[str]] = {
    "Kakisim": {"error", "unknown"},
}

CATEGORY_ORDER = ["lexical", "syntactic", "protocol-level", "user-level"]
PAPER_FONT = "CMU Serif"
PAPER_FONT_SIZE = 13  # annotation labels on data points (kept small to avoid clutter)
_AXIS_FONT = dict(family=PAPER_FONT, size=22)
_TICK_FONT = dict(family=PAPER_FONT, size=18)

N_SAMPLES = 50_000
TESTING_N_SAMPLES = 1_000
DISC_THRESHOLD = 0.5
LABEL_THRESHOLD = 0.5


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def sample_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    return df.sample(n=min(n, len(df)), random_state=2).reset_index(drop=True)


def sample_balanced(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """50/50 attack/normal sample."""
    n_each = min(n // 2, (df["label"] == 0).sum(), (df["label"] == 1).sum())
    normal = df[df["label"] == 0].sample(n=n_each, random_state=2)
    attack = df[df["label"] == 1].sample(n=n_each, random_state=2)
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
    """Train DT to distinguish dataset A from B; evaluate on held-out test split."""
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
    feat_train: pd.DataFrame,
    labels_train: np.ndarray,
    feat_test: pd.DataFrame,
    labels_test: np.ndarray,
    col: str,
) -> float:
    """Train DT on source train; evaluate on source test."""
    X_train = feat_train[[col]].fillna(0).to_numpy(dtype=float)
    X_test = feat_test[[col]].fillna(0).to_numpy(dtype=float)

    if np.std(X_train) == 0:
        return 0.5

    clf = DecisionTreeClassifier(random_state=2)
    clf.fit(X_train, labels_train)
    return float(clf.score(X_test, labels_test))


def _feature_category(col, is_cv: bool) -> str:
    if not isinstance(col, str):
        return "unknown"
    if col in FEATURE_CATEGORIES:
        return FEATURE_CATEGORIES[col]
    if is_cv:
        return (
            "protocol-level"
            if col.upper() in mysql_keywords or col.upper() in mysql_functions
            else "user-level"
        )
    logger.warning(f"No category for feature '{col}'")
    return "unknown"


def plot_scatter(means_df: pd.DataFrame, output_dir: Path):
    """Scatter: mean domain_inv (x) vs mean label_acc (y), colored by category.

    Single PDF aggregating all extractors and sources.
    """
    import plotly.graph_objects as go

    category_colors = {
        "lexical": "#1f77b4",
        "syntactic": "#ff7f0e",
        "protocol-level": "#2ca02c",
        "user-level": "#9b6dce",
        "unknown": "#999999",
    }

    # Cap each extractor to 2000 features before averaging
    def _cap_extractor(df):
        # if len(df) > 2000:
        # return df.sample(n=2000, random_state=2)
        return df

    means_df = (
        means_df.groupby("extractor", group_keys=False)
        .apply(_cap_extractor)
        .reset_index(drop=True)
    )

    # Remove features flagged in FEATURES_TO_HIDE
    for ext, feats in FEATURES_TO_HIDE.items():
        means_df = means_df[
            ~((means_df["extractor"] == ext) & (means_df["feature"].isin(feats)))
        ]

    # Average across all sources and extractors
    agg = (
        means_df.groupby(["feature", "category"])[["mean_domain_inv", "label_acc"]]
        .mean()
        .reset_index()
    )

    # agg = agg[~((agg["label_acc"] < 0.6) & (agg["mean_domain_inv"] > 0.8))]

    fig = go.Figure()
    for cat in CATEGORY_ORDER + ["unknown"]:
        grp = agg[agg["category"] == cat]
        if grp.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=grp["mean_domain_inv"],
                y=grp["label_acc"],
                mode="markers",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color=category_colors.get(cat, "#999999"),
                    line=dict(width=0.8, color="white"),
                ),
                name=cat,
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"Category: {cat}<br>"
                    "Domain indiscriminability: %{x:.3f}<br>"
                    "Label acc: %{y:.3f}<extra></extra>"
                ),
                customdata=grp["feature"],
            )
        )

    # Add text labels with thin lines connecting to markers
    for _, row in agg[agg["feature"].isin(LABELS_TO_DISPLAY)].iterrows():
        fig.add_annotation(
            x=row["mean_domain_inv"],
            y=row["label_acc"],
            text=row["feature"],
            showarrow=True,
            arrowhead=0,
            arrowwidth=0.5,
            arrowcolor="gray",
            ax=0,
            ay=-20,
            font=dict(size=PAPER_FONT_SIZE, family=PAPER_FONT),
        )

    fig.add_vline(x=DISC_THRESHOLD, line=dict(color="gray", dash="dash", width=1))
    fig.add_hline(y=LABEL_THRESHOLD, line=dict(color="gray", dash="dash", width=1))

    # Quadrant type labels
    quadrant_labels = [
        ("Type 1", -0.03, 1.01, "left", "top"),
        ("Type 2", 1.03, 1.01, "right", "top"),
        ("Type 3", 1.03, -0.04, "right", "bottom"),
        ("Type 4", -0.03, -0.04, "left", "bottom"),
    ]
    for label, x, y, xa, ya in quadrant_labels:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            xanchor=xa,
            yanchor=ya,
            font=dict(size=20, family=PAPER_FONT, color="black"),
        )

    fig.update_layout(
        font=dict(family=PAPER_FONT, size=18),
        xaxis=dict(
            title=dict(text="Mean domain indiscriminability", font=_AXIS_FONT),
            tickfont=_TICK_FONT,
            range=[-0.05, 1.05],
        ),
        yaxis=dict(
            title=dict(text="Mean predictive power", font=_AXIS_FONT),
            tickfont=_TICK_FONT,
            range=[-0.05, 1.02],
        ),
        legend=dict(
            title=dict(text="Feature type", font=_AXIS_FONT),
            font=_TICK_FONT,
            x=0.02,
            y=0.5,
            xanchor="left",
            yanchor="middle",
        ),
        plot_bgcolor="white",
        width=900,
        height=750,
        margin=dict(l=60, r=40, t=40, b=60),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee")
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee")

    out_path = output_dir / "fd_all.pdf"
    fig.write_image(str(out_path))
    print(f"Saved scatter: {out_path}")

    # Quadrant counts per category
    quadrant_names = {
        "low-indisc / low-label": (
            agg["mean_domain_inv"] < DISC_THRESHOLD,
            agg["label_acc"] < LABEL_THRESHOLD,
        ),
        "low-indisc / high-label": (
            agg["mean_domain_inv"] < DISC_THRESHOLD,
            agg["label_acc"] >= LABEL_THRESHOLD,
        ),
        "high-indisc / low-label": (
            agg["mean_domain_inv"] >= DISC_THRESHOLD,
            agg["label_acc"] < LABEL_THRESHOLD,
        ),
        "high-indisc / high-label": (
            agg["mean_domain_inv"] >= DISC_THRESHOLD,
            agg["label_acc"] >= LABEL_THRESHOLD,
        ),
    }
    cats = [c for c in CATEGORY_ORDER if c != "unknown"]
    rows = []
    for qname, (x_mask, y_mask) in quadrant_names.items():
        counts = agg[x_mask & y_mask].groupby("category").size()
        row = {"quadrant": qname}
        total = 0
        for cat in cats:
            n = counts.get(cat, 0)
            row[cat] = n
            total += n
        row["total"] = total
        rows.append(row)
    quadrant_df = pd.DataFrame(rows)
    print("\nFeatures per quadrant per category:")
    print(quadrant_df.to_csv(index=False), end="")
    quadrant_csv = output_dir / "fd_quadrant_counts.csv"
    quadrant_df.to_csv(quadrant_csv, index=False)
    print(f"Saved quadrant counts: {quadrant_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Source-in-domain feature discriminability evaluation"
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
        help=f"Subsample to {TESTING_N_SAMPLES} rows for quick iteration",
    )
    parser.add_argument(
        "--from-folder",
        metavar="PATH",
        help="Skip computation; re-plot from all fd_*.csv files in a folder",
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

    if not args.from_folder and not args.dataset:
        parser.error("--dataset is required when --from-folder is not specified")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.from_folder:
        folder = Path(args.from_folder)
        csv_files = sorted(folder.glob("fd_*.csv"))
        if not csv_files:
            parser.error(f"No fd_*.csv files found in {folder}")
        print(f"Loading {len(csv_files)} CSV(s) from {folder} ...")
        results_df = pd.concat([pd.read_csv(p) for p in csv_files], ignore_index=True)
        # This applies exceptions.
        results_df["category"] = results_df.apply(
            lambda r: _feature_category(r["feature"], r["extractor"] == "CountVect"),
            axis=1,
        )
        plot_scatter(results_df, output_dir)
        return

    n_samples = TESTING_N_SAMPLES if args.testing else N_SAMPLES
    if args.testing:
        print(f"[testing] using up to {n_samples} samples per split")

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    all_extractors_cfg = [
        ("Li et al.", "li"),
        ("GAUR (expert)", "gaur_expert"),
        ("GAUR (ChatGPT)", "gaur_chatgpt"),
        ("GAUR (Mistral)", "gaur_mistral"),
        ("Loginov et al.", "loginov"),
        ("Kakisim", "kakisim"),
        ("CountVect", "cv"),
    ]

    if args.fe == "all":
        extractors_cfg = all_extractors_cfg
    else:
        requested = [k.strip() for k in args.fe.split(",")]
        unknown = [k for k in requested if k not in EXTRACTOR_KEYS]
        if unknown:
            parser.error(
                f"Unknown extractor key(s): {unknown}. Valid: {list(EXTRACTOR_KEYS)}"
            )
        selected = {EXTRACTOR_KEYS[k] for k in requested}
        extractors_cfg = [(n, k) for n, k in all_extractors_cfg if n in selected]

    all_results = []

    for src_name, src_df in datasets.items():
        print(f"\n=== Source: {src_name} ===")

        src_train = src_df[src_df["split"] == "train"].reset_index(drop=True)
        src_test = src_df[src_df["split"] == "test"].reset_index(drop=True)

        for ext_name, ext_key in extractors_cfg:
            print(f"  --- {ext_name} ---")

            # Build and fit extractor on source train
            if ext_key == "li":
                extractor = LiExtractor()
            elif ext_key == "gaur_expert":
                extractor = GaurExtractor(use_hybrid=False, mode="expert")
            elif ext_key == "gaur_chatgpt":
                extractor = GaurExtractor(use_hybrid=False, mode="chatgpt")
            elif ext_key == "gaur_mistral":
                extractor = GaurExtractor(use_hybrid=False, mode="mistral")
            elif ext_key == "loginov":
                extractor = LoginovExtractor()
                fit_corpus = (
                    sample_n(src_train, n_samples) if args.testing else src_train
                )
                extractor.prepare_for_training(fit_corpus)
                print(f"    Loginov valid_schars: {len(extractor.valid_schars)}")
            elif ext_key == "kakisim":
                extractor = KakisimExtractor(views=["C"])
            else:  # cv
                extractor = CountVectExtractor(max_features=10_000)
                fit_corpus = (
                    sample_n(src_train, n_samples) if args.testing else src_train
                )
                extractor.vectorizer.fit(fit_corpus["full_query"])
                extractor._fitted = True
                print(f"    CV vocab: {len(extractor.vectorizer.vocabulary_)} tokens")

            def _extract(df: pd.DataFrame, _ext=extractor) -> pd.DataFrame:
                feat = _ext.extract_features(df)
                if hasattr(feat, "toarray"):
                    return pd.DataFrame(
                        feat.toarray(), columns=_ext.get_feature_names_out()
                    )
                return feat if isinstance(feat, pd.DataFrame) else pd.DataFrame(feat)

            #  Label prediction
            # Train split is normal-only; sample balanced pool from test split,
            # then split 30/70 for DT train/eval.
            lp_pool = sample_balanced(src_test, n_samples)
            lp_train_df, lp_test_df = train_test_split(
                lp_pool, test_size=0.7, random_state=2, stratify=lp_pool["label"]
            )

            print(
                f"    Label prediction: train {len(lp_train_df)}, test {len(lp_test_df)}"
            )
            lp_train_feats = _extract(lp_train_df)
            lp_test_feats = _extract(lp_test_df)
            lp_train_labels = lp_train_df["label"].to_numpy()
            lp_test_labels = lp_test_df["label"].to_numpy()

            feature_cols = lp_train_feats.columns.tolist()
            print(f"    Features: {len(feature_cols)}")

            label_acc_per_feat: dict[str, float] = {}
            for col in feature_cols:
                label_acc_per_feat[col] = run_label_prediction(
                    lp_train_feats,
                    lp_train_labels,
                    lp_test_feats,
                    lp_test_labels,
                    col,
                )

            # ── Discriminability vs each target ─────────────────────────────
            # TEMPORARY (uncomment to verify attack leakage hypothesis):
            # src_pool = sample_n(src_test, n_samples)
            # src_disc_train, src_disc_test = train_test_split(
            #     src_pool, test_size=0.5, random_state=2
            # )
            src_pool = sample_n(src_train, n_samples)
            src_disc_train, src_disc_test = train_test_split(
                src_pool, test_size=0.7, random_state=2
            )
            src_disc_train_feats = _extract(src_disc_train)
            src_disc_test_feats = _extract(src_disc_test)

            # dict[col] -> list of (tgt_name, disc_acc, domain_inv)
            disc_per_feat: dict[str, list[tuple]] = defaultdict(list)

            for tgt_name, tgt_df in datasets.items():
                if tgt_name == src_name:
                    continue
                print(f"    Discriminability {src_name} vs {tgt_name} ...")

                tgt_train = tgt_df[tgt_df["split"] == "train"].reset_index(drop=True)
                tgt_pool = sample_n(tgt_train, n_samples)
                tgt_disc_train, tgt_disc_test = train_test_split(
                    tgt_pool, test_size=0.7, random_state=2
                )
                tgt_disc_train_feats = _extract(tgt_disc_train)
                tgt_disc_test_feats = _extract(tgt_disc_test)

                print(
                    f"      DT train {len(src_disc_train)}+{len(tgt_disc_train)}"
                    f" | DT test {len(src_disc_test)}+{len(tgt_disc_test)}"
                )

                for col in feature_cols:
                    disc_acc = run_discriminability(
                        src_disc_train_feats,
                        tgt_disc_train_feats,
                        src_disc_test_feats,
                        tgt_disc_test_feats,
                        col,
                    )
                    disc_per_feat[col].append(
                        (tgt_name, disc_acc, min(1.0, 2 * (1 - disc_acc)))
                    )

            #  Aggregate
            for col in feature_cols:
                pairs_data = disc_per_feat[col]
                raw_accs = [d[1] for d in pairs_data]
                domain_invs = [d[2] for d in pairs_data]
                disc_accs_str = ";".join(f"{d[0]}:{d[1]:.4f}" for d in pairs_data)
                all_results.append(
                    {
                        "source": src_name,
                        "extractor": ext_name,
                        "feature": col,
                        "mean_disc_acc": float(np.mean(raw_accs)),
                        "disc_accs": disc_accs_str,
                        "mean_domain_inv": float(np.mean(domain_invs)),
                        "label_acc": min(
                            1.0, max(0.0, 2 * (label_acc_per_feat[col] - 0.5))
                        ),
                        "category": _feature_category(col, ext_key == "cv"),
                    }
                )

    results_df = pd.DataFrame(all_results)

    unknown_feats = results_df[results_df["category"] == "unknown"]
    if not unknown_feats.empty:
        print("\nFeatures with unknown category:")
        for _, row in (
            unknown_feats[["extractor", "feature"]].drop_duplicates().iterrows()
        ):
            print(f"  [{row['extractor']}] {row['feature']}")

    # One CSV per extractor, named after it
    for ext_name, ext_df in results_df.groupby("extractor"):
        safe_name = ext_name.replace(" ", "_").replace("(", "").replace(")", "")
        csv_path = output_dir / f"fd_{safe_name}.csv"
        ext_df.to_csv(csv_path, index=False)
        print(f"Saved: {csv_path}")

    plot_scatter(results_df, output_dir)


if __name__ == "__main__":
    main()
