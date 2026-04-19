"""
Generic vs Specialised Comparison + Transfer Learning Matrix Report

Generates all comparison plots across all feature extractor types:
  1. Generic vs Specialised: bar charts, ROC curves, recall heatmaps
  2. Transfer Learning matrices: AUROC / AUPRC / F1 for every train×test pair

Expected results directory structure (flat, one directory per model×scenario):

    {results_dir}/
      ae_li_generic/
        ae_li_BCD_on_A/results.csv
        ae_li_BCD_on_A/roc_curves/ae_li_BCD.csv
        ae_li_ABC_on_D/results.csv   ← also used by TL matrix
        ...
      ae_li_specialised/
        ae_li_A_on_A/results.csv
        ae_li_A_on_A/roc_curves/ae_li_A.csv
        ae_li_A_on_B/results.csv     ← cross-domain, used by TL matrix
        ...
      ae_securebert_generic/ ...
      ae_securebert_specialised/ ...
      ...

Output layout:

    {output_dir}/
      generic-vs-specialised/
        metrics_ae_li.png
        roc_ae_li.png
        recall_technique_ae_li.png
        recall_stmt_ae_li.png
        ...
        balanced_accuracy_combined_heatmap.png
      tl-matrix/
        tl_ae_li_auroc.png
        tl_ae_li_auprc.png
        tl_ae_li_f1.png
        ...

Usage:
    # Auto-discover models from results directory
    python experiments/report_generic_vs_specialised.py \\
        --results-dir ~/experiences-results/2026-02-23

    # Specify models explicitly
    python experiments/report_generic_vs_specialised.py \\
        --results-dir ~/experiences-results/2026-02-23 \\
        --models ae_li ae_securebert ae_kakisim_c ae_loginov

    # Custom output directory and format(s)
    python experiments/report_generic_vs_specialised.py \\
        --results-dir ~/experiences-results/2026-02-23 \\
        --output-dir output/results \\
        --format png pdf

    # Include WAFAMOLE (E) as external test-only dataset
    python experiments/report_generic_vs_specialised.py \\
        --results-dir ~/experiences-results/2026-02-23 \\
        --include-wafamole
"""

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DATASETS = ["AdventureWorks", "OHR", "OurAirports", "sakila"]

DATASET_LETTERS = {
    "OurAirports": "A",
    "sakila": "B",
    "AdventureWorks": "C",
    "OHR": "D",
}
ALL_LETTERS = set(DATASET_LETTERS.values())

# Human-readable labels for known model prefixes; unknown prefixes fall back to the prefix itself
KNOWN_LABELS: dict[str, str] = {
    "ae_li": "Li et al.",
    "ae_securebert": "SecureBERT",
    "ae_gaur": "Gaur (Expert)",
    "ae_gaur_chatgpt": "Gaur (ChatGPT)",
    "ae_loginov": "Loginov et al.",
    "ae_codebert": "CodeBERT",
    "ae_sentbert": "SentenceBERT (all-mpnet-base-v2)",
}

COLORS = {"generic": "#636EFA", "specialised": "#EF553B"}
CONCEPT_DRIFT_COLORS = {"origin": "#636EFA", "shifted": "#EF553B"}

METRIC_LABELS: dict[str, str] = {
    "accuracy": "Accuracy (%)",
    "precision": "Precision (%)",
    "recall": "Recall (%)",
    "fone": "F1 Score (%)",
    "rocauc": "AUROC",
    "auprc": "AUPRC",
    "balanced_accuracy_per_technique": "Balanced Accuracy (%)",
}

MAIN_METRICS = ["fone", "rocauc", "auprc", "balanced_accuracy_per_technique"]

TECHNIQUE_COLS: dict[str, str] = {
    "recalltime": "Time-based",
    "recallboolean": "Boolean",
    "recallstacked": "Stacked",
    "recallerror": "Error-based",
    "recallunion": "UNION",
    "recallinsider": "Insider",
    "recallinline": "Inline",
}

STMT_TYPE_COLS: dict[str, str] = {
    "recall_select": "SELECT",
    "recall_insert": "INSERT",
    "recall_update": "UPDATE",
    "recall_delete": "DELETE",
    "recall_insider": "Insider",
}

_PCT_COLS = [
    "fone",
    "accuracy",
    "precision",
    "recall",
    "fpr",
    "balanced_accuracy_per_technique",
]

# TL matrix: leave-one-out training sets and single-dataset training sets
TL_GENERIC_TRAIN_SETS = ["ABC", "ABD", "ACD", "BCD"]
TL_SPECIALISED_TRAIN_SETS = ["D", "C", "B", "A"]
TL_TEST_SETS = ["A", "B", "C", "D"]
TL_METRIC_DISPLAY = {"auroc": "AUROC", "auprc": "AUPRC", "f1": "F1 Score"}


def leave_one_out_complement(letter: str) -> str:
    return "".join(sorted(ALL_LETTERS - {letter}))


def model_label(prefix: str) -> str:
    return KNOWN_LABELS.get(prefix, prefix)


def _parse_pct(val) -> float:
    """Parse a value that may be a '88.53%' string or already a float."""
    if isinstance(val, str):
        return float(val.rstrip("%"))
    return float(val)


def _normalize_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert percent-string columns to float in-place."""
    for col in _PCT_COLS:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].str.rstrip("%").astype(float)
    for col in [c for c in df.columns if c.startswith("recall") and c != "recall"]:
        if df[col].dtype == object:
            df[col] = df[col].str.rstrip("%").astype(float)
    return df


def load_results(results_dir: Path, model_prefix: str) -> pd.DataFrame:
    """Load generic/specialised results for the 4 leave-one-out configurations.

    Paths:
      generic:     {results_dir}/{prefix}_generic/{prefix}_{complement}_on_{letter}/results.csv
      specialised: {results_dir}/{prefix}_specialised/{prefix}_{letter}_on_{letter}/results.csv
    """
    rows = []
    for dataset in DATASETS:
        letter = DATASET_LETTERS[dataset]
        complement = leave_one_out_complement(letter)
        for scenario in ("generic", "specialised"):
            if scenario == "generic":
                run = f"{model_prefix}_{complement}_on_{letter}"
            else:
                run = f"{model_prefix}_{letter}_on_{letter}"
            path = results_dir / f"{model_prefix}_{scenario}" / run / "results.csv"
            if path.exists():
                df = pd.read_csv(path)
                df["dataset"] = dataset
                df["type"] = scenario
                rows.append(df)

    if not rows:
        return pd.DataFrame()

    return _normalize_pct_columns(pd.concat(rows, ignore_index=True))


def discover_models(results_dir: Path) -> list[str]:
    """Return model prefixes found as *_generic/*_specialised pairs in results_dir."""
    prefixes = []
    for p in sorted(results_dir.glob("*_generic")):
        prefix = p.name.removesuffix("_generic")
        if (results_dir / f"{prefix}_specialised").exists():
            prefixes.append(prefix)
    return prefixes


def load_tl_matrix(
    results_dir: Path, model_prefix: str, scenario: str
) -> dict[str, pd.DataFrame]:
    """Load a full train×test results matrix for one scenario.

    Path: {results_dir}/{prefix}_{scenario}/{prefix}_{train}_on_{test}/results.csv

    Returns a dict with keys "auroc", "auprc", "f1", each a DataFrame
    indexed by training set with test-set columns.
    """
    train_sets = (
        TL_GENERIC_TRAIN_SETS if scenario == "generic" else TL_SPECIALISED_TRAIN_SETS
    )
    subdir = results_dir / f"{model_prefix}_{scenario}"

    auroc = pd.DataFrame(index=train_sets, columns=TL_TEST_SETS, dtype=float)
    auprc = pd.DataFrame(index=train_sets, columns=TL_TEST_SETS, dtype=float)
    f1 = pd.DataFrame(index=train_sets, columns=TL_TEST_SETS, dtype=float)

    for train in train_sets:
        for test in TL_TEST_SETS:
            csv_path = subdir / f"{model_prefix}_{train}_on_{test}" / "results.csv"
            if csv_path.exists():
                row = pd.read_csv(csv_path).iloc[0]
                auroc.loc[train, test] = _parse_pct(row["rocauc"])
                auprc.loc[train, test] = _parse_pct(row["auprc"])
                f1.loc[train, test] = _parse_pct(row["fone"])

    return {"auroc": auroc, "auprc": auprc, "f1": f1}


def plot_roc_curves(
    results_df: pd.DataFrame,
    title: str,
    scenarios: list[tuple[str, str, callable]],
):
    """ROC curve grid (one subplot per dataset, dynamically sized).

    scenarios: list of (label, color, roc_dir_fn) where
               roc_dir_fn(letter, complement) -> Path to the roc_curves directory.
    """
    if results_df.empty:
        print(f"  [skip] no data: {title}")
        return None

    n = len(DATASETS)
    ncols = 2
    nrows = math.ceil(n / ncols)
    fig = make_subplots(
        rows=nrows, cols=ncols, subplot_titles=DATASETS, vertical_spacing=0.12
    )
    for idx, dataset in enumerate(DATASETS):
        row, col = idx // ncols + 1, idx % ncols + 1
        letter = DATASET_LETTERS[dataset]
        complement = leave_one_out_complement(letter)

        for label, color, roc_dir_fn in scenarios:
            roc_dir = roc_dir_fn(letter, complement)
            # Use first CSV found (handles any naming convention)
            roc_files = list(roc_dir.glob("*.csv")) if roc_dir.exists() else []
            if not roc_files:
                print(f"  [warn] missing ROC data in: {roc_dir}")
                continue
            roc_df = pd.read_csv(roc_files[0])
            fig.add_trace(
                go.Scatter(
                    x=roc_df["fpr"],
                    y=roc_df["tpr"],
                    mode="lines",
                    name=label.capitalize(),
                    line=dict(color=color),
                    showlegend=(idx == 0),
                    legendgroup=label,
                ),
                row=row,
                col=col,
            )

        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(dash="dash", color="gray"),
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text="FPR", row=row, col=col)
        fig.update_yaxes(title_text="TPR", row=row, col=col)

    fig.update_layout(title=title, height=350 * nrows, width=900)
    return fig


def _heatmap_fig(results_df: pd.DataFrame, col_map: dict[str, str], title: str):
    """Generic Generic/Specialised heatmap for any recall-column mapping."""
    available = {k: v for k, v in col_map.items() if k in results_df.columns}
    if not available:
        print(f"  [skip] no matching columns: {title}")
        return None

    col_keys = list(available.keys())
    col_labels = list(available.values())

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Generic", "Specialised"],
        horizontal_spacing=0.15,
    )
    for col_idx, model_type in enumerate(("generic", "specialised"), 1):
        subset = results_df[results_df["type"] == model_type].sort_values("dataset")
        z = [[row[c] for c in col_keys] for _, row in subset.iterrows()]
        text = [[f"{v:.1f}" for v in row] for row in z]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=col_labels,
                y=subset["dataset"].tolist(),
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmin=0,
                zmax=100,
                showscale=(col_idx == 2),
                colorbar=dict(title="Recall (%)"),
            ),
            row=1,
            col=col_idx,
        )
    fig.update_layout(title=title, height=350, width=1000)
    return fig


def plot_recall_per_technique(results_df: pd.DataFrame, title: str):
    return _heatmap_fig(results_df, TECHNIQUE_COLS, title)


def plot_recall_per_statement_type(results_df: pd.DataFrame, title: str):
    return _heatmap_fig(results_df, STMT_TYPE_COLS, title)


def plot_combined_metric(
    all_results: dict[str, pd.DataFrame],
    models: list[dict],
    metric: str,
    split_col: str,
    categories: list[str],
    subplot_titles: list[str],
    title: str | None = None,
):
    """Heatmap comparing one metric across all models for two categories side-by-side.

    split_col: column that partitions rows (e.g. "type" or "split").
    categories: the two values of split_col to show (one panel each).
    subplot_titles: display names for the two panels.
    """
    rows = []
    for m in models:
        df = all_results.get(m["prefix"])
        if df is None or df.empty or metric not in df.columns:
            continue
        tmp = df[["dataset", split_col, metric]].copy()
        tmp["approach"] = m["label"]
        rows.append(tmp)

    if not rows:
        print(f"  [skip] no data for metric '{metric}'")
        return None

    combined = pd.concat(rows, ignore_index=True)
    label = METRIC_LABELS.get(metric, metric)
    if title is None:
        title = f"{label} across Feature Extractors"

    approach_order = [
        m["label"] for m in models if all_results.get(m["prefix"]) is not None
    ]
    is_pct = combined[metric].max() > 1
    fmt = ".1f" if is_pct else ".4f"
    zmin, zmax = 0, (100 if is_pct else 1)

    fig = make_subplots(
        rows=1, cols=2, subplot_titles=subplot_titles, horizontal_spacing=0.15
    )
    ds_order = sorted(combined["dataset"].unique())
    for col_idx, category in enumerate(categories, 1):
        subset = combined[combined[split_col] == category]
        z, text = [], []
        for ds in ds_order:
            row_z, row_t = [], []
            for approach in approach_order:
                mask = (subset["dataset"] == ds) & (subset["approach"] == approach)
                val = subset.loc[mask, metric]
                v = val.values[0] if len(val) else float("nan")
                row_z.append(v)
                row_t.append(f"{v:{fmt}}")
            z.append(row_z)
            text.append(row_t)
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=approach_order,
                y=ds_order,
                text=text,
                texttemplate="%{text}",
                colorscale="RdYlGn",
                zmin=zmin,
                zmax=zmax,
                showscale=(col_idx == 2),
                colorbar=dict(title=label),
            ),
            row=1,
            col=col_idx,
        )
    fig.update_layout(title=title, height=350, width=700)
    return fig


def _tl_heatmap_combined(
    generic_matrix: pd.DataFrame,
    specialised_matrix: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Side-by-side heatmap: generic (left) and specialised (right) TL matrices."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Generic", "Specialised"],
        horizontal_spacing=0.15,
    )

    for col_idx, matrix in enumerate((generic_matrix, specialised_matrix), 1):
        flat = matrix.values.astype(float)
        has_data = ~pd.isna(flat)
        is_pct = flat[has_data].max() > 1 if has_data.any() else False
        zmin = 50.0 if is_pct else 0.5
        zmax = 100.0 if is_pct else 1.0
        fmt = ".1f" if is_pct else ".3f"
        high_threshold = 80.0 if is_pct else 0.8

        annotations = []
        for train in matrix.index:
            for test in matrix.columns:
                val = matrix.loc[train, test]
                if pd.isna(val):
                    annotations.append(
                        dict(
                            x=test,
                            y=train,
                            text="N/A",
                            font=dict(color="gray", size=12),
                            showarrow=False,
                            xref=f"x{col_idx}" if col_idx > 1 else "x",
                            yref=f"y{col_idx}" if col_idx > 1 else "y",
                        )
                    )
                else:
                    is_generalization = test not in train
                    annotations.append(
                        dict(
                            x=test,
                            y=train,
                            text=f"{val:{fmt}}",
                            font=dict(
                                color="white" if val > high_threshold else "black",
                                size=14,
                                weight="bold" if is_generalization else "normal",
                            ),
                            showarrow=False,
                            xref=f"x{col_idx}" if col_idx > 1 else "x",
                            yref=f"y{col_idx}" if col_idx > 1 else "y",
                        )
                    )

        fig.add_trace(
            go.Heatmap(
                z=flat,
                x=matrix.columns.tolist(),
                y=matrix.index.tolist(),
                colorscale="RdYlGn",
                zmin=zmin,
                zmax=zmax,
                showscale=(col_idx == 2),
            ),
            row=1,
            col=col_idx,
        )
        fig.layout.annotations = list(fig.layout.annotations) + annotations

    fig.update_layout(
        title=title,
        width=900,
        height=400,
    )
    for col_idx in (1, 2):
        fig.update_xaxes(title_text="Test set", row=1, col=col_idx)
        fig.update_yaxes(title_text="Training set", row=1, col=col_idx)

    return fig


def plot_tl_matrices(
    results_dir: Path,
    model_prefix: str,
    label: str,
) -> dict[str, go.Figure]:
    """Generate combined TL matrix figures (generic + specialised side-by-side) for one model."""
    figs: dict[str, go.Figure] = {}

    generic_matrices = load_tl_matrix(results_dir, model_prefix, "generic")
    specialised_matrices = load_tl_matrix(results_dir, model_prefix, "specialised")

    has_generic = any(m.notna().any().any() for m in generic_matrices.values())
    has_specialised = any(m.notna().any().any() for m in specialised_matrices.values())

    if not has_generic and not has_specialised:
        print(f"  [skip] no TL matrix data for {label}")
        return figs

    for metric_key in generic_matrices:
        g_mat = generic_matrices[metric_key]
        s_mat = specialised_matrices[metric_key]
        g_has = g_mat.notna().any().any()
        s_has = s_mat.notna().any().any()

        if not g_has and not s_has:
            print(f"  [skip] {label} {metric_key}: no data")
            continue

        display = TL_METRIC_DISPLAY.get(metric_key, metric_key.upper())
        title = f"{label}: {display} Matrix"
        figs[f"tl_{model_prefix}_{metric_key}"] = _tl_heatmap_combined(
            g_mat, s_mat, title
        )

    return figs


def plot_all_models_tl_matrix(
    results_dir: Path,
    models: list[dict],
    metric_key: str,
) -> go.Figure | None:
    """2-row combined TL matrix for all models: row 1 = generic, row 2 = specialised.

    Each column corresponds to one model. Intended for multi-model comparison.
    """
    n = len(models)
    subplot_titles = [f"{m['label']} [Generic]" for m in models] + [
        f"{m['label']} [Specialised]" for m in models
    ]
    fig = make_subplots(
        rows=2,
        cols=n,
        subplot_titles=subplot_titles,
        horizontal_spacing=0.01,
        vertical_spacing=0.15,
    )

    has_any = False
    for col_idx, m in enumerate(models, 1):
        prefix = m["prefix"]
        all_matrices = {
            "generic": load_tl_matrix(results_dir, prefix, "generic"),
            "specialised": load_tl_matrix(results_dir, prefix, "specialised"),
        }
        for row_idx, scenario in enumerate(("generic", "specialised"), 1):
            matrix = all_matrices[scenario][metric_key]
            flat = matrix.values.astype(float)
            has_data = ~pd.isna(flat)
            if not has_data.any():
                continue
            has_any = True

            is_pct = flat[has_data].max() > 1
            zmin = 50.0 if is_pct else 0.5
            zmax = 100.0 if is_pct else 1.0
            fmt = ".1f" if is_pct else ".3f"
            high_threshold = 80.0 if is_pct else 0.8

            subplot_num = (row_idx - 1) * n + col_idx
            xref = "x" if subplot_num == 1 else f"x{subplot_num}"
            yref = "y" if subplot_num == 1 else f"y{subplot_num}"

            annotations = []
            for train in matrix.index:
                for test in matrix.columns:
                    val = matrix.loc[train, test]
                    if pd.isna(val):
                        annotations.append(
                            dict(
                                x=test,
                                y=train,
                                text="N/A",
                                font=dict(color="gray", size=16),
                                showarrow=False,
                                xref=xref,
                                yref=yref,
                            )
                        )
                    else:
                        annotations.append(
                            dict(
                                x=test,
                                y=train,
                                text=f"{val:{fmt}}",
                                font=dict(
                                    color="white" if val > high_threshold else "black",
                                    size=16,
                                ),
                                showarrow=False,
                                xref=xref,
                                yref=yref,
                            )
                        )

            # Show colorbar only once: bottom row, last column, horizontal below figures
            show_scale = col_idx == n and row_idx == 2
            fig.add_trace(
                go.Heatmap(
                    z=flat,
                    x=matrix.columns.tolist(),
                    y=matrix.index.tolist(),
                    colorscale="RdYlGn",
                    zmin=zmin,
                    zmax=zmax,
                    showscale=show_scale,
                    colorbar=dict(
                        orientation="h",
                        x=0.5,
                        xanchor="center",
                        y=-0.1,
                        yanchor="top",
                        thickness=15,
                        len=0.5,
                        outlinewidth=1,
                        outlinecolor="black",
                        title=dict(
                            text=TL_METRIC_DISPLAY.get(metric_key, metric_key.upper()),
                            side="bottom",
                        ),
                    ),
                ),
                row=row_idx,
                col=col_idx,
            )
            fig.layout.annotations = list(fig.layout.annotations) + annotations

    if not has_any:
        return None

    for col_idx in range(1, n + 1):
        for row_idx in range(1, 3):
            fig.update_xaxes(title_text="Test set", row=row_idx, col=col_idx)
            if col_idx == 1:
                fig.update_yaxes(title_text="Training set", row=row_idx, col=col_idx)
            else:
                fig.update_yaxes(
                    title_text="", showticklabels=False, row=row_idx, col=col_idx
                )
    fig.update_layout(
        title=None,
        width=max(900, 380 * n),
        height=750,
        margin=dict(b=100),
    )
    return fig


def load_concept_drift_results(results_dir: Path, model_prefix: str) -> pd.DataFrame:
    """Load origin/shifted results for all 4 datasets from a concept-drift run.

    Paths: {results_dir}/{prefix}_concept_drift/{prefix}_{letter}_on_{split}/results.csv
    """
    rows = []
    for dataset, letter in DATASET_LETTERS.items():
        if dataset == "wafamole":
            continue
        for split in ("origin", "shifted"):
            cd_dir = results_dir / f"{model_prefix}_concept_drift"
            # Match ae_gaur_A_on_origin or ae_gaur_A_testing_on_origin etc.
            candidates = list(cd_dir.glob(f"{model_prefix}_{letter}*_on_{split}"))
            path = candidates[0] / "results.csv" if candidates else None
            if path is not None and path.exists():
                df = pd.read_csv(path)
                df["dataset"] = dataset
                df["split"] = split
                rows.append(df)

    if not rows:
        return pd.DataFrame()

    return _normalize_pct_columns(pd.concat(rows, ignore_index=True))


def discover_concept_drift_models(results_dir: Path) -> list[str]:
    """Return model prefixes that have a *_concept_drift directory in results_dir."""
    return [
        p.name.removesuffix("_concept_drift")
        for p in sorted(results_dir.glob("*_concept_drift"))
        if p.is_dir()
    ]


def export_auroc_delta_csv(
    all_results: dict[str, pd.DataFrame],
    models: list[dict],
    output_dir: Path,
    split_col: str = "type",
    cat_a: str = "specialised",
    cat_b: str = "generic",
    filename: str = "auroc_generic_vs_specialised.csv",
) -> None:
    """Export a CSV with AUROC for two categories and their delta (cat_a - cat_b) per model×dataset."""
    rows = []
    for m in models:
        df = all_results.get(m["prefix"])
        if df is None or df.empty or "rocauc" not in df.columns:
            continue
        a_vals = df[df[split_col] == cat_a][["dataset", "rocauc"]].set_index("dataset")[
            "rocauc"
        ]
        b_vals = df[df[split_col] == cat_b][["dataset", "rocauc"]].set_index("dataset")[
            "rocauc"
        ]
        for dataset in df["dataset"].unique():
            a = a_vals.get(dataset, float("nan"))
            b = b_vals.get(dataset, float("nan"))
            rows.append(
                {
                    "model": m["prefix"],
                    "dataset": dataset,
                    f"auroc_{cat_a}": a,
                    f"auroc_{cat_b}": b,
                    "delta": b - a,
                }
            )

    if not rows:
        print("  [skip] no AUROC data for CSV export")
        return

    result_df = pd.DataFrame(rows)
    mean_rows = (
        result_df.groupby("model")[[f"auroc_{cat_a}", f"auroc_{cat_b}", "delta"]]
        .mean()
        .reset_index()
        .assign(dataset="mean")
    )
    result_df = pd.concat([result_df, mean_rows], ignore_index=True)

    out = output_dir / filename
    result_df.to_csv(out, index=False, float_format="%.4f")
    print(f"  Exported {out.name}")


def export_figure(fig: go.Figure, stem: Path, formats: list[str]) -> None:
    for fmt in formats:
        out = stem.with_suffix(f".{fmt}")
        kwargs = {"scale": 2} if fmt == "png" else {}
        fig.write_image(out, **kwargs)
        print(f"  Exported {out.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate generic vs specialised comparison + TL matrix reports "
            "for all model types."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        type=Path,
        help=(
            "Root results directory. Must contain flat {prefix}_generic/ and "
            "{prefix}_specialised/ subdirectories (e.g. ae_li_generic/, ae_li_specialised/)."
        ),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="PREFIX",
        default=None,
        help=(
            "Model prefixes to include (e.g. ae_li ae_securebert). "
            "Auto-discovered from --results-dir when omitted."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results"),
        help=(
            "Root output directory (default: output/results). "
            "Figures go to {output_dir}/generic-vs-specialised/ and {output_dir}/tl-matrix/."
        ),
    )
    parser.add_argument(
        "--format",
        nargs="+",
        choices=["png", "pdf", "svg"],
        default=["png"],
        help="Output format(s) (default: png).",
    )
    parser.add_argument(
        "--include-wafamole",
        action="store_true",
        default=False,
        help=(
            "Include WAFAMOLE (E) as an external test-only dataset. "
            "Adds E to test sets only (not to any training sets)."
        ),
    )
    args = parser.parse_args()

    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        print(f"ERROR: --results-dir does not exist: {results_dir}")
        return 1

    if args.include_wafamole:
        DATASET_LETTERS["wafamole"] = "E"
        DATASETS.append("wafamole")
        TL_TEST_SETS.append("E")

    if args.models:
        prefixes = args.models
    else:
        prefixes = discover_models(results_dir)
        if not prefixes:
            if not discover_concept_drift_models(results_dir):
                print(f"ERROR: no *_generic/*_specialised pairs found in {results_dir}")
                return 1
        else:
            print(f"Discovered models: {prefixes}")

    models = [{"prefix": p, "label": model_label(p)} for p in prefixes]

    # Load generic/specialised results upfront
    all_results: dict[str, pd.DataFrame] = {}
    if models:
        print("\nLoading results…")
        for m in models:
            df = load_results(results_dir, m["prefix"])
            all_results[m["prefix"]] = df
            print(f"  {m['prefix']}: {len(df)} rows")

    gvs_dir = args.output_dir / "generic-vs-specialised"
    tl_dir = args.output_dir / "tl-matrix"
    if models:
        gvs_dir.mkdir(parents=True, exist_ok=True)
        tl_dir.mkdir(parents=True, exist_ok=True)
        print("\n=== AUROC delta CSV ===")
        export_auroc_delta_csv(all_results, models, gvs_dir)

    # # Generic vs Specialised (ROC curves + recall plots — commented out, focusing on TL AUROC matrix)
    # for m in models:
    #     prefix, label = m["prefix"], m["label"]
    #     df = all_results[prefix]
    #     print(f"\n=== {label} ===")
    #
    #     per_model_figs = {
    #         f"roc_{prefix}": plot_roc_curves(
    #             df,
    #             f"ROC Curves: Generic vs Specialised ({label})",
    #             scenarios=[
    #                 (
    #                     "generic",
    #                     COLORS["generic"],
    #                     lambda l, c, p=prefix: results_dir
    #                     / f"{p}_generic"
    #                     / f"{p}_{c}_on_{l}"
    #                     / "roc_curves",
    #                 ),
    #                 (
    #                     "specialised",
    #                     COLORS["specialised"],
    #                     lambda l, c, p=prefix: results_dir
    #                     / f"{p}_specialised"
    #                     / f"{p}_{l}_on_{l}"
    #                     / "roc_curves",
    #                 ),
    #             ],
    #         ),
    #         f"recall_technique_{prefix}": plot_recall_per_technique(
    #             df, f"Recall per Attack Technique: Generic vs Specialised ({label})"
    #         ),
    #         f"recall_stmt_{prefix}": plot_recall_per_statement_type(
    #             df, f"Recall per Statement Type: Generic vs Specialised ({label})"
    #         ),
    #     }
    #     for name, fig in per_model_figs.items():
    #         if fig is not None:
    #             export_figure(fig, gvs_dir / name, args.format)

    # Transfer Learning Matrices
    print("\n=== Transfer Learning Matrices ===")
    if len(models) > 1:
        for metric_key in ("auroc", "auprc", "f1"):
            fig = plot_all_models_tl_matrix(results_dir, models, metric_key)
            if fig is not None:
                export_figure(fig, tl_dir / f"tl_combined_{metric_key}", args.format)
            else:
                print(f"  [skip] combined {metric_key}: no data")
    else:
        for m in models:
            print(f"\n=== TL: {m['label']} ===")
            tl_figs = plot_tl_matrices(results_dir, m["prefix"], m["label"])
            for name, fig in tl_figs.items():
                export_figure(fig, tl_dir / name, args.format)

    # Concept Drift (auto-detected)
    cd_prefixes = discover_concept_drift_models(results_dir)
    if cd_prefixes:
        print("\n=== Concept Drift ===")
        print(f"  Found concept-drift models: {cd_prefixes}")
        cd_dir = args.output_dir / "concept-drift"
        cd_dir.mkdir(parents=True, exist_ok=True)

        all_cd_results: dict[str, pd.DataFrame] = {}
        for prefix in cd_prefixes:
            df = load_concept_drift_results(results_dir, prefix)
            all_cd_results[prefix] = df
            print(f"  {prefix}: {len(df)} rows")

        cd_models = [{"prefix": p, "label": model_label(p)} for p in cd_prefixes]

        print("\n=== Concept Drift AUROC delta CSV ===")
        export_auroc_delta_csv(
            all_cd_results,
            cd_models,
            cd_dir,
            split_col="split",
            cat_a="origin",
            cat_b="shifted",
            filename="auroc_concept_drift_delta.csv",
        )

        print(f"  Concept drift          : {cd_dir.resolve()}")

    print(f"\nDone.")
    if models:
        print(f"  Generic vs specialised : {gvs_dir.resolve()}")
        print(f"  TL matrices            : {tl_dir.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
