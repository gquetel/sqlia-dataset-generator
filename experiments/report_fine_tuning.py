"""
Fine-Tuning Experiment Report

Generates a line chart (with shaded std bands) of AUROC vs. k (number of
fine-tuning samples) for each feature extractor.

Results are averaged over the 4 leave-one-out configurations (ABC→D, ABD→C,
ACD→B, BCD→A) per model; the shaded band covers one std across those configs
combined with the within-config std already present in summary.csv.

Expected directory structure:
    {results_dir}/
      ae_<extractor>_fine_tuning/
        ae_<extractor>_ABC_on_D/summary.csv
        ae_<extractor>_ABD_on_C/summary.csv
        ae_<extractor>_ACD_on_B/summary.csv
        ae_<extractor>_BCD_on_A/summary.csv

Usage:
    python experiments/report_fine_tuning.py \\
        --results-dir ~/experiences-results/2026-04-01/fine-tune

    python experiments/report_fine_tuning.py \\
        --results-dir ~/experiences-results/2026-04-01/fine-tune \\
        --output-dir output/fine-tuning \\
        --format png pdf
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

TRAIN_TEST_CONFIGS = ["ABC_on_D", "ABD_on_C", "ACD_on_B", "BCD_on_A"]

# In-domain AUROC: train and test on the same dataset (averaged over A→A, B→B, C→C, D→D)
IN_DOMAIN_AUROC: dict[str, float] = {
    "ae_codebert": 0.9980,
    "ae_codet5": 0.9603,
    "ae_flan_t5": 0.9949,
    "ae_gaur": 0.9910,
    "ae_gaur_chatgpt": 0.9891,
    "ae_gaur_mistral": 0.9895,
    "ae_kakisim_c": 0.8160,
    "ae_li": 0.9899,
    "ae_llm2vec": 0.8562,
    "ae_loginov": 0.9908,
    "ae_roberta": 0.9949,
    "ae_securebert": 0.9946,
    "ae_sentbert": 0.9903,
    "ae_cv": 0.9394,
}

KNOWN_LABELS: dict[str, str] = {
    "ae_codebert": "CodeBERT",
    "ae_codet5": "CodeT5",
    "ae_flan_t5": "Flan-T5",
    "ae_llm2vec": "LLM2Vec",
    "ae_roberta": "RoBERTa",
    "ae_securebert": "SecureBERT",
    "ae_sentbert": "SentenceBERT",
    "ae_li": "Li et al.",
    "ae_loginov": "Loginov et al.",
    "ae_gaur": "Gaur (Expert)",
    "ae_gaur_chatgpt": "Gaur (ChatGPT)",
    "ae_gaur_mistral": "Gaur (Mistral)",
    "ae_kakisim_c": "Kakisim et al.",
    "ae_cv": "CountVect",
}

COLORS = [
    "#DCE775",
    "#E57373",
    "#4FC3F7",
    "#8b7e74",
    "#9fa8da",
    "#66BB6A",
    "#0D47A1",
    "#ffd54f",
    "#7E57C2",
]

MARKERS = ["circle", "x"]

MODEL_ORDER = [
    "ae_li",
    "ae_loginov",
    "ae_cv",
    "ae_gaur",
    "ae_gaur_chatgpt",
    "ae_gaur_mistral",
    "ae_kakisim_c",
    "ae_roberta",
    "ae_securebert",
    "ae_sentbert",
    "ae_flan_t5",
    "ae_llm2vec",
    "ae_codebert",
    "ae_codet5",
]


def model_label(prefix: str) -> str:
    return KNOWN_LABELS.get(prefix, prefix)


def discover_models(results_dir: Path) -> list[str]:
    """Auto-detect model prefixes from directory names matching *_fine_tuning."""
    prefixes = []
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and d.name.endswith("_fine_tuning"):
            prefix = d.name[: -len("_fine_tuning")]
            prefixes.append(prefix)
    return prefixes


def load_model_data(results_dir: Path, model_prefix: str) -> pd.DataFrame | None:
    """Load and aggregate summary.csv across all 4 configs for one model.

    Returns a DataFrame with columns: k, rocauc_mean, rocauc_std
    where mean/std are aggregated across the 4 leave-one-out configs.
    """
    model_dir = results_dir / f"{model_prefix}_fine_tuning"
    per_config: list[pd.DataFrame] = []

    for config in TRAIN_TEST_CONFIGS:
        csv_path = model_dir / f"{model_prefix}_{config}" / "summary.csv"
        if not csv_path.exists():
            print(f"  [warn] missing {csv_path}", file=sys.stderr)
            continue
        df = pd.read_csv(csv_path)
        per_config.append(df[["k", "rocauc_mean", "rocauc_std"]])

    if not per_config:
        return None

    # Concatenate across configs and aggregate per k
    combined = pd.concat(per_config, ignore_index=True)
    # Mean of means; std combines within-config std and cross-config spread
    agg = (
        combined.groupby("k")["rocauc_mean"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "rocauc_mean", "std": "rocauc_std"})
    )
    # Fill NaN std (k=0 has a single value across configs that may all be the
    # same run, so std=NaN → treat as 0)
    agg["rocauc_std"] = agg["rocauc_std"].fillna(0.0)
    return agg


def _k_label(k: int) -> str:
    if k == 0:
        return "0\n(baseline)"
    if k >= 1000:
        return f"{k // 1000}k"
    return str(k)


def build_figure(
    model_data: dict[str, pd.DataFrame],
    show_std: bool = True,
    in_domain: dict[str, float] | None = None,
) -> go.Figure:
    fig = go.Figure()
    in_domain = in_domain or {}

    # Collect the union of k values (as category labels) in sorted order
    all_k = sorted({k for df in model_data.values() for k in df["k"].tolist()})
    x_labels = [_k_label(k) for k in all_k]

    for idx, (prefix, df) in enumerate(model_data.items()):
        color = COLORS[idx % len(COLORS)]
        marker_symbol = MARKERS[idx // len(COLORS) % len(MARKERS)]
        label = model_label(prefix)

        # Align to global k order
        df_indexed = df.set_index("k").reindex(all_k)
        means = df_indexed["rocauc_mean"].tolist()
        stds = df_indexed["rocauc_std"].tolist()

        upper = [m + s if not pd.isna(s) else m for m, s in zip(means, stds)]
        lower = [max(0.0, m - s) if not pd.isna(s) else m for m, s in zip(means, stds)]

        # Shaded std band
        if show_std:
            fig.add_trace(
                go.Scatter(
                    x=x_labels + x_labels[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=color,
                    opacity=0.15,
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=means,
                mode="lines+markers",
                name=label,
                line=dict(color=color, width=1),
                marker=dict(size=8, symbol=marker_symbol),
            )
        )

        # Highlight first k where gap to in-domain <= 0.01
        if prefix in in_domain:
            spec = in_domain[prefix]
            crossover = next(
                (
                    (xl, m)
                    for xl, k, m in zip(x_labels, all_k, means)
                    if not pd.isna(m) and (spec - m) <= 0.01
                ),
                None,
            )
            if crossover:
                fig.add_trace(
                    go.Scatter(
                        x=[crossover[0]],
                        y=[crossover[1]],
                        mode="markers",
                        marker=dict(
                            size=12,
                            color=color,
                            line=dict(color="black", width=1),
                        ),
                        showlegend=False,
                        hovertemplate=f"{model_label(prefix)}: within 0.01 of in-domain at k={crossover[0]}<extra></extra>",
                    )
                )

    _axis_font = dict(family="CMU Serif", size=22)
    _tick_font = dict(family="CMU Serif", size=18)

    fig.update_layout(
        font=dict(family="CMU Serif", size=18),
        xaxis=dict(
            title=dict(text="k (fine-tuning samples)", font=_axis_font),
            tickfont=_tick_font,
            type="category",
        ),
        yaxis=dict(
            title=dict(text="AUROC", font=_axis_font),
            tickfont=_tick_font,
            range=[0.5, 1.0],
        ),
        legend=dict(
            title=dict(text="Feature extractor", font=_axis_font),
            font=_tick_font,
            x=1.02,
            y=1.0,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="lightgrey",
            borderwidth=1,
        ),
        template="plotly_white",
        width=1000,
        height=580,
        margin=dict(t=30),
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning AUROC line chart")
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Path to fine-tuning results (contains *_fine_tuning/ subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/results"),
        help="Root output directory (default: output/results). Figures go to {output_dir}/fine-tuning/.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model prefixes to include (default: auto-discover)",
    )
    parser.add_argument(
        "--format",
        nargs="+",
        default=["png"],
        choices=["png", "pdf", "html", "svg"],
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Disable shaded std band",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser()
    output_dir = args.output_dir / "fine-tuning"
    output_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or discover_models(results_dir)
    if not models:
        print("No models found.", file=sys.stderr)
        sys.exit(1)

    print(f"Models: {models}")
    model_data: dict[str, pd.DataFrame] = {}
    for prefix in models:
        df = load_model_data(results_dir, prefix)
        if df is not None:
            model_data[prefix] = df
            print(f"  {prefix}: {len(df)} k-values loaded")
        else:
            print(f"  {prefix}: no data found, skipping", file=sys.stderr)

    if not model_data:
        print("No data loaded.", file=sys.stderr)
        sys.exit(1)

    # Sort models according to canonical order; unknowns go at the end
    known_order = {p: i for i, p in enumerate(MODEL_ORDER)}
    model_data = dict(
        sorted(
            model_data.items(), key=lambda kv: known_order.get(kv[0], len(MODEL_ORDER))
        )
    )

    # Only pass entries for models actually loaded
    in_domain = {
        p: IN_DOMAIN_AUROC[p]
        for p in model_data
        if p in IN_DOMAIN_AUROC and IN_DOMAIN_AUROC[p] > 0
    }

    # Report first k within 0.01 of in-domain, or failure at k=10000
    for prefix, df in model_data.items():
        if prefix not in in_domain:
            continue
        spec = in_domain[prefix]
        label = model_label(prefix)
        df_indexed = df.set_index("k")
        close = [
            (k, m)
            for k, m in zip(df["k"], df["rocauc_mean"])
            if not pd.isna(m) and (spec - m) <= 0.01
        ]
        if close:
            k_close, m_close = min(close, key=lambda t: t[0])
            print(
                f"  {label}: within 0.01 of in-domain ({spec:.3f}) at k={k_close} (AUROC={m_close:.3f})"
            )
        else:
            m_at_10k = (
                df_indexed.loc[10000, "rocauc_mean"]
                if 10000 in df_indexed.index
                else df_indexed["rocauc_mean"].iloc[-1]
            )
            gap = spec - m_at_10k
            print(
                f"  {label}: never within 0.01 — gap {gap:.3f} at k=10000 (in-domain={spec:.3f}, AUROC={m_at_10k:.3f})"
            )

    fig = build_figure(model_data, show_std=not args.no_std, in_domain=in_domain)

    for fmt in args.format:
        out_path = output_dir / f"fine_tuning_auroc.{fmt}"
        if fmt == "html":
            fig.write_html(str(out_path))
        else:
            fig.write_image(str(out_path))
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
