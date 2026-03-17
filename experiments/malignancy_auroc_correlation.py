#!/usr/bin/env python3
"""Correlate malignancy metrics (topk_roc_auc, topk_fpr) with AUROC generalization scores.

For each malignancy metric, computes Pearson and Spearman correlation with AUROC
across all (model, scenario) pairs.

Aggregation modes:
  --aggregate auc  : normalized AUC over the k_pct axis (trapezoidal integration)
  --aggregate at10 : value at k_pct = 10

Expected inputs:
  --malignancy-dir : contains {model}_malignancy.csv files
  --results-dir    : contains {model}_generic/{model}_{train}_on_{test}/results.csv files

Usage:
    python3 experiments/malignancy_auroc_correlation.py \\
        --malignancy-dir /home/gquetel/experiences-results/2026-03-11/malignancy \\
        --results-dir /home/gquetel/experiences-results/2026-02-16-results \\
        --output-dir output/experiments/malignancy_correlation \\
        --aggregate auc

    # Testing mode (no files written)
    python3 experiments/malignancy_auroc_correlation.py \\
        --malignancy-dir /home/gquetel/experiences-results/2026-03-11/malignancy \\
        --results-dir /home/gquetel/experiences-results/2026-02-16-results \\
        --testing
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

GENERIC_SCENARIOS = {"ABC→D", "ABD→C", "ACD→B", "BCD→A"}

MALIGNANCY_METRICS = ["topk_roc_auc", "topk_fpr"]


def _aggregate_auc_over_k(group: pd.DataFrame, metric: str) -> float:
    """Normalized AUC over k_pct axis via trapezoidal integration."""
    g = group.sort_values("k_pct")
    x = g["k_pct"].values.astype(float)
    y = g[metric].values.astype(float)
    span = x[-1] - x[0]
    return float(np.trapz(y, x) / span) if span > 0 else float(y[0])


def _aggregate_at_k(group: pd.DataFrame, metric: str, k: float) -> float:
    """Value at specific k_pct, or nearest available."""
    idx = (group["k_pct"] - k).abs().idxmin()
    return float(group.loc[idx, metric])


def _load_malignancy_data(malignancy_dir: Path, aggregate: str) -> pd.DataFrame:
    """Load and aggregate malignancy CSVs.

    Files expected: {model}_malignancy.csv (or {model}_malignancy_{scenario}.csv)
    Returns one row per (extractor, scenario) with aggregated metric values.
    """
    csvs = list(malignancy_dir.glob("*_malignancy*.csv"))
    if not csvs:
        print(f"[warn] No malignancy CSV files found in {malignancy_dir}")
        return pd.DataFrame()

    dfs = []
    for csv in csvs:
        try:
            dfs.append(pd.read_csv(csv))
        except Exception as e:
            print(f"[warn] Could not read {csv}: {e}")

    if not dfs:
        return pd.DataFrame()

    raw = pd.concat(dfs, ignore_index=True)

    raw = raw[raw["scenario"].isin(GENERIC_SCENARIOS)].copy()
    if raw.empty:
        print("[warn] No generic scenario rows found in malignancy data")
        return pd.DataFrame()

    rows = []
    for (model, scenario), grp in raw.groupby(["model", "scenario"]):
        row = {"extractor": model, "scenario": scenario}
        for metric in MALIGNANCY_METRICS:
            if metric not in grp.columns:
                row[metric] = float("nan")
                continue
            if aggregate == "auc":
                row[metric] = _aggregate_auc_over_k(grp, metric)
            else:  # at10
                row[metric] = _aggregate_at_k(grp, metric, 10.0)
        rows.append(row)

    return pd.DataFrame(rows)


def _load_auroc_data(results_dir: Path) -> pd.DataFrame:
    """Load AUROC from results CSVs under *_generic/ subdirectories.

    Path convention: {results_dir}/{extractor}_generic/{extractor}_{train}_on_{test}/results.csv
    """
    rows = []
    for generic_dir in results_dir.glob("*_generic"):
        extractor = generic_dir.name.removesuffix("_generic")
        for run_dir in generic_dir.iterdir():
            if not run_dir.is_dir():
                continue
            name = run_dir.name
            prefix = f"{extractor}_"
            if not name.startswith(prefix):
                continue
            remainder = name[len(prefix) :]  # e.g. "ABC_on_D"
            if "_on_" not in remainder:
                continue
            train, test = remainder.split("_on_", 1)
            scenario = f"{train}→{test}"
            if scenario not in GENERIC_SCENARIOS:
                continue
            results_csv = run_dir / "results.csv"
            if not results_csv.exists():
                continue
            try:
                df = pd.read_csv(results_csv)
                rocauc = df["rocauc"].iloc[0]
                if isinstance(rocauc, str):
                    rocauc = float(rocauc.rstrip("%"))
                else:
                    rocauc = float(rocauc)
                rows.append(
                    {"extractor": extractor, "scenario": scenario, "rocauc": rocauc}
                )
            except Exception as e:
                print(f"[warn] Could not parse {results_csv}: {e}")

    return pd.DataFrame(rows)


def _compute_correlations(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute Pearson and Spearman correlations for each malignancy metric vs rocauc."""
    rows = []
    for metric in MALIGNANCY_METRICS:
        sub = merged[["rocauc", metric]].dropna()
        n = len(sub)
        if n < 3:
            print(f"[warn] {metric}: only {n} valid pairs, skipping correlation")
            continue
        pr, pp = pearsonr(sub[metric], sub["rocauc"])
        sr, sp = spearmanr(sub[metric], sub["rocauc"])
        rows.append(
            {
                "metric": metric,
                "pearson_r": round(pr, 4),
                "pearson_p": round(pp, 4),
                "spearman_rho": round(sr, 4),
                "spearman_p": round(sp, 4),
                "n_pairs": n,
            }
        )
    return pd.DataFrame(rows)


def _per_extractor_correlations(merged: pd.DataFrame) -> None:
    """Print per-extractor correlations for diagnostic purposes."""
    for extractor, grp in merged.groupby("extractor"):
        print(f"\n  [{extractor}] n={len(grp)} scenarios")
        for metric in MALIGNANCY_METRICS:
            sub = grp[["rocauc", metric]].dropna()
            if len(sub) < 3:
                continue
            pr, _ = pearsonr(sub[metric], sub["rocauc"])
            sr, _ = spearmanr(sub[metric], sub["rocauc"])
            print(f"    {metric}: Pearson r={pr:.3f}, Spearman ρ={sr:.3f}")


def _scatter_plots(merged: pd.DataFrame, output_dir: Path) -> None:
    """One scatter plot per malignancy metric: x=metric, y=rocauc."""
    for metric in MALIGNANCY_METRICS:
        sub = merged[["extractor", "scenario", "rocauc", metric]].dropna()
        if sub.empty:
            continue

        pr, pp = pearsonr(sub[metric], sub["rocauc"])
        sr, sp = spearmanr(sub[metric], sub["rocauc"])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sub[metric], sub["rocauc"], alpha=0.7, s=60)

        x_vals = sub[metric].values
        y_vals = sub["rocauc"].values
        m, b = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(x_vals.min(), x_vals.max(), 200)
        ax.plot(
            x_line,
            m * x_line + b,
            color="black",
            linewidth=1.5,
            linestyle="--",
            label="OLS fit",
        )

        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['extractor']}\n{row['scenario']}",
                (row[metric], row["rocauc"]),
                fontsize=6,
                ha="center",
                va="bottom",
                xytext=(0, 4),
                textcoords="offset points",
            )

        ax.set_xlabel(metric, fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        ax.set_title(f"Malignancy vs AUROC: {metric}")
        ax.text(
            0.05,
            0.95,
            f"Pearson r={pr:.3f} (p={pp:.3f})\nSpearman ρ={sr:.3f} (p={sp:.3f})",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        fig.tight_layout()
        plot_path = output_dir / f"scatter_{metric}.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Saved {plot_path.name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate malignancy metrics with AUROC generalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--malignancy-dir",
        type=Path,
        default=Path("output/experiments/malignancy"),
        help="Directory containing {model}_malignancy.csv files (default: output/experiments/malignancy)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing {model}_generic/{model}_{train}_on_{test}/results.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/experiments/malignancy_correlation"),
        help="Output directory for correlations CSV and scatter plots (default: output/experiments/malignancy_correlation)",
    )
    parser.add_argument(
        "--aggregate",
        choices=["auc", "at10"],
        default="auc",
        help="Aggregation over k_pct: 'auc' (trapezoidal AUC over k axis) or 'at10' (k=10%%). Default: auc",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Restrict to specific model names (e.g. ae_li ae_securebert). Default: all.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Print summary stats without writing output files",
    )
    args = parser.parse_args()

    # --- Load malignancy data ---
    print(f"Loading malignancy data from {args.malignancy_dir} ...")
    malignancy_df = _load_malignancy_data(args.malignancy_dir, args.aggregate)
    if malignancy_df.empty:
        print("ERROR: no malignancy data found")
        return 1
    print(f"  {len(malignancy_df)} (model, scenario) pairs after aggregation")

    if args.models:
        malignancy_df = malignancy_df[malignancy_df["extractor"].isin(args.models)]
        print(f"  {len(malignancy_df)} pairs after filtering to {args.models}")

    # --- Load AUROC data ---
    print(f"Loading AUROC data from {args.results_dir} ...")
    auroc_df = _load_auroc_data(args.results_dir)
    if auroc_df.empty:
        print("ERROR: no AUROC data found")
        return 1
    print(f"  {len(auroc_df)} (extractor, scenario) pairs found")

    if args.models:
        auroc_df = auroc_df[auroc_df["extractor"].isin(args.models)]
        print(f"  {len(auroc_df)} pairs after filtering to {args.models}")

    # --- Merge ---
    merged = malignancy_df.merge(auroc_df, on=["extractor", "scenario"], how="inner")
    n_merged = len(merged)
    print(f"Merged: {n_merged} matching (model, scenario) pairs")

    if merged.empty:
        print(
            "ERROR: no matching pairs — check that model names match between malignancy CSVs and results dirs"
        )
        print(f"  Malignancy models : {sorted(malignancy_df['extractor'].unique())}")
        print(f"  AUROC extractors  : {sorted(auroc_df['extractor'].unique())}")
        return 1

    n_malignancy_only = len(malignancy_df) - n_merged
    n_auroc_only = len(auroc_df) - n_merged
    if n_malignancy_only > 0:
        print(
            f"  [warn] {n_malignancy_only} malignancy rows had no matching AUROC entry"
        )
    if n_auroc_only > 0:
        print(f"  [warn] {n_auroc_only} AUROC rows had no matching malignancy entry")

    # --- Correlations ---
    corr_df = _compute_correlations(merged)
    print("\nCorrelation Results (all model×scenario pairs):")
    print(corr_df.to_string(index=False))

    print("\nPer-extractor breakdown:")
    _per_extractor_correlations(merged)

    if args.testing:
        print("\n[testing] Skipping file output")
        return 0

    # --- Save outputs ---
    args.output_dir.mkdir(parents=True, exist_ok=True)

    corr_path = args.output_dir / "correlations.csv"
    corr_df.to_csv(corr_path, index=False)
    print(f"\nSaved {corr_path}")

    merged_path = args.output_dir / "merged_data.csv"
    merged.to_csv(merged_path, index=False)
    print(f"Saved {merged_path}")

    print("\nGenerating scatter plots ...")
    _scatter_plots(merged, args.output_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
