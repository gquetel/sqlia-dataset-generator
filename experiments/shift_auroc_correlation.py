#!/usr/bin/env python3
"""Correlate domain shift detection accuracy with AUROC generalization scores.

For each shift metric (NoRed_MMD, NoRed_KS, BBSDs_MMD, BBSDs_KS, DomainClf_Binomial),
computes Pearson and Spearman correlation with AUROC across all (extractor, scenario) pairs.

Expected inputs:
  --shift-dir : contains {extractor}_{pool_key}_distances_scenarios.csv files
  --results-dir: contains {model}_generic/{model}_{train}_on_{test}/results.csv files

Usage:
    python3 experiments/shift_auroc_correlation.py \\
        --shift-dir output/experiments/distribution_distances \\
        --results-dir /home/gquetel/experiences-results/2026-02-16-results \\
        --output-dir output/experiments/shift_correlation \\
        --aggregate auc

    # Testing mode (no files written)
    python3 experiments/shift_auroc_correlation.py \\
        --shift-dir output/experiments/distribution_distances \\
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

# Pure cross-domain scenarios (train on ABC, test on D)
GENERIC_SCENARIOS = {"ABC→D", "ABD→C", "ACD→B", "BCD→A"}

# Mixed-shift scenarios (train on ABC, deployment stream = ABCD)
# Maps mixed label → equivalent pure AUROC scenario key
MIXED_SCENARIOS: dict[str, str] = {
    "ABC→ABCD": "ABC→D",
    "ABD→ABDC": "ABD→C",
    "ACD→ACDB": "ACD→B",
    "BCD→BCDA": "BCD→A",
}

SHIFT_METRICS = [
    "NoRed_MMD_detacc",
    "NoRed_KS_detacc",
    "BBSDs_MMD_detacc",
    "BBSDs_KS_detacc",
    "DomainClf_Binomial_detacc",
]


def _aggregate_auc(group: pd.DataFrame, metric: str) -> float:
    """Normalized AUC over n_samples axis via trapezoidal integration.

    Normalizes by (max_n - min_n) so the result stays in [0, 1].
    """
    g = group.sort_values("n_samples")
    x = g["n_samples"].values.astype(float)
    y = g[metric].values.astype(float)
    span = x[-1] - x[0]
    return float(np.trapz(y, x) / span) if span > 0 else float(y[0])


def _aggregate_max(group: pd.DataFrame, metric: str) -> float:
    return float(group[metric].max())


def _aggregate_at_n(group: pd.DataFrame, metric: str, n: int) -> float:
    """Value at specific n_samples, or nearest available."""
    idx = (group["n_samples"] - n).abs().idxmin()
    return float(group.loc[idx, metric])


def _load_shift_data(
    shift_dir: Path, aggregate: str, at_n: int, mode: str
) -> pd.DataFrame:
    """Load and aggregate shift detection CSVs.

    mode="generic": keeps ABC→D-style rows, scenario column is the AUROC join key.
    mode="mixed":   keeps ABC→ABCD-style rows, remaps scenario to ABC→D for joining.
    """
    csvs = list(shift_dir.glob("*.csv"))
    if not csvs:
        print(f"[warn] No CSV files found in {shift_dir}")
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

    if mode == "generic":
        raw = raw[raw["scenario"].isin(GENERIC_SCENARIOS)].copy()
        if raw.empty:
            print("[warn] No generic scenario rows found in shift data")
            return pd.DataFrame()
    else:  # mixed
        raw = raw[raw["scenario"].isin(MIXED_SCENARIOS)].copy()
        if raw.empty:
            print("[warn] No mixed-shift scenario rows found in shift data")
            return pd.DataFrame()
        # Remap to the AUROC join key (e.g. ABC→ABCD → ABC→D)
        raw["scenario"] = raw["scenario"].map(MIXED_SCENARIOS)

    rows = []
    for (extractor, scenario), grp in raw.groupby(["extractor", "scenario"]):
        row = {"extractor": extractor, "scenario": scenario}
        for metric in SHIFT_METRICS:
            if metric not in grp.columns:
                row[metric] = float("nan")
                continue
            if aggregate == "auc":
                row[metric] = _aggregate_auc(grp, metric)
            elif aggregate == "max":
                row[metric] = _aggregate_max(grp, metric)
            else:  # at_N
                row[metric] = _aggregate_at_n(grp, metric, at_n)
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
            # run_dir.name expected: {extractor}_{train}_on_{test}
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
    """Compute Pearson and Spearman correlations for each shift metric vs rocauc."""
    rows = []
    for metric in SHIFT_METRICS:
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
        for metric in SHIFT_METRICS:
            sub = grp[["rocauc", metric]].dropna()
            if len(sub) < 3:
                continue
            pr, _ = pearsonr(sub[metric], sub["rocauc"])
            sr, _ = spearmanr(sub[metric], sub["rocauc"])
            short = metric.replace("_detacc", "")
            print(f"    {short}: Pearson r={pr:.3f}, Spearman ρ={sr:.3f}")


def _scatter_plots(merged: pd.DataFrame, output_dir: Path) -> None:
    """One scatter plot per shift metric: x=detacc@1000, y=rocauc."""
    for metric in SHIFT_METRICS:
        sub = merged[["extractor", "scenario", "rocauc", metric]].dropna()
        if sub.empty:
            continue

        pr, pp = pearsonr(sub[metric], sub["rocauc"])
        sr, sp = spearmanr(sub[metric], sub["rocauc"])

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(sub[metric], sub["rocauc"], alpha=0.7, s=60)

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

        ax.set_xlabel(f"{metric} @ n=1000", fontsize=11)
        ax.set_ylabel("AUROC", fontsize=11)
        title = metric.replace("_detacc", "")
        ax.set_title(f"Shift vs AUROC: {title}")
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
        description="Correlate domain shift detection accuracy with AUROC generalization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--shift-dir",
        type=Path,
        default=Path("output/experiments/distribution_distances"),
        help="Directory containing shift CSV files (default: output/experiments/distribution_distances)",
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
        default=Path("output/experiments/shift_correlation"),
        help="Output directory for correlations CSV and scatter plots (default: output/experiments/shift_correlation)",
    )
    parser.add_argument(
        "--mode",
        choices=["generic", "mixed"],
        default="mixed",
        help="Which shift scenarios to correlate: 'generic' (ABC→D) or 'mixed' (ABC→ABCD, default: mixed)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        metavar="MODEL",
        default=None,
        help="Restrict to specific extractor names (e.g. ae_li ae_securebert ae_roberta). Default: all.",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Print summary stats without writing output files",
    )
    args = parser.parse_args()

    # --- Load shift data ---
    print(f"Loading shift data from {args.shift_dir} ...")
    shift_df = _load_shift_data(args.shift_dir, "at_N", 1000, args.mode)
    if shift_df.empty:
        print("ERROR: no shift data found")
        return 1
    print(f"  {len(shift_df)} (extractor, scenario) pairs after aggregation")

    if args.models:
        shift_df = shift_df[shift_df["extractor"].isin(args.models)]
        print(f"  {len(shift_df)} pairs after filtering to {args.models}")

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
    merged = shift_df.merge(auroc_df, on=["extractor", "scenario"], how="inner")
    n_merged = len(merged)
    print(f"Merged: {n_merged} matching (extractor, scenario) pairs")

    if merged.empty:
        print(
            "ERROR: no matching pairs — check that extractor names match between shift CSVs and results dirs"
        )
        print(f"  Shift extractors : {sorted(shift_df['extractor'].unique())}")
        print(f"  AUROC extractors : {sorted(auroc_df['extractor'].unique())}")
        return 1

    n_shift_only = len(shift_df) - n_merged
    n_auroc_only = len(auroc_df) - n_merged
    if n_shift_only > 0:
        print(f"  [warn] {n_shift_only} shift rows had no matching AUROC entry")
    if n_auroc_only > 0:
        print(f"  [warn] {n_auroc_only} AUROC rows had no matching shift entry")

    # --- Correlations ---
    corr_df = _compute_correlations(merged)
    print("\nCorrelation Results (all extractor×scenario pairs):")
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
