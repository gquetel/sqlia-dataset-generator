"""
Dataset Statistics Dashboard

Usage:
    python3 dataset_stats.py --dataset A ~/datasets/OurAirports.csv \\
                              --dataset B ~/datasets/sakila.csv \\
                              --dataset C ~/datasets/AdventureWorks.csv \\
                              --dataset D ~/datasets/OHR.csv

This script works as an empirical sanity check. If a metric is anormaly low / high, it
might significate a problem during the dataset generation.
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path


def load_dataset(path: str, chunksize: int = 500_000) -> pd.DataFrame:
    """
    Load dataset using chunked reading for large files.

    Args:
        path: Path to CSV file
        chunksize: Number of rows per chunk

    Returns:
        Complete DataFrame with specified dtypes
    """
    # Specify dtypes to avoid warnings
    dtype_spec = {
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

    chunks = []
    for chunk in pd.read_csv(
        path, chunksize=chunksize, dtype=dtype_spec, low_memory=False
    ):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    print(f"  Loaded {len(df):,} rows from {Path(path).name}")
    return df


def collect_dataset_stats(df: pd.DataFrame, dataset_name: str) -> Dict:
    """
    Collect all statistics for a dataset.

    Args:
        df: Dataset DataFrame
        dataset_name: Name for display purposes

    Returns:
        Dictionary containing all statistics
    """
    stats = {}

    # Sample counts
    attacks = (df["label"] == 1).sum()
    normal = (df["label"] == 0).sum()
    total = len(df)

    stats["attacks"] = attacks
    stats["normal"] = normal
    stats["total"] = total
    stats["attack_pct"] = (attacks / total * 100) if total > 0 else 0
    stats["normal_pct"] = (normal / total * 100) if total > 0 else 0

    # Statement type distribution for normal queries
    df_normal = df[df["label"] == 0]
    stmt_normal = df_normal["statement_type"].value_counts(normalize=True) * 100
    stats["stmt_normal"] = stmt_normal.to_dict()

    # Statement type distribution for attacks
    df_attacks = df[df["label"] == 1]
    if len(df_attacks) > 0:
        stmt_attack = df_attacks["statement_type"].value_counts(normalize=True) * 100
        stats["stmt_attack"] = stmt_attack.to_dict()
    else:
        stats["stmt_attack"] = {}

    # Template counts per split
    df_train = df[df["split"] == "train"]
    df_test = df[df["split"] == "test"]
    df_test_normal = df_test[df_test["label"] == 0]
    df_test_attack = df_test[df_test["label"] == 1]

    stats["templates_train"] = df_train["query_template_id"].nunique()
    stats["templates_test_normal"] = df_test_normal["query_template_id"].nunique()
    stats["templates_test_attack"] = df_test_attack["query_template_id"].nunique()

    # Attack technique distribution
    if len(df_attacks) > 0:
        technique_counts = df_attacks["attack_technique"].value_counts()
        stats["technique_dist"] = technique_counts.to_dict()

        # Exploit vs recon breakdown per technique
        technique_stage = (
            df_attacks.groupby(["attack_technique", "attack_stage"])
            .size()
            .unstack(fill_value=0)
        )
        stats["technique_stages"] = technique_stage
    else:
        stats["technique_dist"] = {}
        stats["technique_stages"] = pd.DataFrame()

    return stats


def display_sample_counts(stats: Dict, dataset_name: str):
    """
    Display attack vs normal sample counts.
    """
    print(f"Attacks: {stats['attacks']:,} ({stats['attack_pct']:.1f}%)")
    print(f"Normal:  {stats['normal']:,} ({stats['normal_pct']:.1f}%)")
    print(f"Total:   {stats['total']:,}")
    print()


def display_template_counts(stats: Dict, dataset_name: str):
    """
    Display unique template counts per split.
    """
    print("--- Unique Templates per Split ---")
    print(f"  Train:       {stats['templates_train']:>6,}")
    print(f"  Test normal: {stats['templates_test_normal']:>6,}")
    print(f"  Test attack: {stats['templates_test_attack']:>6,}")
    print()


def display_statement_distribution(stats: Dict, dataset_name: str, label: int):
    """
    Display statement type distribution for normal or attack queries.

    Args:
        stats: Statistics dictionary
        dataset_name: Name for display
        label: 0 for normal, 1 for attacks
    """
    key = "stmt_normal" if label == 0 else "stmt_attack"
    title = (
        "Normal Queries - Statement Type Distribution"
        if label == 0
        else "Attack Queries - Statement Type Distribution"
    )

    print(f"--- {title} ---")

    stmt_dict = stats[key]
    if not stmt_dict:
        print("  (No data)")
        print()
        return

    # Sort by percentage descending
    sorted_items = sorted(stmt_dict.items(), key=lambda x: x[1], reverse=True)

    for stmt_type, pct in sorted_items:
        print(f"  {stmt_type:<10} {pct:6.2f}%")
    print()


def display_technique_distribution(stats: Dict, dataset_name: str):
    """
    Display attack technique distribution.
    """
    print("--- Attack Technique Distribution ---")

    tech_dict = stats["technique_dist"]
    if not tech_dict:
        print("  (No attacks)")
        print()
        return

    total_attacks = sum(tech_dict.values())

    # Sort by count descending
    sorted_items = sorted(tech_dict.items(), key=lambda x: x[1], reverse=True)

    for technique, count in sorted_items:
        pct = (count / total_attacks * 100) if total_attacks > 0 else 0
        print(f"  {technique:<15} {count:>8,} ({pct:5.1f}%)")
    print()


def display_technique_stage_breakdown(stats: Dict, dataset_name: str):
    """
    Display exploit vs recon breakdown per technique.
    """
    print("--- Exploit vs Recon per Technique ---")

    df = stats["technique_stages"]
    if df.empty:
        print("  (No attacks)")
        print()
        return

    # Ensure columns exist
    if "exploit" not in df.columns:
        df["exploit"] = 0
    if "recon" not in df.columns:
        df["recon"] = 0

    # Calculate totals and sort by total descending
    df["total"] = df["exploit"] + df["recon"]
    df = df.sort_values("total", ascending=False)

    # Display as formatted table
    print(f"  {'technique':<15} {'exploit':>8} {'recon':>8} {'total':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8}")

    for technique, row in df.iterrows():
        print(
            f"  {technique:<15} {int(row['exploit']):>8,} {int(row['recon']):>8,} {int(row['total']):>8,}"
        )
    print()


def export_csv_summary(all_stats: Dict[str, Dict], datasets: Dict, output_path: Path):
    """
    Export structured CSV with one row per dataset.

    Args:
        all_stats: Dictionary mapping dataset_id -> stats dict
        datasets: Dataset configuration dictionary
        output_path: Path to output CSV file
    """
    rows = []

    for dataset_id, config in datasets.items():
        if dataset_id not in all_stats:
            continue

        stats = all_stats[dataset_id]
        row = {
            "dataset_id": dataset_id,
            "dataset_name": config["name"],
            "attacks": stats["attacks"],
            "normal": stats["normal"],
            "total": stats["total"],
            "attack_pct": round(stats["attack_pct"], 2),
            "templates_train": stats["templates_train"],
            "templates_test_normal": stats["templates_test_normal"],
            "templates_test_attack": stats["templates_test_attack"],
        }

        # Statement type percentages for normal queries
        for stmt_type in ["select", "insert", "update", "delete", "admin"]:
            key = f"stmt_normal_{stmt_type}"
            row[key] = round(stats["stmt_normal"].get(stmt_type, 0), 2)

        # Statement type percentages for attacks
        for stmt_type in ["select", "insert", "update", "delete", "insider"]:
            key = f"stmt_attack_{stmt_type}"
            row[key] = round(stats["stmt_attack"].get(stmt_type, 0), 2)

        # Technique counts
        for technique in [
            "boolean",
            "error",
            "union",
            "time",
            "stacked",
            "inline",
            "insider",
        ]:
            key = f"tech_{technique}_total"
            row[key] = stats["technique_dist"].get(technique, 0)

        # Exploit/recon counts per technique
        df = stats["technique_stages"]
        if not df.empty:
            for technique in df.index:
                exploit = (
                    int(df.loc[technique, "exploit"]) if "exploit" in df.columns else 0
                )
                recon = int(df.loc[technique, "recon"]) if "recon" in df.columns else 0
                row[f"tech_{technique}_exploit"] = exploit
                row[f"tech_{technique}_recon"] = recon

        rows.append(row)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(output_path, index=False)
    print(f"  CSV exported with {len(df_out)} rows, {len(df_out.columns)} columns")


def main():
    parser = argparse.ArgumentParser(
        description="Generate dataset statistics for SQL injection detection datasets"
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Dataset name and path to its CSV file (repeatable)",
    )
    parser.add_argument(
        "--output-dir",
        default="../output/results/dataset_stats",
        help="Output directory for reports (default: output/results/dataset_stats)",
    )
    args = parser.parse_args()

    # Build datasets dictionary from command line arguments
    if not args.dataset:
        parser.error("No datasets provided. Use --dataset NAME PATH (can be repeated)")

    datasets = {}
    for name, path in args.dataset:
        datasets[name] = {"name": name, "path": os.path.expanduser(path)}

    # Collect statistics for all datasets
    all_stats = {}

    for dataset_id, config in datasets.items():
        print(f"\n{'='*80}")
        print(f"DATASET {dataset_id}: {config['name']}")
        print(f"{'='*80}\n")

        # Check if file exists
        if not Path(config["path"]).exists():
            print(f"  WARNING: File not found: {config['path']}")
            print(f"  Skipping dataset {dataset_id}\n")
            continue

        # Load dataset
        df = load_dataset(config["path"])

        # Collect all statistics
        stats = collect_dataset_stats(df, config["name"])
        all_stats[dataset_id] = stats

        # Display results
        display_sample_counts(stats, config["name"])
        display_template_counts(stats, config["name"])
        display_statement_distribution(stats, config["name"], label=0)
        display_statement_distribution(stats, config["name"], label=1)
        display_technique_distribution(stats, config["name"])
        display_technique_stage_breakdown(stats, config["name"])

    if not all_stats:
        print("\nERROR: No datasets were processed. Check file paths.")
        return 1

    # Export results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Exporting results...")

    export_csv_summary(all_stats, datasets, output_dir / "dataset_stats.csv")

    print(f"Results exported: {output_dir / 'dataset_stats.csv'}")

    return 0


if __name__ == "__main__":
    exit(main())
