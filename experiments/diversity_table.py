#!/usr/bin/env python3
"""Read diversity metric CSVs and print markdown tables to stdout."""

import csv
from pathlib import Path

BASE_DIR = Path.home() / "experiences-results/2026-02-11-diversity"
LEX_SYN_CSV = BASE_DIR / "diversity_metrics_lex_syn" / "diversity_metrics.csv"
SEM_CSV = BASE_DIR / "diversity_metrics_sem" / "diversity_metrics.csv"

PREFIXES = ["generic", "specialised"]
TYPES = ["train-normal", "test-normal", "test-attack"]
TYPE_LABELS = {
    "train-normal": "Train (Normal)",
    "test-normal": "Test (Normal)",
    "test-attack": "Test (Attack)",
}


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def main():
    lex_syn_rows = read_csv(LEX_SYN_CSV)
    sem_rows = read_csv(SEM_CSV)

    # Index semantic diversity by (dataset, type)
    sem_index = {(r["dataset"], r["type"]): r["div_sem"] for r in sem_rows}

    # Merge and split prefix/db_name
    merged = []
    for r in lex_syn_rows:
        dataset = r["dataset"]
        prefix, db_name = dataset.split("-", 1)
        div_sem = sem_index.get((dataset, r["type"]), "N/A")
        merged.append(
            {
                "prefix": prefix,
                "db_name": db_name,
                "type": r["type"],
                "vocab_size": int(r["vocab_size"]),
                "unique_parse_trees": int(r["unique_parse_trees"]),
                "div_sem": float(div_sem) if div_sem != "N/A" else div_sem,
            }
        )

    for prefix in PREFIXES:
        for typ in TYPES:
            rows = [r for r in merged if r["prefix"] == prefix and r["type"] == typ]
            if not rows:
                continue

            label = "Generic" if prefix == "generic" else "Specialised"
            print(f"### {label} — {TYPE_LABELS[typ]}\n")
            print("| Dataset | Vocab Size | Unique Parse Trees | Semantic Diversity |")
            print("|---------|------------|--------------------|--------------------|")
            for r in rows:
                sem = (
                    f"{r['div_sem']:.4f}"
                    if isinstance(r["div_sem"], float)
                    else r["div_sem"]
                )
                print(
                    f"| {r['db_name']} | {r['vocab_size']} | {r['unique_parse_trees']} | {sem} |"
                )
            print()


if __name__ == "__main__":
    main()
