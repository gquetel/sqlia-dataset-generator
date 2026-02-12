"""
Generate experimental train/test datasets for cross-dataset generalization studies.

Creates generic (cross-dataset) and specialised (same-dataset) experiment files
by sampling from the full generated datasets in output/.

Script generated using Claude Code.
"""

import argparse
import os
import pandas as pd
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = "/home/infres/gquetel/datasets/full"

DATASETS = {
    "OurAirports": os.path.join(REPO_ROOT, "OurAirports.csv"),
    "sakila": os.path.join(REPO_ROOT, "sakila.csv"),
    "AdventureWorks": os.path.join(REPO_ROOT, "AdventureWorks.csv"),
    "OHR": os.path.join(REPO_ROOT, "OHR.csv"),
}

TRAIN_SIZE = 100_000
TEST_SIZE = 1_000_000
TINY_TRAIN_SIZE = 500
TINY_TEST_SIZE = 5_000


def sample_split(path, split, n, seed, chunksize=500_000):
    """Sample n rows from a CSV where split column matches, using chunked reading."""
    rng = np.random.default_rng(seed)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        filtered = chunk[chunk["split"] == split]
        if len(filtered) > 0:
            chunks.append(filtered)

    df = pd.concat(chunks, ignore_index=True)
    if len(df) <= n:
        print(f"  Warning: only {len(df)} rows available (requested {n})")
        return df
    return df.sample(n=n, random_state=rng.integers(2**31))


def main():
    parser = argparse.ArgumentParser(description="Sample experimental datasets")
    parser.add_argument("--output-dir", default=SCRIPT_DIR, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Generate small datasets (500 train, 5000 test) for one target only",
    )
    parser.add_argument(
        "--specialised-only",
        action="store_true",
        help="Only generate specialised (same-dataset) experiments, skip generic",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_size = TINY_TRAIN_SIZE if args.tiny else TRAIN_SIZE
    test_size = TINY_TEST_SIZE if args.tiny else TEST_SIZE
    suffix = "-tiny" if args.tiny else ""
    names = list(DATASETS.keys())

    # In tiny mode, only process one target (OHR)
    targets = ["OHR"] if args.tiny else names

    # Step 1: Sample test sets
    print("=== Step 1: Sampling test sets ===")
    test_sets = {}
    for name in targets:
        print(f"  Sampling test set for {name}...")
        test_sets[name] = sample_split(DATASETS[name], "test", test_size, args.seed)
        print(f"  -> {len(test_sets[name])} test samples")

    # Step 2: Generic datasets
    if args.specialised_only:
        print("\n=== Skipping generic datasets (--specialised-only) ===")
    else:
        print("\n=== Step 2: Generic datasets ===")
        for target in targets:
            sources = [n for n in names if n != target]
            per_source = train_size // len(sources)
            remainder = train_size - per_source * len(sources)

            train_parts = []
            for i, src in enumerate(sources):
                n = per_source + (1 if i < remainder else 0)
                print(f"  Sampling {n} train rows from {src} for generic-{target}...")
                part = sample_split(
                    DATASETS[src], "train", n, args.seed + hash(src) % 2**16
                )
                train_parts.append(part)

            train_df = pd.concat(train_parts, ignore_index=True)
            train_df["split"] = "train"

            test_df = test_sets[target].copy()
            test_df["split"] = "test"

            out = pd.concat([train_df, test_df], ignore_index=True)
            outpath = os.path.join(args.output_dir, f"generic-{target}{suffix}.csv")
            out.to_csv(outpath, index=False)
            print(f"  Saved {outpath} ({len(train_df)} train + {len(test_df)} test)")

    # Step 3: Specialised datasets
    print("\n=== Step 3: Specialised datasets ===")
    for target in targets:
        print(f"  Sampling {train_size} train rows from {target}...")
        train_df = sample_split(DATASETS[target], "train", train_size, args.seed + 1)
        train_df["split"] = "train"

        test_df = test_sets[target].copy()
        test_df["split"] = "test"

        out = pd.concat([train_df, test_df], ignore_index=True)
        outpath = os.path.join(args.output_dir, f"specialised-{target}{suffix}.csv")
        out.to_csv(outpath, index=False)
        print(f"  Saved {outpath} ({len(train_df)} train + {len(test_df)} test)")

    print("\nDone.")


if __name__ == "__main__":
    main()
