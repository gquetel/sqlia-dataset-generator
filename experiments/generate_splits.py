"""
Generate experimental train/test datasets for cross-dataset generalization studies.

Creates LODO (leave-one-dataset-out) and in-domain experiment files
by sampling from the full generated datasets in output/.

Script generated using Claude Code.
"""

import argparse
import os
import pandas as pd
import numpy as np


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = "~/datasets/full"

DATASETS = {
    "OurAirports": os.path.join(REPO_ROOT, "OurAirports.csv"),
    "sakila": os.path.join(REPO_ROOT, "sakila.csv"),
    "AdventureWorks": os.path.join(REPO_ROOT, "AdventureWorks.csv"),
    "OHR": os.path.join(REPO_ROOT, "OHR.csv"),
}

DATASET_LETTERS = {
    "OurAirports": "a",
    "sakila": "b",
    "AdventureWorks": "c",
    "OHR": "d",
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


def get_concept_drift_template_split(path, seed, chunksize=500_000):
    """Split templates 50/50 per statement_type into origin and shifted sets.

    Collects unique (statement_type, query_template_id) pairs from normal train rows,
    then for each statement_type shuffles and splits 50/50 (odd count: extra to origin).
    Returns (origin_ids, shifted_ids) as sets of query_template_id strings.
    """
    pairs = set()
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        mask = (chunk["label"] == 0) & (chunk["split"] == "train")
        filtered = chunk[mask][
            ["statement_type", "query_template_id"]
        ].drop_duplicates()
        for row in filtered.itertuples(index=False):
            pairs.add((row.statement_type, row.query_template_id))

    # Group by statement_type
    by_type = {}
    for stype, tid in pairs:
        by_type.setdefault(stype, []).append(tid)

    rng = np.random.default_rng(seed)
    origin_ids = set()
    shifted_ids = set()
    for stype, tids in by_type.items():
        tids = sorted(tids)  # deterministic before shuffle
        rng.shuffle(tids)
        mid = (len(tids) + 1) // 2  # odd count: extra goes to origin
        origin_ids.update(tids[:mid])
        shifted_ids.update(tids[mid:])

    return origin_ids, shifted_ids


def sample_by_template_ids(path, split, template_ids, n, seed, chunksize=500_000):
    """Sample n rows filtered by split and query_template_id membership."""
    rng = np.random.default_rng(seed)
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        mask = (chunk["split"] == split) & (
            chunk["query_template_id"].isin(template_ids)
        )
        filtered = chunk[mask]
        if len(filtered) > 0:
            chunks.append(filtered)

    if not chunks:
        return pd.DataFrame()
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
        "--in-domain-only",
        action="store_true",
        help="Only generate in-domain (same-dataset) experiments, skip lodo",
    )
    parser.add_argument(
        "--concept-drift",
        action="store_true",
        help="Generate concept-drift splits (origin/shifted per dataset) instead of normal splits",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    train_size = TINY_TRAIN_SIZE if args.tiny else TRAIN_SIZE
    test_size = TINY_TEST_SIZE if args.tiny else TEST_SIZE
    suffix = "-tiny" if args.tiny else ""
    names = list(DATASETS.keys())

    # In tiny mode, only process one target (OHR)
    targets = ["OHR"] if args.tiny else names

    if args.concept_drift:
        print("=== Concept-drift splits ===")
        for name in targets:
            path = DATASETS[name]
            print(f"  Computing template split for {name}...")
            origin_ids, shifted_ids = get_concept_drift_template_split(path, args.seed)
            print(
                f"  -> {len(origin_ids)} origin templates, {len(shifted_ids)} shifted templates"
            )

            print(f"  Sampling origin train rows ({train_size})...")
            origin_train = sample_by_template_ids(
                path, "train", origin_ids, train_size, args.seed + 2
            )
            origin_train["split"] = "train"

            print(f"  Sampling origin test rows ({test_size})...")
            origin_test = sample_by_template_ids(
                path, "test", origin_ids, test_size, args.seed + 3
            )
            origin_test["split"] = "test"

            origin_df = pd.concat([origin_train, origin_test], ignore_index=True)
            out_origin = os.path.join(args.output_dir, f"origin-{name}{suffix}.csv")
            origin_df.to_csv(out_origin, index=False)
            print(
                f"  Saved {out_origin} ({len(origin_train)} train + {len(origin_test)} test)"
            )

            print(f"  Sampling shifted test rows ({test_size})...")
            shifted_test = sample_by_template_ids(
                path, "test", shifted_ids, test_size, args.seed + 4
            )
            shifted_test["split"] = "test"
            out_shifted = os.path.join(args.output_dir, f"shifted-{name}{suffix}.csv")
            shifted_test.to_csv(out_shifted, index=False)
            print(f"  Saved {out_shifted} ({len(shifted_test)} test rows)")
    else:
        # Step 1: Sample test sets
        print("=== Step 1: Sampling test sets ===")
        test_sets = {}
        for name in targets:
            print(f"  Sampling test set for {name}...")
            test_sets[name] = sample_split(DATASETS[name], "test", test_size, args.seed)
            print(f"  -> {len(test_sets[name])} test samples")

        # Step 2: LODO datasets
        if args.in_domain_only:
            print("\n=== Skipping LODO datasets (--in-domain-only) ===")
        else:
            print("\n=== Step 2: LODO datasets ===")
            for target in targets:
                sources = [n for n in names if n != target]
                per_source = train_size // len(sources)
                remainder = train_size - per_source * len(sources)

                train_parts = []
                for i, src in enumerate(sources):
                    n = per_source + (1 if i < remainder else 0)
                    train_letters = "".join(sorted(DATASET_LETTERS[s] for s in sources))
                    target_letter = DATASET_LETTERS[target]
                    print(
                        f"  Sampling {n} train rows from {src} for {train_letters}-{target_letter}..."
                    )
                    part = sample_split(
                        DATASETS[src], "train", n, args.seed + hash(src) % 2**16
                    )
                    train_parts.append(part)

                train_df = pd.concat(train_parts, ignore_index=True)
                train_df["split"] = "train"

                test_df = test_sets[target].copy()
                test_df["split"] = "test"

                out = pd.concat([train_df, test_df], ignore_index=True)
                train_letters = "".join(sorted(DATASET_LETTERS[s] for s in sources))
                target_letter = DATASET_LETTERS[target]
                outpath = os.path.join(
                    args.output_dir, f"{train_letters}-{target_letter}{suffix}.csv"
                )
                out.to_csv(outpath, index=False)
                print(
                    f"  Saved {outpath} ({len(train_df)} train + {len(test_df)} test)"
                )

        # Step 3: In-domain datasets
        print("\n=== Step 3: In-domain datasets ===")
        for target in targets:
            print(f"  Sampling {train_size} train rows from {target}...")
            train_df = sample_split(
                DATASETS[target], "train", train_size, args.seed + 1
            )
            train_df["split"] = "train"

            test_df = test_sets[target].copy()
            test_df["split"] = "test"

            out = pd.concat([train_df, test_df], ignore_index=True)
            letter = DATASET_LETTERS[target]
            outpath = os.path.join(args.output_dir, f"{letter}-{letter}{suffix}.csv")
            out.to_csv(outpath, index=False)
            print(f"  Saved {outpath} ({len(train_df)} train + {len(test_df)} test)")

    print("\nDone.")


if __name__ == "__main__":
    main()
