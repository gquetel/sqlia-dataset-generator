"""Experiment: Cross-domain attack embedding similarity per attack technique.

For each attack technique, computes cosine similarity of embeddings:
- Within-domain: same technique, same dataset
- Cross-domain: same technique, different datasets
Transferability ratio = cross-domain / within-domain similarity.

Uses user_inputs (attack payload only) to isolate the attack pattern
from domain-specific SQL vocabulary (table names, column names, values).

A higher ratio for RoBERTa vs SecureBERT would mean RoBERTa represents the same
attack technique more consistently across domains.
"""

import argparse
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.metrics.pairwise import cosine_similarity

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from extractors.roberta import RobertaExtractor
from extractors.securebert import SecureBERTExtractor

logger = logging.getLogger(__name__)

EXTRACTORS = {
    "roberta-base": RobertaExtractor,
    "securebert": SecureBERTExtractor,
}

ATTACK_TECHNIQUES = [
    "boolean",
    "error",
    "inline",
    "stacked",
    "time",
    "union",
    "insider",
]


def load_attacks(path: str, techniques: list[str], n_samples: int) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        usecols=["user_inputs", "label", "attack_technique", "split"],
        low_memory=False,
    )
    df = df[(df["label"] == 1) & (df["split"] == "test")]
    df = df[df["attack_technique"].isin(techniques)].dropna(subset=["user_inputs"])
    sampled = df.groupby("attack_technique", group_keys=False).apply(
        lambda g: g.sample(n=min(n_samples, len(g)), random_state=2)
    )
    return sampled.reset_index(drop=True)


def extract_embeddings(extractor_cls, device, texts: list[str]) -> np.ndarray:
    # Wrap as full_query since that is what extractors expect
    extractor = extractor_cls(device=device, embeddings_path=None, batch_size=32)
    df = pd.DataFrame({"full_query": texts})
    result = extractor.extract_features(df)
    if isinstance(result, list):
        return np.stack(result)
    return np.asarray(result)


def mean_pairwise_cosine(X: np.ndarray) -> float:
    if len(X) < 2:
        return np.nan
    sim = cosine_similarity(X)
    upper = np.triu(np.ones(sim.shape, dtype=bool), k=1)
    return float(sim[upper].mean())


def mean_cross_cosine(X1: np.ndarray, X2: np.ndarray) -> float:
    if len(X1) == 0 or len(X2) == 0:
        return np.nan
    return float(cosine_similarity(X1, X2).mean())


def compute_stats(
    domain_embeddings: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    """domain_embeddings[technique][domain] = embeddings array."""
    rows = []
    for technique, by_domain in domain_embeddings.items():
        domains = [d for d, X in by_domain.items() if len(X) >= 2]
        if len(domains) < 2:
            continue

        within = np.nanmean([mean_pairwise_cosine(by_domain[d]) for d in domains])
        cross = np.nanmean(
            [
                mean_cross_cosine(by_domain[d1], by_domain[d2])
                for d1, d2 in combinations(domains, 2)
            ]
        )
        rows.append(
            {
                "technique": technique,
                "within_domain_sim": within,
                "cross_domain_sim": cross,
                "transferability": cross / within if within > 0 else np.nan,
                "n_domains": len(domains),
            }
        )
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Compare cross-domain attack embedding similarity for RoBERTa vs SecureBERT"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        metavar="NAME_OR_PATH",
        required=True,
        help="Alternating NAME PATH pairs, e.g. A /path/a.csv B /path/b.csv",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=500,
        help="Max attack samples per technique per dataset (default: 500)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiments/h4_attack_similarity",
    )
    parser.add_argument("--testing", action="store_true")
    args = parser.parse_args()

    tokens = args.dataset
    if len(tokens) % 2 != 0:
        parser.error("--dataset requires an even number of arguments (NAME PATH pairs)")
    datasets = [(tokens[i], tokens[i + 1]) for i in range(0, len(tokens), 2)]

    if args.testing:
        args.n_samples = 20
        print(f"[testing] n_samples={args.n_samples}")

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    techniques = ATTACK_TECHNIQUES

    all_stats = []

    for model_name, extractor_cls in EXTRACTORS.items():
        print(f"\n── {model_name} ──")
        domain_embeddings: dict[str, dict[str, np.ndarray]] = {
            t: {} for t in techniques
        }

        for name, path in datasets:
            print(f"  Loading {name} ...")
            df = load_attacks(path, techniques, args.n_samples)
            for technique, group in df.groupby("attack_technique"):
                texts = group["user_inputs"].tolist()
                print(
                    f"    {technique}: {len(texts)} samples → extracting embeddings ..."
                )
                embs = extract_embeddings(extractor_cls, device, texts)
                domain_embeddings[technique][name] = embs

        stats = compute_stats(domain_embeddings)
        stats.insert(0, "model", model_name)
        all_stats.append(stats)
        print(f"\n  Transferability ratios ({model_name}):")
        print(
            stats[
                [
                    "technique",
                    "within_domain_sim",
                    "cross_domain_sim",
                    "transferability",
                ]
            ].to_string(index=False)
        )

    df_all = pd.concat(all_stats, ignore_index=True)
    csv_path = output_dir / "attack_similarity.csv"
    df_all.to_csv(csv_path, index=False)
    print(f"\nSaved {csv_path}")

    # Summary: mean transferability per model
    print("\nMean transferability ratio per model:")
    for model_name, grp in df_all.groupby("model"):
        print(f"  {model_name}: {grp['transferability'].mean():.4f}")

    # Plot: grouped bar chart, transferability per technique per model
    fig = go.Figure()
    colors = {"roberta-base": "#4C78A8", "securebert": "#F58518"}
    for model_name, grp in df_all.groupby("model"):
        fig.add_trace(
            go.Bar(
                name=model_name,
                x=grp["technique"],
                y=grp["transferability"],
                marker_color=colors.get(model_name),
            )
        )
    fig.update_layout(
        title="Attack embedding transferability (cross-domain / within-domain cosine similarity)",
        xaxis_title="Attack technique",
        yaxis_title="Transferability ratio",
        barmode="group",
        height=500,
        width=1200,
        xaxis_tickangle=-45,
    )
    plot_path = output_dir / "attack_similarity.svg"
    fig.write_image(str(plot_path))
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
