"""Detect distribution shift between train and test feature spaces.

For each extractor and each generic transfer-learning scenario (e.g. ABC→D),
runs shift detectors from Rabanser et al. (2019) as a two-step pipeline:
  DR method → statistical test
From the paper: Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift
https://arxiv.org/pdf/1810.11953

Detection accuracy (empirical power) is reported: the fraction of repetitions
where H0 ("train and test distributions are identical") is correctly rejected
at significance level alpha. This mirrors the paper's evaluation protocol.
"""

import argparse
import logging
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from scipy import sparse
from scipy.stats import binomtest, kstwobign
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize

SCRIPT_DIR = Path(__file__).parent.absolute()
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT / "models"))

from constants import DotDict
from registry import MODEL_CONFIGS, _make_extractor

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

SCENARIOS = {
    "ABC→D": (["A", "B", "C"], "D"),
    "ABD→C": (["A", "B", "D"], "C"),
    "ACD→B": (["A", "C", "D"], "B"),
    "BCD→A": (["B", "C", "D"], "A"),
}

SCENARIO_ORDER = ["ABC→D", "ABD→C", "ACD→B", "BCD→A"]

# DR method → statistical test pipelines
PIPELINES = [
    ("NoRed", "MMD"),
    ("NoRed", "KS"),
    ("BBSDs", "MMD"),
    ("BBSDs", "KS"),
    ("DomainClf", "Binomial"),
]

# Detection accuracy metric names (one per pipeline)
METRICS = [f"{dr}_{test}_detacc" for dr, test in PIPELINES]

# p-value key within each test's result dict
PVAL_KEY: dict[str, str] = {
    "MMD": "MMD_pval",
    "KS": "KS_pval",
    "Binomial": "pval",
}

N_REPS = 100
ALPHA = 0.05
DEFAULT_SAMPLE_SIZES = [10, 20, 50, 100, 200, 500, 1000]


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=CSV_DTYPES, low_memory=False)


def to_dense(X) -> np.ndarray:
    if sparse.issparse(X):
        return X.toarray()
    if isinstance(X, pd.DataFrame):
        return X.to_numpy()
    if isinstance(X, list):
        return np.stack(X)
    return np.asarray(X)


def build_extractor(extractor_name: str, device, embeddings_path: str, cache_dir: str):
    config = MODEL_CONFIGS[extractor_name]
    project_paths = DotDict({"embeddings_path": embeddings_path})
    return _make_extractor(
        config, device=device, project_paths=project_paths, cache_dir=cache_dir
    )


def fit_extractor(extractor, df_train: pd.DataFrame):
    extractor.prepare_for_training(df_train)
    if hasattr(extractor, "_fitted"):
        extractor.extract_features(df_train)


def sample_normal_test(df: pd.DataFrame, n: int) -> pd.DataFrame:
    normal = df[(df["split"] == "test") & (df["label"] == 0)]
    return normal.sample(n=min(n, len(normal)), random_state=2)


# --- DR methods ---


def dr_no_reduction(
    X_train: np.ndarray, X_test: np.ndarray, ctx: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Identity — returns features unchanged."""
    return X_train, X_test


def dr_bbsd_soft(
    X_train: np.ndarray, X_test: np.ndarray, ctx: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Project features through a normal/attack soft classifier (BBSDs).

    Uses a pre-trained logistic regression stored in ctx["BBSDs_clf"] if
    available, otherwise trains one on ctx["X_labeled"] / ctx["y_labeled"].
    """
    clf = ctx.get("BBSDs_clf")
    if clf is None:
        clf = LogisticRegression(max_iter=1000, random_state=2)
        clf.fit(ctx["X_labeled"], ctx["y_labeled"])
    return clf.predict_proba(X_train), clf.predict_proba(X_test)


def dr_domain_clf(
    X_train: np.ndarray, X_test: np.ndarray, ctx: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Train a domain classifier and return eval-half predictions split by true domain.

    Pools source (train) and target (test) features, trains a logistic regression
    on one half to distinguish domains, and returns predictions on the eval half
    separately for source and target samples (used by test_binomial).
    """
    n, m = len(X_train), len(X_test)
    X = np.concatenate([X_train, X_test], axis=0)
    y = np.concatenate([np.zeros(n), np.ones(m)])

    rng = np.random.default_rng(seed=2)
    idx = rng.permutation(n + m)
    half = (n + m) // 2
    train_idx, eval_idx = idx[:half], idx[half:]

    clf = LogisticRegression(max_iter=1000, random_state=2)
    clf.fit(X[train_idx], y[train_idx])

    preds = clf.predict(X[eval_idx])
    source_mask = y[eval_idx] == 0
    target_mask = y[eval_idx] == 1
    return preds[source_mask], preds[target_mask]


# --- Statistical tests ---

N_PERMUTATIONS = 1000


def _kernel_matrix(X_all: np.ndarray) -> np.ndarray:
    """RBF kernel matrix with median bandwidth heuristic.

    Uses ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y> to avoid the O(N^2*K) intermediate
    array produced by naive broadcasting (X[:,None] - X[None,:]), which costs
    N*N*K*8 bytes — e.g. 24 GB for N=2000, K=768.
    """
    sq_norms = np.einsum("ij,ij->i", X_all, X_all)  # (N,)
    sq_dists = sq_norms[:, None] + sq_norms[None, :] - 2.0 * (X_all @ X_all.T)  # (N,N)
    np.clip(sq_dists, 0, None, out=sq_dists)  # numerical safety
    sigma_sq = float(np.median(sq_dists[sq_dists > 0]))
    return np.exp(-sq_dists / sigma_sq)


def _mmd_sq_from_kernel(K: np.ndarray, n: int, m: int) -> float:
    """Unbiased MMD² estimate from a precomputed kernel matrix."""
    K_XX = K[:n, :n].copy()
    K_YY = K[n:, n:].copy()
    K_XY = K[:n, n:]
    np.fill_diagonal(K_XX, 0)
    np.fill_diagonal(K_YY, 0)
    return K_XX.sum() / (n * (n - 1)) + K_YY.sum() / (m * (m - 1)) - 2 * K_XY.mean()


def test_mmd(Z_train: np.ndarray, Z_test: np.ndarray) -> dict:
    """MMD statistic and p-value via vectorized permutation test on the kernel matrix.

    All N_PERMUTATIONS permutations are computed in batch using indicator matrices:
      W_src[p, i] = 1 if index i is assigned to source in permutation p.
    Then K_XX_sum[p] = (W_src @ K * W_src).sum(1) - diagonal correction.
    """
    n, m = len(Z_train), len(Z_test)
    N = n + m
    K = _kernel_matrix(np.concatenate([Z_train, Z_test], axis=0))
    observed = _mmd_sq_from_kernel(K, n, m)

    rng = np.random.default_rng(seed=2)
    perms = np.array([rng.permutation(N) for _ in range(N_PERMUTATIONS)])  # (P, N)

    p_idx = np.arange(N_PERMUTATIONS)[:, None]
    W_src = np.zeros((N_PERMUTATIONS, N))
    W_tgt = np.zeros((N_PERMUTATIONS, N))
    W_src[p_idx, perms[:, :n]] = 1.0
    W_tgt[p_idx, perms[:, n:]] = 1.0

    # WK_src[p, j] = sum_{i in src[p]} K[i, j]
    WK_src = W_src @ K  # (P, N)
    WK_tgt = W_tgt @ K  # (P, N)
    K_diag = np.diag(K)  # (N,)

    # Off-diagonal sums: K_XX_sum[p] = sum_{i≠j, i,j in src[p]} K[i,j]
    K_XX_offdiag = (WK_src * W_src).sum(1) - (W_src * K_diag).sum(1)
    K_YY_offdiag = (WK_tgt * W_tgt).sum(1) - (W_tgt * K_diag).sum(1)
    # Cross sum: sum_{i in src[p], j in tgt[p]} K[i,j]
    K_XY_sum = (WK_src * W_tgt).sum(1)

    mmd_sq_perms = (
        K_XX_offdiag / (n * (n - 1))
        + K_YY_offdiag / (m * (m - 1))
        - 2.0 * K_XY_sum / (n * m)
    )
    count = int((mmd_sq_perms >= observed).sum())

    return {
        "MMD": float(np.sqrt(max(observed, 0))),
        "MMD_pval": (count + 1) / (N_PERMUTATIONS + 1),
    }


def test_ks_bonferroni(Z_train: np.ndarray, Z_test: np.ndarray) -> dict:
    """Per-dimension KS test with Bonferroni correction for multiple testing.

    Vectorized: sorts all K dimensions at once via np.argsort, computes empirical
    CDFs via cumsum, then evaluates asymptotic p-values in bulk with kstwobign.sf.
    Equivalent to calling scipy.ks_2samp(method="asymp") per dimension but ~100x faster
    for high-dimensional features by avoiding K Python-level function calls.
    """
    if Z_train.ndim == 1:
        Z_train = Z_train[:, None]
        Z_test = Z_test[:, None]

    n, K = Z_train.shape
    m = Z_test.shape[0]

    # combined shape: (n+m, K); sort indices per dimension
    combined = np.concatenate([Z_train, Z_test], axis=0)
    sort_idx = np.argsort(combined, axis=0, kind="stable")  # (n+m, K)

    # 1 where the point comes from test, 0 from train
    is_test = np.zeros(n + m, dtype=np.float64)
    is_test[n:] = 1.0
    sorted_is_test = is_test[sort_idx]  # (n+m, K)

    cdf_train = np.cumsum(1.0 - sorted_is_test, axis=0) / n  # (n+m, K)
    cdf_test = np.cumsum(sorted_is_test, axis=0) / m  # (n+m, K)
    ks_stats = np.max(np.abs(cdf_train - cdf_test), axis=0)  # (K,)

    en = np.sqrt(n * m / (n + m))
    pvals = kstwobign.sf(en * ks_stats)  # asymptotic p-values, shape (K,)

    return {
        "KS": float(np.max(ks_stats)),
        "KS_pval": float(min(1.0, float(np.min(pvals)) * K)),
    }


def test_binomial(Z_source_preds: np.ndarray, Z_target_preds: np.ndarray) -> dict:
    """Binomial test on domain classifier accuracy against chance (0.5)."""
    n_correct = int((Z_source_preds == 0).sum() + (Z_target_preds == 1).sum())
    n_total = len(Z_source_preds) + len(Z_target_preds)
    acc = n_correct / n_total if n_total > 0 else 0.0
    p_value = float(binomtest(n_correct, n_total, 0.5, alternative="greater").pvalue)
    return {"acc": acc, "pval": p_value}


DR_FNS: dict[str, Callable] = {
    "NoRed": dr_no_reduction,
    "BBSDs": dr_bbsd_soft,
    "DomainClf": dr_domain_clf,
}

TEST_FNS: dict[str, Callable] = {
    "MMD": test_mmd,
    "KS": test_ks_bonferroni,
    "Binomial": test_binomial,
}


# --- Detection accuracy ---


def _single_rep(
    X_source: np.ndarray,
    X_target: np.ndarray,
    ctx: dict,
    n_samples: int,
    alpha: float,
    seed: int,
) -> dict[str, bool]:
    """One repetition: subsample, run all pipelines, return per-pipeline rejection."""
    rng = np.random.default_rng(seed=seed)
    src_idx = rng.choice(
        len(X_source), size=min(n_samples, len(X_source)), replace=False
    )
    tgt_idx = rng.choice(
        len(X_target), size=min(n_samples, len(X_target)), replace=False
    )
    Z_src = X_source[src_idx]
    Z_tgt = X_target[tgt_idx]
    return {
        f"{dr}_{test}": TEST_FNS[test](*DR_FNS[dr](Z_src, Z_tgt, ctx))[PVAL_KEY[test]]
        < alpha
        for dr, test in PIPELINES
    }


def _timed_single_rep(
    X_source: np.ndarray,
    X_target: np.ndarray,
    ctx: dict,
    n_samples: int,
    seed: int,
) -> dict[str, float]:
    """Run all pipelines sequentially and return per-pipeline wall-clock times (seconds)."""
    rng = np.random.default_rng(seed=seed)
    src_idx = rng.choice(
        len(X_source), size=min(n_samples, len(X_source)), replace=False
    )
    tgt_idx = rng.choice(
        len(X_target), size=min(n_samples, len(X_target)), replace=False
    )
    Z_src = X_source[src_idx]
    Z_tgt = X_target[tgt_idx]
    timings = {}
    for dr, test in PIPELINES:
        key = f"{dr}_{test}"
        t0 = time.perf_counter()
        TEST_FNS[test](*DR_FNS[dr](Z_src, Z_tgt, ctx))
        timings[key] = time.perf_counter() - t0
    return timings


def compute_detection_accuracy(
    X_source: np.ndarray,
    X_target: np.ndarray,
    ctx: dict,
    n_samples: int,
    n_reps: int,
    alpha: float,
    n_workers: int = 1,
    debug: bool = False,
) -> dict[str, float]:
    """Empirical detection power: fraction of repetitions where p-value < alpha.

    Repetitions are run in parallel across n_workers processes. Only BBSDs_clf
    is passed to workers (X_labeled / y_labeled are not needed and not serialized).
    """
    # Strip heavy training arrays from ctx — workers only need the fitted classifier
    worker_ctx = {k: v for k, v in ctx.items() if k not in ("X_labeled", "y_labeled")}
    pipeline_keys = [f"{dr}_{test}" for dr, test in PIPELINES]

    if debug:
        timings = _timed_single_rep(X_source, X_target, worker_ctx, n_samples, seed=2)
        timing_str = "  ".join(f"{k}={v*1000:.1f}ms" for k, v in timings.items())
        logger.info(
            "  [debug] per-pipeline timing (1 rep, n=%d): %s", n_samples, timing_str
        )

    rep_results = Parallel(n_jobs=n_workers, prefer="threads")(
        delayed(_single_rep)(
            X_source, X_target, worker_ctx, n_samples, alpha, seed=2 + i
        )
        for i in range(n_reps)
    )
    counts = {k: sum(r[k] for r in rep_results) for k in pipeline_keys}
    return {f"{k}_detacc": counts[k] / n_reps for k in pipeline_keys}


def main():
    parser = argparse.ArgumentParser(
        description="Compute shift detection accuracy across SQL injection datasets"
    )
    parser.add_argument(
        "--extractor",
        required=True,
        nargs="+",
        choices=list(MODEL_CONFIGS.keys()),
        help="Feature extractor(s) to use",
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        required=True,
        help="Dataset short name (e.g. A) and CSV path (repeatable)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Feature pool size per dataset (must be >= max sample size, default: 2000)",
    )
    parser.add_argument(
        "--sample-sizes",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_SIZES,
        metavar="N",
        help="Sample sizes to evaluate (paper's x-axis), default: 10 20 50 100 200 500 1000",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=N_REPS,
        help=f"Number of repetitions per sample size (default: {N_REPS})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help=f"Significance level for H0 rejection (default: {ALPHA})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/experiments/distribution_distances",
        help="Output directory for CSV results and heatmaps",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(64, max(1, int(os.cpu_count() * 0.8))),
        help="Parallel workers for repetitions (default: 80%% of cores, capped at 64 for OpenBLAS)",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="Reduce sample sizes and repetitions for quick iteration",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log per-pipeline timing for one rep before each parallel batch to identify bottlenecks",
    )
    args = parser.parse_args()

    if args.testing:
        args.sample_sizes = [10, 50]
        args.repetitions = 10
        args.samples = max(args.samples, max(args.sample_sizes))
        print(
            f"[testing] sample_sizes={args.sample_sizes}, repetitions={args.repetitions}"
        )

    max_n = max(args.sample_sizes)
    if args.samples < max_n:
        parser.error(f"--samples ({args.samples}) must be >= max sample size ({max_n})")

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = str(output_dir / "embeddings")
    Path(embeddings_path).mkdir(parents=True, exist_ok=True)
    cache_dir = str(output_dir / "feature_cache")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # BBSDs normal samples: proportional to --samples (10x), capped at 10 000
    bbsds_n = min(10000, args.samples * 10)

    dataset_names = [name for name, _ in args.dataset]

    datasets: dict[str, pd.DataFrame] = {}
    for name, path in args.dataset:
        print(f"Loading {name} from {path} ...")
        datasets[name] = load_dataset(path)

    df_train_combined = pd.concat(
        [df[df["split"] == "train"] for df in datasets.values()], ignore_index=True
    )
    print(f"Combined training set size: {len(df_train_combined)}")

    normal_samples: dict[str, pd.DataFrame] = {}
    for name, df in datasets.items():
        sampled = sample_normal_test(df, args.samples)
        print(f"  {name}: {len(sampled)} normal test samples (pool)")
        normal_samples[name] = sampled

    available = set(dataset_names)
    active_scenarios = {
        scenario: (train_ds, test_ds)
        for scenario, (train_ds, test_ds) in SCENARIOS.items()
        if set(train_ds) | {test_ds} <= available
    }
    if active_scenarios:
        print(f"Active scenarios: {list(active_scenarios.keys())}")
    else:
        print("No complete scenarios available (need all 4 datasets A/B/C/D)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for extractor_name in args.extractor:
        print(f"\n=== Extractor: {extractor_name} ===")
        pool_rows: dict[str, list] = defaultdict(list)
        extractor = build_extractor(
            extractor_name,
            device=device,
            embeddings_path=embeddings_path,
            cache_dir=cache_dir,
        )

        print("Fitting extractor on combined training data ...")
        fit_extractor(extractor, df_train_combined)

        features: dict[str, np.ndarray] = {}
        for name, df in normal_samples.items():
            print(f"Extracting features for dataset {name} ...")
            X_raw = extractor.extract_features(df)
            X = to_dense(X_raw)
            del X_raw
            features[name] = normalize(X, norm="l2")
            del X
            print(f"  {name}: feature matrix {features[name].shape}")

        # Pre-extract BBSDs labeled features per dataset:
        # - normals from train split (for zero overlap with shift detection samples)
        # - attacks from test split
        print("Extracting BBSDs labeled features ...")
        bbsds_normal_features: dict[str, np.ndarray] = {}
        bbsds_attack_features: dict[str, np.ndarray] = {}
        for name, df in datasets.items():
            normal_train = df[(df["split"] == "train") & (df["label"] == 0)]
            normal_sample = normal_train.sample(
                n=min(bbsds_n, len(normal_train)), random_state=2
            )
            X_raw = extractor.extract_features(normal_sample)
            bbsds_normal_features[name] = normalize(to_dense(X_raw), norm="l2")
            del X_raw

            attack_test = df[(df["split"] == "test") & (df["label"] == 1)]
            attack_test = attack_test.sample(
                n=min(bbsds_n, len(attack_test)), random_state=2
            )
            X_raw = extractor.extract_features(attack_test)
            bbsds_attack_features[name] = normalize(to_dense(X_raw), norm="l2")
            del X_raw

            print(
                f"  BBSDs {name}: {len(bbsds_normal_features[name])} normals, "
                f"{len(bbsds_attack_features[name])} attacks"
            )

        for scenario, (train_ds, test_ds) in active_scenarios.items():
            pool_key = "".join(train_ds)
            X_train = np.concatenate([features[n] for n in train_ds], axis=0)
            X_test = features[test_ds]

            ctx: dict = {
                "X_labeled": np.concatenate(
                    [bbsds_normal_features[n] for n in train_ds]
                    + [bbsds_attack_features[n] for n in train_ds]
                ),
                "y_labeled": np.concatenate(
                    [np.zeros(len(bbsds_normal_features[n])) for n in train_ds]
                    + [np.ones(len(bbsds_attack_features[n])) for n in train_ds]
                ),
            }
            # Pre-train BBSDs classifier once (fixed training data across all repetitions)
            bbsds_clf = LogisticRegression(max_iter=1000, random_state=2)
            bbsds_clf.fit(ctx["X_labeled"], ctx["y_labeled"])
            ctx["BBSDs_clf"] = bbsds_clf

            for n_samples in args.sample_sizes:
                detacc = compute_detection_accuracy(
                    X_train,
                    X_test,
                    ctx,
                    n_samples,
                    args.repetitions,
                    args.alpha,
                    args.workers,
                    debug=args.debug,
                )
                row = {
                    "extractor": extractor_name,
                    "scenario": scenario,
                    "n_samples": n_samples,
                    **detacc,
                }
                pool_rows[pool_key].append(row)

                summary = "  ".join(
                    f"{k.replace('_detacc', '')}={v:.2f}" for k, v in detacc.items()
                )
                print(f"  {scenario} n={n_samples:>5}: {summary}")

            # No-shift baseline: ABC vs ABC (same pool, independent subsamples).
            # Expected detacc ≈ alpha — true H0 ground truth.
            base_label = f"{pool_key}→{pool_key}"
            for n_samples in args.sample_sizes:
                detacc = compute_detection_accuracy(
                    X_train,
                    X_train,
                    ctx,
                    n_samples,
                    args.repetitions,
                    args.alpha,
                    args.workers,
                    debug=args.debug,
                )
                row = {
                    "extractor": extractor_name,
                    "scenario": base_label,
                    "n_samples": n_samples,
                    **detacc,
                }
                pool_rows[pool_key].append(row)
                summary = "  ".join(
                    f"{k.replace('_detacc', '')}={v:.2f}" for k, v in detacc.items()
                )
                print(f"  {base_label} n={n_samples:>5}: {summary}")

            # Mixed-shift baseline: ABC vs ABCD (D injected into deployment stream).
            # Tests detection power when the shift is diluted by seen domains.
            X_abcd = np.concatenate([X_train, features[test_ds]], axis=0)
            mixed_label = f"{pool_key}→{pool_key}{test_ds}"
            for n_samples in args.sample_sizes:
                detacc = compute_detection_accuracy(
                    X_train,
                    X_abcd,
                    ctx,
                    n_samples,
                    args.repetitions,
                    args.alpha,
                    args.workers,
                    debug=args.debug,
                )
                row = {
                    "extractor": extractor_name,
                    "scenario": mixed_label,
                    "n_samples": n_samples,
                    **detacc,
                }
                pool_rows[pool_key].append(row)
                summary = "  ".join(
                    f"{k.replace('_detacc', '')}={v:.2f}" for k, v in detacc.items()
                )
                print(f"  {mixed_label} n={n_samples:>5}: {summary}")

        for pool_key, rows in pool_rows.items():
            csv_path = (
                output_dir / f"{extractor_name}_{pool_key}_distances_scenarios.csv"
            )
            pd.DataFrame(rows).to_csv(csv_path, index=False)
            print(f"Saved {csv_path}")

        del extractor, features, bbsds_normal_features, bbsds_attack_features
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
