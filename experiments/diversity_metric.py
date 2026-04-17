import argparse
import hashlib
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sqlglot
import sqlglot.errors
import sqlparse
import sys
import torch
from scipy.stats import gmean
from scipy.spatial.distance import pdist

from transformers import RobertaTokenizerFast, RobertaModel
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import CountVectorizer

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / ".cache" / "diversity_metric"


def print_vocab_size(queries, type: str, name: str, output_dir: Path) -> dict:
    v = CountVectorizer()
    X = v.fit_transform(queries)
    vocab_size = len(v.vocabulary_)
    print(f"Vocabulary size for {name} {type} queries: {vocab_size}")

    token_count = X.sum()
    ttr = vocab_size / token_count if token_count else 0
    print(f"Type-Token Ratio (TTR) for {name} {type} queries: {ttr:.4f}")

    with open(output_dir / f"vocab-{name}-{type}.txt", "w") as f:
        for word, idx in sorted(v.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {word}\n")

    return {"vocab_size": vocab_size, "ttr": float(ttr)}


def print_unique_pts(queries: list, type: str, name: str, output_dir: Path) -> dict:
    pts = {}
    cnt_prserr = 0

    logging.disable(sys.maxsize)
    for q in tqdm(queries):
        try:
            glot_trees = sqlglot.parse(q, dialect="mysql")
            for glot_tree in glot_trees:
                if glot_tree == None or isinstance(glot_tree, sqlglot.exp.Command):
                    # A Command is returned, the tool didn't manage to parse the query
                    # correctly, ignore those.
                    cnt_prserr += 1
                    continue

                # Replace all literals or identifier to get a canonical representation.
                # "Normalize" parse trees.
                for i in glot_tree.find_all(
                    sqlglot.exp.Identifier | sqlglot.exp.Literal | sqlglot.exp.Comment
                ):
                    i.set("this", "I")

                for i in glot_tree.find_all(sqlglot.exp.HexString):
                    i.set("this", "0")

                canon_tree = glot_tree.sql(comments=False)
                if canon_tree not in pts:
                    pts[canon_tree] = 1
                else:
                    pts[canon_tree] += 1
        except sqlglot.errors.ParseError as e:
            cnt_prserr += 1
        except sqlglot.errors.TokenError as e:
            cnt_prserr += 1
        except KeyError as e:
            cnt_prserr += 1

    logging.disable(logging.NOTSET)

    if cnt_prserr > 0:
        print(f"There were {cnt_prserr} parsing errors during processing.")
    s_keys = sorted(pts)
    with open(output_dir / f"parse-trees-{name}-{type}.txt", "w") as f:
        for e in s_keys:
            f.write(f"{e}: {pts[e]}\n")
    print(f"Number of unique parse trees for {name} {type} queries: {len(pts)}")

    return {"unique_parse_trees": len(pts), "parse_errors": cnt_prserr}


def compute_and_save_embeddings(df: pd.DataFrame, output_dir: Path):
    """Compute embeddings of queries (column 'full_query') and cache them.

    Args:
        df (pd.DataFrame): DataFrame with a 'full_query' column.
        output_dir (Path): Directory for caching embedding files.
    """
    # Use caching mechanism.
    str_hash_df = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

    fp_cache = output_dir / f"embeddings-{str_hash_df}.pkl"
    queries = df["full_query"].to_list()

    if os.path.isfile(fp_cache):
        print(f"Loaded already preprocessed embeddings located from {fp_cache}")
        return pd.read_pickle(fp_cache, compression="zstd")
    else:
        bert_model = "ehsanaghaei/SecureBERT"
        tokenizer = RobertaTokenizerFast.from_pretrained(bert_model)
        rb_model = RobertaModel.from_pretrained(bert_model)
        rb_model.eval()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rb_model.to(device)
        # We compute embeddings by batches, they should not be too big because
        # they might be bigger than memory.
        embeddings = []

        batch_size = 64
        with torch.no_grad():
            for i in tqdm(range(0, len(queries), batch_size)):
                batch_queries = queries[i : i + batch_size]

                inputs = tokenizer(
                    batch_queries,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )

                # Move inputs to device and get embeddings.
                inputs = {k: v.to(device) for k, v in inputs.items()}

                # Move back to CPU and convert to numpy
                outputs = rb_model(**inputs, output_hidden_states=True)
                batch_embeddings = outputs.pooler_output.cpu().numpy()
                embeddings.extend(batch_embeddings)

        embeddings = np.array(embeddings)
        print(f"Saved preprocessed embeddings at {fp_cache}")
        pd.to_pickle(embeddings, fp_cache, compression="zstd")

    return embeddings


def print_dataset_tsne(
    df: pd.DataFrame,
    type: str,
    name: str,
    output_dir: Path,
    n_sampling: None | int = None,
):
    if n_sampling:
        df = df.sample(n_sampling, random_state=42)

    queries = df["full_query"].to_list()
    embeddings = compute_and_save_embeddings(df, output_dir)

    # https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
    # Let's use default params as much as possible.
    # We set perplexity to 50 as the doc states that higher dimensions requires
    # higher values.
    tsne = TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(50, len(queries) - 1),
        verbose=1,
        n_jobs=-1,
    )
    tsne_embeddings = tsne.fit_transform(embeddings)

    # Save the results to allow to compute the figure with all datasets later.
    results = {
        "queries": queries,
        "embeddings": embeddings,
        "tsne_embeddings": tsne_embeddings,
        "type": type,
        "name": name,
    }

    print(f"t-SNE results saved to tsne-{name}-{type}.pkl")

    pd.to_pickle(results, output_dir / f"tsne-{name}-{type}.pkl", compression="zstd")

    # Now plot individual results.
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        tsne_embeddings[:, 0],
        tsne_embeddings[:, 1],
        alpha=0.6,
        s=20,
    )

    plt.title(
        f"t-SNE Visualization of {name} {type} \n"
        f"Using SecureBERT Embeddings (n={len(queries)})"
    )
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True, alpha=0.3)
    legend_label = f"{type.capitalize()} Queries"
    plt.legend([scatter], [legend_label])

    plt.tight_layout()
    plt.savefig(output_dir / f"tsne-{name}-{type}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {output_dir / f'tsne-{name}-{type}.png'}")


def print_div_sem(df: pd.DataFrame, type: str, name: str, output_dir: Path) -> dict:
    """Diversity metric from: https://aclanthology.org/2024.findings-naacl.228.pdf

    Args:
        df (pd.DataFrame): DataFrame with a 'full_query' column.
        type (str): Query type (e.g. "normal" or "attack").
        name (str): Dataset name.
        output_dir (Path): Directory for caching embedding files.
    """

    _embeddings = compute_and_save_embeddings(df=df, output_dir=output_dir)
    pairwise_distances = pdist(_embeddings, metric="cosine")
    div_sem = np.mean(pairwise_distances)

    print(
        f"Semantic Diversity of {type} for dataset {name} using cosine distance: {div_sem}"
    )

    return {"div_sem": float(div_sem)}


def load_wafamole_samples():
    """Load WAFAMOLE dataset from cache directory.

    Expects .cache/diversity_metric/wafamole/attacks.sql and sane.sql.
    Returns None with download instructions if files are missing.
    """
    wafamole_dir = CACHE_DIR / "wafamole"
    fp_attacks = wafamole_dir / "attacks.sql"
    fp_sane = wafamole_dir / "sane.sql"

    if not fp_attacks.is_file() or not fp_sane.is_file():
        print("WAFAMOLE dataset not found in cache. To use WAFAMOLE, run:")
        print(
            f"  git clone https://github.com/zangobot/wafamole_dataset {wafamole_dir}"
        )
        print(f"  cd {wafamole_dir}")
        print("  cat attacks.sql.* > attacks.sql")
        print("  cat sane.sql.* > sane.sql")
        return None

    # Parsing the SQL files is slow, so cache the parsed results as pickles.
    fp_patks = CACHE_DIR / "parsed-wafamole-attacks.pkl"
    fp_psane = CACHE_DIR / "parsed-wafamole-sane.pkl"

    if fp_patks.is_file():
        attacks = pd.read_pickle(fp_patks, compression="zstd")
    else:
        attack = open(fp_attacks, "r").read()
        attacks = sqlparse.split(attack)
        pd.to_pickle(attacks, fp_patks, compression="zstd")

    if fp_psane.is_file():
        sanes = pd.read_pickle(fp_psane, compression="zstd")
    else:
        sane = open(fp_sane, "r").read()
        sanes = sqlparse.split(sane)
        pd.to_pickle(sanes, fp_psane, compression="zstd")

    df_sane = pd.DataFrame(sanes, columns=["full_query"])
    df_attack = pd.DataFrame(attacks, columns=["full_query"])

    df_sane = df_sane.assign(label=0)
    df_attack = df_attack.assign(label=1)

    return pd.concat([df_sane, df_attack])


def load_kaggle_dataset():
    """Load Kaggle SQL injection dataset from cache directory.

    Expects .cache/diversity_metric/kaggle/Modified_SQL_Dataset.csv.
    Returns None with download instructions if file is missing.
    """
    kaggle_dir = CACHE_DIR / "kaggle"
    kaggle_path = kaggle_dir / "Modified_SQL_Dataset.csv"

    if not kaggle_path.is_file():
        print("Kaggle dataset not found in cache. To use Kaggle, run:")
        print("  pip install kaggle")
        print(
            f"  kaggle datasets download -d sajid576/sql-injection-dataset -p {kaggle_dir}/"
        )
        print(f"  unzip {kaggle_dir}/sql-injection-dataset.zip -d {kaggle_dir}/")
        return None

    df = pd.read_csv(kaggle_path)
    df = df.rename(columns={"Query": "full_query", "Label": "label"})
    return df


def process_dataset(
    df: pd.DataFrame,
    name: str,
    output_dir: Path,
    samples: int = None,
    vocab: bool = False,
    parse_trees: bool = False,
    div_sem: bool = False,
) -> list[dict]:
    # Build subsets depending on whether the DataFrame has a split column.
    if "split" in df.columns:
        subsets = [
            ("train-normal", df[(df["split"] == "train") & (df["label"] == 0)]),
            ("test-normal", df[(df["split"] == "test") & (df["label"] == 0)]),
            ("test-attack", df[(df["split"] == "test") & (df["label"] == 1)]),
        ]
    else:
        subsets = [
            ("normal", df[df["label"] == 0]),
            ("attack", df[df["label"] == 1]),
        ]

    rows = []
    for label, sub_df in subsets:
        if samples:
            sub_df = sub_df.sample(n=min(samples, len(sub_df)), random_state=42)

        queries = sub_df["full_query"].tolist()
        row = {"dataset": name, "type": label, "n_samples": len(queries)}
        if vocab:
            row.update(print_vocab_size(queries, label, name, output_dir))
        if parse_trees:
            row.update(print_unique_pts(queries, label, name, output_dir))
        if div_sem:
            row.update(print_div_sem(sub_df, label, name, output_dir))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute diversity metrics on SQL datasets"
    )
    parser.add_argument(
        "--dataset",
        nargs=2,
        action="append",
        metavar=("NAME", "PATH"),
        help="Dataset name and path to its CSV file (repeatable)",
    )
    parser.add_argument(
        "--with-wafamole",
        action="store_true",
        help="Include WAFAMOLE baseline comparison",
    )
    parser.add_argument(
        "--with-kaggle",
        action="store_true",
        help="Include Kaggle baseline comparison",
    )
    parser.add_argument(
        "--vocab", action="store_true", help="Compute vocabulary metrics"
    )
    parser.add_argument(
        "--parse-trees", action="store_true", help="Compute unique parse trees"
    )
    parser.add_argument(
        "--div-sem", action="store_true", help="Compute semantic diversity"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Subsample normal and attack queries to N each",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="../output/results/diversity_metric",
        help="Output directory",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    datasets = []  # list of (name, df)

    # We expect provided datasets using the argument to be of our format.
    if args.dataset:
        for name, path in args.dataset:
            df = pd.read_csv(
                path,
                dtype={
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
                },
            )
            datasets.append((name, df))

    # Load WAFAMOLE and Kaggle if requested
    if args.with_wafamole:
        df = load_wafamole_samples()
        if df is not None:
            datasets.append(("WAFAMOLE", df))

    if args.with_kaggle:
        df = load_kaggle_dataset()
        if df is not None:
            datasets.append(("Kaggle", df))

    if not datasets:
        parser.error(
            "No datasets provided. Use --dataset, --with-wafamole, or --with-kaggle."
        )

    all_rows = []
    for name, df in datasets:
        rows = process_dataset(
            df=df,
            name=name,
            output_dir=output_dir,
            samples=args.samples,
            vocab=args.vocab,
            parse_trees=args.parse_trees,
            div_sem=args.div_sem,
        )
        all_rows.extend(rows)

    if all_rows:
        results_df = pd.DataFrame(all_rows)
        csv_path = output_dir / "diversity_metrics.csv"
        results_df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
