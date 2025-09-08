import hashlib
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
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


def print_vocab_size(queries, type: str, name: str):
    v = CountVectorizer()
    X = v.fit_transform(queries)
    vocab_size = len(v.vocabulary_)
    print(f"Vocabulary size for {name} {type} queries: {vocab_size}")

    token_count = X.sum()
    ttr = vocab_size / token_count if token_count else 0
    print(f"Type-Token Ratio (TTR) for {name} {type} queries: {ttr:.4f}")

    with open(f"vocab-{name}-{type}.txt", "w") as f:
        for word, idx in sorted(v.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {word}\n")


def print_unique_pts(queries: list, type: str, name: str) -> dict:
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
                    sqlglot.exp.Identifier
                    | sqlglot.exp.Literal
                    | sqlglot.exp.Comment
                ):
                    i.set("this", "I")

                for i in glot_tree.find_all(sqlglot.exp.HexString):
                    i.set("this", "0")

                # print(repr(glot_tree))
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
            cnt_prserr+=1

    logging.disable(logging.NOTSET)

    if cnt_prserr > 0:
        print(f"There were {cnt_prserr} parsing errors during processing.")
    s_keys = sorted(pts)
    with open(f"parse-trees-{name}-{type}.txt", "w") as f:
        for e in s_keys:
            f.write(f"{e}: {pts[e]}\n")
    print(f"Number of unique parse trees for {name} {type} queries: {len(pts)}")


def compute_and_save_embeddings(df: pd.DataFrame):
    """Compute embeddings of queries (column 'full_query') and cache them.

    Args:
        df (pd.DataFrame): _description_
    """
    # Use caching mechanism.
    str_hash_df = hashlib.sha256(
        pd.util.hash_pandas_object(df, index=True).values
    ).hexdigest()

    fp_cache = "".join(["../output/", "embeddings-", str_hash_df, ".pkl"])
    queries = df["full_query"].to_list()

    if os.path.isfile(fp_cache):
        print(f"Loaded already preprocessed embeddings located from {fp_cache}")
        return pd.read_pickle(fp_cache)
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
        pd.to_pickle(embeddings, fp_cache)

    return embeddings


def print_dataset_tsne(
    df: pd.DataFrame, type: str, name: str, n_sampling: None | int = None
):
    if n_sampling:
        df = df.sample(n_sampling, random_state=42)

    queries = df["full_query"].to_list()
    embeddings = compute_and_save_embeddings(df)

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

    with open(f"../output/tsne-{name}-{type}.pkl", "wb") as f:
        pickle.dump(results, f)

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
    plt.savefig(f"../output/tsne-{name}-{type}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to ../output/tsne-{name}-{type}.png")


def print_div_sem(df: pd.DataFrame, type: str, name: str):
    """Diversity metric from: https://aclanthology.org/2024.findings-naacl.228.pdf

    Args:
        df (pd.DataFrame): _description_
        type (str): _description_
        name (str): _description_
    """

    _embeddings = compute_and_save_embeddings(df=df)
    pairwise_distances = pdist(_embeddings, metric="cosine")
    div_sem = np.mean(pairwise_distances)

    print(
        f"Semantic Diversity of {type} for dataset {name} using cosing distance: {div_sem}"
    )


def load_wafamole_samples(fp_sane: str, fp_attacks: str):
    # This is too long to parse each time, let's also save them as pickles.
    fp_patks = "../output/parsed-wafamole-attacks.pkl"
    fp_psane = "../output/parsed-wafamole-sane.pkl"

    if os.path.isfile(fp_patks):
        attacks = pd.read_pickle(fp_patks)
    else:
        attack = open(fp_attacks, "r").read()
        attacks = sqlparse.split(attack)
        pd.to_pickle(attacks, fp_patks)

    if os.path.isfile(fp_psane):
        sanes = pd.read_pickle(fp_psane)
    else:
        sane = open(fp_sane, "r").read()
        sanes = sqlparse.split(sane)
        pd.to_pickle(sanes, fp_psane)

    df_sane = pd.DataFrame(sanes, columns=["full_query"])
    df_attack = pd.DataFrame(attacks, columns=["full_query"])

    df_sane = df_sane.assign(label=0)
    df_attack = df_attack.assign(label=1)

    return pd.concat([df_sane, df_attack])


def process_dataset(
    df: pd.DataFrame,
    name: str,
    query_column: str = "full_query",
    label_column: str = "label",
    samples_0: int = None,
    samples_1: int = None,
    vocab: bool = False,
    parse_trees: bool = False,
    div_sem: bool = False,
):
    df_0 = df[df[label_column] == 0]
    df_1 = df[df[label_column] == 1]

    if samples_0:
        df_0 = df_0.sample(n=samples_0, random_state=42)
    if samples_1:
        df_1 = df_1.sample(n=samples_1, random_state=42)

    queries_0 = df_0[query_column].tolist()
    queries_1 = df_1[query_column].tolist()

    if vocab:
        print_vocab_size(queries_0, "normal", name)
        print_vocab_size(queries_1, "attack", name)
    if parse_trees:
        print_unique_pts(queries_0, "normal", name)
        print_unique_pts(queries_1, "attack", name)

    df_0 = df_0.rename(columns={query_column: "full_query"})
    df_1 = df_1.rename(columns={query_column: "full_query"})

    if div_sem:
        print_div_sem(df_0, "normal", name)
        print_div_sem(df_1, "attack", name)


def main():
    samples_0 = 5000
    samples_1 = 5000
    Path("../output").mkdir(exist_ok=True, parents=True)

    # From: https://github.com/zangobot/wafamole_dataset
    # cat attacks.sql.* > attacks.sqls
    # cat sane.sql.* > sane.sql
    wafamole_sane_path = "../../original_wafamole_dataset/sane.sql"
    wafamole_attacks_path = "../../original_wafamole_dataset/attacks.sql"

    # From: https://www.kaggle.com/datasets/sajid576/sql-injection-dataset
    kaggle_path = "../Modified_SQL_Dataset.csv"

    # From: 
    anubis_path = "/home/gquetel/experiences-results/dataset-generation/unsupervized-v7/dataset.csv"

    # Diversity metrics to measure:
    compute_vocab = True
    compute_parse_trees = True
    compute_div_sem = True

    
    df_anubis = pd.read_csv(
        anubis_path,
        # dtype is specified to prevent a DtypeWarning
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
    process_dataset(
        df=df_anubis,
        name="ANUBIS",
        # samples_0=samples_0,
        # samples_1=samples_1,
        vocab=compute_vocab,
        parse_trees=compute_parse_trees,
        # div_sem=compute_div_sem,
    )

    df_kaggle = pd.read_csv(kaggle_path)
    process_dataset(
        df_kaggle,
        name="Kaggle",
        query_column="Query",
        label_column="Label",
        # samples_0=samples_0,
        # samples_1=samples_1,
        vocab=compute_vocab,
        parse_trees=compute_parse_trees,
        # div_sem=compute_div_sem,
    )

    df_wafamole = load_wafamole_samples(
        fp_sane=wafamole_sane_path, fp_attacks=wafamole_attacks_path
    )
    process_dataset(
        df=df_wafamole,
        name="WAFAMOLE",
        # samples_0=samples_0,
        # samples_1=samples_1,
        vocab=compute_vocab,
        parse_trees=compute_parse_trees,
        # div_sem=compute_div_sem,
    )


if __name__ == "__main__":
    main()
