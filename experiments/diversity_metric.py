import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import random
import sqlglot
import sqlglot.errors
import sqlparse
import sys
import torch

from typing import Union
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

    logging.disable(logging.NOTSET)

    if cnt_prserr > 0:
        print(f"There were {cnt_prserr} parsing errors during processing.")
    s_keys = sorted(pts)
    with open(f"parse-trees-{name}-{type}.txt", "w") as f:
        for e in s_keys:
            f.write(f"{e}: {pts[e]}\n")
    print(f"Number of unique parse trees for {name} {type} queries: {len(pts)}")


def save_tsne(queries: list, type: str, name: str):
    # Load model
    bert_model = "ehsanaghaei/SecureBERT"
    tokenizer = RobertaTokenizerFast.from_pretrained(bert_model)
    rb_model = RobertaModel.from_pretrained(bert_model)
    rb_model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
            outputs = rb_model(**inputs)

            # Move back to CPU and convert to numpy
            outputs = rb_model(**inputs, output_hidden_states=True)
            batch_embeddings = outputs.pooler_output.cpu().numpy()
            embeddings.extend(batch_embeddings)

    embeddings = np.array(embeddings)
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

    with open(f"tsne-{name}-{type}.pkl", "wb") as f:
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

    # Add color legend
    legend_label = f"{type.capitalize()} Queries"
    plt.legend([scatter], [legend_label])

    plt.tight_layout()
    plt.savefig(f"tsne-{name}-{type}.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to tsne-{name}-{type}.png")


def get_diversity_anubis(
    fp_dataset="../dataset.csv",
    samples_0: Union[int, None] = None,
    samples_1: Union[int, None] = None,
):
    df_anubis = pd.read_csv(
        fp_dataset,
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

    queries_anubis_0 = df_anubis[df_anubis["label"] == 0]["full_query"].to_list()
    queries_anubis_1 = df_anubis[df_anubis["label"] == 1]["full_query"].to_list()

    if samples_0 and samples_1:
        queries_anubis_0 = random.sample(queries_anubis_0, samples_0)
        queries_anubis_1 = random.sample(queries_anubis_1, samples_1)

    # Vocab size
    # print_vocab_size(queries_anubis_0, "normal", "ANUBIS")
    # print_vocab_size(queries_anubis_1, "attack", "ANUBIS")

    # # PTs
    # print_unique_pts(queries_anubis_0, "normal", "ANUBIS")
    # print_unique_pts(queries_anubis_1, "attack", "ANUBIS")

    # T-SNE
    save_tsne(queries_anubis_0, "normal", "ANUBIS")


def get_diversity_wafamole(
    samples_0: Union[int, None] = None,
    samples_1: Union[int, None] = None,
):
    # Paths to merged files as described in documentation.
    fp_sane = "../../orignal_wafamole_dataset/sane.sql"
    sane = open(fp_sane, "r").read()
    sanes = sqlparse.split(sane)

    fp_attacks = "../../orignal_wafamole_dataset/attacks.sql"
    attack = open(fp_attacks, "r").read()
    attacks = sqlparse.split(attack)

    if samples_0 and samples_1:
        attacks = random.sample(attacks, samples_0)
        sanes = random.sample(sanes, samples_1)
    # Vocab size
    print_vocab_size(sanes, "normal", "WAFAMOLE")
    print_vocab_size(attacks, "attack", "WAFAMOLE")

    # PTs
    print_unique_pts(sanes, "normal", "WAFAMOLE")
    print_unique_pts(attacks, "attack", "WAFAMOLE")


def get_diversity_kaggle(
    fp_kaggle: str,
    samples_0: Union[int, None] = None,
    samples_1: Union[int, None] = None,
):  # We used: https://www.kaggle.com/datasets/sajid576/sql-injection-dataset
    # It does not require preprocessing as it is well formatted.
    df_kaggle = pd.read_csv(fp_kaggle)

    queries_kaggle_0 = df_kaggle[df_kaggle["Label"] == 0]["Query"].to_list()
    queries_kaggle_1 = df_kaggle[df_kaggle["Label"] == 1]["Query"].to_list()

    if samples_0 and samples_1:
        queries_kaggle_0 = random.sample(queries_kaggle_0, samples_0)
        queries_kaggle_1 = random.sample(queries_kaggle_1, samples_1)

    # Vocab size
    print_vocab_size(queries_kaggle_0, "normal", "Kaggle")
    print_vocab_size(queries_kaggle_1, "attack", "Kaggle")

    # PTs
    print_unique_pts(queries_kaggle_0, "normal", "Kaggle")
    print_unique_pts(queries_kaggle_1, "attack", "Kaggle")

def build_tsne_figures_all_datasets():
    # Using the saved pickles, build a TSNE figure with all datasets 
    datasets = ["Kaggle", "ANUBIS", "WAFAMOLE"]

    for dataset in datasets:
        with open(f"tsne-{dataset}-normal.pkl", "rb") as f:
            results = pickle.load(f)
        tsne_embeddings = results["tsne_embeddings"]
        

def main():
    samples_0 = None
    samples_1 = None
    kaggle_path = "/home/gquetel/Downloads/Modified_SQL_Dataset.csv"
    # anubis_path = "/home/gquetel/experiences-results/dataset-generation/unsupervized-v6/dataset.csv"

    # anubis_path = "../dataset-small.csv"
    anubis_path = "../10percent-anubis.csv"

    # We want to observe the same metrics with datasets of similar size: we randomly
    # sample from WAFAMOLE and Superviz number of samples present in Kaggle.
    samples_0 = 64
    # samples_1 = 11382

    get_diversity_anubis(anubis_path, samples_0, samples_1)
    # get_diversity_wafamole(samples_0, samples_1)
    # get_diversity_kaggle(kaggle_path, samples_0, samples_1)
    build_tsne_figures_all_datasets()


if __name__ == "__main__":
    main()
