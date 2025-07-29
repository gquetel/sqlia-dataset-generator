import random
from typing import Union
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sqlglot
import sqlparse
import sqlglot.errors
from tqdm import tqdm
import logging
import sys


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
    print_vocab_size(queries_anubis_0, "normal", "ANUBIS")
    print_vocab_size(queries_anubis_1, "attack", "ANUBIS")

    # PTs
    print_unique_pts(queries_anubis_0, "normal", "ANUBIS")
    print_unique_pts(queries_anubis_1, "attack", "ANUBIS")


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


def main():
    samples_0 = None
    samples_1 = None
    kaggle_path = "/home/gquetel/Downloads/Modified_SQL_Dataset.csv"
    anubis_path = "/home/gquetel/experiences-results/dataset-generation/unsupervized-v6/dataset.csv"

    # anubis_path = "../dataset-small.csv"
    # anubis_path = "../10percent-anubis.csv"

    # We want to observe the same metrics with datasets of similar size: we randomly
    # sample from WAFAMOLE and Superviz number of samples present in Kaggle.
    # samples_0 = 19537
    # samples_1 = 11382

    get_diversity_anubis(anubis_path, samples_0, samples_1)
    # get_diversity_wafamole(samples_0, samples_1)
    # get_diversity_kaggle(kaggle_path, samples_0, samples_1)


if __name__ == "__main__":
    main()
