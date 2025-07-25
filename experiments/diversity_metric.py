import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import sqlglot
import sqlparse
import sqlglot.errors
from tqdm import tqdm
import logging
import sys

# MIT License
#
# Copyright (c) 2019 Cong Feng.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# From: https://github.com/neural-dialogue-metrics/Distinct-N

"""
Copied from nltk.ngrams().
"""
from itertools import chain


def pad_sequence(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Returns a padded sequence of items before ngram extraction.

        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        ['<s>', 1, 2, 3, 4, 5, '</s>']
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        ['<s>', 1, 2, 3, 4, 5]
        >>> list(pad_sequence([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [1, 2, 3, 4, 5, '</s>']

    :param sequence: the source data to be padded
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = iter(sequence)
    if pad_left:
        sequence = chain((left_pad_symbol,) * (n - 1), sequence)
    if pad_right:
        sequence = chain(sequence, (right_pad_symbol,) * (n - 1))
    return sequence


def ngrams(
    sequence,
    n,
    pad_left=False,
    pad_right=False,
    left_pad_symbol=None,
    right_pad_symbol=None,
):
    """
    Return the ngrams generated from a sequence of items, as an iterator.
    For example:

        >>> from nltk.util import ngrams
        >>> list(ngrams([1,2,3,4,5], 3))
        [(1, 2, 3), (2, 3, 4), (3, 4, 5)]

    Wrap with list for a list version of this function.  Set pad_left
    or pad_right to true in order to get additional ngrams:

        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, None)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_right=True, right_pad_symbol='</s>'))
        [(1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, left_pad_symbol='<s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5)]
        >>> list(ngrams([1,2,3,4,5], 2, pad_left=True, pad_right=True, left_pad_symbol='<s>', right_pad_symbol='</s>'))
        [('<s>', 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, '</s>')]


    :param sequence: the source data to be converted into ngrams
    :type sequence: sequence or iter
    :param n: the degree of the ngrams
    :type n: int
    :param pad_left: whether the ngrams should be left-padded
    :type pad_left: bool
    :param pad_right: whether the ngrams should be right-padded
    :type pad_right: bool
    :param left_pad_symbol: the symbol to use for left padding (default is None)
    :type left_pad_symbol: any
    :param right_pad_symbol: the symbol to use for right padding (default is None)
    :type right_pad_symbol: any
    :rtype: sequence or iter
    """
    sequence = pad_sequence(
        sequence, n, pad_left, pad_right, left_pad_symbol, right_pad_symbol
    )

    history = []
    while n > 1:
        history.append(next(sequence))
        n -= 1
    for item in sequence:
        history.append(item)
        yield tuple(history)
        del history[0]


def distinct_n_sentence_level(sentence, n):
    """
    Compute distinct-N for a single sentence.
    :param sentence: a list of words.
    :param n: int, ngram.
    :return: float, the metric value.
    """
    if len(sentence) == 0:
        return 0.0  # Prevent a zero division
    distinct_ngrams = set(ngrams(sentence, n))
    return len(distinct_ngrams) / len(sentence)


def distinct_n_corpus_level(sentences, n):
    """
    Compute average distinct-N of a list of sentences (the corpus).
    :param sentences: a list of sentence.
    :param n: int, ngram.
    :return: float, the average value.
    """
    return sum(
        distinct_n_sentence_level(sentence, n) for sentence in sentences
    ) / len(sentences)


def print_distinctn_diversity(queries, type: str, name: str):
    # Higher value = Better
    # It means that the corpus exhibit a high proportion of unique n-grams
    print(
        f"Type-Token Ratio for {name} {type} queries: {distinct_n_corpus_level(queries,1)}"
    )
    print(
        f"Distinct-2 diversity for {name} {type} queries: {distinct_n_corpus_level(queries,2)}"
    )
    print(
        f"Distinct-3 diversity for {name} {type} queries: {distinct_n_corpus_level(queries,3)}"
    )


def print_vocab_size(queries, type: str, name: str):
    # TODO: See if another tokenizer would be more appropriate.
    v = CountVectorizer()
    v.fit_transform(queries)
    print(f"Vocabulary size for {name} {type} queries: {len(v.vocabulary_)}")

    with open(f"vocab-{name}-{type}.txt", "w") as f:
        for word, idx in sorted(v.vocabulary_.items(), key=lambda x: x[1]):
            f.write(f"{idx}: {word}\n")


def print_unique_pts(queries: list, type: str, name: str) -> dict:
    pts = {}
    cnt_prserr = 0

    logging.disable(sys.maxsize)
    for q in tqdm(queries):
        try:
            glot_tree = sqlglot.parse_one(q, dialect="mysql")
            if isinstance(glot_tree, sqlglot.exp.Command):
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

            print(repr(glot_tree))
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


def get_diversity_anubis(fp_dataset="../dataset.csv"):
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

    # Distinct-n
    print_distinctn_diversity(queries_anubis_0, "normal", "ANUBIS")
    print_distinctn_diversity(queries_anubis_1, "attack", "ANUBIS")

    # Vocab size
    print_vocab_size(queries_anubis_0, "normal", "ANUBIS")
    print_vocab_size(queries_anubis_1, "attack", "ANUBIS")

    # PTs
    print_unique_pts(queries_anubis_0, "normal", "ANUBIS")
    print_unique_pts(queries_anubis_1, "attack", "ANUBIS")


def get_diversity_wafamole():
    fp_sane = "../../orignal_wafamole_dataset/sane.sql"
    sane = open(fp_sane, "r").read()
    sanes = sqlparse.split(sane)

    fp_attacks = "../../orignal_wafamole_dataset/attacks.sql"
    attack = open(fp_attacks, "r").read()
    attacks = sqlparse.split(attack)

    # Distinct-n
    print_distinctn_diversity(sanes, "normal", "WAFAMOLE")
    print_distinctn_diversity(attacks, "attack", "WAFAMOLE")

    # Vocab size
    print_vocab_size(sanes, "normal", "WAFAMOLE")
    print_vocab_size(attacks, "attack", "WAFAMOLE")

    # PTs
    print_unique_pts(sanes, "normal", "WAFAMOLE")
    print_unique_pts(attacks, "attack", "WAFAMOLE")


def main():
    # anubis_path = "/home/gquetel/experiences-results/dataset-generation/unsupervized-v6/dataset.csv"
    anubis_path = "../dataset-small.csv"
    # anubis_path = "../10percent-anubis.csv"

    get_diversity_anubis(anubis_path)
    # get_diversity_wafamole()


if __name__ == "__main__":
    main()
