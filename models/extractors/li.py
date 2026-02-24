"""Li et al. feature extractor — 23 boolean/count features from SQL queries."""

import re

import pandas as pd

from base import BaseExtractor


def extract_li_features(query: str) -> dict:
    """Extract SQL keywords and patterns from a query."""
    query_lower = query.lower()

    QUERY_KEYWORDS = {"select", "update", "drop", "insert", "create"}
    DATABASE_KEYWORDS = {
        "sysobjects",
        "msysobjects",
        "version",
        "information_schema",
    }
    CONNECTION_KEYWORDS = {"inner join", "and", "or", "xor"}
    FILE_KEYWORDS = {"load_file", "infile", "outfile", "dumpfile"}
    STRING_FUNCTIONS = {"substr", "substring", "mid", "asc"}
    COMPARISON_OPERATORS = {"=", "<", ">", "<=", ">=", "<>", "!="}

    result = {
        "len_query": len(query),
        "has_null": 0,
        "has_comment": int("--" in query_lower or "#" in query_lower),
        "has_query_keywords": 0,
        "has_union": 0,
        "has_database_keywords": 0,
        "has_connection_keywords": 0,
        "has_file_keywords": 0,
        "has_exec": int(re.search(r"\bexec\b", query_lower) is not None),
        "has_string_functions": 0,
        "c_comparison": 0,
        "has_exist_keyword": int(re.search(r"\bexists\b", query_lower) is not None),
        "has_floor": int(re.search(r"\bfloor\b", query_lower) is not None),
        "has_rand": int(re.search(r"\brand\b", query_lower) is not None),
        "has_group": int(re.search(r"\bgroup\b", query_lower) is not None),
        "has_order": int(re.search(r"\border\b", query_lower) is not None),
        "has_length": int(re.search(r"\blength\b", query_lower) is not None),
        "has_ascii": int(re.search(r"\bascii\b", query_lower) is not None),
        "has_concat": int(re.search(r"\bconcat\b", query_lower) is not None),
        "has_if": int(re.search(r"\bif\b", query_lower) is not None),
        "has_count": int(re.search(r"\bcount\b", query_lower) is not None),
        "has_sleep": int(re.search(r"\bsleep\b", query_lower) is not None),
        "has_tautology": has_tautology(query),
    }

    if re.search(r"\bnull\b", query_lower):
        result["has_null"] = 1

    word_pattern = r"\b{}\b"
    for keyword in QUERY_KEYWORDS:
        if re.search(word_pattern.format(keyword), query_lower):
            result["has_query_keywords"] = 1
            break

    if re.search(r"\bunion\b", query_lower):
        result["has_union"] = 1

    for keyword in CONNECTION_KEYWORDS:
        if keyword in query_lower:
            result["has_connection_keywords"] = 1
            break

    for keyword in FILE_KEYWORDS:
        if keyword in query_lower:
            result["has_file_keywords"] = 1
            break

    for keyword in DATABASE_KEYWORDS:
        if keyword in query_lower:
            result["has_database_keywords"] = 1
            break

    for func in STRING_FUNCTIONS:
        if re.search(word_pattern.format(func), query_lower):
            result["has_string_functions"] = 1
            break

    for op in COMPARISON_OPERATORS:
        result["c_comparison"] += query_lower.count(op)

    return result | get_char_kinds_number(query=query)


def get_escape_char_number(query: str) -> dict:
    p_hex = re.compile(r"\\x[0-9a-fA-F]{2}")
    p_unicode = re.compile(r"\\u[0-9a-fA-F]{4}")
    c_hex = len(p_hex.findall(query))
    c_unicode = len(p_unicode.findall(query))
    return {"c_hex": c_hex, "c_unicode": c_unicode}


def has_tautology(query):
    pattern = r"(\w+)=\1"
    if re.search(pattern=pattern, string=query) == None:
        return 0
    return 1


def get_char_kinds_number(query: str) -> dict:
    c_num = 0
    c_upper = 0
    c_space = 0
    c_special = 0
    c_arith = 0
    c_square_brackets = 0
    c_round_brackets = 0
    c_curly_brackets = 0
    c_quot_within_quot = 0
    is_within_squot = False
    is_within_dquot = False
    has_multiline_comment = 0
    enumerator = enumerate(query)
    for i, c in enumerator:
        if c.isdigit():
            c_num += 1
        elif c.isupper():
            c_upper += 1
        elif c.isspace():
            c_space += 1
        elif c == "/":
            if i + 1 < len(query) and query[i + 1] == "*":
                has_multiline_comment = 1
            else:
                c_arith += 1
        elif c == "*":
            if i + 1 < len(query) and query[i + 1] == "/":
                has_multiline_comment = 1
            else:
                c_arith += 1
        elif c in ["+", "-"]:
            c_arith += 1
        elif c == "[" or c == "]":
            c_square_brackets += 1
        elif c == "(" or c == ")":
            c_round_brackets += 1
        elif c == "{" or c == "}":
            c_curly_brackets += 1
        elif c == "'":
            if is_within_dquot:
                c_quot_within_quot += 1
            else:
                is_within_squot = True
            c_special += 1
        elif c == '"':
            if is_within_squot:
                c_quot_within_quot += 1
            else:
                is_within_dquot = True
            c_special += 1
        elif not c.isalnum():
            c_special += 1

    return {
        "c_num": c_num,
        "c_upper": c_upper,
        "c_space": c_space,
        "c_special": c_special,
        "c_arith": c_arith,
        "c_square_brackets": c_square_brackets,
        "c_round_brackets": c_round_brackets,
        "has_multiline_comment": has_multiline_comment,
        "c_curly_brackets": c_curly_brackets,
    }


def _get_li_features_from_row(s: pd.Series) -> pd.Series:
    return pd.Series(extract_li_features(s["full_query"]))


def pre_process_for_li(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(_get_li_features_from_row, axis=1)


class LiExtractor(BaseExtractor):
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        return pre_process_for_li(df)
