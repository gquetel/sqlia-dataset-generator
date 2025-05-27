import logging
import re
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

logger = logging.getLogger(__name__)

def extract_li_features(query: str) -> dict:
    """ Extract SQL keywords and patterns from a query.
    
    Returns a dictionary of boolean flags indicating presence of various SQL features.
    """
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

    # result.update(get_char_kinds_number(query=query))
    return result | get_char_kinds_number(query=query)


def get_escape_char_number(query: str) -> dict:
    """  Count the number of hexadecimal and Unicode escape sequences in a string.

    Args:
        query (str): The input string to analyze for escape sequences

    Returns:
        dict: A dictionary with two keys:
        - 'c_hex': Number of hexadecimal escape sequences 
        - 'c_unicode': Number of Unicode escape sequences 
    """
    p_hex = re.compile(r"\\x[0-9a-fA-F]{2}")
    p_unicode = re.compile(r"\\u[0-9a-fA-F]{4}")

    c_hex = len(p_hex.findall(query))
    c_unicode = len(p_unicode.findall(query))
    return {"c_hex": c_hex, "c_unicode": c_unicode}


def has_tautology(query):    
    pattern = r"(\w+)=\1"
    if re.search(pattern=pattern, string=query) == None:
        return False
    return True


def get_char_kinds_number(query: str) -> dict:
    """Return, in order, number of numeric chars, uppercase chars, spaces, special chars,a
    arhithmetic operators , square brackets, round brackets, curly brackets,
    Args:
        query (str): _description_

    Returns:
        dict: _description_
    """
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


def _get_RF_li_features_from_query(s: pd.Series) -> pd.Series:
    d_features = extract_li_features(s["full_query"])
    return pd.Series({**s, **d_features})


def pre_process_for_RF_li(df: pd.DataFrame) -> pd.DataFrame:
    # input df has 2 columns: full_query, label
    # output df has the two previous columns and the new features
    df_lexrf = df.copy()

    df_lexrf = df_lexrf.apply(_get_RF_li_features_from_query, axis=1)
    return df_lexrf

class CustomRF_Li:
    def __init__(self, GENERIC, max_depth: int | None = None):
        self.max_depth = max_depth
        self.GENERIC = GENERIC

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df_pped, labels = self.preprocess_for_preds(df)
        preds = self.clf.predict(df_pped.to_numpy())
        return labels, preds
    
    def predict_proba(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        f_matrix, labels = self.preprocess_for_preds(df)
        ppreds = self.clf.predict_proba(f_matrix.to_numpy())
        return labels, ppreds
    
    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_RF_li(df_pped)
        df_pped.drop(
            ["label", "full_query"]
            + [
                "statement_type",
                "query_template_id",
                "attack_payload",
                "attack_id",
                "attack_technique",
                "split",
                "attack_desc",
                "attack_status",
                "attack_stage", 
                "tamper_method", 
            ],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        return df_pped, labels

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pped, train_labels = self.preprocess_for_preds(df)

        rf = RandomForestClassifier(
            random_state=self.GENERIC.RANDOM_SEED, max_depth=self.max_depth, n_jobs=-1
        )
        rf.fit(df_pped.to_numpy(), train_labels)
        self.clf = rf

class CustomDT_Li: 
    def __init__(self, GENERIC, max_depth: int | None = None):
        self.max_depth = max_depth
        self.GENERIC = GENERIC
        self.feature_names = None

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        df_pped, labels = self.preprocess_for_preds(df)
        preds = self.clf.predict(df_pped.to_numpy())
        return labels, preds
    
    def predict_proba(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        f_matrix, labels = self.preprocess_for_preds(df)
        ppreds = self.clf.predict_proba(f_matrix.to_numpy())
        return labels, ppreds

    def preprocess_for_preds(self, df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_RF_li(df_pped)
        df_pped.drop(
            ["label", "full_query"]
            + [
                "statement_type",
                "query_template_id",
                "attack_payload",
                "attack_id",
                "attack_technique",
                "split",
                "attack_desc",
                "attack_status",
                "attack_stage", 
                "tamper_method", 
            ],
            axis=1,
            inplace=True,
            errors="ignore",
        )
        return df_pped, labels

    def train_model(
        self,
        df: pd.DataFrame,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pped, train_labels = self.preprocess_for_preds(df)
        self.feature_names = df_pped.columns

        dt = DecisionTreeClassifier(
            random_state=self.GENERIC.RANDOM_SEED, max_depth=self.max_depth
        )
        dt.fit(df_pped.to_numpy(), train_labels)
        self.clf = dt