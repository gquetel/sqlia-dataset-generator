import logging
import re
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MaxAbsScaler
from constants import MyAutoEncoder, MyAutoEncoderRelu
import numpy as np
from sklearn.svm import OneClassSVM

logger = logging.getLogger(__name__)

import torch.nn as nn
import torch.nn.functional as F
import torch


def extract_li_features(query: str) -> dict:
    """Extract SQL keywords and patterns from a query.

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
    """Count the number of hexadecimal and Unicode escape sequences in a string.

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
        return 0
    return 1


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


def _get_ocsvm_li_features_from_query(s: pd.Series) -> pd.Series:
    d_features = extract_li_features(s["full_query"])
    return pd.Series({**s, **d_features})


def pre_process_for_li(df: pd.DataFrame) -> pd.DataFrame:
    # input df has 2 columns: full_query, label
    # output df has the two previous columns and the new features
    _df = df.copy()
    _df = _df.apply(_get_ocsvm_li_features_from_query, axis=1)
    return _df


class OCSVM_Li:
    def __init__(
        self,
        GENERIC,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str = "scale",
        max_iter: int = -1,
        use_scaler: bool = False,
    ):
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma
        self.clf = None

        self._scaler = StandardScaler()

        self.GENERIC = GENERIC
        self.model_name = None
        self.max_iter = max_iter
        self.use_scaler = use_scaler

    def preprocess_for_preds(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()  # Mem ~OK
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_li(df_pped)

        if drop_og_columns:
            df_pped.drop(
                ["label", "full_query"]
                + [
                    "statement_type",
                    "query_template_id",
                    "user_inputs",
                    "attack_id",
                    "attack_technique",
                    "split",
                    "attack_desc",
                    "attack_status",
                    "attack_stage",
                    "tamper_method",
                    "template_split",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )

        return df_pped, labels

    def preprocess_for_train(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> pd.DataFrame:
        """Preprocess data for training. We ignore label data.

        Args:
            df (pd.DataFrame): _description_
            drop_og_columns (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        df_pped, _ = self.preprocess_for_preds(df=df, drop_og_columns=drop_og_columns)
        return df_pped

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name
        df_pped = self.preprocess_for_train(df)

        self.feature_columns = df_pped.columns.tolist()

        if self.use_scaler:
            df_pped = self._scaler.fit_transform(df_pped.values)

        model = OneClassSVM(
            nu=self.nu,
            kernel=self.kernel,
            gamma=self.gamma,
            max_iter=self.max_iter,
        )

        model.fit(df_pped)
        self.clf = model


class LOF_Li:
    def __init__(
        self,
        GENERIC,
        n_jobs: int = -1,
        contamination: float = 0.1,
        use_scaler: bool = False,
    ):
        self.contamination = contamination
        self.n_jobs = n_jobs

        self._scaler = StandardScaler()

        self.random_state = GENERIC.RANDOM_SEED
        self.clf = None
        self.GENERIC = GENERIC
        self.model_name = None
        self.use_scaler = use_scaler

    def preprocess_for_preds(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_li(df_pped)

        if drop_og_columns:
            df_pped.drop(
                ["label", "full_query"]
                + [
                    "statement_type",
                    "query_template_id",
                    "user_inputs",
                    "attack_id",
                    "attack_technique",
                    "split",
                    "attack_desc",
                    "attack_status",
                    "attack_stage",
                    "tamper_method",
                    "template_split",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )

        return df_pped, labels

    def preprocess_for_train(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> pd.DataFrame:
        """Preprocess data for training. We ignore label data.

        Args:
            df (pd.DataFrame): _description_
            drop_og_columns (bool, optional): _description_. Defaults to True.

        Returns:
            pd.DataFrame: _description_
        """
        df_pped, _ = self.preprocess_for_preds(df=df, drop_og_columns=drop_og_columns)
        self.feature_columns = df_pped.columns.tolist()
        return df_pped

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        self.model_name = model_name

        df_pped = self.preprocess_for_train(df)
        model = LocalOutlierFactor(n_jobs=self.n_jobs, novelty=True)
        if self.use_scaler:
            df_pped = self._scaler.fit_transform(df_pped.values)
            model.fit(df_pped)
        else:
            model.fit(df_pped.values)
        self.clf = model


class AutoEncoder_Li:
    def __init__(
        self,
        GENERIC,
        device,
        learning_rate: float = 0.001,
        epochs: int = 100,
        batch_size: int = 32,
        use_scaler: bool = False,
    ):
        self.random_state = GENERIC.RANDOM_SEED
        self.clf = None
        self.GENERIC = GENERIC
        self.model_name = None

        # Let's use MaxAbsScaler => 0 and 1 because no value can be negative here.
        self._scaler = MaxAbsScaler()

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.use_scaler = use_scaler
        self.device = device

        self.feature_columns = None

    def preprocess_for_preds(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> tuple[pd.DataFrame, np.ndarray]:
        df_pped = df.copy()
        labels = np.array(df_pped["label"])
        df_pped = pre_process_for_li(df_pped)

        if drop_og_columns:
            df_pped.drop(
                ["label", "full_query"]
                + [
                    "statement_type",
                    "query_template_id",
                    "user_inputs",
                    "attack_id",
                    "attack_technique",
                    "split",
                    "attack_desc",
                    "attack_status",
                    "attack_stage",
                    "tamper_method",
                    "template_split",
                ],
                axis=1,
                inplace=True,
                errors="ignore",
            )
        return df_pped, labels

    def _dataframe_to_tensor_batched(self, df, batch_size=4096):
        """
        Used during testing.

        Args:
            df (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 4096.

        Returns:
            _type_: _description_
        """
        n_samples = len(df)
        tensors = []

        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            df_batch = df.iloc[i:batch_end]
            batch_dense = df_batch.values
            if self.use_scaler:
                batch_dense = self._scaler.transform(batch_dense)
            tensors.append(torch.FloatTensor(batch_dense))

        return torch.cat(tensors, dim=0).to(self.device)

    def preprocess_for_train(
        self, df: pd.DataFrame, drop_og_columns: bool = True
    ) -> pd.DataFrame:
        """Preprocess data for training. We ignore label data.
        Args:
            df (pd.DataFrame): Input dataframe
            drop_og_columns (bool, optional): Whether to drop original columns. Defaults to True.
        Returns:
            pd.DataFrame: Preprocessed dataframe
        """
        df_pped, _ = self.preprocess_for_preds(df=df, drop_og_columns=drop_og_columns)
        return df_pped

    def train_model(
        self,
        df: pd.DataFrame,
        project_paths,
        model_name: str = None,
    ):
        # Init variables for training + model
        self.model_name = model_name
        df_pped = self.preprocess_for_train(df)
        self.feature_columns = df_pped.columns.tolist()
        input_dim = len(self.feature_columns)

        # Let's apply Scaler here and not in preprocess, as we want to keep
        # information about the columns
        df_pped = np.array(df_pped)
        assert(df_pped.min() >= 0)

        # If scaling =>
        if self.use_scaler:
            scaled_data = self._scaler.fit_transform(df_pped)
            self._scaler_min = scaled_data.min(axis=None)
            self._scaler_max = scaled_data.max(axis=None)
            train_data = torch.FloatTensor(scaled_data)
            self.clf = MyAutoEncoder(
                input_dim=input_dim,
            )
        else:
            train_data = torch.FloatTensor(df_pped)
            self.clf = MyAutoEncoderRelu(input_dim=input_dim)
        
        self.clf.to(self.device)

        criterion = nn.MSELoss().to(self.device)
        optimizer = torch.optim.Adam(self.clf.parameters(), lr=self.learning_rate)

        self.clf.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i : i + self.batch_size]
                batch = batch.to(self.device)

                optimizer.zero_grad()
                reconstructed = self.clf(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            logger.debug(
                f"Epoch {epoch}/{self.epochs}, Loss: {total_loss/len(train_data):.6f}"
            )
