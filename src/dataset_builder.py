from pathlib import Path
import shutil
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import random
import re

from .payload_generators.sqlmap_generator import sqlmapGenerator

from .endpoints_controller import ServerManager
from .sql_connector import SQLConnector

import src.config_parser as config_parser


def _extract_params(template):
    param_names = re.findall(r"\{([a-zA-Z_]+)\}", template)
    return param_names


class DatasetBuilder:
    def __init__(self, config) -> None:
        self.config = config
        self.seed = config_parser.get_seed(self.config)
        #  Dict holding all possible filler values
        # Keys are tuple of the form (db_name, dictionnary_name)
        self.dictionnaries = {}

        # Array of attack queries not correctly constructed
        self._failed_attacks = []

        # Connection wrapper to SQL server.
        self.sqlc = None

        random.seed(self.seed)
        np.random.seed(self.seed)

        # Dataset output path
        self.outpath = config_parser.get_output_path(config)
        self.populate_dictionnaries()

    def populate_dictionnaries(self):
        # Iterate over all datasets, check under data/databases/$dataset/dicts
        # And load all existing file into self.dictionnaries[(family,dict)]
        used_databases = config_parser.get_used_databases(self.config)

        for db in used_databases:
            dicts_dir = "".join(["./data/databases/", db, "/dicts/"])
            for filename in os.listdir(dicts_dir):
                with open(dicts_dir + filename, "r") as f:
                    self.dictionnaries[(db, filename)] = f.read().splitlines()

    def get_all_templates(self):
        """Return all statements templates."""
        used_databases = config_parser.get_used_databases(self.config)
        statements_type = config_parser.get_statement_types_and_proportions(
            self.config
        )
        _all_templates = pd.DataFrame()

        for db in used_databases:
            dir_path = "".join(["./data/databases/", db, "/queries/"])
            for stmt_type in statements_type:
                # Iterate  statements_type,
                # Load relevant csv file, and sample templates.
                type = stmt_type["type"]
                _t = pd.read_csv(dir_path + type + ".csv")
                _all_templates = pd.concat([_t, _all_templates])

        _all_templates["placeholders"] = _all_templates["template"].apply(
            _extract_params
        )
        return _all_templates

    def populate_normal_templates(self, n_n: int):
        used_databases = config_parser.get_used_databases(self.config)
        statements_type = config_parser.get_statement_types_and_proportions(
            self.config
        )

        n_n_per_db = int(n_n / len(used_databases))
        self._df_templates_n = pd.DataFrame()
        for db in used_databases:
            dir_path = "".join(["./data/databases/", db, "/queries/"])
            for stmt_type in statements_type:
                # Iterate  statements_type,
                # Load relevant csv file, and sample templates.
                type = stmt_type["type"]
                proportion = stmt_type["proportion"]

                df_templates = pd.read_csv(dir_path + type + ".csv")

                # Sample normal queries
                _dft = df_templates.sample(
                    n=int(proportion * n_n_per_db),
                    replace=True,
                )
                self._df_templates_n = pd.concat([self._df_templates_n, _dft])

    def generate_normal_queries(self):
        # Iterate over placeholders, and payload clause for type
        # Randomly choose a value in dict for that placeholder
        # And encapsulate based on type

        # Generate the same number of normal queries that the number of attacks.
        self.populate_normal_templates(self._n_attacks)

        generated_normal_queries = []
        for template_row in tqdm(self._df_templates_n.itertuples()):
            placeholders_pattern = r"\{([^}]*)\}"
            all_placeholders = [
                m.group(1)
                for m in re.finditer(
                    placeholders_pattern, template_row.template
                )
            ]
            all_types = template_row.payload_type.split()

            assert len(all_types) == len(all_placeholders)
            db_name = template_row.ID.split("-")[0]

            # Replace 1 by 1 all placeholders by a randomly choosen dict value
            query = template_row.template
            for placeholder, type in zip(all_placeholders, all_types):
                if placeholder[0] == "!":
                    # Choose value
                    filler = random.choice(
                        self.dictionnaries[(db_name, placeholder[1:])]
                    )
                else:
                    # Some edge cases:
                    if placeholder == "pos_number":
                        filler = random.randint(0, 64000)
                    else:
                        filler = random.choice(
                            self.dictionnaries[(db_name, placeholder)]
                        )
                match type:
                    case "int" | "float":
                        query = query.replace(
                            f"{{{placeholder}}}", str(filler)
                        )
                    case "string":
                        # New templating method, the string should
                        # not be escaped. It is done in the template.

                        # However, we also use double quotes, we need to escape those in filler
                        filler = filler.replace('"', '""')
                        query = query.replace(
                            f"{{{placeholder}}}",
                            f"{filler}",
                        )
                    case _:
                        raise ValueError(f"Unknown payload type: {type}.")
            # Append query, tempalte ID and label to dataset.

            if not self._verify_syntactic_validity_query(query=query):
                raise ValueError("Failed normal query: ", query)

            generated_normal_queries.append(
                {
                    "full_query": query,
                    "label": 0,
                    "query_template_id": template_row.ID,
                    "attack_payload": None,
                    "attack_id": None,
                    "attack_technique": None,
                    "attack_desc": None,
                }
            )
        self.df = pd.concat([self.df, pd.DataFrame(generated_normal_queries)])

    def _verify_syntactic_validity_query(self, query: str):
        if self.sqlc == None:
            self.sqlc = SQLConnector(self.config)
        res = self.sqlc.is_query_syntvalid(query=query)
        return res

    def generate_attack_queries_sqlmapapi(self) -> dict:
        server_port = 8081
        generated_attack_queries = []

        # First, initialize all HTTP endpoints for each template.
        templates = self.get_all_templates()
        if self.sqlc == None:
            self.sqlc = SQLConnector(self.config)
        # Prune all sent_queries for attacks
        _ = self.sqlc.get_and_empty_sent_queries()

        server = ServerManager(
            templates=templates, sqlconnector=self.sqlc, port=server_port
        )
        server.start_server()

        # Now iterate over templates and techniques to generate payloads.
        sqlg = sqlmapGenerator(
            config=self.config,
            templates=templates,
            sqlconnector=self.sqlc,
            placeholders_dictionnaries_list=self.dictionnaries,
            port=server_port,
        )
        generated_attack_queries = sqlg.generate_attacks()
        # input()
        server.stop_server()

        self._n_attacks = len(generated_attack_queries)
        self.df = generated_attack_queries

    def _add_split_column(self, train_size=0.7):
        """
        Add a 'split' column to dataframe with 'train' or 'test' values.
        For attack samples, split is done at the attack_id level.
        For normal samples, random split based on train_size.

        Args:
            train_size: Fraction of data to use for training (default 0.7)
        """

        attack_samples = self.df[self.df["label"] == 1]
        unique_attack_ids = attack_samples["attack_id"].unique()
        if len(unique_attack_ids) <= 1:
            print(
                "_add_split_column: invalid number of columns, cannot split correctly."
            )
            self.df["split"] = ""
            return

        self.df["split"] = "test"

        test_size = (1 - train_size) / 100
        # sample num_test_attack_ids of scenarios
        num_test_attack_ids = max(1, int(len(unique_attack_ids) * test_size))

        test_attack_ids = np.random.choice(
            unique_attack_ids, size=num_test_attack_ids, replace=False
        )

        # Assign 'train' for attack samples that aren't in test_attack_ids
        train_mask = (self.df["label"] == 1) & (
            ~self.df["attack_id"].isin(test_attack_ids)
        )
        self.df.loc[train_mask, "split"] = "train"

        normal_indices = self.df[self.df["label"] == 0].index.tolist()

        if normal_indices:
            num_train = int(len(normal_indices) * train_size)

            train_normal_indices = np.random.choice(
                normal_indices, size=num_train, replace=False
            )

            for idx in train_normal_indices:
                self.df.at[idx, "split"] = "train"

    def _clean_cache_folder(self):
        shutil.rmtree("./cache/", ignore_errors=True)

    def build(
        self,
    ) -> pd.DataFrame:
        train_size = 0.7
        self.generate_attack_queries_sqlmapapi()
        self.generate_normal_queries()
        self._add_split_column(train_size=train_size)

    def save(self):
        self.df.to_csv(self.outpath, index=False)
        self._clean_cache_folder()
