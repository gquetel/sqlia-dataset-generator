import numpy as np
import os
import pandas as pd
import random
import re
import shutil

from tqdm import tqdm
from .payload_generators.sqlmap_generator import sqlmapGenerator

from .endpoints_controller import ServerManager
from .sql_connector import SQLConnector

import src.config_parser as config_parser


def _extract_params(template):
    param_names = re.findall(r"\{([-a-zA-Z_]+)\}", template)
    param_counts = {}
    res = []

    # We artificially suffix parameters with the same name to force
    # the selection of different values.
    for param in param_names:
        if param in param_counts:
            param_counts[param] += 1
            sx_param = f"{param}{param_counts[param]}"
            res.append(sx_param)
        else:
            param_counts[param] = 1
            res.append(param)
    return res


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
        statements_type = config_parser.get_statement_types_and_proportions(self.config)
        _all_templates = pd.DataFrame()

        for db in used_databases:
            dir_path = "".join(["./data/databases/", db, "/queries/"])
            for stmt_type in statements_type:
                # Iterate  statements_type,
                # Load relevant csv file, and sample templates.
                type = stmt_type["type"]
                _t = pd.read_csv(dir_path + type + ".csv")
                _t["statement_type"] = type
                _all_templates = pd.concat([_t, _all_templates])

        _all_templates["placeholders"] = _all_templates["template"].apply(
            _extract_params
        )
        return _all_templates

    def populate_normal_templates(self, n_n: int):
        used_databases = config_parser.get_used_databases(self.config)
        statements_type = config_parser.get_statement_types_and_proportions(self.config)

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
                _dft["statement_type"] = type
                self._df_templates_n = pd.concat([self._df_templates_n, _dft])

    def generate_normal_queries(self):
        # Iterate over placeholders, and payload clause for type
        # Randomly choose a value in dict for that placeholder
        # And encapsulate based on type

        # Generate the same number of normal queries that the number of attacks.
        self.populate_normal_templates(self._n_attacks)
        generated_normal_queries = []
        for template_row in tqdm(self._df_templates_n.itertuples()):
            all_placeholders = _extract_params(template=template_row.template)
            all_types = template_row.payload_type.split()

            # print(all_placeholders,all_types)
            assert len(all_types) == len(all_placeholders)
            db_name = template_row.ID.split("-")[0]

            # Replace 1 by 1 all placeholders by a randomly choosen dict value
            query = template_row.template

            for placeholder, type in zip(all_placeholders, all_types):
                # Remove placeholder's artificial int suffix:
                placeholder = placeholder.rstrip("123456789")
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
                        query = query.replace(f"{{{placeholder}}}", str(filler), 1)
                    case "string":
                        # New templating method, the string should
                        # not be escaped. It is done in the template.

                        # However, we also use double quotes, we need to escape those in filler
                        filler = filler.replace('"', '""')

                        query = query.replace(f"{{{placeholder}}}", f"{filler}", 1)
                    case _:
                        raise ValueError(f"Unknown payload type: {type}.")
            # Append query, template ID and label to dataset.

            if not self._verify_syntactic_validity_query(query=query):
                raise ValueError("Failed normal query: ", query)

            generated_normal_queries.append(
                {
                    "full_query": query,
                    "label": 0,
                    "statement_type": template_row.statement_type,
                    "query_template_id": template_row.ID,
                    "attack_payload": None,
                    "attack_id": None,
                    "attack_technique": None,
                    "attack_desc": None,
                    "sqlmap_status": None,
                }
            )
        self.df = pd.concat(
            [self.df, pd.DataFrame(generated_normal_queries)],
            ignore_index=True,
        )

    def _verify_syntactic_validity_query(self, query: str):
        if self.sqlc == None:
            self.sqlc = SQLConnector(self.config)
        res = self.sqlc.is_query_syntvalid(query=query)
        return res

    def generate_attack_queries_sqlmapapi(self, testing_mode : bool) -> dict:
        server_port = 8080
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
        generated_attack_queries = sqlg.generate_attacks(testing_mode)
        # input()
        server.stop_server()

        self._n_attacks = len(generated_attack_queries)
        self.df = generated_attack_queries

    def _add_split_column_using_statement_type(self, train_size=0.7):
        # Then sample
        # and set their split to train
        self.df["split"] = "train"
        statements_type = config_parser.get_statement_types_and_proportions(self.config)

        # Placing this here seems weird as we already got such a call earlier
        # However, without it, the sampled template changes from one
        # invocation to the other...
        random.seed(self.seed)

        # Iterate over statement types and sample
        # train_size * len(template_statement_type) templates
        for stmt_type in statements_type:
            type = stmt_type["type"]
            _df_type = self.df[self.df["statement_type"] == type]
            templates_ids = _df_type["query_template_id"].unique()
            n_ids_test = int((1 - train_size) * len(templates_ids))
            ids_test = random.sample(templates_ids.tolist(), k=n_ids_test)
            print(ids_test)
            self.df.loc[self.df["query_template_id"].isin(ids_test), "split"] = "test"

    def _clean_cache_folder(self):
        shutil.rmtree("./cache/", ignore_errors=True)

    def build(
        self,
        testing_mode : bool
    ) -> pd.DataFrame:
        train_size = 0.7
        self.generate_attack_queries_sqlmapapi(testing_mode=testing_mode)
        self.generate_normal_queries()
        self._add_split_column_using_statement_type(train_size=train_size)

    def save(self):
        self.df.to_csv(self.outpath, index=False)
        self._clean_cache_folder()
