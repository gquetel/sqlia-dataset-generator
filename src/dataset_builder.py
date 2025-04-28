import os
import pandas as pd
import numpy as np
import random
import re

from .sql_connector import SQLConnector

from .payload_generator import PayloadDistributionManager
import src.config_parser as config_parser


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

        # Payload types specified to be generated
        payloads_type = config_parser.get_payload_types_and_proportions(
            self.config
        )

        # Statements types specified to be generated
        statements_type = config_parser.get_statement_types_and_proportions(
            self.config
        )

        # Proportions of normal, attacks, undefined queries.
        n_n, n_a, n_u = config_parser.get_queries_numbers(config)

        # Initialize Payload generation component
        self.pdm = PayloadDistributionManager(
            payloads_type, n_attack_queries=n_a, config=config
        )

        # Randomly select templates given the config file distribution
        self._df_templates_a = pd.DataFrame()
        self._df_templates_n = pd.DataFrame()
        self._df_templates_u = pd.DataFrame()

        # Select all query templates.
        self.populate_templates(
            n_n=n_n, n_a=n_a, n_u=n_u, statements_type=statements_type
        )

        # Load all dictionnaries in memory to prevent many file read.
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

    def populate_templates(
        self, n_n: int, n_a: int, n_u: int, statements_type: dict
    ):
        used_databases = config_parser.get_used_databases(self.config)

        n_n_per_db = int(n_n / len(used_databases))
        n_a_per_db = int(n_a / len(used_databases))
        n_u_per_db = int(n_u / len(used_databases))

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

                # Sample attack queries
                _dft = df_templates.sample(
                    n=int(proportion * n_a_per_db),
                    replace=True,
                )
                self._df_templates_a = pd.concat([self._df_templates_a, _dft])
                # Sample undefined queries
                _dft = df_templates.sample(
                    n=int(proportion * n_u_per_db),
                    replace=True,
                )
                self._df_templates_u = pd.concat([self._df_templates_u, _dft])

    def generate_normal_queries(self):
        # Iterate over placeholders, and payload clause for type
        # Randomly choose a value in dict for that placeholder
        # And encapsulate based on type
        generated_normal_queries = []
        for template_row in self._df_templates_n.itertuples():
            placeholders_pattern = r"\{([!]?[^}]*)\}"
            all_placeholders = [
                m.group(1)
                for m in re.finditer(
                    placeholders_pattern, template_row.full_query
                )
            ]
            all_types = template_row.payload_type.split()
            assert len(all_types) == len(all_placeholders)
            db_name = template_row.ID.split("-")[0]

            # Replace 1 by 1 all placeholders by a randomly choosen dict value
            query = template_row.full_query
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
                        # String should be escaped
                        escape_char = random.choice(['"', "'"])
                        # escape quoting char in string
                        filler = filler.replace(
                            escape_char, f"{escape_char}{escape_char}"
                        )
                        query = query.replace(
                            f"{{{placeholder}}}",
                            f"{escape_char}{filler}{escape_char}",
                        )
                    case _:
                        raise ValueError(f"Unknown payload type: {type}.")
            # Append query, tempalte ID and label to dataset.
            generated_normal_queries.append(
                {
                    "full_query": query,
                    "label": 0,
                    "template_id": template_row.ID,
                    "malicious_input": None,
                }
            )

        assert self._is_all_queries_syntactically_valid(
            queries=generated_normal_queries
        )
        self.df = pd.DataFrame(generated_normal_queries)

    def _verify_syntactic_validity_query(self, query: str):
        if self.sqlc == None:
            self.sqlc = SQLConnector(self.config)
        return self.sqlc.is_query_syntvalid(query=query)

    def _is_all_queries_syntactically_valid(self, queries: list):
        self.sqlc = SQLConnector(self.config)
        for d in queries:
            query = d["full_query"]
            if not self.sqlc.is_query_syntvalid(query=query):
                print("Invalid query: ", query)
                return False
        return True

    def _get_query_with_payload(self, template_row):
        placeholders_pattern = r"\{([!]?[^}]*)\}"
        all_placeholders = [
            m.group(1)
            for m in re.finditer(placeholders_pattern, template_row.full_query)
        ]
        all_types = template_row.payload_type.split()
        assert len(all_types) == len(all_placeholders)
        db_name = template_row.ID.split("-")[0]

        # Replace 1 by 1 all placeholders by a randomly choosen dict value
        query = template_row.full_query
        for placeholder, type in zip(all_placeholders, all_types):
            if placeholder[0] == "!":
                # First, generate original value
                # It is used to know how to integrate payload in query.
                expected_value = placeholder[1::]
                if expected_value == "pos_number":
                    original_value = random.randint(0, 64000)
                else:
                    original_value = random.choice(
                        self.dictionnaries[(db_name, expected_value)]
                    )

                    # If type == int, then cast, otherwise let as string.
                    if type == "int":
                        original_value = int(original_value)
                    elif type == "float":
                        original_value = float(original_value)

                # Now use PayloadDistributionManager to generate payload
                payload, desc = self.pdm.generate_payload(
                    original_value, template_row.payload_clause
                )
                # Then directly integrate payload.
                query = query.replace(f"{{{placeholder}}}", payload)

            else:
                if placeholder == "pos_number":
                    filler = random.randint(0, 64000)
                else:
                    filler = random.choice(
                        self.dictionnaries[(db_name, placeholder)]
                    )
                # Then, integrate filler:
                match type:
                    case "int" | "float":
                        query = query.replace(
                            f"{{{placeholder}}}", str(filler)
                        )
                    case "string":
                        escape_char = random.choice(['"', "'"])
                        filler = filler.replace(
                            escape_char, f"{escape_char}{escape_char}"
                        )
                        query = query.replace(
                            f"{{{placeholder}}}",
                            f"{escape_char}{filler}{escape_char}",
                        )
                    case _:
                        raise ValueError(f"Unknown payload type: {type}.")
        return {
            "full_query": query,
            "label": 1,
            "template_id": template_row.ID,
            "malicious_input": payload,
            "malicious_input_desc": desc,
        }

    def generate_attack_queries(self) -> dict:
        generated_attack_queries = []
        for template_row in self._df_templates_a.itertuples():
            # Generate query:
            attempt_query = self._get_query_with_payload(
                template_row=template_row
            )

            # Here, verify that query is syntactically valid. If not, add it to undefined ones. Then try to regenerate one from same template 10 times, then give up
            remaining_attempts = 10
            is_valid = self._verify_syntactic_validity_query(
                query=attempt_query["full_query"]
            )

            while remaining_attempts >= 0 and not is_valid:
                self._failed_attacks.append(attempt_query)
                remaining_attempts -= 1
                attempt_query = self._get_query_with_payload(
                    template_row=template_row
                )
                is_valid = self._verify_syntactic_validity_query(
                    query=attempt_query["full_query"]
                )
            if is_valid:
                generated_attack_queries.append(attempt_query)
            else:
                self._failed_attacks.append(attempt_query)
                print(
                    "Could not generate attack query for template:",
                    template_row.full_query,
                )
        self.df = pd.concat([self.df, pd.DataFrame(generated_attack_queries)])

    def generate_undefined_queries(self):
        print(self._failed_attacks)
        # TODO, penser à des scénarios.

        
    def build(
        self,
    ) -> pd.DataFrame:
        self.generate_normal_queries()
        self.generate_attack_queries()
        self.generate_undefined_queries()

    def save(self):
        self.df.to_csv(self.outpath, index=False)
