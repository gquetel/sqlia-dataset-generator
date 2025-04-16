import os
import pandas as pd
import random
import re

from .payload_generator import PayloadDistributionManager
import src.config_parser as config_parser


class DatasetBuilder:
    def __init__(self, config) -> None:
        self.config = config
        self.seed = int(self.config.get("RANDOM", "seed"))
        random.seed(self.seed)

        self.outpath = config_parser.get_output_path(config)
        payloads_type = config_parser.get_payload_types_and_proportions(self.config)
        statements_type = config_parser.get_statement_types_and_proportions(self.config)
        n_n, n_a, n_u = config_parser.get_queries_numbers(config)

        # Initialize Payload generation component
        self.pdm = PayloadDistributionManager(payloads_type, n_attack_queries=n_a)

        # Randomly select templates given the config file distribution
        self._df_templates_a = pd.DataFrame()
        self._df_templates_n = pd.DataFrame()
        self._df_templates_u = pd.DataFrame()

        self.populate_templates(
            n_n=n_n, n_a=n_a, n_u=n_u, statements_type=statements_type
        )

        #  Dict holding all possible filler values
        # Keys are tuple of the form (db_name, dictionnary_name)
        self.dictionnaries = {}
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

    def populate_templates(self, n_n: int, n_a: int, n_u: int, statements_type: dict):
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
                    random_state=self.seed,
                )
                self._df_templates_n = pd.concat([self._df_templates_n, _dft])

                # Sample attack queries
                _dft = df_templates.sample(
                    n=int(proportion * n_a_per_db),
                    replace=True,
                    random_state=self.seed,
                )
                self._df_templates_a = pd.concat([self._df_templates_a, _dft])

                # Sample undefined queries
                _dft = df_templates.sample(
                    n=int(proportion * n_u_per_db),
                    replace=True,
                    random_state=self.seed,
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
                for m in re.finditer(placeholders_pattern, template_row.full_query)
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
                    if placeholder == "number":
                        # TODO, which values to set here ?
                        filler = random.randint(-64000, 64000)
                    else:
                        filler = random.choice(
                            self.dictionnaries[(db_name, placeholder)]
                        )
                match type:
                    case "int":
                        query = query.replace(f"{{{placeholder}}}", str(filler))
                    case "string":
                        # String should be escaped
                        escape_char = random.choice(['"', "'"])
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
        self.df = pd.DataFrame(generated_normal_queries)

    def generate_attack_queries(self):
        generated_attack_queries = []
        for template_row in self._df_templates_a.itertuples():
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
                    # Use PayloadDistributionManager to generate payload
                    payload, id = self.pdm.generate_payload(template_row.payload_clause)
                    filler = payload
                else:
                    # Some edge cases:
                    if placeholder == "number":
                        # TODO, which values to set here ?
                        filler = random.randint(-64000, 64000)
                    else:
                        filler = random.choice(
                            self.dictionnaries[(db_name, placeholder)]
                        )
                match type:
                    case "int":
                        query = query.replace(f"{{{placeholder}}}", str(filler))
                    case "string":
                        # String should be escaped
                        escape_char = random.choice(['"', "'"])
                        query = query.replace(
                            f"{{{placeholder}}}",
                            f"{escape_char}{filler}{escape_char}",
                        )
                    case _:
                        raise ValueError(f"Unknown payload type: {type}.")
            generated_attack_queries.append(
                {
                    "full_query": query,
                    "label": 0,
                    "template_id": template_row.ID,
                    "malicious_input": payload,
                    "malicious_input_id": id,
                }
            )
        self.df = pd.concat([self.df, pd.DataFrame(generated_attack_queries)])

    def generate_undefined_queries(self):
        pass

    def build(
        self,
    ) -> pd.DataFrame:
        self.generate_normal_queries()
        self.generate_attack_queries()
        print(self.pdm.get_final_stats())
        self.generate_undefined_queries()
        

    def save(self):
        self.df.to_csv(self.outpath, index=False)
