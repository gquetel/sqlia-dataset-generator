import pandas as pd
import os
import random

from .payload_generator import PayloadDistributionManager
from .queries_generator import (
    FILEPATHS,
    load_dictionnaries,
    load_payloads,
    load_query_templates,
    pick_from_dict,
    construct_normal_query,
)
import src.config_parser as config_parser


class DatasetBuilder:
    def __init__(self, config) -> None:
        self.seed = config.get("RANDOM", "seed")
        random.seed(self.seed)

        # self.templates_config = config_parser.get_statement_types_and_proportions(
        #     config
        # )
        # self.used_databases = config_parser.get_used_databases(config)
        # self.n_normal_queries, self.n_attack_queries = (
        #     config_parser.get_queries_numbers(config)
        # )
        # self.outpath = config_parser.get_output_path(config)
        payloads_to_use = config_parser.get_payload_types_and_proportions(
            config
        )
        n_n, n_a, n_u = config_parser.get_queries_numbers(config)
        pdm = PayloadDistributionManager(payloads_to_use, n_attack_queries=n_a)
        payload, payload_id = pdm.generate_payload()
        print(payload,payload_id)
        exit(1)
        
        # self.payloads = load_payloads(payloads_to_use)
        # self.verify_paths()

    def verify_paths(self):
        for database in self.used_databases:
            db_fp = FILEPATHS["databasesdir"] + database
            if not os.path.exists(db_fp):
                raise ValueError(
                    f"Specified database folder '{database}' must be found in '{FILEPATHS['databasesdir']}'."
                )

            for template in self.templates_config:
                if template["proportion"] > 0:  # Skip check if unused
                    continue

                query_fp = db_fp + "/queries/" + template["type"]
                if not os.path.exists(query_fp):
                    raise ValueError(
                        f"Template file {template['type']} not found in {db_fp}/queries/."
                    )

    def construct_attack_query(
        self,
        l_available_templates: list[str],
        d_dictionnaries: dict,
    ) -> str:
        picked_template = random.choice(l_available_templates)
        picked_template = self.generator.generate_payload(picked_template)
        for dictionnary in d_dictionnaries:
            picked_template = picked_template.replace(
                f"{{{dictionnary}}}",
                pick_from_dict(d_dictionnaries, dictionnary),
            )
            picked_template = picked_template.replace(
                f"{{!{dictionnary}}}",
                pick_from_dict(d_dictionnaries, dictionnary),
            )

        return picked_template

    def init_generator(self, payload: dict, d_dictionnaries: dict, database):
        if payload["family"] == "sqlmap":
            self.generator = sqlmapGenerator(
                payload["templates"], d_dictionnaries, database
            )
        else:
            raise ValueError(
                f"Payload family '{payload['family']}' is not supported."
            )

    def build(
        self,
    ) -> pd.DataFrame:
        # Compute the number of queries to generate for each database
        n_queries_per_db = self.n_normal_queries // len(self.used_databases)
        n_queries_per_db_attack = self.n_attack_queries // len(
            self.used_databases
        )

        l_n_queries = []
        l_a_queries = []

        for database in self.used_databases:
            d_dictionnaries = load_dictionnaries(database)

            for statement in self.templates_config:
                l_available_templates = load_query_templates(
                    database, statement["type"]
                )

                # Generate normal queries
                n_queries_from_type = int(
                    n_queries_per_db * statement["proportion"]
                )
                while n_queries_from_type > 0:
                    query = construct_normal_query(
                        l_available_templates, d_dictionnaries
                    )
                    l_n_queries.append(query)
                    n_queries_from_type -= 1

                # Generate attack queries
                n_a_queries_from_type = int(
                    n_queries_per_db_attack * statement["proportion"]
                )
                print(
                    "Generating",
                    n_a_queries_from_type,
                    "queries of type",
                    statement["type"],
                )

                for payload in self.payloads:
                    n_queries_from_payload = int(
                        n_a_queries_from_type * payload["proportion"]
                    )
                    self.init_generator(payload, d_dictionnaries, database)

                    while n_queries_from_payload > 0:
                        query = self.construct_attack_query(
                            l_available_templates, d_dictionnaries
                        )
                        l_a_queries.append(query)
                        n_queries_from_payload -= 1

        self.df = pd.concat(
            [
                pd.DataFrame(
                    {"query": l_n_queries, "label": 0}
                ),  # Normal queries
                pd.DataFrame(
                    {"query": l_a_queries, "label": 1}
                ),  # Attack queries
            ]
        )
        return self.df

    def save(self):
        self.df.to_csv(self.outpath, index=False)
