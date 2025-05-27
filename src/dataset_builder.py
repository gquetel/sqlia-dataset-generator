import logging
import numpy as np
import os
import pandas as pd
import random
import secrets
import string
import re
import shutil

from sklearn.model_selection import train_test_split
from tqdm import tqdm
from .sqlmap_generator import sqlmapGenerator

from .sql_query_server import TemplatedSQLServer
from .db_cnt_manager import SQLConnector
from . import config_parser

logger = logging.getLogger(__name__)


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
        # Object attributes initialisation

        self.config = config
        self.seed = config_parser.get_seed(self.config)

        #  Dict holding all possible filler values, Keys are tuple of the form:
        #  (schema_name, dictionnary_name)
        self.dictionaries = {}

        # Dataset output path
        self.outpath = config_parser.get_output_path(config)

        # Connection wrapper to SQL server.
        self.sqlc = None

        # Dataframe holding the sampled templates of normal queries to fill.
        self._df_templates_n = None
        # Dataframe holding sampled templates introduced in the test set (not present
        # in the train set).
        self.df_templates_test = None
        # Dataframe holding the templates selected for being normal-only templates.
        self.df_tno = None
        # Dataframe holding the templates of administration queries
        self.df_tadmin = None

        # Initialisation code.
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.populate_dictionaries()

    def populate_dictionaries(self):
        """Load dictionaries of legitimate values for placeholders.

        The function iterates over all datasets, checks under
        data/databases/$dataset/dicts and load all existing file into
        self.dictionaries[(dataset_name, placeholder_id)]
        """
        used_databases = config_parser.get_used_databases(self.config)

        for db in used_databases:
            dicts_dir = "".join(["./data/databases/", db, "/dicts/"])
            for filename in os.listdir(dicts_dir):
                with open(dicts_dir + filename, "r") as f:
                    self.dictionaries[(db, filename)] = f.read().splitlines()

    def get_all_templates(self) -> pd.DataFrame:
        """Return all statements templates from generation settings."""
        used_databases = config_parser.get_used_databases(self.config)
        statements_type = config_parser.get_statement_types_and_proportions(self.config)
        _all_templates = pd.DataFrame()

        for db in used_databases:
            dir_path = "".join(["./data/databases/", db, "/queries/"])
            for stmt_type in statements_type:
                # Iterate over statements_type, load relevant csv file
                # And then add necessary fields.
                _t = pd.read_csv(dir_path + stmt_type["type"] + ".csv")

                _t["proportion"] = stmt_type["proportion"]
                _t["statement_type"] = stmt_type["type"]
                _all_templates = pd.concat([_t, _all_templates])

        _all_templates["placeholders"] = _all_templates["template"].apply(
            _extract_params
        )
        return _all_templates

    def select_templates(self, testing_mode: bool):
        """Modify self.templates according to the generation settings.

        - This function randomly samples templates that will only be present for testing.
        - This function randomly samples templates that will only be used to generate
        normal samples
        - If testing mode is enabled, this reduce the number of templates for generation.

        Args:
            testing_mode (bool): _description_
        """

        self.templates = self.get_all_templates()

        # First, remove administrator statements to not mess with ratios computations
        self.df_tadmin = self.templates[self.templates["ID"].str.contains("admin")]
        self.templates = self.templates[~self.templates["ID"].str.contains("admin")]

        # Testing settings, allows for quick iteration over templates.
        if testing_mode:
            n_templates = 10
            self.templates = self.templates.sample(n=n_templates)
            logger.warning(
                f"Testing mode enabled, using {n_templates} templates and error technique"
            )

        # Samples templates for normal only generation:
        ratio_tno = 0.1  # TODO: add this in config
        n_tno = round(self.templates.shape[0] * ratio_tno)
        self.df_tno = self.templates.sample(n=n_tno)
        self.templates = self.templates.drop(self.df_tno.index)

        # Sample templates for df_test
        # 20% of the templates will be kept for test split.
        ratio_tt = 0.2
        n_tt = round(self.templates.shape[0] * ratio_tt)
        self.df_templates_test = self.templates.sample(n=n_tt)

    def _add_split_column_exclude(self):
        """Add a split column with test set templates being disjoinct from train set.

        The split is made according to the already made list of test templates
        self.df_templates_test.
        """
        ids_ttest = self.df_templates_test["ID"].to_list()
        self.df.loc[self.df["query_template_id"].isin(ids_ttest), "split"] = "test"
        self.df.loc[~self.df["query_template_id"].isin(ids_ttest), "split"] = "train"

    def _add_split_column_include(self, train_size=0.7):
        """Add a split column with test set including templates of train set.

        The templates that should be considered for test are already sampled. Hence we
        only need to randomly sample 1 - train_size * len(df) to be considered as test
        set. Then we remove from the remainder all samples belonging to the train split.

        Doing so allows to keep a distribution of queries where test_set only queries
        are not more represented than the others (which would happen if we selected
        all template_test queries for test set and then add some queries from other
        templates to reach 1 - train_size * len(df))

        Args:
            train_size (float, optional): _description_. Defaults to 0.7.
        """
        ids_ttest = self.df_templates_test["ID"].to_list()
        _df_train, _df_test = train_test_split(self.df, train_size=train_size)
        _df_test["split"] = "test"
        _df_train = _df_train[~_df_train["query_template_id"].isin(ids_ttest)]
        _df_train["split"] = "train"

        self.df = pd.concat([_df_test, _df_train])

    def _augment_test_set_normal_queries(self):
        """We augment the number of normal queries in test set."""
        atk_ratio = config_parser.get_attacks_ratio(self.config)

        n_attack_test_set = self.df[
            (self.df["split"] == "test") & (self.df["label"] == 1)
        ].shape[0]
        target_n_normal_test = int(n_attack_test_set / atk_ratio)

        # Here, we upsample from: 
        # - all templates used to generate attacks (self.templates)
        # - Plus those sampled by  self.select_templates to be considered as normal
        #   only templates.
        # - Plus the administrative queries
        
        l_normal_templates = (
            list(self.templates["ID"].unique())
            + list(self.df_tno["ID"].unique())
            + list(self.df_tadmin["ID"].unique())
        )

        self.populate_normal_templates(
            n_n=target_n_normal_test, templates_list=l_normal_templates
        )
        self.generate_normal_queries()
        # Now all queries without a split value should go to test set
        self.df.loc[self.df["split"].isna(), "split"] = "test"

    def populate_normal_templates(self, n_n: int, templates_list: list):
        """Randomly sample n_n templates with given the templates_list array

        Args:
            n_n (int): _description_
            templates_list (list): _description_.
        """

        _df_all_templates = self.get_all_templates()
        # Only keep those which match templates_list:
        _dft = _df_all_templates[_df_all_templates["ID"].isin(templates_list)].copy()
        self._df_templates_n = _dft.sample(n=n_n, replace=True, weights="proportion")

    def generate_normal_queries(self):
        # Iterate over placeholders, and payload clause for type
        # Randomly choose a value in dict for that placeholder
        # And encapsulate based on type

        generated_normal_queries = []
        for template_row in tqdm(self._df_templates_n.itertuples()):
            all_placeholders = _extract_params(template=template_row.template)

            # template_row.payload_type is na when no input needs filling.
            if(not pd.isna(template_row.payload_type)):
                all_types = template_row.payload_type.split()
            else:
                all_types = []
            
            # print(all_types,all_placeholders)
            assert len(all_types) == len(all_placeholders)
            schema_name = template_row.ID.split("-")[0]

            # Replace 1 by 1 all placeholders by a randomly choosen dict value
            query = template_row.template
            for placeholder, type in zip(all_placeholders, all_types):
                # Remove placeholder's artificial int suffix:
                placeholder = placeholder.rstrip("123456789")

                if placeholder == "rand_pos_number":
                    filler = random.randint(0, 64000)
                elif placeholder == "rand_string":
                    alphabet = string.ascii_letters + string.digits
                    filler = "".join(secrets.choice(alphabet) for i in range(20))
                else:
                    filler = random.choice(
                        self.dictionaries[(schema_name, placeholder)]
                    )

                match type:
                    case "int" | "float":
                        query = query.replace(f"{{{placeholder}}}", str(filler), 1)
                    case "string":
                        # New templating method, the string should
                        # not be escaped. It is done in the template.

                        # However, we also use double quotes, we need to escape those in filler
                        filler = str(filler).replace('"', '""')

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
                    "attack_status": None,
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

    def generate_attack_queries_sqlmapapi(
        self, testing_mode: bool, debug_mode: bool
    ) -> dict:
        server_port = 8080
        generated_attack_queries = []

        # First, initialize all HTTP endpoints for each template.
        # templates are already selected / sampled in self.templates

        if self.sqlc == None:
            self.sqlc = SQLConnector(self.config)
        # Prune all sent_queries for attacks
        _ = self.sqlc.get_and_empty_sent_queries()

        server = TemplatedSQLServer(
            templates=self.templates, sqlconnector=self.sqlc, port=server_port
        )
        server.start_server()

        # Now iterate over templates and techniques to generate payloads.
        sqlg = sqlmapGenerator(
            config=self.config,
            templates=self.templates,
            sqlconnector=self.sqlc,
            placeholders_dictionaries_list=self.dictionaries,
            port=server_port,
        )
        generated_attack_queries = sqlg.generate_attacks(testing_mode, debug_mode)
        server.stop_server()

        self._n_attacks = len(generated_attack_queries)
        self.df = generated_attack_queries

    def _clean_cache_folder(self):
        shutil.rmtree("./cache/", ignore_errors=True)

    def build(self, testing_mode: bool, debug_mode: bool):
        train_size = 0.7

        # First, sample queries templates according to scenario.
        self.select_templates(testing_mode=testing_mode)
        self.generate_attack_queries_sqlmapapi(
            testing_mode=testing_mode, debug_mode=debug_mode
        )

        # List of templates to create normal queries from. Corresponds to :
        # - all templates used to generate attacks (self.templates)
        # - Plus those sampled by  self.select_templates to be considered as normal
        #   only templates.
        # - Plus the administrative queries
        l_normal_templates = (
            list(self.templates["ID"].unique())
            + list(self.df_tno["ID"].unique())
            + list(self.df_tadmin["ID"].unique())
        )
        self.populate_normal_templates(self._n_attacks, l_normal_templates)
        self.generate_normal_queries()

        self._add_split_column_include(train_size=train_size)
        self._augment_test_set_normal_queries()

    def save(self):
        self.df.to_csv(self.outpath, index=False)
        # self._clean_cache_folder()
