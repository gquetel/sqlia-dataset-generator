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

    def change_split(self, args):
        testing_mode = args.testing
        do_syn_check = not args.no_syn_check
        train_size = 0.7

        # Load dataset
        _df = pd.read_csv(self.outpath)

        # Only keep attacks
        self.df = _df[_df["label"] == 1]
        self._n_attacks = len(self.df)

        # Init templates
        self.select_templates(testing_mode=testing_mode)

        l_normal_templates = (
            list(self.templates["ID"].unique())
            + list(self.df_tno["ID"].unique())
            + list(self.df_tadmin["ID"].unique())
        )
        self.populate_normal_templates(self._n_attacks, l_normal_templates)
        self.generate_normal_queries(do_syn_check)

        self._add_split_column_include(train_size=train_size)
        self._augment_test_set_normal_queries(do_syn_check)

        self._add_template_split_info()

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
        as23_template = self.templates[self.templates["ID"] == "airport-S23"]

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

        # Also, a special case for template airport-S23,
        # for which we do not generate attacks either.
        self.df_tno = pd.concat([self.df_tno, as23_template])
        self.templates = self.templates.drop(self.df_tno.index)

        # Sample templates for df_test: DEPRECATED
        # 0% of the templates will be kept for test split.
        ratio_tt = 0.0
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

    def _add_split_column(self):
        """Add a split column information for unsupervised dataset generation

        At the time this function is called. self.df contains the same number of attacks
        as normal samples. Normal samples will be used for training. Attack samples for
        testing.

        Args:
            train_size (float, optional): _description_. Defaults to 0.7.
        """
        atk_mask = self.df["label"] == 1
        self.df.loc[atk_mask, "split"] = "test"
        self.df.loc[~atk_mask, "split"] = "train"

    def _augment_test_set_normal_queries(self, do_syn_check: bool):
        """We augment the number of normal queries in test set."""
        atk_ratio = config_parser.get_attacks_ratio(self.config)

        n_attack_test_set = self.df[
            (self.df["split"] == "test") & (self.df["label"] == 1)
        ].shape[0]

        target_n_normal_test = int(n_attack_test_set / atk_ratio) - n_attack_test_set

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
        self.generate_normal_queries(do_syn_check)
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

        # We should not sample given "proportion" directly, we should normalize based on
        # the number of templates for that statement type (given by column statement_type)

        type_counts = _dft.groupby("statement_type").size()
        _dft["normalized_weight"] = _dft.apply(
            lambda row: row["proportion"] / type_counts[row["statement_type"]], axis=1
        )
        self._df_templates_n = _dft.sample(
            n=n_n, replace=True, weights="normalized_weight"
        )

    def fill_placeholder(
        self, query: str, placeholder: str, schema_name: str, count: int = 1
    ) -> tuple[str, str]:
        if placeholder == "rand_pos_number":
            filler = random.randint(0, 64000)
        elif placeholder == "rand_medium_pos_number":
            filler = random.randint(1000, 6400)
        elif placeholder == "rand_small_pos_number":
            filler = random.randint(2, 5)
        elif placeholder == "rand_string":
            alphabet = string.ascii_letters + string.digits
            filler = "".join(secrets.choice(alphabet) for i in range(20))
        else:
            filler = random.choice(self.dictionaries[(schema_name, placeholder)])
        filler = str(filler).replace('"', '""')
        return (query.replace(f"{{{placeholder}}}", f"{filler}", 1), filler)

    def fill_condition_randomly(
        self, query: str, template_info: dict
    ) -> tuple[str, list]:
        """Randomly choose conditions to insert in query.

        This functions mimic the behavior of a page with multiple possible
        search conditions, that might not all be used.

        For now, templates allows this are: ['airport-S23']
        Args:
            template_info (dict): _description_
        Returns:
            tuple[str, list]: The string with all random conditions,
                and the syntetics user inputs
        """
        possible_fields = [
            "type",
            "name",
            "latitude",
            "longitude",
            "elevation_feet",
            "scheduled_service",
            "geo",  # This covers continent, iso_country, iso_region, municipality
        ]
        n_conds = random.randint(2, 7)
        choosen_conds = random.sample(possible_fields, n_conds)

        conditions = []
        user_inputs = []

        for field in choosen_conds:
            condition = None
            if field == "type":
                type_patterns = [
                    'type = "{airports_type}"',
                    'type LIKE "{airports_type}"',
                    'type IN ("{airports_type}","{airports_type}")',
                    'type IN ("{airports_type}","{airports_type}","{airports_type}")',
                ]
                condition = random.choice(type_patterns)

            elif field == "name":
                name_patterns = [
                    'name LIKE "%{rand_string}%"',
                    'name LIKE "%{rand_string}%" OR name LIKE "%{rand_string}%"',
                ]
                condition = random.choice(name_patterns)

            elif field == "latitude":
                lat_patterns = [
                    "latitude >= {airports_latitude_deg}",
                    "latitude <= {airports_latitude_deg}",
                    "latitude BETWEEN {airports_latitude_deg} AND {airports_latitude_deg}",
                ]
                condition = random.choice(lat_patterns)

            elif field == "longitude":
                lng_patterns = [
                    "longitude >= {airports_longitude_deg}",
                    "longitude <= {airports_longitude_deg}",
                    "longitude BETWEEN {airports_longitude_deg} AND {airports_longitude_deg}",
                ]
                condition = random.choice(lng_patterns)

            elif field == "elevation_feet":
                elev_patterns = [
                    "elevation_feet >= {airports_elevation_ft}",
                    "elevation_feet <= {airports_elevation_ft}",
                    "elevation_feet BETWEEN {airports_elevation_ft} AND {airports_elevation_ft}",
                ]
                condition = random.choice(elev_patterns)

            elif field == "scheduled_service":
                service_patterns = [
                    'scheduled_service = "{airports_scheduled_service}"',
                    'scheduled_service LIKE "{airports_scheduled_service}"',
                ]
                condition = random.choice(service_patterns)

            elif field == "geo":
                geo_fields = [
                    'continent LIKE "{airports_continent}"',
                    'continent = "{airports_continent}"',
                    'iso_country = "{airports_iso_country}"',
                    'iso_country LIKE "{airports_iso_country}"',
                    'iso_region = "{airports_iso_region}"',
                    'iso_region LIKE "{airports_iso_region}"',
                    'municipality = "{airports_municipality}"',
                    'municipality LIKE "{airports_municipality}"',
                    '( FIND_IN_SET("{rand_string}", keywords_field) > 0 OR FIND_IN_SET("{rand_string}", keywords_field) > 0 OR FIND_IN_SET("{rand_string}", keywords_field) > 0)',
                    '( FIND_IN_SET("{rand_string}", keywords_field) > 0 OR FIND_IN_SET("{rand_string}", keywords_field) > 0)',
                    '( FIND_IN_SET("{rand_string}", keywords_field) > 0)',
                ]
                condition = random.choice(geo_fields)
            # Now fill condition with actual placeholder and keep their value in user_inputs.
            all_placeholders = re.findall(r"\{([-a-zA-Z_]+)\}", condition)
            schema_name = template_info.ID.split("-")[0]

            for placeholder in all_placeholders:
                condition, filler = self.fill_placeholder(
                    query=condition,
                    placeholder=placeholder,
                    schema_name=schema_name,
                    count=1,
                )
                # Add generated fillers to user_input array
                user_inputs.append(filler)

            # Then add it to conditions array
            conditions.append(condition)

        condition_string = " AND ".join(conditions)
        query = query.replace(f"{{conditions}}", f"{condition_string}", 1)

        return query, user_inputs

    def generate_normal_queries(self, do_syn_check: bool):
        # Iterate over placeholders, and payload clause for type
        # Randomly choose a value in dict for that placeholder
        # And encapsulate based on type

        generated_normal_queries = []
        for template_row in tqdm(self._df_templates_n.itertuples()):
            all_placeholders = _extract_params(template=template_row.template)
            schema_name = template_row.ID.split("-")[0]

            query = template_row.template
            user_inputs = []

            # Replace 1 by 1 all placeholders by a randomly choosen dict value
            for placeholder in all_placeholders:
                # Remove placeholder's artificial int suffix:
                placeholder = placeholder.rstrip("123456789")

                # Special case for query on which conditions vary for each
                # input (airport-S23).
                if placeholder == "conditions":
                    query, fillers = self.fill_condition_randomly(
                        query=query, template_info=template_row
                    )
                    user_inputs = user_inputs + fillers
                else:
                    query, filler = self.fill_placeholder(
                        query=query,
                        placeholder=placeholder,
                        schema_name=schema_name,
                        count=1,
                    )
                    user_inputs.append(filler)

            if do_syn_check:
                if not self._verify_syntactic_validity_query(query=query):
                    raise ValueError("Failed normal query: ", query)
            user_inputs = [str(u) for u in user_inputs]
            generated_normal_queries.append(
                {
                    "full_query": query,
                    "label": 0,
                    "statement_type": template_row.statement_type,
                    "query_template_id": template_row.ID,
                    "user_inputs": " ".join(user_inputs),
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
        # input()
        server.stop_server()

        self._n_attacks = len(generated_attack_queries)
        self.df = generated_attack_queries

    def _clean_cache_folder(self):
        shutil.rmtree("./cache/", ignore_errors=True)

    def _add_template_split_info(self):
        """Add a column which specify wether the sample comes from the 'original'
        or the 'challenging' set of templates (the latter being those not seen during
        training).
        """
        ids_challenging = self.df_templates_test["ID"].to_list()
        self.df["template_split"] = "original"
        self.df.loc[
            self.df["query_template_id"].isin(ids_challenging), "template_split"
        ] = "challenging"

    def _remove_contradictions(self):
        """Remove contradictory samples from the dataset."""
        mask_atk = self.df["label"] == 1
        mask_n = self.df["label"] == 0
        df_a = self.df[mask_atk]
        df_n = self.df[mask_n]
        contradictions = set(df_a["full_query"]) & set(df_n["full_query"])

        _init_len = len(self.df)
        self.df = self.df[~self.df["full_query"].isin(contradictions)]
        logger.info(f"Removed {_init_len - len(self.df)} generated contradictions.")

    def _remove_user_input_admin(self):
        admin_ids = list(self.df_tadmin["ID"].unique())
        mask_admin_samples = self.df["query_template_id"].isin(admin_ids)
        self.df.loc[mask_admin_samples, "user_inputs"] = ""

    def build(self, args):
        testing_mode = args.testing
        debug_mode = args.debug
        do_syn_check = not args.no_syn_check

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
        self.generate_normal_queries(do_syn_check)
        self._add_split_column()

        self._augment_test_set_normal_queries(do_syn_check)
        self._add_template_split_info()

        self._remove_contradictions()
        self._remove_user_input_admin()

    def save(self):
        self.df.to_csv(self.outpath, index=False)
        # self._clean_cache_folder()
