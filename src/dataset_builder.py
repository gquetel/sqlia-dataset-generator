import logging
import numpy as np
import os
import pandas as pd
import random
import secrets
import string
import re
import shutil

from tqdm import tqdm

from .condition_generator import ConditionGenerator
from .ithreat_generator import iThreatGenerator
from .sqlia_generator import sqlmapGenerator

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

        # config is made of a "general" attribute containing information generic to
        # the whole app (such as seed info), and "dataset" that is specific to the
        # dataset currently generated.
        self.config = config
        self.seed = config["general"]["seed"]

        self.dataset_config = self.config["dataset"]
        self.dataset_name = self.dataset_config["name"]

        #  Dict holding all possible filler values, Keys are placeholder names
        self.dictionaries = {}

        # Connection wrapper to SQL server.
        self.sqlc = None

        # Sampled templates of normal queries to fill.
        self._df_templates_n = None

        # Template DataFrames partitioned by usage:
        # Templates of administration queries
        self.df_tadmin = None
        # Templates used to generate attacks
        self.df_atk_templates = None
        # Templates used to generate normal queries (atk, normal-only and admin)
        self.df_norm_templates = None

        # Dataframe holding SQL-sourced statements (for normal query generation only)
        self.df_sql_statements = None

        # Initialisation code.
        random.seed(self.seed)
        np.random.seed(self.seed)
        self.populate_dictionaries()

        dataset_path = f"./data/datasets/{self.dataset_name}"
        self.condition_generator = ConditionGenerator(
            dataset_path=dataset_path,
            fill_placeholder_fn=self.fill_placeholder,
        )

    def _get_sql_connector(self):
        """Get or initialize the SQL connector lazily.

        Returns:
            SQLConnector: The SQL connector instance
        """
        if self.sqlc is None:
            self.sqlc = SQLConnector(self.config, self.dataset_name)
        return self.sqlc

    def populate_dictionaries(self):
        """Load dictionaries of legitimate values for placeholders.

        The function checks under data/datasets/$dataset/dicts and loads all
        existing files into self.dictionaries[placeholder_id]
        """
        dicts_dir = f"./data/datasets/{self.dataset_name}/dicts/"
        for filename in os.listdir(dicts_dir):
            with open(dicts_dir + filename, "r") as f:
                self.dictionaries[filename] = f.read().splitlines()

    def _load_statements_from_sql_script(
        self, sql_file_path: str, statement_type: str
    ) -> pd.DataFrame:
        """Parse SQL script file and extract statements with template ID annotations.
        We expect each line to end with a comment and the template-ID corresponding
        to the query.

        Args:
            sql_file_path: Path to the SQL script file
            statement_type: Type of statement (insert, select, update, delete, admin)

        Returns:
            DataFrame with columns: template, ID, description
        """
        with open(sql_file_path, "r") as f:
            content = f.read()

        # Pattern to match: SQL statement ending with '; -- TEMPLATE-ID'
        # Captures: (statement) ; -- (template_id)
        # Supports formats: 'OHR-I1', 'airport-S1', 'airport-admin1'
        pattern = r"(.+?);[\s]*--[\s]*([A-Za-z]+-(?:[A-Z]-?|admin)\d+)"

        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

        statements = []
        auto_id_counter = 1

        for statement, template_id in matches:
            statement = statement.strip()

            if not statement or statement.startswith("--"):
                continue

            id_pattern = r"^[A-Za-z]+-(?:[A-Z]-?|admin)\d+$"
            if not re.match(id_pattern, template_id):
                logger.warning(
                    f"Invalid template ID format in {sql_file_path}: '{template_id}'. "
                    f"Expected format: {{dataset}}-{{Type}}[-]{{number}} (e.g., OHR-I-1, airport-S1, or airport-admin1). "
                    f"Statement will be skipped."
                )
                continue

            statements.append(
                {
                    "template": statement + ";",
                    "ID": template_id,
                    "description": f"SQL statement from {statement_type}.sql",
                }
            )
        return pd.DataFrame(statements)

    def _calculate_template_distribution(self, statement_type: str) -> dict:
        """Calculate distribution of template types from SQL statements.

        Groups statements by template ID and calculates the proportion of each
        template type within the given statement type.

        Args:
            statement_type: Type of statement (used for filtering and logging)

        Returns:
            Dictionary mapping template_id -> proportion (e.g., {'OHR-I-1': 0.021})
        """
        # Filter to only SQL-sourced statements for this statement type
        sql_statements = self.df_sql_statements[
            self.df_sql_statements["statement_type"] == statement_type
        ]

        if sql_statements.empty:
            return {}

        # Count occurrences of each template ID
        template_counts = sql_statements["ID"].value_counts()

        # Calculate total number of SQL statements
        total_statements = len(sql_statements)

        # Calculate proportions
        distribution = {}
        for template_id, count in template_counts.items():
            proportion = count / total_statements
            distribution[template_id] = proportion

        logger.info(f"Template distribution for {statement_type}:")
        for template_id, proportion in sorted(
            distribution.items(), key=lambda x: x[1], reverse=True
        ):
            count = template_counts[template_id]
            logger.info(
                f"  - {template_id}: {count} statements ({proportion*100:.1f}%)"
            )

        return distribution

    def load_templates_and_stmts(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load all templates from generation settings and optional SQL script statements.

        Loads templates from either .sql files (with template annotations),
        .csv template files, or both. For each statement type, at least one
        source must exist.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
                - DataFrame with all CSV templates
                - DataFrame with all SQL-sourced statements
        """
        statements_type = config_parser.get_statement_types_and_proportions(
            self.dataset_config
        )
        _all_templates = pd.DataFrame()
        _all_sql_statement = pd.DataFrame()

        dataset_dir = f"./data/datasets/{self.dataset_name}"
        queries_dir = f"{dataset_dir}/queries/"

        for stmt_type in statements_type:
            statement_type = stmt_type["type"]
            sql_file_path = f"{dataset_dir}/{statement_type}.sql"
            csv_file_path = f"{queries_dir}{statement_type}.csv"

            sql_exists = os.path.exists(sql_file_path)
            csv_exists = os.path.exists(csv_file_path)

            if not sql_exists and not csv_exists:
                raise FileNotFoundError(
                    f"No SQL file or template CSV found for statement type '{statement_type}'. "
                    f"Expected either:\n"
                    f"  - {sql_file_path} (with template annotations), or\n"
                    f"  - {csv_file_path}\n"
                    f"At least one must exist."
                )

            if sql_exists:
                sql_statements = self._load_statements_from_sql_script(
                    sql_file_path, statement_type
                )
                logger.info(
                    f"Loaded {len(sql_statements)} statements from {statement_type}.sql "
                    f"with {sql_statements['ID'].nunique()} template types"
                )
                sql_statements["statement_type"] = stmt_type["type"]

            templates = pd.read_csv(csv_file_path)
            logger.info(f"Loaded {len(templates)} templates from {statement_type}.csv")

            templates["proportion"] = stmt_type["proportion"]
            templates["statement_type"] = stmt_type["type"]

            # Extract placeholders from templates, later use to create sqlmap attacks
            templates["placeholders"] = templates["template"].apply(_extract_params)

            # If not an admin statement, we check that there is placeholders in queries
            if statement_type != "admin":
                templates_without_placeholders = templates[
                    templates["placeholders"].apply(lambda x: len(x) == 0)
                ]

                if not templates_without_placeholders.empty:
                    error_details = []
                    for stmt_type in templates_without_placeholders[
                        "statement_type"
                    ].unique():
                        templates_of_type = templates_without_placeholders[
                            templates_without_placeholders["statement_type"]
                            == stmt_type
                        ]
                        template_ids = templates_of_type["ID"].tolist()
                        error_details.append(f"  - {stmt_type.upper()}: {template_ids}")

                    raise ValueError(
                        f"Found templates without placeholders. Templates must have at least one placeholder.\n"
                        f"Problematic template IDs:\n" + "\n".join(error_details) + "\n"
                    )

            # Check for template ID mismatches if both exist
            if sql_exists and csv_exists:
                sql_ids = set(sql_statements["ID"].unique())
                csv_ids = set(templates["ID"].unique())

                missing_in_csv = sql_ids - csv_ids
                if missing_in_csv:
                    raise ValueError(
                        f"Template IDs in SQL annotations don't match CSV template IDs for {statement_type}:\n"
                        f"  - SQL references: {sorted(sql_ids)}\n"
                        f"  - CSV contains: {sorted(csv_ids)}\n"
                        f"  - Missing in CSV: {sorted(missing_in_csv)}\n"
                    )
                # Only concat if no issue are found
                _all_sql_statement = pd.concat([_all_sql_statement, sql_statements])

            _all_templates = pd.concat([_all_templates, templates])

        # Reset indices to avoid duplicates from concatenating multiple CSVs
        _all_templates = _all_templates.reset_index(drop=True)
        if not _all_sql_statement.empty:
            _all_sql_statement = _all_sql_statement.reset_index(drop=True)
        return _all_templates, _all_sql_statement

    def init_templates(self, testing_mode: bool):
        """Initialize templates variables according to the generation settings.

        - This function randomly samples templates that will only be used to generate
        normal samples
        - If testing mode is enabled, this reduce the number of templates for generation.

        Args:
            testing_mode (bool): If True, reduces attack templates to 1 for faster testing
        """

        templates, statements = self.load_templates_and_stmts()
        self.df_sql_statements = statements

        # First, save all templates as they all will be used to generate normal queries
        self.df_norm_templates = templates

        # Then, identify and remove admin statements.
        self.df_tadmin = templates[templates["ID"].str.contains("admin")]
        templates = templates[~templates["ID"].str.contains("admin")]

        # Now, initialise df_atk_templates. Before we need to decide which templates
        # are used for generating only attacks.
        ratio_tno = config_parser.get_normal_only_template_ratio(self.config)
        if ratio_tno > 0:
            n_tno = round(templates.shape[0] * ratio_tno)
            df_tno = templates.sample(n=n_tno)
            self.df_atk_templates = templates.drop(df_tno.index)
        else:
            self.df_atk_templates = templates

        # Instantiate {conditions} templates for attack generation.
        # Instead of excluding them, we freeze one concrete set of conditions
        # so sqlmap can attack the resulting placeholders normally.
        conditions_mask = self.df_atk_templates["template"].str.contains(
            r"\{conditions\}", regex=True
        )
        if conditions_mask.any():
            for idx in self.df_atk_templates[conditions_mask].index:
                template_str = self.df_atk_templates.at[idx, "template"]
                template_id = self.df_atk_templates.at[idx, "ID"]
                table_name = self._extract_table_name(template_str)
                if table_name and self.condition_generator.has_conditions(table_name):
                    instantiated_conds = self.condition_generator.instantiate_template(
                        table_name
                    )
                    new_template = template_str.replace(
                        "{conditions}", instantiated_conds, 1
                    )
                    self.df_atk_templates.at[idx, "template"] = new_template
                    self.df_atk_templates.at[idx, "placeholders"] = _extract_params(
                        new_template
                    )
                    logger.info(
                        f"Instantiated {{conditions}} in {template_id}: {new_template}"
                    )

        if testing_mode:
            n_templates = 1
            # We want shorter atck time so we reduce the number of templates
            self.df_atk_templates = self.df_atk_templates.sample(n=n_templates)
            # Other templates don't take so much time so we can let them as is.
            logger.warning(
                f"Testing mode enabled, reduced the number of attack templates to {n_templates}."
            )

    def _assign_splits_to_dataset(self, n_normal_train: int):
        """Assign train/test splits to the complete dataset.

        All attacks go to the test set. Normal samples are split with the first
        n_normal_train samples going to training and the rest to testing.

        Args:
            n_normal_train: Number of normal samples to assign to training set
        """
        # All attacks go to test set
        atk_mask = self.df["label"] == 1
        self.df.loc[atk_mask, "split"] = "test"

        # For normal samples: shuffle then split to ensure statement type diversity
        normal_mask = self.df["label"] == 0
        normal_indices = self.df[normal_mask].index
        shuffled = (
            normal_indices.to_series().sample(frac=1, random_state=self.seed).index
        )

        train_indices = shuffled[:n_normal_train]
        test_indices = shuffled[n_normal_train:]

        self.df.loc[train_indices, "split"] = "train"
        self.df.loc[test_indices, "split"] = "test"

    def populate_normal_templates(self, n_n: int):
        """Sample n_n templates with distribution preservation from SQL files.

        Behavior:
        1. Use SQL statements from script first (if they exist)
        2. Calculate remaining samples needed
        3. Supplement using templates, maintaining SQL distribution proportions
        4. If only templates exist, sample with normalized weights

        Args:
            n_n: Number of normal samples to generate
        """
        # df_norm_templates holds the templates from which to generate.
        csv_templates = self.df_norm_templates
        sql_statements = self.df_sql_statements

        normal_samples = pd.DataFrame()
        # We iterate over statement types and:
        # - compute how many statements must be generated
        # - If we find statements from sql script we sampled from there
        # - Else, or if there isn't enough statements in the script, we rely
        #   on templates to generate samples.
        for statement_type in csv_templates["statement_type"].unique():
            stmt_proportion = csv_templates[
                csv_templates["statement_type"] == statement_type
            ]["proportion"].iloc[0]
            target_samples = int(n_n * stmt_proportion)
            if target_samples == 0:
                continue

            csv_tmpls = csv_templates[csv_templates["statement_type"] == statement_type]

            # Check if SQL statements exist and filter by statement type
            if not sql_statements.empty:
                sql_stmts = sql_statements[
                    sql_statements["statement_type"] == statement_type
                ]
            else:
                sql_stmts = pd.DataFrame()

            if not sql_stmts.empty:
                normal_samples = pd.concat([normal_samples, sql_stmts])
                remaining_samples = target_samples - len(sql_stmts)
                if remaining_samples > 0:
                    # We generate the next queries according to the template
                    # distribution in the script (that we want to imitate).
                    template_distribution = self._calculate_template_distribution(
                        statement_type
                    )

                    supplemented = pd.DataFrame()

                    for template_id, proportion in template_distribution.items():
                        target_count = int(remaining_samples * proportion)

                        if target_count == 0:
                            continue

                        matching_csv = csv_tmpls[csv_tmpls["ID"] == template_id]
                        # Note: Using replace=True allows duplicates when target_count exceeds available templates.
                        # This is intentional - we need to generate the target number of samples even with limited templates.
                        sampled = matching_csv.sample(n=target_count, replace=True)
                        supplemented = pd.concat([supplemented, sampled])

                    normal_samples = pd.concat([normal_samples, supplemented])
                    logger.info(
                        f"Used {len(sql_stmts)} from SQL script file + {len(supplemented)} template-generated"
                    )
                elif remaining_samples < 0:
                    logger.warning(
                        f"Sampled {target_samples} {statement_type} from SQL script file."
                    )
                    # We keep other statement type but sample ours.
                    normal_samples = normal_samples[
                        normal_samples["statement_type"] != statement_type
                    ]
                    sampled = sql_stmts.sample(n=target_samples, replace=False)
                    normal_samples = pd.concat([normal_samples, sampled])

            else:
                # Sample templates with equal probability within this statement type
                sampled = csv_tmpls.sample(n=target_samples, replace=True)
                normal_samples = pd.concat([normal_samples, sampled])

        # It's ok to merge statements and templates here, no placeholders are present
        # they will be ignored
        self._df_templates_n = normal_samples.reset_index(drop=True)

    def fill_placeholder(
        self, query: str, placeholder: str, count: int = 1
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
            filler = random.choice(self.dictionaries[placeholder])
        filler = str(filler).replace('"', '""')
        return (query.replace(f"{{{placeholder}}}", f"{filler}", 1), filler)

    @staticmethod
    def _extract_table_name(template: str) -> str | None:
        """Extract the main table name from a SQL template.

        Looks for FROM <table> WHERE pattern to identify the table used
        in condition generation.

        Returns:
            Table name or None if not found.
        """
        match = re.search(r"\bFROM\s+(\w+)\s+WHERE\b", template, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def fill_condition_randomly(
        self, query: str, template_info: dict
    ) -> tuple[str, list]:
        """Randomly choose conditions to insert in query.

        This function mimics the behavior of a page with multiple possible
        search conditions, that might not all be used. Condition definitions
        are loaded from the dataset's conditions.toml file.

        Args:
            query (str): The query template containing {conditions} placeholder
            template_info (dict): Template row information (used to extract table name)
        Returns:
            tuple[str, list]: The query with conditions filled in,
                and the synthetic user inputs
        """
        table_name = self._extract_table_name(query)
        if table_name is None or not self.condition_generator.has_conditions(
            table_name
        ):
            raise ValueError(
                f"Template uses {{conditions}} but no condition definition found "
                f"for table '{table_name}' in conditions.toml"
            )

        condition_string, user_inputs = self.condition_generator.generate(table_name)
        query = query.replace("{conditions}", condition_string, 1)
        return query, user_inputs

    def generate_normal_queries(self, do_syn_check: bool):
        """Generate normal queries from templates.

        New behavior:
        - SQL-sourced statements: Use as-is (already complete)
        - Template-sourced statements: Fill placeholders with random values
        """
        generated_normal_queries = []
        for template_row in tqdm(self._df_templates_n.itertuples()):
            query = template_row.template
            user_inputs = []
            all_placeholders = _extract_params(template=template_row.template)

            # Replace 1 by 1 all placeholders by a randomly chosen dict value
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
                        count=1,
                    )
                    user_inputs.append(filler)

            # Validate syntax if requested
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
                    "attack_status": None,
                }
            )
        self.df = pd.concat(
            [self.df, pd.DataFrame(generated_normal_queries)],
            ignore_index=True,
        )

    def _verify_syntactic_validity_query(self, query: str):
        sqlc = self._get_sql_connector()
        res = sqlc.is_query_syntvalid(query=query)
        return res

    def generate_attack_queries_sqlmapapi(
        self, testing_mode: bool, debug_mode: bool
    ) -> dict:
        base_http_port = 8080
        server_port = base_http_port + self.dataset_config.get("port_offset", 0)
        generated_attack_queries = []

        # First, initialize all HTTP endpoints for each template.
        # templates are already selected in self.df_atk_templates

        sqlc = self._get_sql_connector()
        # Prune all sent_queries for attacks
        _ = sqlc.get_and_empty_sent_queries()

        server = TemplatedSQLServer(
            templates=self.df_atk_templates, sqlconnector=sqlc, port=server_port
        )
        server.start_server()
        # Now iterate over templates and techniques to generate payloads.
        sqlg = sqlmapGenerator(
            dataset_config=self.dataset_config,
            templates=self.df_atk_templates,
            sqlconnector=sqlc,
            placeholders_dictionaries_list=self.dictionaries,
            port=server_port,
            seed=self.seed,
            testing_mode=testing_mode,
            debug_mode=debug_mode,
        )
        generated_attack_queries = sqlg.generate_attacks()
        server.stop_server()

        self._n_attacks = len(generated_attack_queries)
        self.df = generated_attack_queries

    def _clean_cache_folder(self):
        shutil.rmtree(f"./.cache/{self.dataset_name}/", ignore_errors=True)

    def _remove_contradictions(self):
        """Remove contradictory samples from the dataset."""
        mask_atk = self.df["label"] == 1
        mask_n = self.df["label"] == 0
        df_a = self.df[mask_atk]
        df_n = self.df[mask_n]
        contradictions = set(df_a["full_query"]) & set(df_n["full_query"])

        _init_len = len(self.df)
        self.df = self.df[~self.df["full_query"].isin(contradictions)]
        logger.warning(f"Removed {_init_len - len(self.df)} generated contradictions.")

    def _remove_user_input_admin(self):
        admin_ids = list(self.df_tadmin["ID"].unique())
        mask_admin_samples = self.df["query_template_id"].isin(admin_ids)
        self.df.loc[mask_admin_samples, "user_inputs"] = ""

    def generate_ithreat(self, args):
        """Generate insider threat attack samples using sqlmap.

        Args:
            args: Command-line arguments containing testing mode flag

        Returns:
            pd.DataFrame: DataFrame containing insider threat attack samples
        """
        sqlc = self._get_sql_connector()

        itg = iThreatGenerator(self.config, sqlc, args.testing)

        # df_metasploit = itg.perform_insider_attack_metasploit()
        df_sqlmap = itg.perform_insider_attack_sqlmap()

        return df_sqlmap

    def build(self, args):
        """Build the complete dataset with attacks and normal samples.

        Orchestrates the full dataset generation process:
        1. Initialize and partition templates (attack vs normal-only)
        2. Generate SQL injection attacks using sqlmap
        3. Generate insider threat attacks
        4. Calculate required normal samples to achieve target attack ratio
        5. Generate normal queries from templates
        6. Assign train/test splits (attacks to test, normal samples split)
        7. Remove contradictory samples (queries appearing in both attack and normal)
        8. Remove user inputs from admin queries

        Args:
            args: Command-line arguments containing:
                - testing: Enable fast testing mode (reduced templates)
                - debug: Enable debug output from sqlmap
                - no_syn_check: Skip syntax validation of generated queries
        """
        # If testing_mode begins to be too annoying to pass around,
        # transform it into a class attribute.
        testing_mode = args.testing
        debug_mode = args.debug
        do_syn_check = not args.no_syn_check

        # First, we identify all templates that should be used for generation.
        self.init_templates(testing_mode=testing_mode)
        self.generate_attack_queries_sqlmapapi(
            testing_mode=testing_mode, debug_mode=debug_mode
        )
        # Then, generate insider attacks.
        df_ithreat = self.generate_ithreat(args)
        self.df = pd.concat([self.df, df_ithreat])

        # Calculate all normal queries needed upfront
        n_attacks = len(self.df[self.df["label"] == 1])
        atk_ratio = config_parser.get_attacks_ratio(self.config)

        # Train set: normal samples equal to total attack count (for balanced training)
        n_normal_train = n_attacks
        # Test set: remaining normal samples to achieve target attack ratio
        n_normal_test = int(n_attacks / atk_ratio) - n_attacks
        total_normal = n_normal_train + n_normal_test

        # Generate all normal queries in one pass
        self.populate_normal_templates(total_normal)
        self.generate_normal_queries(do_syn_check)

        # Assign train/test splits
        self._assign_splits_to_dataset(n_normal_train)

        self._remove_contradictions()
        self._remove_user_input_admin()

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        outpath = os.path.join(output_dir, f"{self.dataset_name}.csv")

        self.df.to_csv(outpath, index=False)
        self._clean_cache_folder()
