import logging
import os
from pathlib import Path
from subprocess import STDOUT, Popen, PIPE
import pandas as pd
import urllib.parse
import urllib.request
import urllib.error
import mysql.connector
import re
import string
import secrets

import random

from .db_cnt_manager import SQLConnector

logger = logging.getLogger(__name__)


class sqlmapGenerator:
    def __init__(
        self,
        dataset_config: dict,
        templates: pd.DataFrame,
        sqlconnector: SQLConnector,
        placeholders_dictionaries_list: list,
        port: int,
        seed: int,
        debug_mode: bool = False,
        testing_mode: bool = False,
    ):
        """Initialize data structures for payload generation."""

        self.templates = templates.to_dict("records")
        self.dataset_config = dataset_config
        self.port = port
        self.sqlc = sqlconnector
        self.debug_mode = debug_mode
        self.testing_mode = testing_mode

        # List of dictionaries of values
        self.pdl = placeholders_dictionaries_list

        # Extract dataset name from dataset-specific config
        self.dataset_name = dataset_config["name"]

        # List of tamper scripts that can be used during the attack, 1 is choosen at
        # random for each sqlmap invocation amongst this attribute.
        self._tamper_scripts = [
            "commentbeforeparentheses",  # Prepends (inline) comment before parentheses
            "equaltolike",  # Replaces all occurrences of operator = by 'LIKE'
            "lowercase",  # Replaces each keyword character with lower case value
            "multiplespaces",  # Adds multiple spaces around SQL keywords
            "randomcase",  # (e.g. SELECT -> SEleCt)
            "sleep2getlock",  # Replace 'SLEEP(5)' with stuff like "GET_LOCK('ETgP',5)"
            "space2comment",  # Replaces space character (' ') with comments '/**/'
            "space2mysqlblank",
        ]

        # "space2dash", # Replaces(' ') with  ('--'), a random string and ('\n')
        #  space2dash tamper script lead to sqlmap being unable to identify injections
        # let's not use it for generation.

        self.seed = seed

        self.generated_attacks = pd.DataFrame()
        self._scenario_id = 0

    def _run_pt_kill(self):
        socket_path = f"/tmp/mysql-{self.dataset_name}/mysql.sock"
        socket_opt = f" --socket={socket_path}"
        ptkill_command = (
            f"pt-kill --kill-query --user=root --password={self.sqlc.priv_pwd} --interval 1"
            f" --host={self.sqlc.host} --port={self.sqlc.port} --database "
            f"{self.sqlc.database} --busy-time 5s --run-time 10s --print{socket_opt}"
        )
        logger.debug(f"{ptkill_command}")
        proc = Popen(
            ptkill_command,
            shell=True,
            stdout=PIPE,
            stderr=STDOUT,
        )

        for line in proc.stdout:
            logger.warning(f"pt-kill matched the following query: {line.rstrip()}")
        proc.wait()

    def _clean_db(self):
        """Clean-up function to call after each sqlmap invocation to reduce side-effects"""

        # TODO:  Some attacks generate heavy queries, blocking the execution of
        # further ones. Changing MAX_EXECUTION_TIME does not seem to affect
        # this, nor lowering --risk value (lowering --risk value still lead to the
        # execution of heavy queries, this seem to be a bug from sqlmap.)

        # Hence, first run pt-kill to kill all blocking queries.
        self._run_pt_kill()

        # Then init a new connection
        self.sqlc.init_new_cnx()

        # Then we clean the content of tables. Sometimes sqlmap inserts some data,
        # letting it there incrementally increase the number of commands required to
        # dump data from the DBMS as we invoke more and more sqlmap.

        # Dynamically discover tables (not views) in the current dataset's database
        try:
            result = self.sqlc.execute_query(
                f"SHOW FULL TABLES WHERE Table_type = 'BASE TABLE';"
            )
            # execute_query returns a list of result sets, we need the first one
            if result and len(result) > 0:
                tables = [row[0] for row in result[0]]
            else:
                tables = []
        except Exception as e:
            logger.warning(f"Failed to retrieve table list from database: {str(e)}")
            tables = []

        for table in tables:
            try:
                self.sqlc.execute_query(
                    f"SET FOREIGN_KEY_CHECKS = 0;" f"TRUNCATE TABLE  {table} ;"
                )
            # 3. Still, for some reason the TRUNCATE call used after the pt-kill would
            # sometimes hang, so we also added  'connection_timeout': 10,  to the
            # connection settings. We need to catch that exception.
            except mysql.connector.errors.ReadTimeoutError:
                logger.critical(
                    f"TRUNCATE call hanged for more than 10 seconds, the database might"
                    f" not be clean."
                )
            except mysql.connector.errors.DatabaseError as e:
                logger.critical(f"TRUNCATE call failed: {str(e.msg)}")

        # Clear query cache
        _ = self.sqlc.get_and_empty_sent_queries()

    def call_sqlmap_subprocess(self, command) -> int:
        """Call the provided sqlmap command and return its success code

        Args:
            command (_type_): sqlmap command to run

        Returns:
            int: 0 if the attack succeeded, 1 if not correct payload was
                found to be injected.
        """
        self._clean_db()
        proc = Popen(
            command,
            shell=True,
            stdout=PIPE,
            stderr=STDOUT,
            universal_newlines=True,
        )

        output = ""
        for line in proc.stdout:
            logger.info(line.rstrip())
            output += line

        proc.wait()

        if "all tested parameters do not appear to be injectable." in output:
            return 1
        return 0

    def get_default_query_for_path(self, url) -> str:
        try:
            _ = urllib.request.urlopen(url).read()
        except urllib.error.URLError:
            logger.critical(
                f"get_default_query_for_path: failed to GET url: {url}, this is abnormal."
            )

        queries = self.sqlc.get_and_empty_sent_queries()
        if len(queries) > 0:
            return queries[-1]
        return []

    def _construct_eval_option(self, dataset_name: str, parameters: list[str]) -> str:
        """_summary_

        Args:
            dataset_name (str): _description_
            parameters (list[str]): _description_

        Returns:
            str: _description_
        """

        """ 
        From: https://github.com/sqlmapproject/sqlmap/wiki/Usage

        In case that user wants to change (or add new) parameter values, most probably 
        because of some known dependency, he can provide to sqlmap a custom python code 
        with option --eval that will be evaluated just before each request.

        For example:

        $ python sqlmap.py -u "http://www.target.com/vuln.php?id=1&hash=c4ca4238a0b9238\
        20dcc509a6f75849b" --eval="import hashlib;hash=hashlib.md5(id).hexdigest()"

        So we randomly select 100 values for the endpoint's parameters (given by the 
        parameters arguments) and before the creation of each query, sqlmap will 
        randomly sample one of them.

        Because this will be given as an option to the sqlmap command, we need to make 
        sure that the whole string is properly escaped: escape double quotes. 
        """

        e_str = ' --eval="'

        # First, we want to somehow control the random in here.
        e_str += f"import random;"

        for param in parameters:
            param_no_sx = param.rstrip("123456789")
            if param_no_sx == "rand_pos_number":
                ran_values = [random.randint(0, 64000) for _ in range(10)]
            elif param_no_sx == "rand_medium_pos_number":
                ran_values = [random.randint(1000, 6400) for _ in range(10)]
            elif param_no_sx == "rand_small_pos_number":
                ran_values = [random.randint(2, 5) for _ in range(10)]
            elif param_no_sx == "rand_string":
                alphabet = string.ascii_letters + string.digits
                ran_values = [
                    "".join(secrets.choice(alphabet) for i in range(20))
                    for _ in range(10)
                ]
            else:
                ran_values = random.choices(self.pdl[param_no_sx], k=10)

            values_str = str(ran_values).replace('"', '\\"')
            e_str += f"{param}=random.choice({values_str});"

        e_str += '" '
        return e_str

    def get_attack_payloads(self, queries: list, template_info: dict) -> list:
        """Extract sqlmap generated payloads from queries given the orignal template.

        Args:
            queries (list): List of queries to extract payloads from
            template_info (dict): template

        Returns:
            list: sqlmap-generated payloads
        """
        template = template_info["template"]

        # We construct a list of the fixed parts to remove from queries
        param_names = re.findall(r"(\{[-a-zA-Z_]+\})", template)
        regex_pattern = "|".join(map(re.escape, param_names))
        fixed_parts = re.split(regex_pattern, template)
        fixed_parts = [f for f in fixed_parts if f != ""]
        res = []

        for q in queries:
            _p = q
            for f in fixed_parts:
                _p = _p.replace(f, " ")
            res.append(_p)

        return res

    def perform_recognition(
        self,
        url: str,
        settings_tech: str,
        params: list[str],
        dataset_name: str,
        template_info: dict,
    ) -> pd.DataFrame:
        """Executes the reconnaissance phase of an SQL injection attack using SQLMap.

        This function identifies potential SQL injection vulnerabilities in the target
        URL without actually exploiting them. It runs SQLMap with reconnaissance
        settings and collects all queries sent during the process.

        Args:
            url (str): The target URL to test for SQL injection vulnerabilities.
            settings_tech (str): A string containing the SQLMap technique to use
                followed by arguments specifying information to exfiltrate. Only the
                technique is used in this phase.

        Returns:
            pd.DataFrame: A DataFrame containing the queries attempted during
                reconnaissance,
        """

        # The settings_tech is a string, first composed of the technique to use
        # And then arguments specifying the information to exfiltrate, we ignore those
        # for the recon phase:
        settings_tech = settings_tech.split()[0]

        _df = pd.DataFrame()
        default_query = self.get_default_query_for_path(url=url)

        # Columns that are the same for all sqlmap calls for this template.
        label = 1

        # When testing, only test a single parameter
        if self.testing_mode:
            params = [params[0]]

        for i, param in enumerate(params):
            # Iterate over existing params.
            _t_params = params.copy()
            _t_params.remove(param)
            settings_eval = self._construct_eval_option(
                dataset_name=dataset_name, parameters=_t_params
            )
            tamper_script = random.choice(self._tamper_scripts)
            settings_verbose = "-v 3 " if self.debug_mode else "-v 0 "

            recon_settings = (
                f"{settings_verbose} --skip-waf --level=5 --risk=1 --batch "
                f"--skip='user-agent,referer,host' {settings_eval} "
                f" -p '{param}' "
                f' -tamper="{tamper_script}" '
                f'{settings_tech} -u "{url}" '
            )

            # For first parameter, make sure that no previous session file exists.
            if i == 0:
                recon_settings += "--flush-session "

            recon_command = "sqlmap " + recon_settings
            logger.info(f">> Using recon command: {recon_command}")

            retcode = self.call_sqlmap_subprocess(command=recon_command)
            # Fetch all queries for current parameter
            recon_queries = self.sqlc.get_and_empty_sent_queries()
            full_queries = list(filter(lambda a: a != default_query, recon_queries))
            attack_status = "success" if retcode == 0 else "failure"
            attack_payloads = self.get_attack_payloads(full_queries, template_info)
            _df = pd.concat(
                [
                    _df,
                    pd.DataFrame(
                        {
                            "full_query": full_queries,
                            "label": label,
                            "user_inputs": attack_payloads,
                            "attack_stage": "recon",
                            "tamper_method": tamper_script,
                            "attack_status": attack_status,
                            # "attacked_parameter": param,
                        }
                    ),
                ]
            )

        return _df

    def perform_exploit(
        self, url: str, settings_tech: str, template_info: dict
    ) -> pd.DataFrame:
        """Executes the exploitation phase of an SQL injection attack using SQLMap.

        This function attempts to actually exploit vulnerabilities discovered during the
        reconnaissance phase. It runs SQLMap without the --flush-session flag to resume
        the attack and collects all queries sent during the exploitation.

        Args:
            url (str): The target URL to exploit.
            settings_tech (str): A string containing the SQLMap technique and arguments
                                to use for exploitation.

        Returns:
            pd.DataFrame: A DataFrame containing the queries attempted during
                exploitation.
        """

        # We do not provide --eval here: parameters for which a value is set in --eval
        # override the payloads injected by sqlmap. Since we cannot know for certain
        # which parameter worked during recon, we might override the successfull
        # parameter, making the attack fail.

        # I also don't want to spend any more time on adding variation on parameters
        # this is not the priority.

        tamper_script = random.choice(self._tamper_scripts)
        settings_verbose = "-v 3 " if self.debug_mode else "-v 0 "

        exploit_settings = (
            f"{settings_verbose} --skip-waf --level=5 --risk=1 --batch "
            f"--skip='user-agent,referer,host'"
            f' -tamper="{tamper_script}" '
            f'{settings_tech} -u "{url}"'
        )
        command = "sqlmap " + settings_tech + exploit_settings

        logger.info(f">> Using exploit command: {command}")
        retcode = self.call_sqlmap_subprocess(command=command)
        exploit_queries = self.sqlc.get_and_empty_sent_queries()

        default_query = self.get_default_query_for_path(url=url)
        full_queries = list(filter(lambda a: a != default_query, exploit_queries))

        attack_payloads = self.get_attack_payloads(full_queries, template_info)

        label = 1
        attack_status = "success" if retcode == 0 else "failure"

        _df = pd.DataFrame(
            {
                "full_query": full_queries,
                "label": label,
                "user_inputs": attack_payloads,
                "attack_stage": "exploit",
                "attack_status": attack_status,
                "tamper_method": tamper_script,
            }
        )
        return _df

    def perform_attack(self, technique: tuple, template_info: dict):
        """Orchestrates a full SQLI attack for a given technique and query template.

        This function runs both reconnaissance and exploitation phases of an SQL
        injection attack using SQLMap. It constructs a the endpoints' URL,
        executes both phases, and appends the result to the internel generated_attacks
        DataFrame object.

        Args:
            technique (tuple): A tuple containing the technique name and the
                corresponding SQLMap settings.
            template_info (dict): Information about the query template to use,
                including ID, placeholders, and statement type.

        Returns:
            None: Updates the insider generated_attacks DataFrame with the attack
                results.
        """
        # Load all information to build the sqlmap command.
        name_tech, settings_tech = technique
        params = {}
        for i, param in enumerate(template_info["placeholders"]):
            param_no_sx = param.rstrip("123456789")

            if param_no_sx == "rand_pos_number":
                random_param_value = random.randint(0, 64000)
            elif param_no_sx == "rand_medium_pos_number":
                random_param_value = random.randint(1000, 6400)
            elif param_no_sx == "rand_small_pos_number":
                random_param_value = random.randint(2, 5)
            elif param_no_sx == "rand_string":
                alphabet = string.ascii_letters + string.digits
                random_param_value = "".join(
                    secrets.choice(alphabet) for i in range(20)
                )
            else:
                random_param_value = random.choice(self.pdl[param_no_sx])

            params[param] = random_param_value

        encoded_params = urllib.parse.urlencode(params)
        url = f"http://localhost:{self.port}/{template_info['ID']}?{encoded_params}"

        # Url is built. Invoke sqlmap for recognition
        _df_recon = self.perform_recognition(
            url=url,
            settings_tech=settings_tech,
            params=template_info["placeholders"],
            dataset_name=self.dataset_name,
            template_info=template_info,
        )

        # Variables independant of attack_status:
        atk_id = f"{name_tech}-{self._scenario_id}"
        template_id = template_info["ID"]

        # If any recognition sqlmap attack succeeded, we can find at least a row where
        # `attack_status` is set to success in _df_recon.
        if (_df_recon["attack_status"] == "success").any():
            # At least a vulnerable endpoint has been found.
            # Start exploit
            _df_exploit = self.perform_exploit(
                url=url,
                settings_tech=settings_tech,
                template_info=template_info,
            )

            _df = pd.concat([_df_recon, _df_exploit])

            # This is anormal:
            if _df_exploit.iloc[0]["attack_status"] == "failure":
                logger.critical(
                    f"perform_attack: An vulnerable endpoint was found"
                    f" but exploit failed."
                )
                # Set all to failure.
                _df["attack_status"] = "failure"
            else:
                # Set all to success
                _df["attack_status"] = "success"

        else:
            # No vulnerable endpoint has been found, do not launch exploit.
            _df = _df_recon

            # Failure is already set to all samples of df.
        _df["statement_type"] = template_info["statement_type"]
        _df["query_template_id"] = template_id
        _df["attack_id"] = atk_id
        _df["attack_technique"] = name_tech

        self.generated_attacks = pd.concat([self.generated_attacks, _df])
        self._scenario_id += 1

    def generate_attacks(self):
        """Generates SQL injection attacks for all templates using multiple techniques.

        This function iterates through all combinations of templates and SQLI
        techniques, performing the attack for each combination.

        Returns:
            pd.DataFrame: The complete generated_attacks DataFrame containing data
                from all performed attacks across all templates and techniques.
        """
        # Data exfiltration scope is limited to reduce generation time.
        # The current configuration uses specific flags (--users, --banner,
        # --schema, --current-user) instead of --all to reduce dataset generation
        # time, especially when databases are populated.
        #
        # If you need to include payloads that dump all database information
        # (tables, columns, data), replace the targeted flags below with --all for the
        # desired techniques.

        # techniques = {
        #     "boolean": "--technique=B --all ",
        #     "error": "--technique=E --all ",
        #     "union": "--technique=U --all  ",
        #     "stacked": "--technique=S --users --banner ",
        #     "time": "--technique=T --current-user ",
        #     "inline": "--technique=Q --all ",
        # }

        techniques = {
            "boolean": "--technique=B --users --banner --schema ",
            "error": "--technique=E --users --banner --schema ",
            "union": "--technique=U --users --banner --schema  ",
            "stacked": "--technique=S --users --banner ",
            "time": "--technique=T --current-user ",
            "inline": "--technique=Q --users --banner --schema ",
        }

        cache_dir = f"./.cache/{self.dataset_name}"
        Path(cache_dir).mkdir(parents=True, exist_ok=True)

        # Template's number is reduced, we also only consider the error technique.
        if self.testing_mode:
            techniques = {"error": "--technique=E --schema "}

        # Build ordered list of (template, technique, cache_filepath) combos.
        combos = [
            (template, tech, f"{cache_dir}/{template['ID']}-{tech[0]}")
            for template in self.templates
            for tech in techniques.items()
        ]

        # Find the last cached file (scan in reverse) and load only that one,
        # since each cache file is a cumulative snapshot of all prior attacks.
        last_cached_idx = -1
        for idx in range(len(combos) - 1, -1, -1):
            if Path(combos[idx][2]).is_file():
                last_cached_idx = idx
                break

        if last_cached_idx >= 0:
            cache_path = combos[last_cached_idx][2]
            self.generated_attacks = pd.read_csv(cache_path)
            last_atkid = self.generated_attacks.iloc[-1]["attack_id"]
            self._scenario_id = int(last_atkid.split("-")[1]) + 2
            logger.info(
                f">> Resuming from cached file {cache_path}"
                f" with {self._scenario_id - 1} launched attacks."
            )

        prev_cache = None
        for idx, (template, tech, cache_filepath) in enumerate(combos):
            if idx <= last_cached_idx:
                prev_cache = cache_filepath  # track last existing cache
                continue

            self.perform_attack(tech, template)
            self.generated_attacks.to_csv(cache_filepath, index=False)
            if prev_cache is not None:
                Path(prev_cache).unlink(missing_ok=True)
            prev_cache = cache_filepath

        return self.generated_attacks
