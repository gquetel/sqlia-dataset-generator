import configparser
import logging
from pathlib import Path
from subprocess import STDOUT, Popen, PIPE
import pandas as pd
import urllib.parse
import urllib.request
import urllib.error

import random

from .db_cnt_manager import SQLConnector
from .config_parser import get_seed

logger = logging.getLogger(__name__)


class sqlmapGenerator:
    def __init__(
        self,
        config: configparser.ConfigParser,
        templates: pd.DataFrame,
        sqlconnector: SQLConnector,
        placeholders_dictionnaries_list: list,
        port: int,
    ):
        """Initialize data structures for payload generation."""

        self.templates = templates.to_dict("records")
        self.config = config
        self.port = port
        self.sqlc = sqlconnector
        # List of dictionnaries of values
        self.pdl = placeholders_dictionnaries_list

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

        self.seed = get_seed(self.config)

        # We want some kind of reproducibility in the resulting dataset,
        # however, using the same seed  for all sqlmap invocation would lead to
        # the same payloads for each of endpoints.  So we add an offset at each
        # invocation.
        self.seed_offset = 0
        self.generated_attacks = pd.DataFrame()
        self._scenario_id = 0

    def _clean_db(self):
        """Clean-up function to call after each sqlmap invocation to reduce side-effects"""

        # TODO:  Some attacks generate heavy queries, blocking the execution of
        # further ones. Changing MAX_EXECUTION_TIME does not seem to affect
        # this, nor lowering --risk value (lowering --risk value still lead to the
        # execution of heavy queries, this seem to be a bug from sqlmap.)

        # Hence, we first init a new connection.
        self.sqlc.init_new_cnx()

        # Then, we kill queries with a time > 5 and username tata. These queries
        # get a lock on some table which mess with the nexts sqlmap invocations.
        # This procedure does just that.

        time = 5
        query = f"""
        DROP PROCEDURE IF EXISTS killqueries; 

        CREATE PROCEDURE killqueries()
        BEGIN
        DECLARE done INT DEFAULT FALSE;
        DECLARE kill_id INT;
        DECLARE cur CURSOR FOR
            SELECT id FROM information_schema.processlist WHERE user = '{self.sqlc.user}' and time > {time};
        DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

        OPEN cur;

        read_loop: LOOP
            FETCH cur INTO kill_id;
            IF done THEN
            LEAVE read_loop;
            END IF;
            SET @s = CONCAT('KILL QUERY ', kill_id);
            PREPARE stmt FROM @s;
            EXECUTE stmt;
            DEALLOCATE PREPARE stmt;
        END LOOP;

        CLOSE cur;
        END;
        
        call killqueries();
        """
        self.sqlc.execute_query(query)
        
        # Then we clean the content of tables. Sometimes sqlmap inserts some data,
        # letting it there incrementally increase the number of commands required to
        # dump data from the DBMS as we invoke more and more sqlmap.
        
        tables = [
            "regions",
            "countries",
            "navaids",
            "runways",
            "airport_frequencies",
            "airport",
        ]

        for table in tables:
            self.sqlc.execute_query(
                f"SET FOREIGN_KEY_CHECKS = 0;" f"TRUNCATE TABLE  {table} ;"
            )
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
        # If any queries have been sent in the meantime, ignore those and retrieve last one.
        return queries[-1]

    def _construct_eval_option(self, db_name: str, parameters: list[str]) -> str:
        """_summary_

        Args:
            db_name (str): _description_
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
        # e_str = f' --eval="import random; random.seed({self.seed + self.seed_offset})"'
        e_str += f"import random;"

        for param in parameters:
            param_no_sx = param.rstrip("123456789")
            ran_values = random.choices(self.pdl[(db_name, param_no_sx)], k=10)

            values_str = str(ran_values).replace('"', '\\"')
            e_str += f"{param}=random.choice({values_str});"

        e_str += '" '
        return e_str

    def perform_recognition(
        self,
        url: str,
        settings_tech: str,
        params: list[str],
        db_name: str,
        debug_mode: bool,
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
        desc = ""  # TODO
        malicious_input = ""  # TODO
        label = 1

        for i, param in enumerate(params):
            # Iterate over existing params.
            _t_params = params.copy()
            _t_params.remove(param)
            settings_eval = self._construct_eval_option(
                db_name=db_name, parameters=_t_params
            )
            tamper_script = random.choice(self._tamper_scripts)
            settings_verbose = "-v 3 " if debug_mode else "-v 0 "

            recon_settings = (
                f"{settings_verbose} --skip-waf -D dataset --level=5 --risk=1 --batch "
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
            # We ignore retcode, we do not expect to find the string here.
            _ = self.call_sqlmap_subprocess(command=recon_command)

            # Fetch all queries for current parameter
            recon_queries = self.sqlc.get_and_empty_sent_queries()
            full_queries = list(filter(lambda a: a != default_query, recon_queries))

            _df = pd.concat(
                [
                    _df,
                    pd.DataFrame(
                        {
                            "full_query": full_queries,
                            "label": label,
                            "attack_payload": malicious_input,
                            "attack_desc": desc,
                            "attack_stage": "recon",
                            "tamper_method": tamper_script,
                            # "attacked_parameter": param, # TODO ?
                        }
                    ),
                ]
            )
        return _df

    def perform_exploit(
        self, url: str, settings_tech: str, debug_mode: bool
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
        #
        # I also don't want to spend any more time on adding variation on parameters
        # this is not the priority.
        tamper_script = random.choice(self._tamper_scripts)
        settings_verbose = "-v 3 " if debug_mode else "-v 0 "

        exploit_settings = (
            f"{settings_verbose} --skip-waf -D dataset --level=5 --risk=1 --batch "
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

        malicious_input = ""  # TODO
        label = 1
        sqlmap_status = "success" if retcode == 0 else "failure"

        _df = pd.DataFrame(
            {
                "full_query": full_queries,
                "label": label,
                "attack_payload": malicious_input,
                "attack_stage": "exploit",
                "sqlmap_status": sqlmap_status,
                "tamper_method": "None",
            }
        )
        return _df

    def perform_attack(self, technique: tuple, template_info: dict, debug_mode: bool):
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
            None: Updates the internal generated_attacks DataFrame with the attack
                results.
        """
        # Load all information to build the sqlmap command.
        name_tech, settings_tech = technique
        db_name = template_info["ID"].split("-")[0]
        params = {}
        for i, param in enumerate(template_info["placeholders"]):
            param_no_sx = param.rstrip("123456789")
            random_param_value = random.choice(self.pdl[(db_name, param_no_sx)])
            params[param] = random_param_value

        encoded_params = urllib.parse.urlencode(params)
        url = f"http://localhost:{self.port}/{template_info['ID']}?{encoded_params}"

        # Url is built. Invoke sqlmap for recognition
        _df_recon = self.perform_recognition(
            url=url,
            settings_tech=settings_tech,
            params=template_info["placeholders"],
            db_name=db_name,
            debug_mode=debug_mode,
        )

        # Now invoke sqlmap for exploitation
        _df_exploit = self.perform_exploit(
            url=url, settings_tech=settings_tech, debug_mode=debug_mode
        )

        atk_id = f"{name_tech}-{self._scenario_id}"
        template_id = template_info["ID"]
        desc = ""  # TODO

        # Fetch retcode of exploit phase to set it to recon one.
        sqlmap_status = _df_exploit.iloc[0]["sqlmap_status"]
        _df_exploit["sqlmap_status"] = sqlmap_status

        # Merge df of both steps and set all remaining columns.
        _df = pd.concat([_df_recon, _df_exploit])
        _df["statement_type"] = template_info["statement_type"]
        _df["query_template_id"] = template_id
        _df["attack_id"] = atk_id
        _df["attack_technique"] = name_tech
        _df["attack_desc"] = desc

        self.generated_attacks = pd.concat([self.generated_attacks, _df])
        self._scenario_id += 1
        self.seed_offset += 1

    def generate_attacks(self, testing_mode: bool, debug_mode: bool):
        """Generates SQL injection attacks for all templates using multiple techniques.

        This function iterates through all combinations of templates and SQLI
        techniques, performing the attack for each combination.

        Returns:
            pd.DataFrame: The complete generated_attacks DataFrame containing data
                from all performed attacks across all templates and techniques.
        """
        techniques = {
            "boolean": "--technique=B --all ",
            "error": "--technique=E --all ",
            "union": "--technique=U --all  ",
            "stacked": "--technique=S --users --banner ",
            "time": "--technique=T --current-user ",
            "inline": "--technique=Q --all ",
        }

        Path("./cache/").mkdir(parents=True, exist_ok=True)

        # Template's number is reduced, we also only consider the error technique.
        if testing_mode:
            techniques = {"error": "--technique=E --users "}

        for template in self.templates:
            for i in techniques.items():
                # example of cache file: ./cache/airport-I1-union
                cache_filepath = f"./cache/{template['ID']}-{i[0]}"
                if Path(cache_filepath).is_file():
                    self.generated_attacks = pd.read_csv(cache_filepath)
                    last_atkid = self.generated_attacks.iloc[-1]["attack_id"]
                    # Increment, as the retrieved value is the latest number of attacks.
                    # We want the next one to be different.
                    self._scenario_id = int(last_atkid.split("-")[1]) + 1
                    logger.info(
                        f">> Found cached file for scenario {template['ID']}-{i[0]}"
                        f" with {self._scenario_id} launched attacks."
                    )
                    self._scenario_id += 1
                    continue
                self.perform_attack(i, template, debug_mode)
                self.generated_attacks.to_csv(cache_filepath, index=False)

        return self.generated_attacks
