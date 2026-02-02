import logging
from pathlib import Path
from subprocess import STDOUT, Popen, PIPE
import shutil

import pandas as pd

from .db_cnt_manager import SQLConnector
from .config_parser import get_mysql_info

logger = logging.getLogger(__name__)


class iThreatGenerator:
    def __init__(
        self, config: dict, sqlconnector: SQLConnector, testing_mode: bool = False
    ):
        self.sqlc = sqlconnector
        self.config = config
        self.testing_mode = testing_mode

    def enable_query_logging(self):
        """Send a query to the DBMS to enable the query logging."""
        self.sqlc.execute_priv_query(
            "SET GLOBAL general_log = 'ON';"
            "SET GLOBAL log_output = 'TABLE'; TRUNCATE TABLE mysql.general_log;"
        )
        pass

    def disable_query_logging(self):
        """Send a query to the DBMS to disable the query logging."""
        self.sqlc.execute_priv_query("SET GLOBAL general_log = 'OFF';")

    def clear_general_log(self):
        self.sqlc.execute_priv_query("TRUNCATE TABLE mysql.general_log;")

    def collect_general_log(self):
        """Collect all queries in the general_log table."""
        res_set = self.sqlc.execute_priv_query(
            "SELECT command_type,argument FROM mysql.general_log;"
        )
        queries = []
        for r in res_set[0]:
            if r[0] == "Query":
                queries.append(r[1].decode("UTF-8"))

        # Remove 3 last ones that comes from the connection and the SELECT logfile.
        # SET NAMES 'utf8mb4' COLLATE 'utf8mb4_0900_ai_ci'
        # SET @@session.autocommit = OFF
        # SELECT command_type,argument FROM mysql.general_log
        return queries[:-3]

    def perform_insider_attack_sqlmap(self):
        self.enable_query_logging()

        _, _, host, port, priv_user, priv_pwd = get_mysql_info(config=self.config)

        database = self.config["dataset"]["name"]

        connect_string = f"mysql://{priv_user}:{priv_pwd}@{host}:{str(port)}/{database}"
        # We don't want to use session files, no interaction either.
        base_command = f"sqlmap --fresh-queries  --batch -d '{connect_string}' "

        if self.testing_mode:
            objectives = ["--schema"]
        else:
            objectives = [
                "--dump",  # Dump DBMS database table entries
                "--passwords",  #  Enumerate DBMS users password hashes
                "--schema",  #  Enumerate DBMS users privileges
                "--all",  # Retrieve everything
            ]

        df_res = pd.DataFrame()
        cnt_obj = 0

        for obj in objectives:
            command = base_command + obj
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

            queries = self.collect_general_log()
            self.clear_general_log()

            _df = pd.DataFrame(
                {
                    "full_query": queries,
                    "label": 1,
                    "user_inputs": "",
                    "attack_stage": "exploit",
                    # TODO, we could find a way to check whethe exploitation has been
                    # performed correctly. For now, there is no errors from the sqlmap
                    # logs so we hard code their status to "success"...
                    "attack_status": "success",
                    "tamper_method": "",
                    "statement_type": "insider",
                    "query_template_id": "",
                    "attack_id": f"ithreat-sqlmap-{cnt_obj}",
                    "attack_technique": "insider",
                    "split": "test",
                }
            )

            cnt_obj += 1
            df_res = pd.concat([df_res, _df])

        self.disable_query_logging()
        return df_res

    def perform_insider_attack_metasploit(self):
        if shutil.which("msfconsole") is None:
            logger.critical(
                "msfconsole command not found, skipping metasploit generation"
            )
            return pd.DataFrame()

        self.enable_query_logging()
        _, _, host, port, priv_user, priv_pwd = get_mysql_info(config=self.config)
        df_res = pd.DataFrame()
        cnt_obj = 0

        # extracts the schema information from a MySQL DB server
        sc_schemadump = (
            f'"use auxiliary/scanner/mysql/mysql_schemadump; '
            f"set USERNAME {priv_user}; "
            f"set PASSWORD {priv_pwd}; "
            f"run mysql://{host}:{port};"
            f'exit"'
        )

        # extracts the usernames and encrypted password hashes from a MySQL server
        sc_userdump = (
            f'"use auxiliary/scanner/mysql/mysql_hashdump; '
            f"set USERNAME {priv_user}; "
            f"set PASSWORD {priv_pwd}; "
            f"run mysql://{host}:{port};"
            f'exit"'
        )

        # Other scenarios does not fit with insider threats
        scenarios = [sc_schemadump, sc_userdump]
        # Quiet, command mode
        base_command = "msfconsole -q -x "

        for sc in scenarios:
            command = base_command + sc
            logger.debug(f"Running command: {command}")

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

            queries = self.collect_general_log()
            self.clear_general_log()

            _df = pd.DataFrame(
                {
                    "full_query": queries,
                    "label": 1,
                    "user_inputs": "",
                    "attack_stage": "exploit",
                    "attack_status": "success",
                    "tamper_method": "",
                    "statement_type": "insider",
                    "query_template_id": "",
                    "attack_id": f"ithreat-msf-{cnt_obj}",
                    "attack_technique": "insider",
                }
            )
            cnt_obj += 1
            df_res = pd.concat([df_res, _df])
        self.disable_query_logging()

        return df_res
