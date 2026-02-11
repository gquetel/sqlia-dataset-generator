import logging
import os
from pathlib import Path
from subprocess import PIPE, Popen

import mysql.connector
import mysql

from .config_parser import get_mysql_info, get_dataset_port

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset-level database isolation helpers
# ---------------------------------------------------------------------------

# Resolve the project root and datasets directory relative to this file.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATASETS_DIR = _PROJECT_ROOT / "data" / "datasets"
_SCRIPTS_DIR = _PROJECT_ROOT / "scripts"


def _mysql_env(dataset_name: str, port: int) -> dict:
    """Build the environment variables needed by mysql-start / mysql-stop."""
    mysql_home = f"/tmp/mysql-{dataset_name}"
    env = os.environ.copy()
    env["MYSQL_HOME"] = mysql_home
    env["MYSQL_DATADIR"] = f"{mysql_home}/data"
    env["MYSQL_UNIX_PORT"] = f"{mysql_home}/mysql.sock"
    env["MYSQL_PORT"] = str(port)
    # Ensure our scripts directory is on PATH
    env["PATH"] = f"{_SCRIPTS_DIR}:{env.get('PATH', '')}"
    return env


def start_mysql_server(config: dict, dataset_name: str) -> int:
    """Start an isolated MySQL instance for *dataset_name*.

    Returns the port the server is listening on.
    """
    port = get_dataset_port(config, dataset_name)
    env = _mysql_env(dataset_name, port)

    # Clean any leftover state first.
    proc = Popen(
        [str(_SCRIPTS_DIR / "mysql-stop"), "--clean"],
        env=env,
        stdout=PIPE,
        stderr=PIPE,
    )
    proc.communicate()

    # Start a fresh instance.
    logger.info(
        f"Starting MySQL for '{dataset_name}' on port {port} "
        f"(MYSQL_HOME={env['MYSQL_HOME']})"
    )
    proc = Popen(
        [str(_SCRIPTS_DIR / "mysql-start")],
        env=env,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(
            f"mysql-start failed for '{dataset_name}':\n"
            f"{stderr.decode()}\n{stdout.decode()}"
        )

    logger.info(f"MySQL for '{dataset_name}' started on port {port}.")
    return port


def stop_mysql_server(config: dict, dataset_name: str) -> None:
    """Stop and clean up the isolated MySQL instance for *dataset_name*."""
    port = get_dataset_port(config, dataset_name)
    env = _mysql_env(dataset_name, port)

    logger.info(f"Stopping MySQL for '{dataset_name}'.")
    proc = Popen(
        [str(_SCRIPTS_DIR / "mysql-stop"), "--clean"],
        env=env,
        stdout=PIPE,
        stderr=PIPE,
    )
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        logger.warning(f"mysql-stop failed for '{dataset_name}': {stderr.decode()}")


def init_dataset_db(config: dict, dataset_name: str) -> None:
    """Create a single dataset's database (schema + data) via the MySQL CLI."""
    port = get_dataset_port(config, dataset_name)
    _, _, _, _, priv_user, priv_pwd = get_mysql_info(config=config)
    dataset_dir = _DATASETS_DIR / dataset_name
    sql_file = "init_db.sql"

    sql_path = dataset_dir / sql_file
    if not sql_path.is_file():
        raise FileNotFoundError(f"Expected SQL file not found: {sql_path}")

    command = (
        f"mysql --port={port} --protocol=tcp "
        f"-u {priv_user} -p{priv_pwd} < {sql_path}"
    )

    logger.info(f"Initializing database: sourcing {sql_path}")
    proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to source {sql_path}: {stderr.decode()}")

    logger.info(f"Database '{dataset_name}' initialized successfully.")


class SQLConnector:
    def __init__(self, config: dict, database: str = None):
        user, pwd, host, port, priv_user, priv_pwd = get_mysql_info(config=config)
        self.user = user
        self.pwd = pwd
        self.host = host
        self.port = port
        self.priv_user = priv_user
        self.priv_pwd = priv_pwd
        self.database = database if database else "dataset"
        self.init_new_cnx()

        # Array of sent queries by self.execute_query
        self.sent_queries = []

    def init_new_cnx(self):
        self.cnx = mysql.connector.connect(
            user=self.user,
            password=self.pwd,
            host=self.host,
            port=self.port,
            database=self.database,
            read_timeout=10,
        )

    def get_and_empty_sent_queries(self) -> list:
        res = self.sent_queries.copy()
        self.sent_queries = []
        return res

    def execute_query(self, query):
        # https://dev.mysql.com/doc/connector-python/en/connector-python-multi.html
        if self.cnx is None or not self.cnx.is_connected():
            self.init_new_cnx()

        results = []
        self.sent_queries.append(query)
        with self.cnx.cursor(buffered=True) as cur:
            # Set maximum execution time of 10 sec (only applies to SELECT statements).
            cur.execute("SET SESSION MAX_EXECUTION_TIME=10000")
            cur.execute(query)
            for _, result_set in cur.fetchsets():
                results.append(result_set)
        return results

    def execute_priv_query(self, query):
        # Initialize a privileged connection.
        cnx = mysql.connector.connect(
            user=self.priv_user,
            password=self.priv_pwd,
            host=self.host,
            port=self.port,
            database=self.database,
            read_timeout=10,
        )

        results = []
        self.sent_queries.append(query)
        with cnx.cursor(buffered=True) as cur:
            cur.execute(query)
            for _, result_set in cur.fetchsets():
                results.append(result_set)
        return results

    def is_query_syntvalid(self, query: str) -> bool:
        if self.cnx is None or not self.cnx.is_connected():
            self.init_new_cnx()

        with self.cnx.cursor(buffered=True) as cursor:
            try:
                # Set maximum execution time of 10 sec.
                cursor.execute("SET SESSION MAX_EXECUTION_TIME=10000")
                cursor.execute(query, map_results=True)
            except mysql.connector.Error as e:
                if e.errno == 1064:
                    return False
        return True
