import mysql.connector
import configparser
import mysql

from .config_parser import get_mysql_info, get_used_databases

class SQLConnector:
    def __init__(self, config: configparser.ConfigParser):
        user, pwd, socket_path = get_mysql_info(config=config)
        # self.db_name = get_used_databases(config=config)
        self.user = user
        self.pwd = pwd
        self.socket_path = socket_path
        self.database = "dataset"
        self.init_new_cnx()

        # Array of sent queries by self.execute_query
        self.sent_queries = []

    def init_new_cnx(self):
        self.cnx = mysql.connector.connect(
            user=self.user,
            password=self.pwd,
            unix_socket=self.socket_path,
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
