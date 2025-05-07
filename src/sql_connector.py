import mysql.connector
import configparser
import mysql

from .config_parser import get_mysql_info, get_used_databases

class SQLConnector:
    def __init__(self, config: configparser.ConfigParser):
        user, pwd, socket_path = get_mysql_info(config=config)
        self.db_name = get_used_databases(config=config)
        self.user = user
        self.pwd = pwd
        self.socket_path = socket_path
        self.init_new_cnx()

    def init_new_cnx(self):
        self.cnx = mysql.connector.connect(
            user=self.user,
            password=self.pwd,
            unix_socket=self.socket_path,
            database="dataset",
        )

    def execute_query(self, query):
        # https://dev.mysql.com/doc/connector-python/en/connector-python-multi.html
        if self.cnx is None or not self.cnx.is_connected():
            self.init_new_cnx()
        try:            
            results = []
            with self.cnx.cursor(buffered=True) as cur:
                cur.execute(query)
                for _, result_set in cur.fetchsets():
                    results.append(result_set)
            return results

        except mysql.connector.Error as err:
            raise Exception(f"Database error: {err}")
        
    def is_query_syntvalid(self, query: str) -> bool:
        if self.cnx is None or not self.cnx.is_connected():
            self.init_new_cnx()

        with self.cnx.cursor(buffered=True) as cursor:
            try:
                # Set maximum execution time of 2 sec. 
                # When query hang -> They are executed, return True
                
                cursor.execute("SET SESSION MAX_EXECUTION_TIME=2000")
                cursor.execute(query, map_results=True)
            except mysql.connector.Error as e:
                if e.errno == 1064:
                    return False
        return True
