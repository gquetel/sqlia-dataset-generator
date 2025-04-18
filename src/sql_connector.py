import mysql.connector
import configparser
import mysql


from .config_parser import get_mysql_info
class SQLConnector:
    def __init__(self, config: configparser.ConfigParser):
        user, pwd, socket_path = get_mysql_info(config=config)
        self.user = user
        self.pwd = pwd
        self.socket_path = socket_path

        self.init_new_cnx()

    def init_new_cnx(self):
        self.cnx = mysql.connector.connect(
            user=self.user, password=self.pwd, unix_socket=self.socket_path
        )

    def is_query_syntvalid(self, query: str) -> bool:
        if self.cnx is None or not self.cnx.is_connected():
            self.init_new_cnx()

        with self.cnx.cursor(buffered=True) as cursor:
            try:
                cursor.execute(query)
            except mysql.connector.Error as e:
                if e.errno == 1064:
                    return False
                else:
                    return True
