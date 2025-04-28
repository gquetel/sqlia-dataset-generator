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
        # TODO: Find a way to asses syntactic validity of whole multi-statements.
        if self.cnx is None or not self.cnx.is_connected():
            self.init_new_cnx()

        with self.cnx.cursor(buffered=True) as cursor:
            r = None
            try:
                r = cursor.execute(query, multi=True)
            except mysql.connector.Error as e:
                if e.errno == 1064:
                    return False

            while r != None:
                # Iterate over components of generator if any.
                try:
                    next(r)
                except StopIteration:
                    # Exhaustion
                    return True

                except mysql.connector.connection_cext.MySQLInterfaceError as e:
                    # Error in on of the multiple statement
                    if "You have an error in your SQL syntax;" in e.msg:
                        return False
        return True
