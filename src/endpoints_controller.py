from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
import pandas as pd
import threading
import logging

from .sql_connector import SQLConnector

logger = logging.getLogger(__name__)


class ServerManager:
    def __init__(
        self,
        templates: pd.DataFrame,
        sqlconnector: SQLConnector,
        port: int = 8080,
    ):
        self.port = port
        self.templates = templates
        self.sqlconnector = sqlconnector
        self.httpd = None

    def start_server(self):
        server_address = ("", self.port)

        SQLQueryHandler.set_context(self.templates, self.sqlconnector)
        httpd = ThreadingHTTPServer(server_address, SQLQueryHandler)

        # Using threading.Thread allows for non-blocking execution.
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True
        server_thread.start()

        self.httpd = httpd
        logger.info(f"Endpoints available at http://localhost:{self.port}/")

    def stop_server(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class SQLQueryHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _set_response(self):
        self.send_response(200)
        self.send_header("Content-type", "text/plain")
        self.end_headers()

    @classmethod
    def set_context(cls, templates, sqlconnector):
        cls.query_templates = templates.to_dict("records")
        cls.sqlconnector = sqlconnector

    def log_message(self, format, *args):
        logger.debug(
            f"{self.client_address[0]} - - [{self.log_date_time_string()}] {format % args}"
        )

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path.strip("/")
        query_params = parse_qs(parsed_url.query)

        # Find the query template that matches the path
        matching_template = None
        for template in self.query_templates:
            if template["ID"] == path:
                matching_template = template
                break

        if matching_template:
            self.process_query(matching_template, query_params)
        else:
            self._set_response()
            response = "Endpoint not found"
            self.wfile.write(bytes(response, "UTF-8"))

    def process_query(self, template, query_params):
        try:
            missing_params = []
            for param in template["placeholders"]:
                if param not in query_params:
                    missing_params.append(param)

            if missing_params:
                # This is most of a time a problem with the
                # template creation, this is not normal
                # raise exception.
                self._set_response()
                response = f"Missing required parameters: {', '.join(missing_params)}"
                self.wfile.write(bytes(response, "UTF-8"))
                raise ValueError(
                    f"Some request with missing parameters {missing_params} has been generated, this is abnormal."
                )
                return

            # Build the SQL query by replacing template parameters
            query = template["template"]
            for param in template["placeholders"]:
                param_value = query_params.get(param, [""])[0]
                query = query.replace(f"{{{param}}}", param_value)

            results = self.sqlconnector.execute_query(query)
            self._set_response()

            try:
                self.wfile.write(bytes(str(results), "UTF-8"))
            except BrokenPipeError as e:
                print(f"Broken Pipe error for query {query}")

        except Exception as e:
            # Mimic information leak. Required for several sqlmap techniques.
            self._set_response()
            try:
                self.wfile.write(bytes(str(e), "UTF-8"))
            except BrokenPipeError as e:
                print(f"Broken Pipe error for query {query}")
