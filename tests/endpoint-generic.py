from http.server import BaseHTTPRequestHandler, HTTPServer
import mysql.connector
from urllib.parse import urlparse, parse_qs
import os
import re
import csv
import threading

# Database configuration
DB_CONFIG = {
    "unix_socket": "/home/gquetel/tmp/scylla/mysqld_1/socket",
    "user": "root",
    "password": "root",
    "database": "dataset",
}


def load_query_templates(csv_file):
    """Load query templates from a CSV file"""
    templates = []

    with open(csv_file, "r") as f:
        reader = csv.reader(f)
        _ = next(reader)

        for row in reader:
            if len(row) < 6:
                continue

            template = row[0]
            query_id = row[1]
            description = row[2]
            payload_type = row[3]
            payload_clause = row[4]
            expected_column_number = int(row[5])

            # Extract parameter names from the template using regex
            param_names = re.findall(r"\{(!?)([a-zA-Z_]+)\}", template)
            params = [name for _, name in param_names]

            templates.append(
                {
                    "template": template,
                    "id": query_id,
                    "description": description,
                    "payload_type": payload_type,
                    "payload_clause": payload_clause,
                    "expected_column_number": expected_column_number,
                    "params": params,
                }
            )

    return templates


class SQLQueryHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        self.csv_queries_path = "queries.csv"

        self.check_and_create_csv()
        self.query_templates = load_query_templates(
            "../data/databases/airport/queries/select.csv"
        )
        super().__init__(*args, **kwargs)

    def check_and_create_csv(self):
        headers = [
            "full_query",
            "label",
            "template_id",
            "malicious_input",
            "malicious_input_desc",
        ]
        csv_path = self.csv_queries_path
        if not os.path.isfile(csv_path):
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
            print(
                f"Created new CSV file at {csv_path} with headers: {headers}"
            )
            return True
        else:
            print(f"CSV file already exists at {csv_path}")
            return False

    def log_query(self, query, template_id: int):
        """Log SQL query and whether it encountered an error"""
        # headers = full_query,label,template_id,malicious_input,malicious_input_desc
        label = 1
        malicious_input = "Unknown"
        malicious_input_desc = "Unknown"
        fields = [
            query,
            label,
            template_id,
            malicious_input,
            malicious_input_desc,
        ]

        with open(self.csv_queries_path, "a") as f:
            writer = csv.writer(f)
            writer.writerow(fields)

    def do_GET(self):
        parsed_url = urlparse(self.path)
        path = parsed_url.path.strip("/")
        query_params = parse_qs(parsed_url.query)

        # Find the query template that matches the path
        matching_template = None
        for template in self.query_templates:
            if template["id"] == path:
                matching_template = template
                break

        if matching_template:
            self.process_query(matching_template, query_params)
        else:
            self.send_response(404)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            response = "Endpoint not found"
            self.wfile.write(bytes(response), "UTF-8")

    def process_query(self, template, query_params):
        """Process a query template with the provided parameters"""
        try:
            # Check if all required parameters are present
            missing_params = []
            for param in template["params"]:
                if param not in query_params:
                    missing_params.append(param)

            if missing_params:
                self.send_response(400)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                response = (
                    f"Missing required parameters: {', '.join(missing_params)}"
                )
                self.wfile.write(bytes(response, "UTF-8"))
                return

            # Build the SQL query by replacing template parameters
            query = template["template"]
            for param in template["params"]:
                param_value = query_params.get(param, [""])[0]
                # Replace both {param} and {!param} patterns
                query = query.replace(f"{{!{param}}}", param_value)
                query = query.replace(f"{{{param}}}", param_value)

            # Execute the query
            try:
                results = self.execute_query(query)
                self.log_query(query=query, template_id=template["id"])

                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(bytes(str(results), "UTF-8"))

            except Exception as e:
                self.log_query(query=query, template_id=template["id"])
                self.send_response(200)
                self.send_header("Content-type", "text/plain")
                self.end_headers()
                self.wfile.write(bytes(str(e), "UTF-8"))

        except Exception as e:
            self.send_response(200)
            self.send_header("Content-type", "text/plain")
            self.end_headers()
            self.wfile.write(bytes(str(e), "UTF-8"))

    def execute_query(self, query):
        """Execute a SQL query and return the results"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            cursor.execute(query)
            results = cursor.fetchall()

            cursor.close()
            conn.close()

            return results

        except mysql.connector.Error as err:
            raise Exception(f"Database error: {err}")


def run_server(port=8080):
    """Start the HTTP server"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, SQLQueryHandler)
    print(f"Starting server on port {port}...")

    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True
    server_thread.start()

    return httpd


def invoke_sqlmap_instances():
    urls = ['"http://localhost:8080/airport-S1?airports_icao_code=AGBT"']
    settings = ["--technique=U --level=5 --risk=3 --schema --users --current-db -- tables  --batch --flush-session -u "]
    for url in urls:
        for setting in settings:
            command = "".join(["sqlmap ", setting, url])
            #os.popen(command)
            print(command)

if __name__ == "__main__":
    httpd = run_server()

    try:
        invoke_sqlmap_instances()
        input("Press Enter to stop the server...\n")
    finally:
        print("Shutting down server...")
        httpd.shutdown()
        httpd.server_close()
