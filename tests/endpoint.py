from http.server import BaseHTTPRequestHandler, HTTPServer
import mysql.connector
from urllib.parse import urlparse, parse_qs
import json
import datetime
import os

# Script to observe whether sqlmap can exploit vulnerable INSERT queries.
# Modified to include logging of SQL queries and errors

# Database configuration
DB_CONFIG = {
    "unix_socket": "/home/gquetel/tmp/hydra/mysqld_1/socket",
    "user": "root",
    "password": "root",  # Change to your actual password
    "database": "dataset",  # Change to your actual database name
}

# Log file configuration
LOG_FILE = "sql_queries.log"


def log_query(query, error=None):
    """Log SQL query and whether it encountered an error"""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    log_dir = os.path.dirname(LOG_FILE)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if error == 1064:
        log_entry = f"[{timestamp}] QUERY UNDEFINED: {query}\n"
    elif error:
        log_entry = f"[{timestamp}] QUERY ERROR {error}: {query}\n"
    else:
        log_entry = f"[{timestamp}] QUERY SUCCESS: {query}\n"

    with open(LOG_FILE, "a") as f:
        f.write(log_entry)


class AirportInsertHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed_url = urlparse(self.path)

        if parsed_url.path == "/insert_airport":
            query_params = parse_qs(parsed_url.query)

            try:
                ident = query_params.get("ident", [""])[0]
                airport_type = query_params.get("type", [""])[0]
                name = query_params.get("name", [""])[0]
                iso_country = query_params.get("iso_country", [""])[0]
                iso_region = query_params.get("iso_region", [""])[0]
                keywords = query_params.get("keywords", [""])[0]

                if not all([ident, airport_type, name, iso_country, iso_region]):
                    self.send_error(400, "Missing required parameters")
                    return

                res = ""
                try:
                    res = self.insert_airport(
                        ident,
                        airport_type,
                        name,
                        iso_country,
                        iso_region,
                        keywords,
                    )
                except Exception as e:
                    self.send_error(500, str(e))
                    return

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"status": "success", "message": f"{res}"}
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(response).encode())

        elif parsed_url.path == "/search_airport":
            query_params = parse_qs(parsed_url.query)

            try:
                icao_code = query_params.get("icao_code", [""])[0]

                if not icao_code:
                    self.send_error(400, "Missing ICAO code parameter")
                    return

                try:
                    results = self.search_airport_by_icao(icao_code)
                except Exception as e:
                    self.send_error(500, str(e))
                    return

                self.send_response(200)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"status": "success", "data": results}
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_response(500)
                self.send_header("Content-type", "application/json")
                self.end_headers()
                response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(response).encode())
        else:
            self.send_error(404, "Endpoint not found")

    def insert_airport(
        self, ident, airport_type, name, iso_country, iso_region, keywords
    ):
        """Insert airport data into the MySQL database"""
        query = f"""
        INSERT INTO airport (ident, type, name, iso_country, iso_region, keywords) 
        VALUES ('{ident}', '{airport_type}', '{name}', '{iso_country}', '{iso_region}', '{keywords}')
        """

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            cursor.execute(query)

            conn.commit()

            res = ""
            res += str(cursor.fetchall())
            print(f"RES : {res}")
            while cursor.nextset():
                res += str(cursor.fetchall())

            log_query(query)

            cursor.close()
            conn.close()
            return res

        except mysql.connector.Error as err:
            log_query(query, error=str(err))
            raise Exception(f"Database error: {err}")

    def search_airport_by_icao(self, icao_code):
        """Search for airport by ICAO code with a vulnerable SQL query"""
        query = f"SELECT * FROM airport WHERE icao_code = '{icao_code}'"

        print(f"Executing query: {query}")

        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor(dictionary=True)

            cursor.execute(query)

            # Fetch all results
            results = cursor.fetchall()

            log_query(query)

            cursor.close()
            conn.close()

            return results

        except mysql.connector.Error as err:
            log_query(query, error=str(err.errno))
            raise Exception(f"Database error: {err}")


def run_server(port=8080):
    """Start the HTTP server"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, AirportInsertHandler)
    print(f"Starting server on port {port}...")
    print(f"Logging SQL queries to {os.path.abspath(LOG_FILE)}")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
