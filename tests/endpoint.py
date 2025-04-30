from http.server import BaseHTTPRequestHandler, HTTPServer
import mysql.connector
from urllib.parse import urlparse, parse_qs
import json

# Script to observe whether sqlmap can exploit vulnerable INSERT queries.


# Database configuration
DB_CONFIG = {
    "unix_socket": "/home/gquetel/tmp/scylla/mysqld_1/socket",
    "user": "root",
    "password": "root",  # Change to your actual password
    "database": "dataset",  # Change to your actual database name
}


class AirportInsertHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Parse URL and query parameters
        parsed_url = urlparse(self.path)

        # Check if the path is for airport insertion
        if parsed_url.path == "/insert_airport":
            query_params = parse_qs(parsed_url.query)

            # Extract parameters from the query string
            try:
                ident = query_params.get("ident", [""])[0]
                airport_type = query_params.get("type", [""])[0]
                name = query_params.get("name", [""])[0]
                iso_country = query_params.get("iso_country", [""])[0]
                iso_region = query_params.get("iso_region", [""])[0]
                keywords = query_params.get("keywords", [""])[0]

                # Validate required parameters
                if not all(
                    [ident, airport_type, name, iso_country, iso_region]
                ):
                    self.send_error(400, "Missing required parameters")
                    return

                # Insert into databas
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

                # Send successful response
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
        else:
            self.send_error(404, "Endpoint not found")

    def insert_airport(
        self, ident, airport_type, name, iso_country, iso_region, keywords
    ):
        """Insert airport data into the MySQL database"""
        try:
            conn = mysql.connector.connect(**DB_CONFIG)
            cursor = conn.cursor()

            # Build query by directly inserting values into the SQL string
            query = f"""
            INSERT INTO airport (ident, type, name, iso_country, iso_region, keywords) 
            VALUES ('{ident}', '{airport_type}', '{name}', '{iso_country}', '{iso_region}', '{keywords}')
            """

            # Execute the vulnerable query
            cursor.execute(query)

            # Commit the transaction
            conn.commit()

            res = ""
            res += str(cursor.fetchall())
            print(f"RES : {res}")
            while cursor.nextset():
                res += str(cursor.fetchall())

            # Close connections
            cursor.close()
            conn.close()
            return res

        except mysql.connector.Error as err:
            raise Exception(f"Database error: {err}")


def run_server(port=8080):
    """Start the HTTP server"""
    server_address = ("", port)
    httpd = HTTPServer(server_address, AirportInsertHandler)
    print(f"Starting server on port {port}...")
    httpd.serve_forever()


if __name__ == "__main__":
    run_server()
