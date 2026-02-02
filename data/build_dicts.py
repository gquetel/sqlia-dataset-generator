#!/usr/bin/env python3
"""
Standalone script to extract dictionary files from a given MySQL database.

This script connects to the database, discovers all tables and columns,
and extracts unique values from each column into separate dictionary files.

Output: One file per column in ./dicts/{table}_{column} format.
Each file contains one unique value per line.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import mysql.connector

# ============================================================================
# HARDCODED CONFIGURATION
# ============================================================================

MYSQL_HOST = "localhost"
MYSQL_PORT = 61337
MYSQL_USER = "tata"
MYSQL_PASSWORD = "tata"
MYSQL_DATABASE = "AdventureWorks"

OUTPUT_DIR = "./dicts/"

# Column types to skip (binary/complex data not useful for SQL templates)
DEFAULT_SKIP_TYPES = [
    "BLOB",
    "BINARY",
    "JSON",
    "GEOMETRY",
    "VARBINARY",
    "MEDIUMBLOB",
    "LONGBLOB",
    "TINYBLOB",
]

# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================


def get_all_tables(conn):
    """
    Get list of all tables in the database.

    Args:
        conn: MySQL connection object

    Returns:
        list: Table names
    """
    cursor = conn.cursor()
    cursor.execute("SHOW TABLES")
    tables = [row[0] for row in cursor.fetchall()]
    cursor.close()
    return tables


def get_table_columns(conn, table):
    """
    Get all columns and their types for a given table.

    Args:
        conn: MySQL connection object
        table: Table name

    Returns:
        list: List of tuples (column_name, data_type)
    """
    cursor = conn.cursor()
    query = """
        SELECT COLUMN_NAME, DATA_TYPE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
    """
    cursor.execute(query, (MYSQL_DATABASE, table))
    columns = cursor.fetchall()
    cursor.close()
    return columns


def should_skip_column(column_type, skip_types):
    """
    Check if a column type should be skipped.

    Args:
        column_type: MySQL data type (e.g., 'VARCHAR', 'INT')
        skip_types: List of types to skip

    Returns:
        bool: True if column should be skipped
    """
    return column_type.upper() in [t.upper() for t in skip_types]


def extract_column_values(conn, table, column, limit=None):
    """
    Extract unique values from a column.

    Args:
        conn: MySQL connection object
        table: Table name
        column: Column name
        limit: Optional limit on number of values to extract

    Yields:
        str: Unique values from the column
    """
    cursor = conn.cursor()

    # Use backticks to handle reserved keywords and special characters
    query = f"SELECT DISTINCT `{column}` FROM `{table}` WHERE `{column}` IS NOT NULL"

    if limit:
        query += f" LIMIT {limit}"

    try:
        cursor.execute(query)
        for row in cursor:
            # Convert to string, handle different data types
            value = row[0]
            if value is not None:
                yield str(value)
    finally:
        cursor.close()


def write_dict_file(output_dir, table, column, values, overwrite=False):
    """
    Write dictionary values to a file.

    Args:
        output_dir: Output directory path
        table: Table name
        column: Column name
        values: Iterable of values to write
        overwrite: Whether to overwrite existing files

    Returns:
        int: Number of values written
    """
    filename = f"{table}_{column}"
    filepath = os.path.join(output_dir, filename)

    # Check if file exists and overwrite is False
    if os.path.exists(filepath) and not overwrite:
        logging.info(
            f"  - Skipping {table}.{column} (file exists, use --overwrite to replace)"
        )
        return 0

    count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for value in values:
            f.write(f"{value}\n")
            count += 1

    return count


# ============================================================================
# MAIN EXTRACTION LOGIC
# ============================================================================


def extract_dictionaries(
    conn, output_dir, skip_types, limit=None, overwrite=False, dry_run=False
):
    """
    Main function to extract all dictionaries from the database.

    Args:
        conn: MySQL connection object
        output_dir: Output directory path
        skip_types: List of column types to skip
        limit: Optional limit on values per column
        overwrite: Whether to overwrite existing files
        dry_run: If True, show what would be done without writing files
    """
    # Get all tables
    tables = get_all_tables(conn)
    logging.info(f"Found {len(tables)} tables in database '{MYSQL_DATABASE}'")

    if dry_run:
        logging.info("DRY RUN MODE - No files will be written")

    # Statistics
    total_tables = 0
    total_columns_extracted = 0
    total_columns_skipped = 0
    total_values = 0
    skipped_by_type = {}

    # Process each table
    for table in tables:
        columns = get_table_columns(conn, table)
        logging.info(f"Processing table '{table}' ({len(columns)} columns)")

        total_tables += 1

        for column_name, column_type in columns:
            # Check if we should skip this column type
            if should_skip_column(column_type, skip_types):
                logging.info(f"  - Skipping {table}.{column_name} ({column_type})")
                total_columns_skipped += 1
                skipped_by_type[column_type] = skipped_by_type.get(column_type, 0) + 1
                continue

            # Extract values
            try:
                values_list = list(
                    extract_column_values(conn, table, column_name, limit)
                )

                if not values_list:
                    logging.info(
                        f"  - {table}.{column_name}: 0 values (empty/NULL only)"
                    )
                    continue

                # Write to file (unless dry run)
                if dry_run:
                    logging.info(
                        f"  - {table}.{column_name}: {len(values_list)} values → {table}_{column_name} (DRY RUN)"
                    )
                    total_values += len(values_list)
                else:
                    count = write_dict_file(
                        output_dir, table, column_name, values_list, overwrite
                    )
                    if count > 0:
                        logging.info(
                            f"  - {table}.{column_name}: {count} values → {table}_{column_name}"
                        )
                        total_columns_extracted += 1
                        total_values += count

            except mysql.connector.Error as e:
                logging.warning(f"  - Failed to extract {table}.{column_name}: {e}")
                continue

    # Print summary statistics
    logging.info("")
    logging.info("=" * 60)
    logging.info("Extraction complete!")
    logging.info("=" * 60)
    logging.info(f"Tables processed: {total_tables}")
    logging.info(f"Columns extracted: {total_columns_extracted}")
    logging.info(f"Columns skipped: {total_columns_skipped}")

    if skipped_by_type:
        logging.info("Skipped by type:")
        for col_type, count in sorted(skipped_by_type.items()):
            logging.info(f"  - {col_type}: {count}")

    logging.info(f"Total unique values: {total_values:,}")

    if not dry_run:
        logging.info(f"Output directory: {os.path.abspath(output_dir)}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Extract dictionary files from a MySQL database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be extracted
  python build_dicts.py --dry-run

  # Extract all dictionaries
  python build_dicts.py

  # Extract with verbose logging
  python build_dicts.py --verbose

  # Limit to 1000 values per column
  python build_dicts.py --limit 1000

  # Overwrite existing files
  python build_dicts.py --overwrite
        """,
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of unique values per column (default: unlimited)",
    )

    parser.add_argument(
        "--skip-types",
        type=str,
        default=None,
        help=f"Comma-separated list of MySQL types to skip (default: {','.join(DEFAULT_SKIP_TYPES)})",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dictionary files (default: skip existing)",
    )

    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be extracted without writing files",
    )

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    # Parse skip types
    skip_types = DEFAULT_SKIP_TYPES
    if args.skip_types:
        skip_types = [t.strip() for t in args.skip_types.split(",")]

    # Create output directory if it doesn't exist
    if not args.dry_run:
        os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Connect to database
    logging.info(
        f"Connecting to database '{MYSQL_DATABASE}' on {MYSQL_HOST}:{MYSQL_PORT}"
    )

    try:
        conn = mysql.connector.connect(
            host=MYSQL_HOST,
            port=MYSQL_PORT,
            user=MYSQL_USER,
            password=MYSQL_PASSWORD,
            database=MYSQL_DATABASE,
        )
    except mysql.connector.Error as e:
        logging.error(f"Failed to connect to database: {e}")
        logging.error(
            f"Check that MySQL is running and database '{MYSQL_DATABASE}' exists"
        )
        sys.exit(1)

    try:
        # Extract dictionaries
        extract_dictionaries(
            conn=conn,
            output_dir=OUTPUT_DIR,
            skip_types=skip_types,
            limit=args.limit,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    finally:
        conn.close()
        logging.info("Database connection closed")


if __name__ == "__main__":
    main()
