"""
Database Schema Validation Tests
Generated using Claude Code.

These tests validate that database initialization scripts (init_db.sql) create
databases with proper structure. Tests are intentionally simple and non-brittle:
- Check that databases exist
- Check that databases contain tables (not empty)
- Check that user permissions work
- Check that init_db.sql files are properly integrated

Test Strategy:
- Uses connection settings from config.toml (same as production code)
- Focuses on smoke tests that catch initialization failures
- Easy to extend for new datasets
"""

import pytest
import mysql.connector
import toml
from pathlib import Path
from typing import List


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="session")
def config():
    """
    Load configuration from config.toml.

    Note: If these tests fail due to connection issues, verify that:
    1. The nix shell is running (starts MySQL on port 61337)
    2. config.toml has correct MySQL connection settings
    3. bootstrap.sql has been executed (happens automatically in nix shell)
    """
    config_path = Path("config.toml")
    assert config_path.exists(), "config.toml not found - run from repo root"
    return toml.load(config_path)


@pytest.fixture(scope="session")
def mysql_config(config):
    """
    MySQL connection parameters from config.toml.

    Returns:
        dict: Connection parameters for root user
    """
    mysql_cfg = config["mysql"]
    return {
        "host": mysql_cfg["host"],
        "port": mysql_cfg["port"],
        "user": mysql_cfg["priv_user"],  # Use root for admin queries
        "password": mysql_cfg["priv_pwd"],
    }


@pytest.fixture(scope="session")
def unprivileged_mysql_config(config):
    """
    MySQL connection parameters for unprivileged user.
    """
    mysql_cfg = config["mysql"]
    return {
        "host": mysql_cfg["host"],
        "port": mysql_cfg["port"],
        "user": mysql_cfg["user"],  # tata
        "password": mysql_cfg["password"],
    }


@pytest.fixture(scope="session")
def mysql_connection(mysql_config):
    """
    Session-scoped MySQL connection for administrative queries.
    """
    cnx = mysql.connector.connect(**mysql_config)
    yield cnx
    cnx.close()


@pytest.fixture
def unprivileged_connection(unprivileged_mysql_config):
    """
    Test-scoped connection using the unprivileged user from config.
    """
    cnx = mysql.connector.connect(**unprivileged_mysql_config)
    yield cnx
    cnx.close()


@pytest.fixture(scope="session")
def dataset_names(config):
    """
    Extract all dataset names from config.toml.

    Returns:
        List[str]: Dataset names (also database names)
    """
    return [dataset["name"] for dataset in config["datasets"]]


# ============================================================================
# Helper Functions
# ============================================================================


def database_exists(cursor, database: str) -> bool:
    """Check if a database exists."""
    cursor.execute("SHOW DATABASES LIKE %s", (database,))
    return cursor.fetchone() is not None


def get_table_count(cursor, database: str) -> int:
    """Get number of tables in a database."""
    cursor.execute(
        "SELECT COUNT(*) FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'",
        (database,),
    )
    return cursor.fetchone()[0]


def get_table_names(cursor, database: str) -> List[str]:
    """Get all table names in a database."""
    cursor.execute(
        "SELECT TABLE_NAME FROM information_schema.TABLES "
        "WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE' "
        "ORDER BY TABLE_NAME",
        (database,),
    )
    return [row[0] for row in cursor.fetchall()]


def user_has_privileges(cursor, user: str, database: str) -> bool:
    """
    Check if a user has basic privileges on a database.

    Returns True if user has at least SELECT privilege.
    """
    cursor.execute(f"SHOW GRANTS FOR '{user}'@'localhost'")
    grants = [row[0].upper() for row in cursor.fetchall()]

    # Look for database-specific or ALL privileges
    # Handle both `database` and database formats, and uppercase DB name
    database_upper = database.upper()
    for grant in grants:
        # Check for: ON `Database`.* or ON Database.* or ON *.*
        if (
            f"ON `{database_upper}`.*" in grant
            or f"ON {database_upper}.*" in grant
            or "ON *.*" in grant
        ):
            if "ALL PRIVILEGES" in grant or "SELECT" in grant:
                return True
    return False


# ============================================================================
# Core Database Tests
# ============================================================================


def test_config_file_exists():
    """Verify config.toml exists and is readable."""
    config_path = Path("config.toml")
    assert config_path.exists(), "config.toml not found - run tests from repo root"


def test_mysql_connection(mysql_connection):
    """Verify that we can connect to MySQL using config.toml settings."""
    cursor = mysql_connection.cursor()
    cursor.execute("SELECT VERSION()")
    version = cursor.fetchone()[0]
    cursor.close()

    assert version is not None, "Failed to query MySQL version"
    # Expected MySQL 8.4.5 in nix shell, but don't enforce exact version
    assert version.startswith("8."), f"Expected MySQL 8.x, got {version}"


def test_unprivileged_user_exists(mysql_connection, unprivileged_mysql_config):
    """Verify that the unprivileged user from config exists."""
    cursor = mysql_connection.cursor()
    user = unprivileged_mysql_config["user"]

    cursor.execute(
        "SELECT 1 FROM mysql.user WHERE User = %s AND Host = 'localhost'", (user,)
    )
    result = cursor.fetchone()
    cursor.close()

    assert result is not None, f"User '{user}'@'localhost' does not exist"


def test_unprivileged_user_can_connect(unprivileged_connection):
    """Verify that the unprivileged user can connect and execute queries."""
    cursor = unprivileged_connection.cursor()
    cursor.execute("SELECT 1 AS test")
    result = cursor.fetchone()
    cursor.close()

    assert result[0] == 1, "Unprivileged user cannot execute basic queries"


@pytest.mark.parametrize(
    "dataset_name", ["OurAirports", "OHR", "sakila", "AdventureWorks"]
)
def test_dataset_database_exists(mysql_connection, dataset_name):
    """Verify that the dataset database exists."""
    cursor = mysql_connection.cursor()
    exists = database_exists(cursor, dataset_name)
    cursor.close()

    assert exists, (
        f"Database '{dataset_name}' does not exist. "
        f"Check that:\n"
        f"  1. nix shell is running (initializes databases)\n"
        f"  2. data/bootstrap.sql sources datasets/{dataset_name}/init_db.sql\n"
        f"  3. datasets/{dataset_name}/init_db.sql is valid SQL"
    )


@pytest.mark.parametrize(
    "dataset_name", ["OurAirports", "OHR", "sakila", "AdventureWorks"]
)
def test_dataset_has_tables(mysql_connection, dataset_name):
    """Verify that the dataset database contains tables (not empty)."""
    cursor = mysql_connection.cursor()
    table_count = get_table_count(cursor, dataset_name)
    cursor.close()

    assert table_count > 0, (
        f"Database '{dataset_name}' exists but contains no tables. "
        f"Check datasets/{dataset_name}/init_db.sql for errors."
    )


@pytest.mark.parametrize(
    "dataset_name", ["OurAirports", "OHR", "sakila", "AdventureWorks"]
)
def test_unprivileged_user_has_access(
    mysql_connection, unprivileged_mysql_config, dataset_name
):
    """Verify that the unprivileged user has privileges on the dataset."""
    cursor = mysql_connection.cursor()
    user = unprivileged_mysql_config["user"]
    has_privs = user_has_privileges(cursor, user, dataset_name)
    cursor.close()

    assert has_privs, (
        f"User '{user}' lacks privileges on database '{dataset_name}'. "
        f"Check that datasets/{dataset_name}/init_db.sql includes:\n"
        f"  GRANT ALL PRIVILEGES ON {dataset_name}.* TO '{user}'@'localhost';\n"
        f"  FLUSH PRIVILEGES;"
    )


@pytest.mark.parametrize(
    "dataset_name", ["OurAirports", "OHR", "sakila", "AdventureWorks"]
)
def test_unprivileged_user_can_query_dataset(unprivileged_connection, dataset_name):
    """Verify that unprivileged user can execute queries on the dataset."""
    cursor = unprivileged_connection.cursor()

    # Get a table name to query
    cursor.execute(f"USE {dataset_name}")
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()

    assert len(tables) > 0, f"No tables found in {dataset_name}"

    # Try a simple SELECT on the first table
    table_name = tables[0][0]
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    result = cursor.fetchone()

    cursor.close()

    # We don't care about the count value, just that the query succeeded
    assert result is not None, f"Failed to query {dataset_name}.{table_name}"


# ============================================================================
# Dataset Configuration Consistency
# ============================================================================


def test_config_datasets_have_init_files(config):
    """Verify that all datasets in config.toml have init_db.sql files."""
    datasets_dir = Path("data/datasets")

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        init_file = datasets_dir / dataset_name / "init_db.sql"

        assert init_file.exists(), (
            f"Dataset '{dataset_name}' is in config.toml but missing init_db.sql.\n"
            f"Expected file: {init_file}"
        )


def test_init_files_are_in_bootstrap(config):
    """Verify that bootstrap.sql sources all init_db.sql files from config."""
    bootstrap_file = Path("data/bootstrap.sql")
    assert bootstrap_file.exists(), "Missing data/bootstrap.sql"

    content = bootstrap_file.read_text()

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        expected_source = f"SOURCE ./datasets/{dataset_name}/init_db.sql"

        assert expected_source in content, (
            f"bootstrap.sql does not source {dataset_name}/init_db.sql\n"
            f"Add this line to data/bootstrap.sql:\n"
            f"  {expected_source};"
        )


# ============================================================================
# Schema Sanity Checks
# ============================================================================


@pytest.mark.parametrize(
    "dataset_name", ["OurAirports", "OHR", "sakila", "AdventureWorks"]
)
def test_dataset_has_reasonable_table_count(mysql_connection, dataset_name):
    """
    Verify that the dataset database is not empty.

    This catches catastrophic failures like init_db.sql failing to execute.
    """
    cursor = mysql_connection.cursor()
    table_count = get_table_count(cursor, dataset_name)
    cursor.close()

    assert table_count > 0, (
        f"Database '{dataset_name}' is empty (no tables). "
        f"Check if init_db.sql executed successfully."
    )


# ============================================================================
# Template Placeholder Validation
# ============================================================================


def extract_placeholders(template: str) -> List[str]:
    """
    Extract placeholder names from a template string.

    Matches the logic from src/dataset_builder.py::_extract_params.
    Returns unique placeholder names (without suffixes for duplicates).
    """
    import re

    param_names = re.findall(r"\{([-a-zA-Z_]+)\}", template)
    return list(set(param_names))  # Return unique placeholders


@pytest.mark.parametrize(
    "dataset_name", ["OurAirports", "OHR", "sakila", "AdventureWorks"]
)
def test_template_placeholders_are_valid(dataset_name):
    """
    Verify that all placeholders in template CSV files are either:
    1. Valid filenames in the dataset's dicts/ directory
    2. Special placeholders: rand_string, rand_small_pos_number,
       rand_medium_pos_number, rand_pos_number

    This prevents typos and missing dictionary files that would cause
    runtime errors during dataset generation.
    """
    import pandas as pd

    # Define special placeholders (from src/dataset_builder.py::fill_placeholder)
    SPECIAL_PLACEHOLDERS = {
        "rand_string",
        "rand_small_pos_number",
        "rand_medium_pos_number",
        "rand_pos_number",
        "conditions",
    }

    dataset_dir = Path("data/datasets") / dataset_name
    queries_dir = dataset_dir / "queries"
    dicts_dir = dataset_dir / "dicts"

    # Ensure directories exist
    assert dataset_dir.exists(), f"Dataset directory not found: {dataset_dir}"
    assert queries_dir.exists(), f"Queries directory not found: {queries_dir}"
    assert dicts_dir.exists(), f"Dicts directory not found: {dicts_dir}"

    # Get all dictionary files (without extension)
    dict_files = {f.stem for f in dicts_dir.iterdir() if f.is_file()}

    # Get all CSV template files
    csv_files = list(queries_dir.glob("*.csv"))
    assert len(csv_files) > 0, f"No CSV template files found in {queries_dir}"

    # Collect all invalid placeholders
    invalid_placeholders = {}

    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            assert False, f"Failed to read {csv_file}: {e}"

        assert "template" in df.columns, (
            f"CSV file {csv_file.name} missing 'template' column. "
            f"Available columns: {df.columns.tolist()}"
        )

        for idx, row in df.iterrows():
            template = row["template"]
            template_id = row.get("ID", f"row_{idx}")

            placeholders = extract_placeholders(template)

            for placeholder in placeholders:
                # Check if placeholder is valid
                is_dict_file = placeholder in dict_files
                is_special = placeholder in SPECIAL_PLACEHOLDERS

                if not (is_dict_file or is_special):
                    # Track invalid placeholder
                    if placeholder not in invalid_placeholders:
                        invalid_placeholders[placeholder] = []
                    invalid_placeholders[placeholder].append(
                        f"{csv_file.name}:{template_id}"
                    )

    # Assert no invalid placeholders found
    if invalid_placeholders:
        error_msg = f"Invalid placeholders found in {dataset_name} templates:\n\n"
        for placeholder, locations in sorted(invalid_placeholders.items()):
            error_msg += f"  '{placeholder}' used in:\n"
            for loc in locations:
                error_msg += f"    - {loc}\n"

        error_msg += (
            f"\nValid placeholders are:\n"
            f"  - Files in {dicts_dir}/ (without extension)\n"
            f"  - Special placeholders: {', '.join(sorted(SPECIAL_PLACEHOLDERS))}\n"
        )

        assert False, error_msg
