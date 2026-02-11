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
- Each dataset gets its own isolated MySQL instance (unique port + datadir)
- MySQL instances are started/stopped automatically via mysql-start / mysql-stop
- Focuses on smoke tests that catch initialization failures
- Easy to extend for new datasets
"""

import copy

import pytest
import mysql.connector
import toml
from pathlib import Path
from typing import List

from src.config_parser import get_dataset_port
from src.db_cnt_manager import (
    init_dataset_db,
    start_mysql_server,
    stop_mysql_server,
)

# All datasets under test.  Each one gets its own MySQL instance.
ALL_DATASETS = ["OurAirports", "OHR", "sakila", "AdventureWorks"]

# Port offsets used by the test suite (hardcoded, independent of config.toml).
# These must not collide with the dev server port (61337) or with each other.
_TEST_PORT_OFFSETS = {
    "OurAirports": 100,
    "OHR": 101,
    "sakila": 102,
    "AdventureWorks": 103,
}


# ============================================================================
# Pytest Fixtures
# ============================================================================


@pytest.fixture(scope="session", params=ALL_DATASETS)
def dataset_name(request):
    """Parametrize all dependent tests over every dataset."""
    return request.param


@pytest.fixture(scope="session")
def config():
    """Load configuration from config.toml."""
    config_path = Path("config.toml")
    assert config_path.exists(), "config.toml not found - run from repo root"
    return toml.load(config_path)


def _test_config(config: dict, dataset_name: str) -> dict:
    """Return a *copy* of config with port_offset set for the test suite.

    The test suite uses its own port offsets (starting at 100) so that test
    MySQL instances never collide with the development server.
    """
    cfg = copy.deepcopy(config)

    # Inject all datasets with test port offsets so that
    # get_dataset_port() can resolve any dataset_name.
    datasets_by_name = {d["name"]: d for d in cfg.get("datasets", [])}
    for name, offset in _TEST_PORT_OFFSETS.items():
        if name in datasets_by_name:
            datasets_by_name[name]["port_offset"] = offset
        else:
            # Dataset not in config.toml (commented out) – add a minimal entry.
            cfg.setdefault("datasets", []).append({"name": name, "port_offset": offset})

    return cfg


@pytest.fixture(scope="session")
def dataset_db(config, dataset_name):
    """Start an isolated MySQL instance, bootstrap it, source init_db.sql.

    Yields a dict with the patched config and connection parameters.
    Tears down (mysql-stop --clean) after all tests for this dataset_name.
    """
    cfg = _test_config(config, dataset_name)
    port = get_dataset_port(cfg, dataset_name)

    # Start a clean MySQL instance for this dataset.
    start_mysql_server(cfg, dataset_name)

    # Source the dataset schema.
    init_dataset_db(cfg, dataset_name)

    mysql_cfg = cfg["mysql"]
    priv_params = {
        "host": mysql_cfg["host"],
        "port": port,
        "user": mysql_cfg["priv_user"],
        "password": mysql_cfg["priv_pwd"],
    }
    unpriv_params = {
        "host": mysql_cfg["host"],
        "port": port,
        "user": mysql_cfg["user"],
        "password": mysql_cfg["password"],
    }

    yield {
        "config": cfg,
        "port": port,
        "priv_params": priv_params,
        "unpriv_params": unpriv_params,
    }

    stop_mysql_server(cfg, dataset_name)


@pytest.fixture
def mysql_connection(dataset_db):
    """Privileged MySQL connection scoped to the per-dataset instance."""
    cnx = mysql.connector.connect(**dataset_db["priv_params"])
    yield cnx
    cnx.close()


@pytest.fixture
def unprivileged_connection(dataset_db):
    """Unprivileged MySQL connection scoped to the per-dataset instance."""
    cnx = mysql.connector.connect(**dataset_db["unpriv_params"])
    yield cnx
    cnx.close()


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
    """Verify that we can connect to the per-dataset MySQL instance."""
    cursor = mysql_connection.cursor()
    cursor.execute("SELECT VERSION()")
    version = cursor.fetchone()[0]
    cursor.close()

    assert version is not None, "Failed to query MySQL version"
    assert version.startswith("8."), f"Expected MySQL 8.x, got {version}"


def test_unprivileged_user_exists(mysql_connection, dataset_db):
    """Verify that the unprivileged user from config exists."""
    cursor = mysql_connection.cursor()
    user = dataset_db["unpriv_params"]["user"]

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


def test_dataset_database_exists(mysql_connection, dataset_name):
    """Verify that the dataset database exists."""
    cursor = mysql_connection.cursor()
    exists = database_exists(cursor, dataset_name)
    cursor.close()

    assert exists, (
        f"Database '{dataset_name}' does not exist. "
        f"Check that:\n"
        f"  1. datasets/{dataset_name}/init_db.sql is valid SQL\n"
        f"  2. The dataset_db fixture ran successfully"
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


def test_unprivileged_user_has_access(mysql_connection, dataset_db, dataset_name):
    """Verify that the unprivileged user has privileges on the dataset."""
    cursor = mysql_connection.cursor()
    user = dataset_db["unpriv_params"]["user"]
    has_privs = user_has_privileges(cursor, user, dataset_name)
    cursor.close()

    assert has_privs, (
        f"User '{user}' lacks privileges on database '{dataset_name}'. "
        f"Check that datasets/{dataset_name}/init_db.sql includes:\n"
        f"  GRANT ALL PRIVILEGES ON {dataset_name}.* TO '{user}'@'localhost';\n"
        f"  FLUSH PRIVILEGES;"
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


def test_unprivileged_user_can_truncate(unprivileged_connection, dataset_name):
    """
    Verify that the unprivileged user can TRUNCATE tables.

    This is critical for attack generation: sqlia_generator._clean_db() runs
    TRUNCATE TABLE on every table between sqlmap runs. If this privilege is
    missing, no exploit samples are generated.
    """
    cursor = unprivileged_connection.cursor()
    table_name = "_test_truncate_check"
    try:
        cursor.execute(f"USE {dataset_name}")
        cursor.execute(f"CREATE TABLE {table_name} (id INT)")
        cursor.execute(f"INSERT INTO {table_name} VALUES (1)")
        cursor.execute(f"TRUNCATE TABLE {table_name}")
    except mysql.connector.Error as e:
        pytest.fail(
            f"User lacks TRUNCATE privilege on '{dataset_name}': {e}\n"
            f"Check that init_db.sql includes:\n"
            f"  GRANT ALL PRIVILEGES ON {dataset_name}.* TO '<user>'@'localhost';\n"
            f"  FLUSH PRIVILEGES;"
        )
    finally:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {dataset_name}.{table_name}")
        except mysql.connector.Error:
            pass
        cursor.close()


def test_unprivileged_user_can_dml(unprivileged_connection, dataset_name):
    """
    Verify that the unprivileged user can execute INSERT, UPDATE, DELETE.

    Sqlmap payloads may execute DML statements. The HTTP server runs queries
    via the unprivileged user, so these privileges are required for attack
    generation.
    """
    cursor = unprivileged_connection.cursor()
    table_name = "_test_dml_check"
    try:
        cursor.execute(f"USE {dataset_name}")
        cursor.execute(f"CREATE TABLE {table_name} (id INT, val VARCHAR(50))")
        cursor.execute(f"INSERT INTO {table_name} VALUES (1, 'test')")
        cursor.execute(f"UPDATE {table_name} SET val = 'updated' WHERE id = 1")
        cursor.execute(f"DELETE FROM {table_name} WHERE id = 1")
    except mysql.connector.Error as e:
        pytest.fail(
            f"User lacks DML privileges on '{dataset_name}': {e}\n"
            f"Check that init_db.sql includes:\n"
            f"  GRANT ALL PRIVILEGES ON {dataset_name}.* TO '<user>'@'localhost';\n"
            f"  FLUSH PRIVILEGES;"
        )
    finally:
        try:
            cursor.execute(f"DROP TABLE IF EXISTS {dataset_name}.{table_name}")
        except mysql.connector.Error:
            pass
        cursor.close()


def test_unprivileged_user_can_access_information_schema(
    unprivileged_connection, dataset_name
):
    """
    Verify that the unprivileged user can query information_schema.

    Sqlmap uses information_schema for enumeration (techniques U, E, B).
    This is essential for reconnaissance and exploit phases.
    """
    cursor = unprivileged_connection.cursor()
    try:
        cursor.execute(
            "SELECT TABLE_NAME FROM information_schema.TABLES "
            "WHERE TABLE_SCHEMA = %s LIMIT 1",
            (dataset_name,),
        )
        tables_result = cursor.fetchone()
        assert (
            tables_result is not None
        ), f"information_schema.TABLES returned no results for '{dataset_name}'"

        cursor.execute(
            "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA = %s LIMIT 1",
            (dataset_name,),
        )
        columns_result = cursor.fetchone()
        assert (
            columns_result is not None
        ), f"information_schema.COLUMNS returned no results for '{dataset_name}'"
    finally:
        cursor.close()


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


def test_init_files_exist_for_all_datasets(config):
    """Verify that each configured dataset has an init_db.sql file.

    Note: bootstrap.sql no longer sources these files directly. Databases
    are created on-demand by init_dataset_db() during generation and by
    the ``dataset_db`` test fixture.
    """
    datasets_dir = Path("data/datasets")

    for dataset_config in config["datasets"]:
        dataset_name = dataset_config["name"]
        init_file = datasets_dir / dataset_name / "init_db.sql"
        assert (
            init_file.exists()
        ), f"Dataset '{dataset_name}' is missing init_db.sql at {init_file}"


# ============================================================================
# Schema Sanity Checks
# ============================================================================


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


def fill_template(template: str, dicts: dict) -> str:
    """
    Fill a template string with deterministic values for testing.

    Uses the first value from each dictionary file, or hardcoded values
    for special placeholders.
    """
    import re

    SPECIAL_VALUES = {
        "rand_pos_number": "1000",
        "rand_medium_pos_number": "100",
        "rand_small_pos_number": "3",
        "rand_string": "testvalue",
        "conditions": "1=1",
    }

    def replacer(match):
        name = match.group(1)
        if name in SPECIAL_VALUES:
            return SPECIAL_VALUES[name]
        if name in dicts and dicts[name]:
            return dicts[name][0]
        return match.group(0)  # Leave unreplaced if no value found

    return re.sub(r"\{([-a-zA-Z_]+)\}", replacer, template)


def test_templates_execute_without_errors(
    unprivileged_connection, dataset_name, subtests
):
    """
    Verify that every CSV template can be filled and executed against the database.

    Fills each template with real dictionary values (first entry) and executes
    it inside a transaction that is always rolled back. This catches:
    - Wrong table names (e.g., dot-notation vs underscore)
    - Wrong column names or typos
    - Schema drift between init_db.sql and templates
    - SQL syntax errors in templates
    """
    import pandas as pd

    dataset_dir = Path("data/datasets") / dataset_name
    queries_dir = dataset_dir / "queries"
    dicts_dir = dataset_dir / "dicts"

    # Load all dictionary files
    dicts = {}
    for dict_file in dicts_dir.iterdir():
        if dict_file.is_file():
            lines = dict_file.read_text().strip().splitlines()
            dicts[dict_file.stem] = [line.strip() for line in lines if line.strip()]

    # Get all CSV template files
    csv_files = sorted(queries_dir.glob("*.csv"))

    cursor = unprivileged_connection.cursor()
    cursor.execute(f"USE {dataset_name}")

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)

        for _, row in df.iterrows():
            template = row["template"]
            template_id = row.get("ID", "unknown")

            filled_query = fill_template(template, dicts)

            with subtests.test(msg=f"{csv_file.stem}/{template_id}"):
                try:
                    cursor.execute("START TRANSACTION")
                    cursor.execute(filled_query)
                    # Consume any result set to avoid "Unread result found"
                    try:
                        cursor.fetchall()
                    except mysql.connector.errors.InterfaceError:
                        pass  # No result set (INSERT/UPDATE/DELETE)
                    cursor.execute("ROLLBACK")
                except mysql.connector.Error as e:
                    cursor.execute("ROLLBACK")
                    # Privilege errors (ER_ACCESS_DENIED_ERROR,
                    # ER_SPECIFIC_ACCESS_DENIED_ERROR, ER_KILL_DENIED_ERROR,
                    # etc.) are expected for admin templates executed by
                    # the unprivileged user.
                    TOLERATED_ERRNOS = {
                        # Privilege errors: expected for admin templates
                        # executed by the unprivileged user.
                        1044,  # ER_DBACCESS_DENIED_ERROR
                        1045,  # ER_ACCESS_DENIED_ERROR
                        1095,  # ER_KILL_DENIED_ERROR
                        1142,  # ER_TABLEACCESS_DENIED_ERROR
                        1143,  # ER_COLUMNACCESS_DENIED_ERROR
                        1227,  # ER_SPECIFIC_ACCESS_DENIED_ERROR
                        1370,  # ER_PROCACCESS_DENIED_ERROR
                        1410,  # ER_GRANT_WRONG_HOST_OR_USER (no perm)
                        # FK constraint violations: the test fills
                        # placeholders with the first dict entry, which
                        # may reference parent rows absent from the DB, or
                        # may try to delete rows that are referenced.
                        # The templates themselves are correct.
                        1451,  # ER_ROW_IS_REFERENCED (FK constraint child exists)
                        1452,  # ER_NO_REFERENCED_ROW_2 (FK constraint parent missing)
                        1094,  # Unknown thread id
                        # Duplicate key errors: the test fills placeholders
                        # with the first dict entry, which may already exist
                        # in the pre-populated database from insert.sql.
                        1062,  # ER_DUP_ENTRY (Duplicate entry for key)
                    }
                    if e.errno in TOLERATED_ERRNOS:
                        pass  # Expected for admin templates
                    # Special cases (maybe they could be treated another way?)
                    elif e.errno == 1064 and template_id == "AW-I22":
                        # We have a syntax error because the injected content is not
                        # correctly escaped. We have a dilmena:
                        # - We expect the user inputs to all be syntactically valid
                        # - Patching this query to have something syntactically valid
                        # means we modify application for normal query generation to
                        # make sure that no injection takes place, however we provide
                        # injection attacks samples. For now we allow this query to create
                        # syntactically invalid queries.
                        pass
                    else:
                        pytest.fail(
                            f"Template {template_id} ({csv_file.name}) failed:\n"
                            f"  Error: {e}\n"
                            f"  Filled query: {filled_query}"
                        )

    cursor.close()
