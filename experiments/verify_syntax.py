"""
Verify syntax validity of normal queries in a generated dataset CSV.

Reads the CSV, filters to normal queries (label=0), starts a MySQL instance
with all needed database schemas loaded, and validates each query using
SQLConnector.is_query_syntvalid().

Usage:
    python3 verify_syntax.py <csv_path> [--config-file config.toml] [--verbose]
"""

import argparse
import csv
import os
import sys
from pathlib import Path
from subprocess import PIPE, Popen

import toml

# Resolve paths relative to this script (experiments/) → repo root.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

# Append project root so we can import src modules.
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_parser import get_mysql_info
from src.db_cnt_manager import SQLConnector


# -- MySQL helpers (adapted from src/db_cnt_manager.py) -----------------------

INSTANCE_NAME = "syntax-check"


def _mysql_env(port: int) -> dict:
    """Build env vars for mysql-start/mysql-stop with a dedicated instance."""
    mysql_home = f"/tmp/mysql-{INSTANCE_NAME}"
    env = os.environ.copy()
    env["MYSQL_HOME"] = mysql_home
    env["MYSQL_DATADIR"] = f"{mysql_home}/data"
    env["MYSQL_UNIX_PORT"] = f"{mysql_home}/mysql.sock"
    env["MYSQL_PORT"] = str(port)
    env["PATH"] = f"{SCRIPTS_DIR}:{env.get('PATH', '')}"
    return env


def start_mysql(port: int) -> None:
    env = _mysql_env(port)

    # Clean any leftover state.
    proc = Popen(
        [str(SCRIPTS_DIR / "mysql-stop"), "--clean"],
        env=env,
        stdout=PIPE,
        stderr=PIPE,
    )
    proc.communicate()

    # Start fresh instance.
    proc = Popen(
        [str(SCRIPTS_DIR / "mysql-start")],
        env=env,
        stdout=PIPE,
        stderr=PIPE,
    )
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"mysql-start failed:\n{stderr.decode()}\n{stdout.decode()}")


def stop_mysql(port: int) -> None:
    env = _mysql_env(port)
    proc = Popen(
        [str(SCRIPTS_DIR / "mysql-stop"), "--clean"],
        env=env,
        stdout=PIPE,
        stderr=PIPE,
    )
    proc.communicate()


def source_init_db(port: int, priv_user: str, priv_pwd: str, dataset_name: str) -> None:
    """Source a dataset's init_db.sql into the running MySQL instance."""
    sql_path = DATASETS_DIR / dataset_name / "init_db.sql"
    if not sql_path.is_file():
        raise FileNotFoundError(f"Expected SQL file not found: {sql_path}")

    command = (
        f"mysql --port={port} --protocol=tcp "
        f"-u {priv_user} -p{priv_pwd} < {sql_path}"
    )
    proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to source {sql_path}: {stderr.decode()}")


# -- Template ID → dataset name mapping ---------------------------------------


def build_template_prefix_map() -> dict[str, str]:
    """Build a mapping from template ID prefix to dataset directory name.

    Scans all CSV template files under data/datasets/*/queries/ and extracts
    the unique prefixes from the ID column.  For example, "airport-S1" yields
    prefix "airport" mapped to dataset "OurAirports".
    """
    prefix_map: dict[str, str] = {}
    for dataset_dir in sorted(DATASETS_DIR.iterdir()):
        queries_dir = dataset_dir / "queries"
        if not queries_dir.is_dir():
            continue
        for csv_path in queries_dir.glob("*.csv"):
            with open(csv_path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    template_id = row.get("ID", "")
                    if "-" in template_id:
                        prefix = template_id.split("-")[0]
                        if prefix not in prefix_map:
                            prefix_map[prefix] = dataset_dir.name
    return prefix_map


def get_dataset_for_template(
    template_id: str, prefix_map: dict[str, str]
) -> str | None:
    """Return the dataset name for a given template ID."""
    if "-" in template_id:
        prefix = template_id.split("-")[0]
        return prefix_map.get(prefix)
    return None


# -- Main ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Verify syntax of normal queries in a generated dataset CSV."
    )
    parser.add_argument("csv_path", help="Path to generated dataset CSV")
    parser.add_argument(
        "--config-file",
        default=str(PROJECT_ROOT / "config.toml"),
        help="Path to config.toml (default: ../config.toml relative to script)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print each failing query",
    )
    args = parser.parse_args()

    # Load config for MySQL credentials.
    config = toml.load(args.config_file)
    user, pwd, host, port, priv_user, priv_pwd = get_mysql_info(config)

    # Read CSV.
    print(f"Reading {args.csv_path} ...")
    import pandas as pd

    df = pd.read_csv(args.csv_path, low_memory=False)
    normal_df = df[df["label"] == 0]
    print(f"  Total rows: {len(df)}, normal queries (label=0): {len(normal_df)}")

    if len(normal_df) == 0:
        print("No normal queries to check.")
        return

    # Build template prefix → dataset mapping.
    prefix_map = build_template_prefix_map()

    # Identify needed datasets.
    needed_datasets: set[str] = set()
    for tid in normal_df["query_template_id"].unique():
        ds = get_dataset_for_template(tid, prefix_map)
        if ds is None:
            print(f"  WARNING: cannot map template '{tid}' to a dataset, skipping")
        else:
            needed_datasets.add(ds)

    if not needed_datasets:
        print("ERROR: Could not identify any datasets from template IDs.")
        sys.exit(1)

    print(f"  Datasets needed: {sorted(needed_datasets)}")

    # Start MySQL and load schemas.
    print(f"\nStarting MySQL instance '{INSTANCE_NAME}' on port {port} ...")
    start_mysql(port)

    try:
        # Source bootstrap.sql first (creates unprivileged user).
        bootstrap_path = PROJECT_ROOT / "data" / "bootstrap.sql"
        if bootstrap_path.is_file():
            command = (
                f"mysql --port={port} --protocol=tcp "
                f"-u {priv_user} -p{priv_pwd} < {bootstrap_path}"
            )
            proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE)
            _, stderr = proc.communicate()
            if proc.returncode != 0:
                raise RuntimeError(f"Failed to source bootstrap.sql: {stderr.decode()}")
            print("  Sourced bootstrap.sql")

        # Source each needed dataset's init_db.sql.
        for ds_name in sorted(needed_datasets):
            print(f"  Loading schema for '{ds_name}' ...")
            source_init_db(port, priv_user, priv_pwd, ds_name)

        # Create one SQLConnector per dataset.
        config_override = {
            "mysql": {
                "user": user,
                "password": pwd,
                "host": host,
                "port": port,
                "priv_user": priv_user,
                "priv_pwd": priv_pwd,
            }
        }
        connectors: dict[str, SQLConnector] = {}
        for ds_name in sorted(needed_datasets):
            connectors[ds_name] = SQLConnector(config_override, database=ds_name)

        # Validate each normal query.
        print(f"\nValidating {len(normal_df)} normal queries ...")
        passed = 0
        failed = 0
        failures: list[tuple[str, str, str]] = []  # (template_id, query, dataset)

        for _, row in normal_df.iterrows():
            tid = row["query_template_id"]
            query = row["full_query"]
            ds = get_dataset_for_template(tid, prefix_map)
            if ds is None or ds not in connectors:
                continue

            if connectors[ds].is_query_syntvalid(query):
                passed += 1
            else:
                failed += 1
                failures.append((tid, query, ds))

        # Report results.
        total = passed + failed
        print(f"\nResults: {total} checked, {passed} passed, {failed} failed")
        if failed > 0:
            print(f"\n--- {failed} FAILING QUERIES ---")
            for tid, query, ds in failures:
                print(f"  [{ds}] {tid}")
                if args.verbose:
                    print(f"    {query}")
            if not args.verbose:
                print("\n  (use --verbose to see full queries)")
            sys.exit(1)
        else:
            print("All normal queries are syntactically valid.")

    finally:
        print(f"\nStopping MySQL instance '{INSTANCE_NAME}' ...")
        stop_mysql(port)


if __name__ == "__main__":
    main()
