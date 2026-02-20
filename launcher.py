import argparse
import logging
import os
from pathlib import Path
import sys
import tomllib

import pandas as pd

from src.dataset_builder import DatasetBuilder
from src.config_parser import get_dataset_port
from src.db_cnt_manager import (
    init_dataset_db,
    start_mysql_server,
    stop_mysql_server,
)

logger = logging.getLogger(__name__)


def init_logging(debug_mode: bool):
    Path("./logs/").mkdir(exist_ok=True, parents=True)

    # We usually do not need the logs of previous generation
    # -> mode = 'w'
    lf = logging.FileHandler("./logs/generation.log", "w")

    lf.setLevel(logging.DEBUG)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=logging.DEBUG, handlers=[lf, lstdo])


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Creates a dataset of SQL queries specific to a domain, containing both normal queries and queries with injections attacks."
    )

    parser.add_argument(
        "--testing",
        action="store_true",
        help="Enable testing mode, for fast generation of a smaller dataset.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode, output will be VERY verbose.",
    )

    parser.add_argument(
        "--no-syn-check",
        action="store_true",
        help="The correct syntax of normal queries will not be verified, this speed up their generation.",
    )

    parser.add_argument(
        "--ithreat-only",
        action="store_true",
        help="Only generate insider threat queries.",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        dest="config_file",
        default="config.toml",
        help="Filepath to the dataset generation configuration file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        default="./output/",
        help="Filepath to the directory that will contain all generated datasets.",
    )

    return parser.parse_args()


def init_toml_config(args: argparse.Namespace) -> dict:
    """Load and parse the TOML configuration file."""
    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    return config


def validate_datasets_config(config: dict):
    """Validate that all dataset names and their statement sources are properly configured.

    New behavior: Accepts SQL files with template annotations OR CSV templates OR both.

    Args:
        config: The loaded TOML configuration

    Raises:
        ValueError: If any dataset name doesn't have a corresponding folder,
                   or if neither SQL file nor CSV template exists for a statement type,
                   or if template annotations are malformed
    """
    # Validate required general config fields
    import src.config_parser as config_parser
    import re

    try:
        config_parser.get_attacks_ratio(config)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Configuration validation failed: {e}")

    try:
        config_parser.get_normal_only_template_ratio(config)
    except (KeyError, ValueError) as e:
        raise ValueError(f"Configuration validation failed: {e}")

    datasets_dir = Path("data/datasets")

    if not datasets_dir.exists():
        raise ValueError(f"Dataset directory not found.")

    available_folders = {
        folder.name for folder in datasets_dir.iterdir() if folder.is_dir()
    }

    configured_datasets = config.get("datasets", [])

    if not configured_datasets:
        raise ValueError("No datasets configured in configuration file")

    # Check for missing folders and validate statement sources (SQL and/or CSV)
    for dataset_config in configured_datasets:
        dataset_name = dataset_config.get("name")

        if dataset_name not in available_folders:
            raise ValueError(
                f"Dataset folder not found in {datasets_dir}: {dataset_name}\n"
                f"Available folders: {', '.join(sorted(available_folders)) if available_folders else '(none)'}"
            )

        # Validate statement sources (SQL files and/or CSV templates)
        statements = dataset_config.get("statements", {})
        if not statements:
            raise ValueError(f"No statements configured for dataset '{dataset_name}'")

        dataset_dir = datasets_dir / dataset_name
        queries_dir = dataset_dir / "queries"

        for statement_name in statements.keys():
            sql_file = dataset_dir / f"{statement_name}.sql"
            csv_file = queries_dir / f"{statement_name}.csv"

            sql_exists = sql_file.exists()

            if not csv_file.exists():
                raise ValueError(
                    f"Statement CSV file not found for dataset '{dataset_name}': {csv_file}\n"
                    f"Expected file for statement '{statement_name}'"
                )

            sql_template_ids = set()
            csv_template_ids = set()

            # Validate SQL file if it exists
            if sql_exists:
                try:
                    with open(sql_file, "r") as f:
                        content = f.read()

                    # Validate it contains appropriate SQL keywords
                    statement_keyword = statement_name.upper()
                    if statement_keyword not in content.upper():
                        raise ValueError(
                            f"SQL file for statement '{statement_name}' does not contain "
                            f"expected keyword '{statement_keyword}': {sql_file}"
                        )

                    # Parse and validate template annotations
                    # Pattern: SQL statement ending with '; -- TEMPLATE-ID'
                    # Supports formats: 'OHR-I-1', 'airport-S1', 'airport-admin1'
                    pattern = r";[\s]*--[\s]*([A-Za-z]+-(?:[A-Z]-?|admin)\d+)"
                    matches = re.findall(pattern, content)

                    if matches:
                        # Validate template ID format
                        id_pattern = r"^[A-Za-z]+-(?:[A-Z]-?|admin)\d+$"
                        for template_id in matches:
                            if not re.match(id_pattern, template_id):
                                raise ValueError(
                                    f"Invalid template ID format in {sql_file}: '{template_id}'. "
                                    f"Expected format: {{dataset}}-{{Type}}[-]{{number}} (e.g., OHR-I-1, airport-S1, or airport-admin1)"
                                )
                            sql_template_ids.add(template_id)

                        logger.info(
                            f"Found {len(matches)} annotated statements in {sql_file} "
                            f"with {len(sql_template_ids)} unique template types"
                        )
                    else:
                        logger.warning(
                            f"SQL file {sql_file} exists but contains no template annotations. "
                            f"Consider adding annotations in format: '; -- {{TEMPLATE-ID}}'"
                        )

                except Exception as e:
                    if isinstance(e, ValueError):
                        raise
                    raise ValueError(f"Error reading SQL file '{sql_file}': {e}")

            # Validate CSV file if it exists

            try:
                import pandas as pd

                df = pd.read_csv(csv_file)

                # Validate required columns
                required_columns = ["template", "ID"]
                missing_columns = [
                    col for col in required_columns if col not in df.columns
                ]
                if missing_columns:
                    raise ValueError(
                        f"Template CSV missing required columns for dataset '{dataset_name}': {csv_file}\n"
                        f"Missing columns: {missing_columns}\n"
                        f"Required columns: {required_columns}"
                    )

                # Check if CSV has at least one entry
                if len(df) < 1:
                    raise ValueError(
                        f"Statement CSV file is empty for dataset '{dataset_name}': {csv_file}\n"
                        f"File must contain at least one query template"
                    )

                # Validate template IDs in CSV
                id_pattern = r"^[A-Za-z]+-(?:[A-Z]-?|admin)\d+$"
                for template_id in df["ID"]:
                    if not re.match(id_pattern, str(template_id)):
                        raise ValueError(
                            f"Invalid template ID format in {csv_file}: '{template_id}'. "
                            f"Expected format: {{dataset}}-{{Type}}[-]{{number}} (e.g., OHR-I-1, airport-S1, or airport-admin1)"
                        )
                    csv_template_ids.add(template_id)

                logger.info(
                    f"Found {len(df)} templates in {csv_file} "
                    f"with {len(csv_template_ids)} unique template IDs"
                )

            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"Error reading template CSV file '{csv_file}': {e}")

            # If both exist, check for template ID mismatches
            if sql_exists and sql_template_ids:
                missing_in_csv = sql_template_ids - csv_template_ids
                if missing_in_csv:
                    logger.warning(
                        f"Template IDs in SQL annotations don't match CSV template IDs for {statement_name}:\n"
                        f"  - SQL references: {sorted(sql_template_ids)}\n"
                        f"  - CSV contains: {sorted(csv_template_ids)}\n"
                        f"  - Missing in CSV: {sorted(missing_in_csv)}\n"
                        f"Supplementation for missing templates will be skipped."
                    )

        logger.info(f"Validation successful for dataset '{dataset_name}'")


def merge_datasets(config: dict, output_dir: str) -> None:
    """Merge all generated individual datasets into a single combined dataset.

    Args:
        config: The loaded TOML configuration
        output_dir: Directory containing individual dataset CSV files
    """
    datasets = config.get("datasets", [])

    logger.info(f"Merging {len(datasets)} dataset(s) into a single file...")

    merged_df = pd.DataFrame()

    for dataset_config in datasets:
        dataset_name = dataset_config.get("name", "unknown")
        dataset_path = os.path.join(output_dir, f"{dataset_name}.csv")

        if not os.path.exists(dataset_path):
            logger.warning(f"Dataset file not found: {dataset_path}, skipping...")
            continue

        logger.info(f"Loading dataset: {dataset_name}")
        df = pd.read_csv(dataset_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
        logger.info(f"Added {len(df)} samples from {dataset_name}")

    if merged_df.empty:
        logger.warning("No datasets found to merge!")
        return

    # Save merged dataset
    output_path = config["general"].get("output_path", "dataset.csv")
    merged_path = os.path.join(output_dir, output_path)
    merged_df.to_csv(merged_path, index=False)

    logger.info(f"Merged dataset saved to: {merged_path}")
    logger.info(f"Total samples: {len(merged_df)}")
    logger.info(f"  - Normal samples: {len(merged_df[merged_df['label'] == 0])}")
    logger.info(f"  - Attack samples: {len(merged_df[merged_df['label'] == 1])}")


def main():
    args = init_args()
    init_logging(args.debug)
    config = init_toml_config(args)
    validate_datasets_config(config)

    datasets = config.get("datasets", [])

    # TODO: launcher should orchestrate parallel dataset generation natively
    for dataset_config in datasets:
        dataset_name = dataset_config.get("name", "unknown")
        logger.info(f"Building dataset: {dataset_name}")

        # Start an isolated MySQL instance for this dataset.
        start_mysql_server(config, dataset_name)

        # Create ONLY this dataset's database for isolation: sqlmap will only
        # see one schema during enumeration instead of all datasets' schemas.
        init_dataset_db(config, dataset_name)

        # We create a unified config for each dataset to be given to DatasetBuilder.
        # Override the port so the builder connects to this dataset's MySQL instance.
        mysql_cfg = {**config["mysql"], "port": get_dataset_port(config, dataset_name)}
        dataset_specific_config = {
            "general": config["general"],
            "mysql": mysql_cfg,
            "dataset": dataset_config,
        }

        db = DatasetBuilder(dataset_specific_config)

        if args.ithreat_only:
            db.df = db.generate_ithreat(args)
        else:
            db.build(args)

        db.save(args.output_dir)
        logger.info(f"Dataset {dataset_name} saved successfully")

        # Stop the MySQL instance and clean up all data.
        # No need to DROP DATABASE first — mysql-stop --clean removes
        # the entire datadir, and DROP DATABASE can hang on leftover locks.
        stop_mysql_server(config, dataset_name)

    # Merge all generated datasets into a single file
    merge_datasets(config, args.output_dir)


if __name__ == "__main__":
    main()
