import argparse
import logging
from pathlib import Path
import sys
import tomllib

from src.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)


def init_logging(debug_mode : bool):
    Path("./logs/").mkdir(exist_ok=True, parents=True)

    # We usually do not need the logs of previous generation
    # -> mode = 'w'
    lf = logging.FileHandler("./logs/generation.log", "w")

    logging_lvl = logging.DEBUG if debug_mode else logging.INFO
    lf.setLevel(logging_lvl)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(logging_lvl)
    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=logging_lvl, handlers=[lf, lstdo])


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
        help="Filepath to the dataset generation configuration file."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        dest="output_dir",
        default="./output/",
        help="Filepath to the directory that will contain all generated datasets."
    )


    return parser.parse_args()


def init_toml_config(args: argparse.Namespace) -> dict:
    """Load and parse the TOML configuration file."""
    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    return config


def validate_datasets_config(config: dict):
    """Validate that all dataset names and their statement CSV files are properly configured.

    Args:
        config: The loaded TOML configuration

    Raises:
        ValueError: If any dataset name doesn't have a corresponding folder,
                   or if any statement CSV file is missing or empty
    """
    datasets_dir = Path("data/datasets")

    if not datasets_dir.exists():
        raise ValueError(f"Dataset directory not found.")

    available_folders = {
        folder.name for folder in datasets_dir.iterdir()
        if folder.is_dir()
    }

    configured_datasets = config.get("datasets", [])

    # Check if any datasets are configured
    if not configured_datasets:
        raise ValueError("No datasets configured in configuration file")

    # Check for missing folders and validate statement CSV files
    for dataset_config in configured_datasets:
        dataset_name = dataset_config.get("name")

        # Check if dataset folder exists
        if dataset_name not in available_folders:
            raise ValueError(
                f"Dataset folder not found in {datasets_dir}: {dataset_name}\n"
                f"Available folders: {', '.join(sorted(available_folders)) if available_folders else '(none)'}"
            )

        # Validate statement CSV files
        statements = dataset_config.get("statements", {})
        if not statements:
            raise ValueError(f"No statements configured for dataset '{dataset_name}'")

        queries_dir = datasets_dir / dataset_name / "queries"
        if not queries_dir.exists():
            raise ValueError(f"Queries directory not found for dataset '{dataset_name}': {queries_dir}")

        for statement_name in statements.keys():
            csv_file = queries_dir / f"{statement_name}.csv"

            # Check if CSV file exists
            if not csv_file.exists():
                raise ValueError(
                    f"Statement CSV file not found for dataset '{dataset_name}': {csv_file}\n"
                    f"Expected file for statement '{statement_name}'"
                )

            # Check if CSV file has at least one entry (excluding header)
            try:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    if len(lines) < 2:  # Need at least header + one data row
                        raise ValueError(
                            f"Statement CSV file is empty or has no entries for dataset '{dataset_name}': {csv_file}\n"
                            f"File must contain at least one query template (excluding header row)"
                        )
            except Exception as e:
                if isinstance(e, ValueError):
                    raise
                raise ValueError(f"Error reading statement CSV file '{csv_file}': {e}")

def main():
    args = init_args()
    init_logging(args.debug)
    config = init_toml_config(args)
    validate_datasets_config(config)

    datasets = config.get("datasets", [])
    
    for dataset_config in datasets:
        dataset_name = dataset_config.get("name", "unknown")
        logger.info(f"Building dataset: {dataset_name}")

        # We create a unified config for each dataset to be given to DatasetBuilder.
        dataset_specific_config = {
            "general": config["general"],
            "mysql": config["mysql"],
            "dataset": dataset_config
        }

        db = DatasetBuilder(dataset_specific_config)

        if args.ithreat_only:
            db.build_ithreat(args)
        else:
            db.build(args)

        db.save(args.output_dir)
        logger.info(f"Dataset {dataset_name} saved successfully")
    # TODO: Function call that merges all generated datasets.

if __name__ == "__main__":
    main()
