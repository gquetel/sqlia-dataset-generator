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


    return parser.parse_args()


def init_toml_config(args: argparse.Namespace) -> dict:
    """Load and parse the TOML configuration file."""
    with open(args.config_file, "rb") as f:
        config = tomllib.load(f)
    return config

def main():
    args = init_args()
    init_logging(args.debug)
    config = init_toml_config(args)

    datasets = config.get("datasets", [])

    if not datasets:
        logger.error("No datasets found in configuration file")
        return

    for dataset_config in datasets:
        dataset_name = dataset_config.get("name", "unknown")
        logger.info(f"Building dataset: {dataset_name}")

        # We create a unified config for each dataset to be given to DatasetBuilder.
        dataset_specific_config = {
            "general": config["general"],
            "mysql": config["mysql"],
            "datasets": [dataset_config]
        }

        db = DatasetBuilder(dataset_specific_config)

        if args.ithreat_only:
            db.build_ithreat(args)
        else:
            db.build(args)

        db.save()
        logger.info(f"Dataset {dataset_name} saved successfully")
    

if __name__ == "__main__":
    main()
