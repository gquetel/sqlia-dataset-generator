import argparse
import configparser
import logging
from pathlib import Path
import sys

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
        "-ini",
        type=str,
        dest="ini",
        required=True,
        help="Filepath to the .ini configuration file.",
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


    return parser.parse_args()


def init_config(args: argparse.Namespace) -> configparser.ConfigParser:
    # Interpolation required to use values from other sections in the same file:
    # https://docs.python.org/3/library/configparser.html
    config = configparser.ConfigParser(interpolation=configparser.BasicInterpolation())
    config.read(args.ini)
    return config


def main():
    args = init_args()
    init_logging(args.debug)
    config = init_config(args)

    db = DatasetBuilder(config)
    db.build(args)
    db.save()


if __name__ == "__main__":
    main()
