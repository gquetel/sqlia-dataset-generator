import argparse
import configparser
import logging
from pathlib import Path
import sys

from src.dataset_builder import DatasetBuilder

logger = logging.getLogger(__name__)

def init_logging():
    Path("./logs/").mkdir(exist_ok=True, parents=True)

    # We usually do not need the logs of previous generation
    # -> mode = 'w'
    lf = logging.FileHandler(
        "./logs/generation.log",
        'w'
    )
    lf.setLevel(logging.INFO)
    lstdo = logging.StreamHandler(sys.stdout)
    lstdo.setLevel(logging.INFO)

    lstdof = logging.Formatter(" %(message)s")
    lstdo.setFormatter(lstdof)
    logging.basicConfig(level=logging.INFO, handlers=[lf, lstdo])


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
    return parser.parse_args()


def init_config(args: argparse.Namespace) -> configparser.ConfigParser:
    # Interpolation required to use values from other sections in the same file:
    # https://docs.python.org/3/library/configparser.html
    config = configparser.ConfigParser(
        interpolation=configparser.BasicInterpolation()
    )
    config.read(args.ini)
    return config

def main():
    args = init_args()
    init_logging()
    config = init_config(args)

    db = DatasetBuilder(config)
    db.build()
    db.save()


if __name__ == "__main__":
    main()
