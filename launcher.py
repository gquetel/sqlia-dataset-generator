import argparse
import configparser
from src.dataset_builder import DatasetBuilder


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
    config = configparser.ConfigParser(interpolation=configparser.BasicInterpolation())
    config.read(args.ini)

    return config


def main():
    args = init_args()
    config = init_config(args)

    db = DatasetBuilder(config)
    db.build()
    db.save()


if __name__ == "__main__":
    main()
