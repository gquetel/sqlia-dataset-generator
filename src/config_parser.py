import configparser
from fractions import Fraction


def get_mysql_info(config: configparser.ConfigParser):
    user = config.get("MYSQL", "user")
    pwd = config.get("MYSQL", "password")
    host = config.get("MYSQL", "host")
    port = config.get("MYSQL", "port")
    priv_user = config.get("MYSQL", "priv_user")
    priv_pwd = config.get("MYSQL", "priv_pwd")
    return user, pwd, host, port, priv_user, priv_pwd


def get_seed(config: configparser.ConfigParser):
    return int(config.get("GENERAL", "seed"))


def get_attacks_ratio(config: configparser.ConfigParser):
    return float(Fraction(config.get("GENERAL", "attacks_ratio")))


def get_output_path(config: configparser.ConfigParser):
    return config.get("GENERAL", "output_path")


def get_used_databases(config: configparser.ConfigParser):
    return config.get("GENERAL", "databases").split()


def get_statement_types_and_proportions(config: configparser.ConfigParser):
    stmts = []

    for section in config.sections():
        if section == "NORMAL_TRAFFIC_TARGETS":
            for key, value in config.items(section):
                stmts.append({"type": key, "proportion": float(Fraction(value))})

    if abs(sum([stmt["proportion"] for stmt in stmts]) - 1.0) > 1e-10:
        raise ValueError(
            f"Proportions of queries types must sum up to 1. Current is {sum([stmt['proportion'] for stmt in stmts])}"
        )

    return stmts
